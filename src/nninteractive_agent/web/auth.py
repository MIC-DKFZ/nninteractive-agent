"""Login for the web demo: one shared password, signed session cookies, a self-signed certificate.

The password and the cookie-signing secret live in ``~/.config/nninteractive_agent/`` (mode 600).
Without a password file a random password is generated on first start and printed.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import secrets
import socket
import time
from pathlib import Path

CONFIG_DIR = Path(
    os.environ.get("NNINTERACTIVE_AGENT_CONFIG", Path.home() / ".config" / "nninteractive_agent")
)
COOKIE = "nni_session"
COOKIE_TTL = 12 * 3600
MAX_FAILURES = 5  # per client address before a lockout
LOCKOUT_S = 300


def _private_file(path: Path, make) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(make() + "\n")
    return path.read_text().strip()


def load_password(path: Path | None = None) -> tuple:
    """(password, path, created): from $NNINTERACTIVE_WEB_PASSWORD, else the password file."""
    if os.environ.get("NNINTERACTIVE_WEB_PASSWORD"):
        return os.environ["NNINTERACTIVE_WEB_PASSWORD"], None, False
    path = path or CONFIG_DIR / "web_password"
    created = not path.exists()
    password = _private_file(path, lambda: secrets.token_urlsafe(12))
    if not password:
        raise ValueError(f"{path} is empty")
    return password, path, created


class Auth:
    def __init__(self, password: str, secret: str | None = None):
        self._pw_hash = hashlib.sha256(password.encode()).digest()
        self._secret = (
            secret or _private_file(CONFIG_DIR / "web_secret", lambda: secrets.token_hex(32))
        ).encode()
        self._failures: dict = {}  # client -> (count, first failure time, locked until)

    # ------------------------------------------------------------------ password
    def locked_for(self, client: str) -> float:
        until = self._failures.get(client, (0, 0.0, 0.0))[2]
        return max(0.0, until - time.time())

    def check_password(self, password: str, client: str) -> bool:
        ok = hmac.compare_digest(hashlib.sha256(password.encode()).digest(), self._pw_hash)
        if ok:
            self._failures.pop(client, None)
            return True
        count, first, _ = self._failures.get(client, (0, time.time(), 0.0))
        if time.time() - first > LOCKOUT_S:
            count, first = 0, time.time()
        count += 1
        until = time.time() + LOCKOUT_S if count >= MAX_FAILURES else 0.0
        self._failures[client] = (count, first, until)
        return False

    # ------------------------------------------------------------------- cookies
    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()

    def make_cookie(self, sid: str | None = None) -> tuple:
        """(cookie value, session id): a new session unless ``sid`` is given."""
        sid = sid or secrets.token_urlsafe(16)
        payload = f"{sid}.{int(time.time()) + COOKIE_TTL}"
        return f"{payload}.{self._sign(payload)}", sid

    def session_id(self, cookie: str | None) -> str | None:
        """The session id of a valid, unexpired cookie, else None."""
        if not cookie or cookie.count(".") != 2:
            return None
        sid, expires, sig = cookie.split(".")
        if not hmac.compare_digest(sig, self._sign(f"{sid}.{expires}")):
            return None
        if not expires.isdigit() or int(expires) < time.time():
            return None
        return sid


# ------------------------------------------------------------------------- TLS
def host_names() -> tuple:
    """(DNS names, IP addresses) this machine is reachable under."""
    names = {"localhost", socket.gethostname(), socket.getfqdn()}
    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.getfqdn(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:  # the address used for outgoing traffic (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(n for n in names if n), sorted(ips)


def ensure_certificate(cert: Path | None = None, key: Path | None = None) -> tuple:
    """Paths of a TLS certificate and key; creates a self-signed pair for this host if needed."""
    if cert and key:
        return cert, key
    tls = CONFIG_DIR / "tls"
    cert, key = tls / "cert.pem", tls / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    names, ips = host_names()
    pkey = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.getfqdn()[:64])])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    crt = (
        x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(pkey.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=730))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(pkey, hashes.SHA256())
    )  # fmt: skip
    tls.mkdir(parents=True, exist_ok=True)
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pkey.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))  # fmt: skip
    cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    return cert, key
