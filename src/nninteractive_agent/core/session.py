"""Creating an nnInteractive session: remote (nninteractive-server) or local (in-process)."""

from __future__ import annotations

import os

LOCAL_HINT = (
    "local inference needs the nnInteractive package with PyTorch: pip install 'nninteractive-agent[local]' "
    "(or point --server / $NNINTERACTIVE_SERVER_URL at an nninteractive-server)"
)


def resolve_spec(spec: str | None = None) -> str:
    """``spec``, else ``$NNINTERACTIVE_SERVER_URL``, else ``local``."""
    return spec or os.environ.get("NNINTERACTIVE_SERVER_URL") or "local"


def model_license(session) -> str:
    """The license of the nnInteractive model behind a local or remote session, as nnInteractive
    reports it (the first line of the model folder's LICENSE file)."""
    lic = getattr(session, "license", None)
    if not lic or lic == "!!MISSING!!":
        return "unknown"  # the model folder has no LICENSE; check where the weights came from
    return lic


def _license_file(folder) -> str:
    f = os.path.join(folder, "LICENSE")
    if os.path.isfile(f):
        with open(f, encoding="utf-8") as fh:
            line = next((x.strip() for x in fh if x.strip()), "")
        if line:
            return line
    return "unknown until the model is loaded"


def probe(spec: str | None = None) -> dict:
    """Check the nnInteractive backend without an image and report its model license. Local: the
    weights (downloaded if needed). Server: a short session that is released again."""
    spec = resolve_spec(spec)
    if spec.startswith(("http://", "https://")):
        session = _make_remote(spec, wait_seconds=30.0)
        try:
            return {"spec": spec, "where": spec, "license": model_license(session)}
        finally:
            session.close()
    try:
        from nnInteractive.model_management import ensure_model_available, get_default_model_id
    except ImportError as e:
        raise RuntimeError(f"{LOCAL_HINT} ({e})") from e
    model_id = spec.partition(":")[2] or get_default_model_id()
    folder = ensure_model_available(model_id)
    return {"spec": spec, "where": f"{model_id} in {folder}", "license": _license_file(folder)}


class Reconnecting:
    """A session that survives the server dropping it (idle timeout, server restart): on the server's
    ``SessionExpiredError`` it calls ``restore()`` for a new, fully restored session and repeats the
    call once. Attributes and ``close`` pass straight through."""

    def __init__(self, session, restore):
        self._session, self._restore = session, restore

    def __getattr__(self, name):
        attr = getattr(self._session, name)
        if name == "close" or not callable(attr):
            return attr

        def call(*args, **kwargs):
            try:
                return getattr(self._session, name)(*args, **kwargs)
            except Exception as e:
                if type(e).__name__ != "SessionExpiredError":
                    raise
                old, self._session = self._session, self._restore()
                try:  # the lease is gone; this stops the old client's heartbeat and connection
                    old.close()
                except Exception:  # noqa: BLE001, S110 - best effort
                    pass
                return getattr(self._session, name)(*args, **kwargs)

        return call


def make_session(spec: str | None = None):
    """Return a session object with the nnInteractiveInferenceSession API.

    ``spec`` is a server URL (``http://host:port``) for a remote session, or ``local`` /
    ``local:<model_id>`` for in-process inference (needs the ``[local]`` extra; a GPU is strongly
    recommended). Defaults to ``$NNINTERACTIVE_SERVER_URL``, then ``local``.
    """
    spec = resolve_spec(spec)
    if spec.startswith(("http://", "https://")):
        return _make_remote(spec)
    if spec == "local" or spec.startswith("local:"):
        return _make_local(spec.partition(":")[2] or None)
    raise ValueError(f"session spec must be a server URL or 'local[:model_id]', got {spec!r}")


def _make_remote(url: str, wait_seconds: float = 120.0):
    """Claim a session, waiting for a free slot if the (shared) server is full."""
    import time

    from nnInteractive.inference.remote import ServerAtCapacityError, nnInteractiveRemoteInferenceSession

    deadline, delay = time.time() + wait_seconds, 5.0
    while True:
        try:
            # Agent turns can take minutes between calls; give slow predictions room as well.
            return nnInteractiveRemoteInferenceSession(url, read_timeout=180.0)
        except ServerAtCapacityError as e:
            if time.time() + delay > deadline:
                raise RuntimeError(
                    f"the nnInteractive server {url} is full (all sessions in use) and no slot became free within "
                    f"{wait_seconds:.0f} s; try again later"
                ) from e
            time.sleep(delay)
            delay = min(delay * 1.5, 20.0)


def _make_local(model_id: str | None):
    try:
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        from nnInteractive.model_management import ensure_model_available, get_default_model_id
    except ImportError as e:
        raise RuntimeError(f"{LOCAL_HINT} ({e})") from e

    folder = ensure_model_available(model_id or get_default_model_id())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    session = nnInteractiveInferenceSession(
        device=device, use_torch_compile=False, torch_n_threads=os.cpu_count() or 8, verbose=False
    )
    session.initialize_from_trained_model_folder(folder, 0, "checkpoint_final.pth")
    return session
