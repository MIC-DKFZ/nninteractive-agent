"""Password-protected web demo of the agent: chat on the right, 2D slices and 3D surfaces on the left.

    nninteractive-agent-web [--examples DIR] [--server URL|local] [--host 127.0.0.1] [--port 8443]
                            [--llm claude|claude-api|codex|codex-api|server] [--model M] [--effort E]
                            [--python full|guarded|off]

A demonstrator of the nninteractive agent tools. Each logged-in browser gets its own image, workspace
and agent conversation. By default it serves this machine only (http://127.0.0.1); with --host 0.0.0.0
it is reachable from the network over HTTPS with a password. The agent's LLM:
  claude      Claude on the subscription (Claude Code login) of the account that starts the server
  claude-api  Claude with an Anthropic API key ($ANTHROPIC_API_KEY or --api-key-file)
  codex       Codex on the ChatGPT subscription (`codex login`) of that account
  codex-api   Codex with an OpenAI API key ($OPENAI_API_KEY or --api-key-file)
  server      a model on an Anthropic-compatible server such as vLLM, on this machine (default
              http://localhost:8000) or another (--llm-url); --model; a key, if it needs one, in
              $LLM_API_KEY or --api-key-file
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ..core.pyguard import PYTHON_MODES, describe, resolve_mode
from ..core.session import probe, resolve_spec
from .auth import COOKIE, COOKIE_TTL, Auth, ensure_certificate, host_names, load_password
from .llm import BACKENDS, DEFAULT_LLM_URL, EFFORT_ORDER, LLMError, make_backend
from .session import Client, SessionManager, UserError

WEB_DIR = Path(__file__).parent
STATIC = WEB_DIR / "static"
MAX_UPLOAD = 2 * 1024**3
VOLUME_FILES = ("image.nii.gz", "image3d.nii.gz", "zeros_uint8.nii.gz", "zeros_uint16.nii.gz")
UPLOAD_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
EXAMPLE_SUFFIXES = UPLOAD_SUFFIXES + (".npy",)
_SUFFIXES_2D = (".png", ".jpg", ".jpeg")
# Variables of an enclosing Claude Code session that must not leak into the agent's Claude Code.
_PARENT_SESSION_VARS = ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT")


def load_examples(path: Path | None) -> list:
    """The example images on the start page.

    ``path`` is a directory, where every image is an example, or a JSON file listing examples. Both use
    the same format, ``{"cases": [{"path", "title", "subtitle", "modality", "tasks", "id"}]}`` (all but
    ``path`` optional; relative paths are resolved against the JSON file's directory, ``$VARS`` are
    expanded). In a directory, an ``examples.json`` with that content describes some or all of its images.
    """
    if path is None:
        return []
    path = Path(path).expanduser()
    if path.is_dir():
        cases = _read_cases(path / "examples.json") if (path / "examples.json").is_file() else []
        listed = {c["file"].resolve() for c in cases}
        for f in sorted(path.iterdir()):
            if f.is_file() and f.name.lower().endswith(EXAMPLE_SUFFIXES) and f.resolve() not in listed:
                cases.append(_case({"path": f.name}, f))
    elif path.is_file():
        cases = _read_cases(path)
    else:
        raise FileNotFoundError(f"--examples {path}: not a directory or JSON file")
    seen: set = set()
    for c in cases:
        base = c["id"]
        n = 2
        while c["id"] in seen:
            c["id"], n = f"{base}-{n}", n + 1
        seen.add(c["id"])
    return cases


def _read_cases(f: Path) -> list:
    out = []
    for c in json.loads(f.read_text())["cases"]:
        p = Path(os.path.expandvars(c["path"])).expanduser()
        out.append(_case(c, p if p.is_absolute() else f.parent / p))
    return out


def _case(c: dict, file: Path) -> dict:
    """A case with defaults for what it does not say; ``auto`` lists the fields the thumbnail pass
    may fill in once it has looked at the image."""
    name = file.name
    stem = next((name[: -len(x)] for x in sorted(EXAMPLE_SUFFIXES, key=len, reverse=True) if name.lower().endswith(x)),
                name)  # fmt: skip
    auto = [k for k in ("subtitle", "modality") if k not in c]
    c = {"title": re.sub(r"[_-]+", " ", stem).strip() or name, "subtitle": name,
         "modality": "2D" if name.lower().endswith(_SUFFIXES_2D) else "", "tasks": [], **c}  # fmt: skip
    c["id"] = _safe(str(c.get("id") or stem))
    c["file"], c["available"], c["auto"] = file, file.is_file(), auto
    if c["available"]:
        st = file.stat()
        c["thumb_key"] = hashlib.sha1(f"{file.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:16]
    return c


def create_app(
    manager: SessionManager, auth: Auth, secure_cookie: bool = True, thumbs_dir: Path | None = None
) -> FastAPI:
    thumbs_dir = thumbs_dir or manager.root / "thumbs"

    @contextlib.asynccontextmanager
    async def lifespan(app):
        reaper = asyncio.create_task(manager.reaper())
        thumbs = asyncio.create_task(asyncio.to_thread(_make_thumbnails, manager.cases, thumbs_dir))
        yield
        reaper.cancel()
        thumbs.cancel()
        await manager.shutdown()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def limit_upload(request: Request, call_next):
        # The multipart parser spools the whole body before the handler runs: refuse early.
        if (
            request.url.path == "/api/upload"
            and int(request.headers.get("content-length") or 0) > MAX_UPLOAD + 2**20
        ):
            return Response("The file is larger than 2 GB.", status_code=413)
        return await call_next(request)

    def session_of(request: Request):
        sid = auth.session_id(request.cookies.get(COOKIE))
        if sid is None:
            raise HTTPException(401, "login required")
        s = manager.get(sid)
        s.touch()
        return s

    # ------------------------------------------------------------------ pages
    @app.get("/")
    async def index(request: Request, key: str | None = None):
        if key is not None:  # the login link printed at startup
            return await _login(request, key)
        if auth.session_id(request.cookies.get(COOKIE)) is None:
            return RedirectResponse("login", status_code=303)
        return HTMLResponse((STATIC / "index.html").read_text(), headers={"Cache-Control": "no-store"})

    @app.get("/login")
    async def login_page():
        return HTMLResponse((STATIC / "login.html").read_text(), headers={"Cache-Control": "no-store"})

    @app.post("/login")
    async def login(request: Request, password: str = Form("")):
        return await _login(request, password)

    async def _login(request: Request, password: str):
        client = request.client.host if request.client else "?"
        if auth.locked_for(client) > 0:
            return RedirectResponse("login?error=locked", status_code=303)
        if not auth.check_password(password, client):
            await asyncio.sleep(1.0)
            return RedirectResponse(
                "login?error=" + ("locked" if auth.locked_for(client) > 0 else "wrong"), status_code=303
            )
        cookie, _ = auth.make_cookie()
        resp = RedirectResponse("./", status_code=303)
        resp.set_cookie(
            COOKIE, cookie, max_age=COOKIE_TTL, httponly=True, secure=secure_cookie, samesite="strict"
        )
        return resp

    @app.post("/logout")
    async def logout(request: Request):
        sid = auth.session_id(request.cookies.get(COOKIE))
        if sid is not None:
            await manager.drop(sid)
        resp = RedirectResponse("login", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    # -------------------------------------------------------------------- api
    @app.get("/api/cases/{case_id}/thumb.png")
    async def case_thumb(case_id: str, request: Request):
        session_of(request)
        case = next((c for c in manager.cases if c["id"] == case_id and c["available"]), None)
        f = thumbs_dir / f"{case['thumb_key']}.png" if case else None
        if f is None or not f.exists():
            raise HTTPException(404)
        return FileResponse(f, headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/open")
    async def open_case(request: Request):
        s = session_of(request)
        body = await request.json()
        case = next((c for c in manager.cases if c["id"] == body.get("case") and c["available"]), None)
        if case is None:
            raise HTTPException(404, "unknown case")
        try:
            await s.open_image(case["file"], case["title"], case.get("tasks"))
        except UserError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile = File(...)):
        s = session_of(request)
        name = _safe(Path(file.filename or "image").name)
        if not name.lower().endswith(UPLOAD_SUFFIXES):
            raise HTTPException(415, "Unsupported file type. Use NIfTI, NRRD, MHA, TIFF or PNG/JPG.")
        d = s.dir / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{time.strftime('%H%M%S')}_{name}"
        size = 0
        f = await asyncio.to_thread(open, dest, "wb")
        try:
            while chunk := await file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "The file is larger than 2 GB.")
                await asyncio.to_thread(f.write, chunk)
        except HTTPException:
            f.close()
            dest.unlink(missing_ok=True)
            raise
        finally:
            f.close()
        try:
            await s.open_image(dest, name)
        except UserError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError, RuntimeError) as e:
            raise HTTPException(422, f"Could not read the image: {e}") from e
        return {"ok": True}

    @app.get("/api/volume/{key}/{name}")
    async def volume(key: str, name: str, request: Request):
        s = session_of(request)
        if s.image_meta is None or s.image_meta["key"] != key or name not in VOLUME_FILES:
            raise HTTPException(404)
        return FileResponse(manager.volumes_dir / key / name, media_type="application/octet-stream",
                            headers={"Cache-Control": "private, max-age=86400"})  # fmt: skip

    @app.get("/api/mesh/{key}")
    async def mesh(key: str, request: Request):
        s = session_of(request)
        m = s.meshes.get(key)
        if m is None:  # replaced or removed since the page was told about it
            return Response(status_code=204)
        return Response(
            m["blob"], media_type="application/octet-stream", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/img/{name}")
    async def agent_image(name: str, request: Request, run: str | None = None):
        s = session_of(request)
        f = s.dir / "images" / name
        if not re.fullmatch(r"\d{5}\.png", name) or not f.is_file() or (run is not None and run != s.dir.name):
            raise HTTPException(404)  # a URL from another run of this browser session: never its image
        return FileResponse(f, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/download")
    async def download(request: Request):
        s = session_of(request)
        try:
            z = await asyncio.to_thread(s.download_zip)
        except UserError as e:
            raise HTTPException(409, str(e)) from e
        return FileResponse(z, media_type="application/zip", filename=z.name)

    @app.websocket("/api/ws")
    async def websocket(ws: WebSocket):
        sid = auth.session_id(ws.cookies.get(COOKIE))
        origin, host = ws.headers.get("origin", ""), ws.headers.get("host", "")
        if sid is None or (origin and re.sub(r"^https?://", "", origin) != host):
            await ws.close(code=4401)
            return
        await ws.accept()
        s = manager.get(sid)
        client = Client(ws)
        pump = asyncio.create_task(client.pump())
        try:
            await s.connect(client)
            while True:
                msg = await ws.receive_json()
                s.touch()
                try:
                    await _command(s, msg)
                except UserError as e:
                    client.put({"t": "notice", "text": str(e)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.disconnect(client)
            pump.cancel()

    return app


async def _command(s, msg: dict) -> None:
    cmd = msg.get("cmd")
    if cmd == "send":
        await s.send(str(msg.get("text", ""))[:20000])
    elif cmd == "stop":
        await s.stop()
    elif cmd == "new_conversation":
        await s.new_conversation()
    elif cmd == "settings":
        await s.settings(msg.get("model"), msg.get("effort"))
    elif cmd == "ping":
        pass


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120] or "file"


def _make_thumbnails(cases: list, out: Path) -> None:
    """A preview per example, cached by file; also fills in the subtitle (size) and modality where the
    examples do not give them."""
    from ..core.io import load_image
    from ..core.workspace import _looks_like_ct
    from .thumbs import thumbnail_png

    out.mkdir(parents=True, exist_ok=True)
    for c in [c for c in cases if c["available"]]:
        png, meta = out / f"{c['thumb_key']}.png", out / f"{c['thumb_key']}.json"
        try:
            if not (png.exists() and meta.exists()):
                img = load_image(c["file"])
                shape = img.data.shape[1:] if img.is_2d else img.data.shape
                info = {"subtitle": f"{c['file'].name} · {' × '.join(str(n) for n in shape)}",
                        "modality": "CT" if _looks_like_ct(img.data) else ("2D" if img.is_2d else "3D")}  # fmt: skip
                png.write_bytes(thumbnail_png(img))
                meta.write_text(json.dumps(info))
            info = json.loads(meta.read_text())
            c.update({k: info[k] for k in c.get("auto", ()) if k in info})
        except Exception as e:  # noqa: BLE001 - a missing thumbnail is cosmetic
            print(f"[web] no thumbnail for {c['file']}: {e}", file=sys.stderr)


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, default=None, help="directory of example images for the start page "
                    "(an examples.json there can add titles and suggested tasks), or a JSON file listing them")  # fmt: skip
    ap.add_argument("--server", default=None, help="nninteractive-server URL or 'local' (default: "
                    "$NNINTERACTIVE_SERVER_URL, else local inference on this machine)")  # fmt: skip
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1: this machine only, plain HTTP (default); "
                    "0.0.0.0: reachable from the network, HTTPS with a password")  # fmt: skip
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--max-sessions", type=int, default=None, help="browsers that can work at the same time "
                    "(default: 1 with local inference, which loads one model per browser; else 10)")  # fmt: skip
    ap.add_argument("--idle-minutes", type=int, default=30, help="pause a session (frees its nnInteractive slot) "
                    "after this much inactivity")  # fmt: skip
    ap.add_argument("--llm", default="claude", choices=BACKENDS, help="the agent's LLM and how it signs in "
                    "(default: claude, the Claude subscription)")  # fmt: skip
    ap.add_argument("--llm-url", default=None, help="the Anthropic-compatible server for --llm server, on this "
                    f"machine or another (default: {DEFAULT_LLM_URL}, vLLM's default port)")  # fmt: skip
    ap.add_argument("--model", default=None, help="default model (claude: opus or sonnet; codex: Codex's "
                    "default; server: required if the server has several)")  # fmt: skip
    ap.add_argument("--effort", default="high", choices=EFFORT_ORDER, help="default effort (the nearest one "
                    "the model takes)")  # fmt: skip
    ap.add_argument("--api-key-file", type=Path, default=None, help="file with the API key for claude-api, "
                    "codex-api or server")  # fmt: skip
    ap.add_argument("--python", choices=PYTHON_MODES, default=None, help="the agent's python tool: full "
                    "(default), guarded (computations only) or off; see the README's Security section")  # fmt: skip
    ap.add_argument("--password-file", type=Path, default=None)
    ap.add_argument("--certfile", type=Path, default=None, help="TLS certificate (default: self-signed)")
    ap.add_argument("--keyfile", type=Path, default=None)
    ap.add_argument(
        "--insecure-http", action="store_true", help="serve plain HTTP on the network (only behind TLS termination)"
    )
    args = ap.parse_args(argv)

    import uvicorn

    for k in [k for k in os.environ if k in _PARENT_SESSION_VARS or k.startswith("CLAUDE_CODE_")]:
        del os.environ[k]
    try:
        llm = make_backend(args.llm, args.model, args.effort, args.llm_url, args.api_key_file)
    except (LLMError, OSError) as e:
        sys.exit(f"nninteractive-agent-web: {e}")
    llm.prepare_environment()
    try:
        examples = load_examples(args.examples)
        python_mode = resolve_mode(args.python)
    except (OSError, ValueError) as e:
        sys.exit(f"nninteractive-agent-web: {e}")
    password, pw_file, created = load_password(args.password_file)
    auth = Auth(password)
    spec = resolve_spec(args.server)
    local = spec.startswith("local")
    label = "local GPU" if local else re.sub(r"^https?://", "", spec).split("/")[0]
    max_sessions = args.max_sessions or (1 if local else 10)
    loopback = _is_loopback(args.host)
    plain_http = loopback or args.insecure_http
    manager = SessionManager(spec, label, examples, max_sessions, args.idle_minutes * 60, llm, python_mode=python_mode)
    print(f"nnInteractive: {'getting the weights' if local else f'connecting to {spec}'} ...", file=sys.stderr, flush=True)
    try:
        backend = probe(spec)
        manager.model_license = backend["license"]
    except Exception as e:  # noqa: BLE001 - the first session tries again and shows the error on the page
        if local and isinstance(e, RuntimeError) and "pip install" in str(e):
            sys.exit(f"nninteractive-agent-web: {e}")
        backend = {"where": f"{spec} (not reachable now: {type(e).__name__}: {e})", "license": "unknown"}
    app = create_app(manager, auth, secure_cookie=not plain_http)

    ssl = {}
    if not plain_http:
        cert, key = ensure_certificate(args.certfile, args.keyfile)
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    scheme = "http" if plain_http else "https"
    print("\nnnInteractive agent web demo", file=sys.stderr)
    if loopback:
        print(f"  open {scheme}://127.0.0.1:{args.port}/?key={password}  (this machine only)", file=sys.stderr)
    else:
        names, ips = host_names()
        for n in [n for n in names if "." in n] + [i for i in ips if i != "127.0.0.1"]:
            print(f"  {scheme}://{n}:{args.port}/", file=sys.stderr)
        if pw_file:
            print(f"  password: {'new, ' if created else ''}in {pw_file}" + (f"  ->  {password}" if created else ""),
                  file=sys.stderr)  # fmt: skip
    why = " (each loads its own model; --max-sessions to change)" if local and args.max_sessions is None else ""
    print(f"  nnInteractive: {backend['where']}; up to {max_sessions} session{'s' if max_sessions != 1 else ''}{why}",
          file=sys.stderr)  # fmt: skip
    print(f"  nnInteractive model license: {backend['license']}", file=sys.stderr)
    print(f"  examples: {len(examples)}" + (f" from {args.examples}" if args.examples else " (--examples DIR adds some)"),
          file=sys.stderr)  # fmt: skip
    subagents = "; verifier subagents" if llm.subagents else ""
    print(f"  agent: {llm.label}; model {llm.model}, effort {llm.effort}{subagents}", file=sys.stderr)
    if llm.name in ("claude", "codex") and not loopback:
        print("  WARNING: other people who log in use your personal subscription. Anthropic's and OpenAI's terms "
              "allow a subscription only for your own use; to host the demo for others, use --llm server, "
              "codex-api or claude-api (README, 'Web demo').", file=sys.stderr)  # fmt: skip
    print(f"  {describe(python_mode)}", file=sys.stderr)
    if python_mode == "full" and not loopback:
        print("  WARNING: anyone who can log in can make the agent run code on this machine as this user. "
              "Consider --python guarded, or run the demo in a container or VM.", file=sys.stderr)  # fmt: skip
    print(file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws_max_size=1 << 20, **ssl)


if __name__ == "__main__":
    main()
