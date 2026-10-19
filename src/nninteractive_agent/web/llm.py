"""Which LLM drives the agent and how it signs in, chosen when a front end starts.

    claude      Claude Code on the Claude subscription (login) of the account running the host
    claude-api  Claude Code with an Anthropic API key
    codex       Codex on the ChatGPT subscription (login) of that account
    codex-api   Codex with an OpenAI API key
    server      Claude Code on an Anthropic-compatible server: open models, e.g. vLLM on this machine
                (the default URL, http://localhost:8000) or on another one

``make_backend`` checks the choice at startup (login present, key accepted, server and model
reachable, which efforts the model takes), so a demo does not fail at its first message. The
``Backend`` it returns lists the models and efforts for the page and creates agent sessions.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .agent_sdk import AgentSession

BACKENDS = ("claude", "claude-api", "codex", "codex-api", "server")
DEFAULT_LLM_URL = "http://localhost:8000"  # vLLM's default port
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
CLAUDE_EFFORTS = ["low", "medium", "high", "xhigh"]
CLAUDE_MODELS = (("opus", "Opus"), ("sonnet", "Sonnet"))
# Codex keeps the API key of codex-api here, away from the ChatGPT login in ~/.codex.
CODEX_API_HOME = Path.home() / ".cache" / "nninteractive_agent" / "codex-api"
# Variables that make Claude Code use something other than the subscription login.
_ANTHROPIC_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


class LLMError(RuntimeError):
    """The chosen LLM cannot be used; the message says why and what to do."""


@dataclass
class Backend:
    name: str  # one of BACKENDS
    label: str  # shown in the page
    models: list  # [{"id", "label", "efforts"}]
    model: str  # default model
    effort: str  # default effort
    env: dict = field(default_factory=dict, repr=False)  # for the agent process; may hold a key
    unset: tuple = ()  # removed from this process's environment (the Claude SDK passes it on)
    codex_home: Path | None = None
    subagents: bool = False  # Codex: verifier subagents (needs the multi_agent_v2_dynamic_tools feature)

    @property
    def engine(self) -> str:
        return "codex" if self.name.startswith("codex") else "claude"

    def efforts(self, model: str) -> list:
        return next((list(m["efforts"]) for m in self.models if m["id"] == model), [])

    def effort_for(self, model: str, wanted: str) -> str:
        """``wanted`` if the model takes it, else the nearest effort it takes (the lower one on a tie)."""
        have = self.efforts(model)
        if wanted in have or not have:
            return wanted
        rank = EFFORT_ORDER.index(wanted) if wanted in EFFORT_ORDER else EFFORT_ORDER.index("high")
        return min(have, key=lambda e: (abs(EFFORT_ORDER.index(e) - rank), EFFORT_ORDER.index(e)))

    def describe(self) -> dict:
        """What the page shows and offers."""
        return {"name": self.name, "label": self.label, "models": self.models}

    def prepare_environment(self) -> None:
        for k in self.unset:
            os.environ.pop(k, None)

    def session(self, ws, on_event, model: str | None = None, effort: str | None = None,
                host_prompt: str = ""):  # fmt: skip
        """A new agent conversation (``AgentSession`` or ``CodexSession``) about ``ws``."""
        model = model or self.model
        effort = self.effort_for(model, effort or self.effort)
        if self.engine == "codex":
            from .agent_codex import CodexSession

            return CodexSession(ws, on_event, model=model, effort=effort, host_prompt=host_prompt,
                                codex_home=self.codex_home, subagents=self.subagents)  # fmt: skip
        return AgentSession(ws, on_event, model=model, effort=effort, host_prompt=host_prompt, env=self.env,
                            report_cost=self.name != "server")  # fmt: skip


def claude_backend(model: str = "opus", effort: str = "high") -> Backend:
    """Claude Code on the subscription login, unchecked (also what tests and older callers get)."""
    return Backend("claude", "Claude (subscription)", _claude_models(), model, effort, unset=_ANTHROPIC_VARS)


def make_backend(name: str = "claude", model: str | None = None, effort: str | None = None,
                 url: str | None = None, api_key_file: Path | None = None) -> Backend:  # fmt: skip
    """The backend ``name`` with its defaults, after checking that it works. Raises LLMError."""
    effort = effort or "high"
    if name == "claude":
        _check_claude_login()
        b = claude_backend(_pick(model or "opus", [m for m, _ in CLAUDE_MODELS]), effort)
    elif name == "claude-api":
        key = _read_key(api_key_file, "ANTHROPIC_API_KEY")
        _check_key("https://api.anthropic.com/v1/models", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                   "Anthropic")  # fmt: skip
        b = Backend(name, "Claude (API key)", _claude_models(), _pick(model or "opus", [m for m, _ in CLAUDE_MODELS]),
                    effort, env={"ANTHROPIC_API_KEY": key}, unset=_ANTHROPIC_VARS)  # fmt: skip
    elif name == "server":
        b = _server_backend(url or DEFAULT_LLM_URL, model, effort, _read_key(api_key_file, "LLM_API_KEY", False))
    elif name in ("codex", "codex-api"):
        b = _codex_backend(name, model, effort, api_key_file)
    else:
        raise LLMError(f"unknown LLM {name!r}; choose one of {', '.join(BACKENDS)}")
    b.effort = b.effort_for(b.model, b.effort)
    return b


# --------------------------------------------------------------------------- Claude Code
def _claude_models() -> list:
    return [{"id": m, "label": label, "efforts": CLAUDE_EFFORTS} for m, label in CLAUDE_MODELS]


def _check_claude_login() -> None:
    exe = shutil.which("claude")
    if exe is None:  # the SDK's bundled CLI then finds the login on its own
        return
    env = {k: v for k, v in os.environ.items() if k not in _ANTHROPIC_VARS}
    try:
        out = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, text=True, timeout=30, env=env,
                             check=False)  # fmt: skip
        status = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return  # an older CLI without `auth status`: the first message will tell
    if not status.get("loggedIn"):
        raise LLMError(
            "Claude Code is not logged in: run `claude` once and log in with the subscription account"
        )


def _server_backend(url: str, model: str | None, effort: str, key: str | None) -> Backend:
    """Claude Code on an Anthropic-compatible server. Claude Code sends the effort, and servers
    take different ones (vLLM with Qwen: low, medium, xhigh), so each is tried once."""
    url = url.rstrip("/")
    host = urllib.parse.urlparse(url).hostname or url
    for var in ("NO_PROXY", "no_proxy"):  # internal hosts are often not reachable through a proxy
        os.environ[var] = ",".join(x for x in (os.environ.get(var), host) if x)
    key = key or "none"
    headers = {"x-api-key": key, "authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
    status, body = _http(f"{url}/v1/models", headers)
    if status != 200:
        raise LLMError(f"{url}/v1/models answered {status}: {str(body)[:300]}")
    ids = [m.get("id") for m in (body.get("data") or []) if m.get("id")]
    if model is None and len(ids) == 1:
        model = ids[0]
    if model not in ids:
        raise LLMError(f"choose a model of {url} with --model: {', '.join(ids) or 'none listed'}")
    efforts = []
    for e in CLAUDE_EFFORTS:
        payload = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}],
                   "output_config": {"effort": e}}  # fmt: skip
        status, body = _http(f"{url}/v1/messages", headers, payload)
        if status == 200:
            efforts.append(e)
        elif status in (404, 405):
            raise LLMError(f"{url} has no Anthropic-compatible /v1/messages endpoint")
        elif not (status == 400 and "effort" in str(body).lower()):
            raise LLMError(f"{url}/v1/messages answered {status} for {model}: {str(body)[:300]}")
    if not efforts:
        raise LLMError(f"{model} on {url} takes none of the efforts {', '.join(CLAUDE_EFFORTS)}")
    env = {
        "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": key,
        # Subagents and Claude Code's own small requests use the same model.
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ENABLE_TOOL_SEARCH": "false", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }  # fmt: skip
    models = [{"id": model, "label": model.split("/")[-1], "efforts": efforts}]
    return Backend(
        "server", f"{model} on {host}", models, model, effort, env=env, unset=("ANTHROPIC_AUTH_TOKEN",)
    )


# --------------------------------------------------------------------------- Codex
def _codex_backend(name: str, model: str | None, effort: str, api_key_file: Path | None) -> Backend:
    from .agent_codex import probe

    if shutil.which("codex") is None:
        raise LLMError("codex is not installed (npm install -g @openai/codex)")
    key = _read_key(api_key_file, "OPENAI_API_KEY") if name == "codex-api" else None
    if key:
        _check_key("https://api.openai.com/v1/models", {"authorization": f"Bearer {key}"}, "OpenAI")
    home = CODEX_API_HOME if key else None
    try:
        info = asyncio.run(probe(home, key))
    except Exception as e:
        raise LLMError(f"could not start codex app-server: {type(e).__name__}: {e}") from e
    account = info["account"] or {}
    if name == "codex" and account.get("type") != "chatgpt":
        raise LLMError("Codex is not logged in with ChatGPT: run `codex login`")
    if name == "codex-api" and account.get("type") != "apiKey":
        raise LLMError("Codex did not accept the API key")
    models = [{k: m[k] for k in ("id", "label", "efforts")} for m in info["models"] if m["efforts"]]
    if not models:
        raise LLMError("Codex lists no model that takes images")
    default = next((m["id"] for m in info["models"] if m["default"]), models[0]["id"])
    plan = account.get("planType")
    label = f"Codex (ChatGPT {plan} subscription)" if name == "codex" and plan else \
        "Codex (ChatGPT subscription)" if name == "codex" else "Codex (API key)"  # fmt: skip
    model = _pick(model or default, [m["id"] for m in models])
    subagents = "multi_agent_v2_dynamic_tools" in info["features"]
    return Backend(name, label, models, model, effort, codex_home=home, subagents=subagents)


# --------------------------------------------------------------------------- helpers
def _pick(model: str, ids: list) -> str:
    if model not in ids:
        raise LLMError(f"unknown model {model!r}; choose one of {', '.join(ids)}")
    return model


def _read_key(path: Path | None, var: str, required: bool = True) -> str | None:
    key = Path(path).expanduser().read_text().strip() if path else os.environ.get(var, "").strip()
    if required and not key:
        raise LLMError(f"no API key: set ${var} or pass --api-key-file")
    return key or None


def _check_key(url: str, headers: dict, provider: str) -> None:
    status, body = _http(url, headers)
    if status in (401, 403):
        raise LLMError(f"the {provider} API rejected the key ({status})")
    if status != 200:
        raise LLMError(f"could not check the {provider} API key: {url} answered {status}: {str(body)[:200]}")


def _http(url: str, headers: dict, payload: dict | None = None, timeout: float = 60) -> tuple:
    """(status, parsed JSON or text); LLMError if the server cannot be reached."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data, {**headers, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise LLMError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from e
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")
