"""Running the agent with Codex (ChatGPT subscription or OpenAI API key) through ``codex app-server``.

The app server speaks JSON-RPC over stdio, one JSON object per line. The tools from ``tools.py`` are
registered as dynamic tools (an experimental part of the protocol): Codex sends ``item/tool/call``
back and the tool runs here, on the ``AgentWorkspace`` the host shows, as with the in-process MCP
server of ``agent_sdk``. Codex's own tools (shell, patches, plugins, browser, ...) are switched off;
its web search stays, and so does code mode: the current models (``tool_mode: code_mode_only``)
call tools from JavaScript in ``exec``, run by the ``codex-code-mode-host`` that ships with Codex.
Without it they still try ``exec`` and give up.

Verifier subagents need Codex's ``multi_agent_v2_dynamic_tools`` (under development in codex-cli
0.162.1): a subagent then calls our tools on its own thread, and only the read-only ones are
allowed. Subagents also need a persisted thread, which ``close()`` deletes again (with its
subagents' threads). Without the feature the agent checks its objects itself.

Written against codex-cli 0.162.1.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Callable
from pathlib import Path

from .. import __version__
from ..core.workspace import AgentWorkspace
from ..skill import skill_prompt, verifier
from ..tools import READ_ONLY_TOOLS, call_tool, tools_for
from .agent_sdk import AgentEvent

# Codex features switched off for the agent: everything that reaches beyond our tools and web search.
# Not "code_mode_host": code mode is how the current models call tools.
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "remote_plugin", "browser_use",
    "browser_use_external", "computer_use", "in_app_browser", "image_generation", "view_image", "goals",
    "sleep_tool", "tool_suggest", "skill_search", "skill_mcp_dependency_install", "workspace_dependencies",
    "memories", "hooks",
)  # fmt: skip
# Switched on for verifier subagents (or "multi_agent" off without them).
SUBAGENT_FEATURES = ("multi_agent", "multi_agent_v2_dynamic_tools")
# Streaming deltas the hosts do not show (they show whole messages).
_QUIET = (
    "item/agentMessage/delta", "item/reasoning/textDelta", "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded", "item/plan/delta", "item/commandExecution/outputDelta",
    "item/fileChange/outputDelta",
)  # fmt: skip
# Offered efforts ("ultra" delegates to subagents, which are off here).
EFFORTS = ("low", "medium", "high", "xhigh", "max")

CODEX_HOST_NOTE = """
# Tools in this host

Your tools are the nninteractive tools and web search. There is no shell, no file access and there
are no subagents: the reference files the skill mentions are included at the end, and where the
skill suggests verifier subagents, check the objects yourself (`montage`, `view`, `stats`).
"""

CODEX_VERIFIER_NOTE = """
# Tools in this host

Your tools are the nninteractive tools and web search; there is no shell and no file access. The
reference files the skill mentions are included at the end.

Where the skill suggests `nninteractive-verifier` subagents, spawn them with
`collaboration.spawn_agent`, all at once so they run in parallel, named verifier_1, verifier_2, ...
(letters, digits and underscores only). They get these instructions too, so start each prompt with
"You are a verifier: follow the verifier instructions.", then give the object ids, what each object
should be and what to watch for. Verifiers can only use the read-only tools (`image_info`, `view`,
`montage`, `projection`, `render3d`, `stats`, `list_objects`). Wait for their reports, then make the corrections
yourself.
"""


def verifier_instructions() -> str:
    """The plugin's verifier agent (one source of truth), for Codex subagents."""
    body = verifier()["body"]
    return f"\n# Verifier instructions (only for verifier subagents)\n\n{body}\n"


class CodexError(RuntimeError):
    """A failed request to the app server, or the app server is gone."""


def tool_specs(python_mode: str = "full", exclude: tuple = ("load_image",)) -> list:
    return [
        {"type": "function", "name": t.name, "description": t.description, "inputSchema": t.input_schema}
        for t in tools_for(python_mode, exclude)
    ]


class AppServer:
    """One ``codex app-server`` process. ``on_request(method, params)`` (async) answers the server's
    requests (``None`` declines them); ``on_notification(method, params)`` receives everything else."""

    def __init__(self, codex_home: Path | None = None, on_notification: Callable | None = None,
                 on_request: Callable | None = None, log_file: Path | None = None,
                 command: tuple = ("codex",), subagents: bool = False):  # fmt: skip
        self.codex_home = codex_home
        self.subagents = subagents
        self.on_notification = on_notification
        self.on_request = on_request
        self.log_file = log_file
        self.command = tuple(command)
        self.proc = None
        self.closed = asyncio.Event()
        self._reader = None
        self._pending: dict = {}
        self._tasks: set = set()
        self._next_id = 0
        self._write_lock = asyncio.Lock()

    async def start(self) -> dict:
        # The API key of codex-api is stored in its own CODEX_HOME at login; keys in the environment
        # must not override the ChatGPT login of the default home.
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")}
        if self.codex_home is not None:
            Path(self.codex_home).mkdir(parents=True, exist_ok=True, mode=0o700)
            env["CODEX_HOME"] = str(self.codex_home)
        args = [*self.command, "app-server"] + [a for f in DISABLED_FEATURES for a in ("--disable", f)]
        if self.subagents:
            args += [a for f in SUBAGENT_FEATURES for a in ("--enable", f)]
            args += ["-c", "suppress_unstable_features_warning=true"]
        else:
            args += ["--disable", "multi_agent"]
        # The child keeps its own copy of the log's file descriptor.
        log = open(self.log_file, "ab") if self.log_file else None  # noqa: ASYNC230, SIM115
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log or asyncio.subprocess.DEVNULL, env=env, limit=1 << 28,
            )  # fmt: skip
        finally:
            if log is not None:
                log.close()
        self._reader = asyncio.create_task(self._read())
        info = await self.request("initialize", {
            "clientInfo": {"name": "nninteractive_agent", "title": "nnInteractive agent", "version": __version__},
            "capabilities": {"experimentalApi": True, "optOutNotificationMethods": list(_QUIET)},
        })  # fmt: skip
        await self._send({"method": "initialized"})
        return info

    async def request(self, method: str, params: dict, timeout: float | None = None) -> dict:
        if self.closed.is_set():
            raise CodexError(f"codex app-server is not running{self._log_tail()}")
        self._next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[self._next_id] = fut
        await self._send({"id": self._next_id, "method": method, "params": params})
        return await asyncio.wait_for(fut, timeout)

    async def _send(self, msg: dict) -> None:
        async with self._write_lock:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            await self.proc.stdin.drain()

    async def _read(self) -> None:
        try:
            while line := await self.proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "method" not in msg:  # a response
                    fut = self._pending.pop(msg.get("id"), None)
                    if fut is not None and not fut.done():
                        if "error" in msg:
                            fut.set_exception(CodexError(msg["error"].get("message") or str(msg["error"])))
                        else:
                            fut.set_result(msg.get("result") or {})
                elif "id" in msg:  # a request from the server
                    task = asyncio.create_task(self._answer(msg))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                elif self.on_notification is not None:
                    self.on_notification(msg["method"], msg.get("params") or {})
        except (OSError, ValueError):  # a broken pipe or an oversized line ends the connection
            pass
        finally:
            self.closed.set()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(CodexError(f"codex app-server exited{self._log_tail()}"))
            self._pending.clear()

    async def _answer(self, msg: dict) -> None:
        method = msg["method"]
        try:
            result = await self.on_request(method, msg.get("params") or {}) if self.on_request else None
            reply = {"id": msg["id"], "result": result} if result is not None else \
                {"id": msg["id"], "error": {"code": -32601, "message": f"{method} is not available in this host"}}  # fmt: skip
        except Exception as e:  # noqa: BLE001 - reported to Codex, which tells the model
            reply = {"id": msg["id"], "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
        try:
            await self._send(reply)
        except (OSError, RuntimeError):
            pass

    def _log_tail(self) -> str:
        if self.log_file is None or not Path(self.log_file).exists():
            return ""
        lines = [s for s in Path(self.log_file).read_text(errors="replace").splitlines() if s.strip()]
        return (": " + " | ".join(lines[-3:])[-600:]) if lines else ""

    async def close(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), 3)
            except (TimeoutError, OSError, RuntimeError):
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), 3)
                except TimeoutError:
                    self.proc.kill()
                    await self.proc.wait()
        if self._reader is not None:
            await asyncio.gather(self._reader, return_exceptions=True)


async def probe(
    codex_home: Path | None = None, api_key: str | None = None, command: tuple = ("codex",)
) -> dict:
    """Sign in with an API key (stored in ``codex_home``) or read the login, and list the models that
    take images and the feature flags: {"account": {...} | None, "models": [{"id", "label", "efforts",
    "default"}], "features": [names]}."""
    server = AppServer(codex_home, command=command)
    await server.start()
    try:
        if api_key:
            await server.request("account/login/start", {"type": "apiKey", "apiKey": api_key}, timeout=60)
        account = (await server.request("account/read", {}, timeout=60)).get("account")
        data, cursor = [], None
        while True:
            res = await server.request("model/list", {"cursor": cursor} if cursor else {}, timeout=60)
            data += res.get("data") or []
            cursor = res.get("nextCursor")
            if not cursor:
                break
        features = []
        try:
            res = await server.request("experimentalFeature/list", {"limit": 500}, timeout=60)
            features = [f.get("name") for f in res.get("data") or []]
        except CodexError:
            pass
    finally:
        await server.close()
    models = []
    for m in data:
        if m.get("hidden") or "image" not in (m.get("inputModalities") or ["image"]):
            continue
        efforts = [e.get("reasoningEffort") for e in m.get("supportedReasoningEfforts") or []]
        models.append({"id": m["id"], "label": m.get("displayName") or m["id"],
                       "efforts": [e for e in EFFORTS if e in efforts], "default": bool(m.get("isDefault"))})  # fmt: skip
    return {"account": account, "models": models, "features": features}


class CodexSession:
    """One conversation with the agent through Codex; the interface of ``agent_sdk.AgentSession``."""

    def __init__(self, ws: AgentWorkspace, on_event: Callable[[AgentEvent], None], model: str | None = None,
                 effort: str = "high", host_prompt: str = "", codex_home: Path | None = None,
                 command: tuple = ("codex",), subagents: bool = False):  # fmt: skip
        self.ws = ws
        self.on_event = on_event
        self.model = model
        self.effort = effort
        self.host_prompt = host_prompt
        self.codex_home = codex_home
        self.command = command
        self.subagents = subagents
        self.server: AppServer | None = None
        self.thread_id: str | None = None
        self._agents: dict = {}  # subagent thread id -> {"id", "name", "text", "status"}
        self._thread_tokens: dict = {}
        self._turn_id: str | None = None
        self._turn: dict | None = None
        self._turn_done: asyncio.Event | None = None
        self._last_error = ""
        self._interrupted = False
        self._tokens = 0

    async def start(self) -> None:
        self.ws.workdir.mkdir(parents=True, exist_ok=True)
        self.server = AppServer(self.codex_home, self._on_notification, self._on_request,
                                self.ws.workdir / "codex.log", self.command, self.subagents)  # fmt: skip
        note = CODEX_VERIFIER_NOTE if self.subagents else CODEX_HOST_NOTE
        instructions = self.host_prompt + note + skill_prompt(inline_references=True)
        if self.subagents:
            instructions += verifier_instructions()
        try:
            await self.server.start()
            res = await self.server.request("thread/start", {
                "cwd": str(self.ws.workdir), "model": self.model, "sandbox": "read-only", "approvalPolicy": "never",
                # subagents cannot be spawned from an ephemeral thread
                "ephemeral": not self.subagents, "dynamicTools": tool_specs(self.ws.python_mode), "config": {"web_search": "live"},
                "developerInstructions": instructions,
            })  # fmt: skip
        except BaseException:
            await self.close()
            raise
        self.thread_id = res["thread"]["id"]

    async def send(self, text: str) -> None:
        """Send a user message and stream the agent's work until its turn ends."""
        if self.server is not None and self.server.closed.is_set():
            await self.close()
            self.on_event(AgentEvent("error", "Codex had stopped and was restarted; the conversation starts fresh "
                                     "(the objects are kept)."))  # fmt: skip
        if self.server is None:
            await self.start()
        self._interrupted, self._last_error = False, ""
        self._turn, self._turn_id, self._turn_done = None, None, asyncio.Event()
        tokens0 = self._tokens
        try:
            res = await self.server.request("turn/start", {
                "threadId": self.thread_id, "effort": self.effort,
                "input": [{"type": "text", "text": text, "text_elements": []}],
            })  # fmt: skip
            self._turn_id = self._turn_id or (res.get("turn") or {}).get("id")
            done = asyncio.ensure_future(self._turn_done.wait())
            gone = asyncio.ensure_future(self.server.closed.wait())
            await asyncio.wait({done, gone}, return_when=asyncio.FIRST_COMPLETED)
            done.cancel()
            gone.cancel()
            if not self._turn_done.is_set():
                raise CodexError(f"codex app-server exited{self.server._log_tail()}")
        except CodexError as e:
            self._turn = {"status": "failed", "error": {"message": str(e)}}
        turn = self._turn or {}
        status = turn.get("status")
        stopped = self._interrupted or status == "interrupted"
        error = (turn.get("error") or {}).get("message") or self._last_error
        if status != "completed" and error and not stopped:
            self.on_event(AgentEvent("error", f"Codex: {error}"))
        text = "stopped" if stopped else ("success" if status == "completed" else status or "ended")
        self.on_event(AgentEvent("done", text=text, is_error=status != "completed" and not stopped,
                                 data={"tokens": self._tokens - tokens0}))  # fmt: skip

    # ------------------------------------------------------------------ app server
    def _on_notification(self, method: str, params: dict) -> None:
        thread = params.get("threadId")
        if method == "thread/tokenUsage/updated":
            total = ((params.get("tokenUsage") or {}).get("total") or {}).get("totalTokens")
            if thread and total is not None:
                self._thread_tokens[thread] = total
                self._tokens = sum(self._thread_tokens.values())
            return
        if thread not in (None, self.thread_id):  # one app server per session: other threads are subagents
            self._on_subagent_notification(method, thread, params)
            return
        if method in ("item/started", "item/completed"):
            item = params.get("item") or {}
            kind = item.get("type")
            if method == "item/completed" and kind == "subAgentActivity":
                self._subagent_activity(item)
            elif method == "item/started" and kind == "dynamicToolCall":
                self.on_event(AgentEvent("tool_call", tool=item.get("tool", ""), args=_args(item.get("arguments")),
                                         data={"id": item.get("id"), "parent": None}))  # fmt: skip
            elif method == "item/completed" and kind == "agentMessage" and (item.get("text") or "").strip():
                self.on_event(AgentEvent("text", item["text"]))
            elif method == "item/completed" and kind == "webSearch":
                args = {"query": item.get("query") or ""}
                self.on_event(AgentEvent("tool_call", tool="WebSearch", args=args,
                                         data={"id": item.get("id"), "parent": None}))  # fmt: skip
                self.on_event(AgentEvent("tool_result", _search_text(item), "WebSearch", args))
        elif method == "turn/started":
            self._turn_id = self._turn_id or (params.get("turn") or {}).get("id")
        elif method == "turn/completed":
            turn = params.get("turn") or {}
            if self._turn_done is not None and self._turn_id in (None, turn.get("id")):
                self._turn = turn
                self._turn_done.set()
        elif method == "error" and not params.get("willRetry"):
            self._last_error = (params.get("error") or {}).get("message", "")

    # ------------------------------------------------------------------ subagents
    def _agent(self, thread: str, path: str | None = None) -> dict:
        """The subagent on ``thread``; a new one is announced like a Claude Code subagent."""
        agent = self._agents.get(thread)
        name = (path or "").rstrip("/").rsplit("/", 1)[-1]
        if agent is not None and name and agent["name"] == "verifier":  # its first events came first
            agent["name"] = name
        if agent is None:
            name = name or "verifier"
            agent = self._agents[thread] = {
                "id": f"agent-{thread}",
                "name": name,
                "text": "",
                "status": "started",
            }
            self.on_event(AgentEvent("tool_call", tool="Agent", args={"description": name},
                                     data={"id": agent["id"], "parent": None}))  # fmt: skip
            self.on_event(AgentEvent("task", f"started: {name}", data={
                "task_id": thread, "tool_use_id": agent["id"], "status": "started", "description": name,
            }))  # fmt: skip
        return agent

    def _subagent_activity(self, item: dict) -> None:
        thread, kind = item.get("agentThreadId"), item.get("kind")
        if thread:
            agent = self._agent(thread, item.get("agentPath"))
            self._subagent_status(
                thread, agent, {"completed": "completed", "interrupted": "stopped"}.get(kind)
            )

    def _subagent_status(self, thread: str, agent: dict, status: str | None) -> None:
        """Report a subagent that finished, or that works again (the main agent can give it more work)."""
        if status is None or agent["status"] == status:
            return
        agent["status"] = status
        data = {"status": status, "description": agent["name"]}  # the name may have come late
        if status != "started":
            data["summary"] = agent["text"]
        first = (agent["text"].strip().splitlines() or [""])[0] if status != "started" else agent["name"]
        self.on_event(AgentEvent("task", f"{status}: {first}",
                                 data={"task_id": thread, "tool_use_id": agent["id"], **data}))  # fmt: skip

    def _on_subagent_notification(self, method: str, thread: str, params: dict) -> None:
        # Only its tool calls, messages and turn ends: a subagent's first events can come before the
        # main thread announces it (with its name).
        item = params.get("item") or {}
        if method == "turn/completed":  # the end of each piece of work
            status = (params.get("turn") or {}).get("status")
            self._subagent_status(
                thread, self._agent(thread), "stopped" if status == "interrupted" else "completed"
            )
        elif method == "item/started" and item.get("type") == "dynamicToolCall":
            agent = self._agent(thread)
            self._subagent_status(thread, agent, "started")
            self.on_event(AgentEvent("tool_call", tool="verifier: " + item.get("tool", ""),
                                     args=_args(item.get("arguments")),
                                     data={"id": item.get("id"), "parent": agent["id"]}))  # fmt: skip
        elif (
            method == "item/completed"
            and item.get("type") == "agentMessage"
            and (item.get("text") or "").strip()
        ):
            self._agent(thread)["text"] = item["text"]

    async def _on_request(self, method: str, params: dict) -> dict | None:
        if method != "item/tool/call":
            return None
        name, args = params.get("tool", ""), _args(params.get("arguments"))
        if params.get("threadId") != self.thread_id and name not in READ_ONLY_TOOLS:
            text = f"Verifiers only look: `{name}` is for the main agent. Report the problem instead."
            self.on_event(AgentEvent("tool_result", text, name, args, is_error=True))
            return {"contentItems": [{"type": "inputText", "text": text}], "success": False}
        out = await asyncio.to_thread(call_tool, self.ws, name, args)
        self.on_event(AgentEvent("tool_result", out.text, name, args, out.images, out.is_error))
        items = [{"type": "inputText", "text": out.text}]
        items += [{"type": "inputImage", "imageUrl": "data:image/png;base64," + base64.b64encode(p).decode()}
                  for p in out.images]  # fmt: skip
        return {"contentItems": items, "success": not out.is_error}

    async def interrupt(self) -> None:
        if self.server is not None and self.thread_id and self._turn_id:
            self._interrupted = True
            try:
                await self.server.request("turn/interrupt", {"threadId": self.thread_id, "turnId": self._turn_id},
                                          timeout=15)  # fmt: skip
            except (CodexError, TimeoutError):
                pass

    async def close(self) -> None:
        if self.server is not None:
            server, self.server = self.server, None
            if self.thread_id and self.subagents and not server.closed.is_set():
                try:  # a persisted thread: delete it (with its subagents' threads) from the Codex history
                    await server.request("thread/delete", {"threadId": self.thread_id}, timeout=15)
                except (CodexError, TimeoutError):
                    pass
            await server.close()


def _args(arguments) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return {}
    return dict(arguments) if isinstance(arguments, dict) else {}


def _search_text(item: dict) -> str:
    lines = [f"searched: {item.get('query') or ''}"]
    for r in (item.get("results") or [])[:8]:
        if isinstance(r, dict) and (r.get("title") or r.get("url")):
            lines.append(f"- {r.get('title') or ''} {r.get('url') or ''}".rstrip())
    return "\n".join(lines)
