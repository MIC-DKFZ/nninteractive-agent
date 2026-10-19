"""The Codex agent session against a fake app server, and the LLM backends (no network, no login)."""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from test_workspace import FakeSession

from nninteractive_agent.core.io import LoadedImage
from nninteractive_agent.core.workspace import AgentWorkspace
from nninteractive_agent.web.agent_codex import CodexSession, probe
from nninteractive_agent.web.llm import LLMError, claude_backend, make_backend

FAKE = (sys.executable, str(Path(__file__).parent / "fake_codex.py"))


@pytest.fixture
def ws(tmp_path):
    w = AgentWorkspace(session=FakeSession(), workdir=tmp_path / "ws")
    w.set_image(LoadedImage(np.random.default_rng(0).normal(100, 10, (10, 40, 40)).astype(np.float32),
                            (2.0, 1.0, 1.0), "fake.npy"))  # fmt: skip
    return w


def run_session(ws, messages, model="m-a", effort="high", **kw):
    """Send ``messages`` in order ("slow" is interrupted) and return the events."""
    events = []

    async def go():
        s = CodexSession(ws, events.append, model=model, effort=effort, command=FAKE, **kw)
        try:
            for m in messages:
                if m == "slow":
                    task = asyncio.create_task(s.send("slow"))
                    while not s._turn_id:
                        await asyncio.sleep(0.01)
                    await s.interrupt()
                    await task
                else:
                    await s.send(m)
        finally:
            await s.close()

    asyncio.run(go())
    return events


def test_codex_session_runs_our_tools_and_reports(ws):
    ev = run_session(ws, ["segment it", "and again"])
    kinds = [e.kind for e in ev]
    assert kinds.count("done") == 2 and kinds.count("tool_call") == 4  # image_info + web search per turn
    call = next(e for e in ev if e.kind == "tool_call")
    result = next(e for e in ev if e.kind == "tool_result")
    assert call.tool == result.tool == "image_info" and call.args == result.args == {} and result.images
    texts = [e.text for e in ev if e.kind == "text"]
    assert texts[0] == (f"tools=18 note=True images={len(result.images)} model=m-a effort=high subagents=False "
                        "verifier=False ephemeral=True")  # fmt: skip
    search = [e for e in ev if e.tool == "WebSearch"]
    assert [e.kind for e in search[:2]] == ["tool_call", "tool_result"] and "Liver" in search[1].text
    done = [e for e in ev if e.kind == "done"]
    assert [d.text for d in done] == ["success", "success"] and not any(d.is_error for d in done)
    assert [d.data["tokens"] for d in done] == [1500, 1500] and done[0].cost_usd is None


def test_codex_interrupt_and_crash_restart(ws):
    ev = run_session(ws, ["slow", "crash", "after the crash"])
    done = [e for e in ev if e.kind == "done"]
    assert done[0].text == "stopped" and not done[0].is_error
    assert done[1].is_error and any(e.kind == "error" and "exited" in e.text for e in ev)
    assert done[2].text == "success"  # a fresh app server took over
    assert any(e.kind == "error" and "restarted" in e.text for e in ev)


def test_verifier_subagents_look_but_do_not_touch(ws, tmp_path):
    ev = run_session(ws, ["verify"], subagents=True, codex_home=tmp_path / "home")
    text = next(e.text for e in ev if e.kind == "text")
    # the subagent could look (list_objects) but not change anything (commit); verifier instructions were given
    assert text.endswith("subagents=True verifier=True ephemeral=False sub_stats=True sub_commit=False")
    card = next(e for e in ev if e.kind == "tool_call" and e.tool == "Agent")
    assert card.args == {"description": "verifier_1"}
    sub = [e for e in ev if e.kind == "tool_call" and e.tool.startswith("verifier: ")]
    assert [e.tool for e in sub] == ["verifier: list_objects", "verifier: commit", "verifier: list_objects"]
    assert all(e.data["parent"] == card.data["id"] for e in sub)
    refused = [e for e in ev if e.kind == "tool_result" and e.tool == "commit"]
    assert len(refused) == 1 and refused[0].is_error and not ws.objects
    tasks = [e.data for e in ev if e.kind == "task"]
    # started once though reported twice, then a second round on the same subagent
    assert [t["status"] for t in tasks] == ["started", "completed", "started", "completed"]
    assert tasks[1]["tool_use_id"] == card.data["id"] and tasks[1]["summary"].startswith("#1 object: OK")
    assert tasks[3]["summary"] == "#1 object: OK after the fix"
    done = next(e for e in ev if e.kind == "done")
    assert done.data["tokens"] == 1500 + 700 and not done.is_error  # subagent tokens count too
    assert (tmp_path / "home" / "deleted-th-1").exists()  # the persisted thread is deleted on close


def test_probe_lists_image_models_with_offered_efforts(tmp_path):
    info = asyncio.run(probe(tmp_path / "home", "sk-test", command=FAKE))
    assert info["account"]["type"] == "chatgpt" and "multi_agent_v2_dynamic_tools" in info["features"]
    assert info["models"] == [
        {"id": "m-a", "label": "Model A", "efforts": ["low", "medium", "high"], "default": True}
    ]


def test_effort_falls_back_to_the_nearest_one():
    b = claude_backend()
    b.models = [{"id": "q", "label": "q", "efforts": ["low", "medium", "xhigh"]}]
    assert b.effort_for("q", "medium") == "medium"
    assert b.effort_for("q", "high") == "medium"  # tie between medium and xhigh: the lower one
    assert b.effort_for("q", "max") == "xhigh"
    assert claude_backend().describe()["models"][0]["id"] == "opus"


class FakeLLMServer(BaseHTTPRequestHandler):
    """Anthropic-compatible endpoints like vLLM with Qwen: efforts high and above are refused."""

    def log_message(self, *a):
        pass

    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._json(200, {"data": [{"id": "org/vision-27b"}, {"id": "org/text-only"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        effort = body.get("output_config", {}).get("effort")
        if effort in ("high", "xhigh", "max"):
            self._json(400, {"error": {"message": f"Unexpected reasoning effort {effort}."}})
        else:
            self._json(200, {"content": [{"type": "text", "text": "h"}]})


def test_server_backend_probes_models_and_efforts(monkeypatch):
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "localhost")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLMServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    try:
        with pytest.raises(LLMError, match="choose a model"):
            make_backend("server", url=url)
        b = make_backend("server", model="org/vision-27b", url=url)
    finally:
        httpd.shutdown()
    assert b.models == [{"id": "org/vision-27b", "label": "vision-27b", "efforts": ["low", "medium"]}]
    assert b.effort == "medium" and b.engine == "claude"
    assert (
        b.env["ANTHROPIC_BASE_URL"] == url.rstrip("/")
        and b.env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "org/vision-27b"
    )
    assert "127.0.0.1" in __import__("os").environ["NO_PROXY"]


def test_api_key_backends_need_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_backend("claude-api")
