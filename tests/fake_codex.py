"""A stand-in for ``codex app-server`` (JSON-RPC over stdio) for test_codex.py: no network, no login.

Every turn calls the dynamic tool `image_info`, searches the web once and reports in its message
what it received (tool count, host note, images, model, effort, subagents). The message "slow"
waits for turn/interrupt, "crash" exits, "verify" starts a verifier subagent on thread th-sub that
calls `list_objects` and then `commit` (which the host must refuse). thread/delete leaves a file
`deleted-<id>` in $CODEX_HOME.
"""

import json
import os
import sys
from pathlib import Path

SUBAGENTS = "multi_agent_v2_dynamic_tools" in sys.argv
thread: dict = {}
turns = 0
next_request = 900


def write(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def notify(method, thread_id, **params):
    write({"method": method, "params": {"threadId": thread_id, **params}})


def item(method, thread_id, item_type, item_id, **fields):
    notify(method, thread_id, item={"type": item_type, "id": item_id, **fields})


def tool_call(thread_id, tid, tool, args):
    """Announce a dynamic tool call, ask the host to run it and return its reply."""
    global next_request
    next_request += 1
    call_id = f"call-{next_request}"
    item("item/started", thread_id, "dynamicToolCall", call_id, tool=tool, arguments=args, status="inProgress")
    write({"id": next_request, "method": "item/tool/call",
           "params": {"threadId": thread_id, "turnId": tid, "callId": call_id, "tool": tool, "arguments": args}})  # fmt: skip
    while (reply := read()).get("id") != next_request:
        pass
    return reply["result"]


def turn(tid, params):
    text = params["input"][0]["text"]
    notify("turn/started", "th-1", turn={"id": tid})
    if text == "crash":
        sys.exit(3)
    if text == "slow":
        while True:
            msg = read()
            if msg.get("method") == "turn/interrupt":
                write({"id": msg["id"], "result": {}})
                notify("turn/completed", "th-1", turn={"id": tid, "status": "interrupted"})
                return
    report = ""
    if text == "verify":
        activity = {"agentThreadId": "th-sub", "agentPath": "/root/verifier_1", "kind": "started"}
        item("item/started", "th-1", "subAgentActivity", "a1", **activity)
        item("item/completed", "th-1", "subAgentActivity", "a1", **activity)
        stats = tool_call("th-sub", "sub-turn", "list_objects", {})
        commit = tool_call("th-sub", "sub-turn", "commit", {})
        item("item/completed", "th-sub", "agentMessage", "sm", text="#1 object: OK\nnothing else")
        notify("thread/tokenUsage/updated", "th-sub", tokenUsage={"total": {"totalTokens": 700}})
        notify("turn/completed", "th-sub", turn={"id": "sub-turn", "status": "completed"})
        item("item/completed", "th-1", "subAgentActivity", "a2", **{**activity, "kind": "completed"})
        # a second round on the same subagent (the main agent asks it to re-check)
        item("item/completed", "th-1", "subAgentActivity", "a3", **{**activity, "kind": "interacted"})
        tool_call("th-sub", "sub-turn-2", "list_objects", {})
        item("item/completed", "th-sub", "agentMessage", "sm2", text="#1 object: OK after the fix")
        notify("turn/completed", "th-sub", turn={"id": "sub-turn-2", "status": "completed"})
        report = f" sub_stats={stats['success']} sub_commit={commit['success']}"
    result = tool_call("th-1", tid, "image_info", {})
    images = sum(i["type"] == "inputImage" and i["imageUrl"].startswith("data:image/png;base64,")
                 for i in result["contentItems"])  # fmt: skip
    item("item/completed", "th-1", "webSearch", f"ws-{tid}", query="liver CT",
         results=[{"title": "Liver", "url": "https://x"}])  # fmt: skip
    note = "Tools in this host" in thread["developerInstructions"]
    verifier = "Verifier instructions" in thread["developerInstructions"]
    report = (f"tools={len(thread['dynamicTools'])} note={note} images={images} model={thread['model']} "
              f"effort={params['effort']} subagents={SUBAGENTS} verifier={verifier} "
              f"ephemeral={thread['ephemeral']}" + report)  # fmt: skip
    item("item/completed", "th-1", "agentMessage", f"m-{tid}", text=report)
    notify("thread/tokenUsage/updated", "th-1", tokenUsage={"total": {"totalTokens": 1500 * turns}})
    notify("turn/completed", "th-1", turn={"id": tid, "status": "completed"})


while True:
    msg = read()
    method, params = msg.get("method"), msg.get("params") or {}
    if "id" not in msg:
        continue
    if method == "initialize":
        result = {"userAgent": "fake"}
    elif method == "account/read":
        result = {"account": {"type": "chatgpt", "planType": "plus"}}
    elif method == "account/login/start":
        result = {"type": "apiKey"}
    elif method == "model/list":
        efforts = [{"reasoningEffort": e} for e in ("low", "medium", "high", "ultra")]
        result = {"data": [
            {"id": "m-a", "displayName": "Model A", "isDefault": True, "hidden": False,
             "inputModalities": ["text", "image"], "supportedReasoningEfforts": efforts},
            {"id": "m-text", "displayName": "Text only", "isDefault": False, "hidden": False,
             "inputModalities": ["text"], "supportedReasoningEfforts": efforts},
        ], "nextCursor": None}  # fmt: skip
    elif method == "experimentalFeature/list":
        result = {"data": [{"name": "multi_agent", "enabled": True}, {"name": "multi_agent_v2_dynamic_tools"}]}
    elif method == "thread/start":
        thread = params
        result = {"thread": {"id": "th-1"}}
    elif method == "thread/delete":
        if os.environ.get("CODEX_HOME"):
            (Path(os.environ["CODEX_HOME"]) / f"deleted-{params['threadId']}").touch()
        result = {}
    elif method == "turn/start":
        turns += 1
        tid = f"turn-{turns}"
        write({"id": msg["id"], "result": {"turn": {"id": tid}}})
        turn(tid, params)
        continue
    else:
        write({"id": msg["id"], "error": {"code": -32601, "message": f"unknown method {method}"}})
        continue
    write({"id": msg["id"], "result": result})
