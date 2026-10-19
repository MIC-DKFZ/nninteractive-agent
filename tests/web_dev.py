"""Run the web front end with fakes (no Claude, no nnInteractive): for UI work and screenshots.

    python tests/web_dev.py [--port 8780] [--password dev] [--examples DIR|FILE.json]

Plain HTTP on localhost only. Messages are answered by a scripted agent that runs real tools on
a fake nnInteractive session.
"""

import argparse
import os
import tempfile
from pathlib import Path

import uvicorn
from web_fakes import GroundTruthAgent, ScriptedAgent, fake_workspace

from nninteractive_agent.web.auth import Auth
from nninteractive_agent.web.server import create_app, load_examples
from nninteractive_agent.web.session import SessionManager

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8780)
ap.add_argument("--password", default="dev")
ap.add_argument("--delay", type=float, default=0.3, help="seconds per scripted tool call")
ap.add_argument("--gt", action="store_true", help="any message commits the reference labels as objects")
ap.add_argument("--examples", type=Path, default=None, help="example images, as for nninteractive-agent-web")
ap.add_argument("--root", type=Path, default=Path(tempfile.gettempdir()) / "nni_web_dev")
args = ap.parse_args()

os.environ.setdefault("NNINTERACTIVE_AGENT_CONFIG", str(args.root / "config"))
manager = SessionManager(
    "fake", "fake nnInteractive", load_examples(args.examples), max_sessions=50, root=args.root,
    workspace_factory=fake_workspace,
    agent_factory=(lambda *a, **k: GroundTruthAgent(*a, **k)) if args.gt else
    (lambda *a, **k: ScriptedAgent(*a, delay=args.delay, **{x: v for x, v in k.items() if x != "delay"})),
)  # fmt: skip
app = create_app(manager, Auth(args.password, secret="dev-secret"), secure_cookie=False)
uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
