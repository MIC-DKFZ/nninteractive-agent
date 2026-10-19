"""``nninteractive-agent setup``: give an agent host the nninteractive tools and skill.

    nninteractive-agent setup codex|claude|print [--server URL|local] [--python full|guarded|off]
                              [--workdir DIR] [--skip-model] [--uninstall]

codex   registers the MCP server with Codex (``codex mcp add``) and copies the skill to
        ``$CODEX_HOME/skills`` (default ``~/.codex/skills``)
claude  registers it with Claude Code for this user (``claude mcp add -s user``) and copies the skill
        and the verifier agent to ``~/.claude/skills`` and ``~/.claude/agents``
print   prints an ``mcpServers`` entry and the skill's location for any other MCP client

The server is registered with this Python interpreter (``python -m nninteractive_agent.mcp_server``),
so it works without an activated environment. For local inference (the default) the command checks
PyTorch and downloads the nnInteractive weights now, so the agent's first call does not wait for them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .core.pyguard import PYTHON_MODES, describe
from .core.session import LOCAL_HINT, probe, resolve_spec
from .skill import PLUGIN_TOOL_PREFIX, SKILL_DIR, SKILL_NAME, VERIFIER_FILE

NAME = "nninteractive"
# Loading the model and predicting on a CPU can take minutes.
CODEX_TIMEOUTS = {"startup_timeout_sec": 120, "tool_timeout_sec": 1800}


def server_command(spec: str, python_mode: str, workdir: str | None = None) -> list:
    cmd = [sys.executable, "-m", "nninteractive_agent.mcp_server", "--server", spec, "--python", python_mode]
    return cmd + (["--workdir", str(Path(workdir).expanduser().resolve())] if workdir else [])


def prepare_inference(spec: str) -> None:
    """Local inference: check PyTorch and the GPU, download the weights. A server: connect once. Both
    report the model's license."""
    if spec.startswith("local"):
        try:
            import torch
        except ImportError as e:
            sys.exit(f"{LOCAL_HINT}\n({e})")
        if torch.cuda.is_available():
            print(f"nnInteractive: local inference on {torch.cuda.get_device_name(0)}")
        else:
            print("nnInteractive: local inference on the CPU (no CUDA GPU found): this works, but each prompt is slow")
        print("nnInteractive: getting the weights ...", flush=True)
    else:
        print(f"nnInteractive: connecting to {spec} ...", flush=True)
    try:
        info = probe(spec)
    except Exception as e:  # noqa: BLE001 - say what failed, in one line
        sys.exit(f"nnInteractive: {type(e).__name__}: {e}")
    print(f"nnInteractive: {info['where']}")
    print(f"nnInteractive model license: {info['license']}")


def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and out.returncode != 0:
        sys.exit(f"`{' '.join(cmd[:4])} ...` failed:\n{out.stderr or out.stdout}")
    return out


def _host_exe(name: str, how: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        sys.exit(f"`{name}` was not found on PATH; install it first ({how})")
    return exe


def _copy_skill(skills_dir: Path) -> Path:
    dest = skills_dir / SKILL_NAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SKILL_DIR, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def _set_codex_timeouts(config: Path) -> None:
    """Add the timeouts to the server's table in config.toml (``codex mcp add`` has no flags for them)."""
    text = config.read_text()
    header = f"[mcp_servers.{NAME}]\n"
    if header not in text:
        return
    head, rest = text.split(header, 1)
    lines = "".join(
        f"{k} = {v}\n" for k, v in CODEX_TIMEOUTS.items() if not re.search(rf"^{k}\s*=", rest, re.M)
    )
    config.write_text(head + header + lines + rest)


def setup_codex(cmd: list, uninstall: bool) -> None:
    codex = _host_exe("codex", "npm install -g @openai/codex")
    home = codex_home()
    home.mkdir(parents=True, exist_ok=True)  # codex refuses a CODEX_HOME that does not exist
    _run([codex, "mcp", "remove", NAME], check=False)
    _remove(home / "skills" / SKILL_NAME)
    if uninstall:
        print(f"Removed the {NAME} MCP server and the {SKILL_NAME} skill from Codex.")
        return
    _run([codex, "mcp", "add", NAME, "--", *cmd])
    _set_codex_timeouts(home / "config.toml")
    dest = _copy_skill(home / "skills")
    print(f"Codex: MCP server '{NAME}' registered in {home / 'config.toml'}; skill in {dest}")
    print('Try:   codex "Segment the spleen in /path/to/ct.nii.gz and save the result"')


def setup_claude(cmd: list, uninstall: bool) -> None:
    claude = _host_exe("claude", "see https://docs.claude.com/en/docs/claude-code")
    home = claude_home()
    _run([claude, "mcp", "remove", "-s", "user", NAME], check=False)
    agent_file = home / "agents" / VERIFIER_FILE.name
    _remove(home / "skills" / SKILL_NAME)
    _remove(agent_file)
    if uninstall:
        print(
            f"Removed the {NAME} MCP server, the {SKILL_NAME} skill and the verifier agent from Claude Code."
        )
        return
    _run([claude, "mcp", "add", "-s", "user", NAME, "--", *cmd])
    dest = _copy_skill(home / "skills")
    agent_file.parent.mkdir(parents=True, exist_ok=True)
    # Outside the plugin the server's tools are called mcp__nninteractive__<tool>.
    agent_file.write_text(VERIFIER_FILE.read_text().replace(PLUGIN_TOOL_PREFIX, f"mcp__{NAME}__"))
    print(
        f"Claude Code: MCP server '{NAME}' registered for this user; skill in {dest}; verifier agent in {agent_file}"
    )
    print('Try:   claude "Segment the spleen in /path/to/ct.nii.gz and save the result"')


def print_config(cmd: list) -> None:
    print(json.dumps({"mcpServers": {NAME: {"command": cmd[0], "args": cmd[1:]}}}, indent=2))
    print(f"\nSkill: {SKILL_DIR}\n  Copy this folder to where your agent looks for skills. Agents without skills can read "
          f"the MCP prompt '{SKILL_NAME}' the server offers.")  # fmt: skip


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="nninteractive-agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup", help="register the tools and the skill with Codex or Claude Code",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    s.add_argument("host", choices=("codex", "claude", "print"))
    s.add_argument("--server", default=None, help="nninteractive-server URL or 'local' (default: "
                   "$NNINTERACTIVE_SERVER_URL, else local inference)")  # fmt: skip
    s.add_argument("--python", choices=PYTHON_MODES, default="full", help="the agent's python tool: full "
                   "(default), guarded (computations only) or off; see the README's Security section")  # fmt: skip
    s.add_argument("--workdir", default=None, help="where renders, logs and results go (default: ./runs/<time> "
                   "in the directory the agent runs in)")  # fmt: skip
    s.add_argument(
        "--skip-model", action="store_true", help="do not check PyTorch or download the weights now"
    )
    s.add_argument("--uninstall", action="store_true", help="remove what setup added")
    args = ap.parse_args(argv)

    spec = resolve_spec(args.server)
    cmd = server_command(spec, args.python, args.workdir)
    if args.host == "print":
        print_config(cmd)
        return
    if not (args.uninstall or args.skip_model):
        prepare_inference(spec)
    (setup_codex if args.host == "codex" else setup_claude)(cmd, args.uninstall)
    if not args.uninstall:
        print(describe(args.python))


if __name__ == "__main__":
    main()
