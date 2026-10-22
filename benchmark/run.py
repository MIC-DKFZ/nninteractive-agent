"""Run the benchmark and score it: an agent, or nnInteractive alone, on every image of data/.

    python benchmark/run.py --track guided --agent claude --model opus --effort medium
    python benchmark/run.py --track unguided --agent codex --model <model> --effort medium
    python benchmark/run.py --track guided --agent claude --llm-url http://localhost:8000 --model <open model>
    python benchmark/run.py --track point
    python benchmark/run.py --track oracle-scribbles      # or oracle-points
    python benchmark/run.py --track voxtell --voxtell "conda run -n voxtell voxtell-predict"
    python benchmark/run.py summary [RESULTS_DIR ...]

Tracks:

- ``unguided``: the agent gets the image and the list of targets by name (the request).
- ``guided``: the same, plus one voxel inside each target (the point deepest inside its reference).
- ``point``: no agent. One positive point at that voxel per target: what nnInteractive gives for
  the same location with no agent around it. Takes a minute.
- ``oracle-points``, ``oracle-scribbles``: no agent. Simulated prompts from the reference mask
  (``oracle.py``), ``--prompts`` per target (default 5), each aimed at the largest error so far.
  The target's Dice after every prompt is recorded, so the agents can be compared with a user who
  knows the answer, at any number of prompts.
- ``voxtell``: no agent. VoxTell, a text-promptable 3D segmentation model, with each target's request
  text as its prompt (all targets of an image in one call of VoxTell's CLI, which runs in its own
  environment: ``--voxtell``; ``--voxtell-group`` splits them over several calls, for less GPU memory).

The agent is Claude Code (``claude -p``) with this package's plugin, or ``codex exec`` with the MCP
server and the skill, configured for the run only (your own settings are not touched). The request
lists the targets in order and asks for each to be committed before the next; each target is scored
as its object was at its own last commit, so a later object that takes voxels from an earlier one
(a tumour inside the liver) does not count against the earlier one. nnInteractive runs on this
machine with the official weights unless ``--server`` names an nninteractive-server.
``--no-image-info`` (agent tracks) tells the agent nothing about the image: the request says only
"This is an image", and the agent works in a folder and on a link to the image whose names do not say
what it shows (results in ``<config>-noinfo``).

Each run gets ``<results>/<config>/<image>/rep<k>/`` with the request, the agent's transcript, its
workspace (tool log, renders, saved masks) and ``score.json``. ``summary`` prints the mean Dice per
configuration, per tier and per image (for the oracles also after each prompt), and writes
``summary.json`` next to the runs. Every mean is taken per image first: the mean over an image's
targets, then the mean over images, so an image with many targets weighs as much as one with one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle  # noqa: E402
import prepare  # noqa: E402  (checksums of the built data)

TOOL_PREFIX = "mcp__plugin_nninteractive_nninteractive__"
AGENT_TRACKS = ("unguided", "guided")
TRACKS = (*AGENT_TRACKS, "point", *(f"oracle-{k}" for k in oracle.KINDS), "voxtell")
INSTRUCTIONS = (
    "You segment images with the nninteractive tools. Before the first nninteractive tool call, {load} and "
    "follow it. Work autonomously: there is no human to answer questions during the run."
)


# ------------------------------------------------------------------------------------------- request
def request(case: dict, image_path: Path, guided: bool, describe: bool = True) -> str:
    lines = [f"This is {case['what'] if describe else 'an image'}: {image_path}."]
    if guided:
        lines.append("Segment each of the following structures as its own object, named exactly by its id. For each, a "
                     "voxel inside it is given ([axis0, axis1, axis2], the coordinates your tools use).")  # fmt: skip
    else:
        lines.append("Segment each of the following structures as its own object, named exactly by its id.")
    lines.append("Work through them in this order and commit each object before you start the next one:")
    for t in case["targets"]:
        lines.append(f"- {t['id']}: {t['request']}" + (f", contains voxel {t['hint']}" if guided else ""))
    lines.append("Save the result when you are done.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- agents
def _plugin_dir() -> Path:
    from nninteractive_agent.skill import PLUGIN_DIR

    return PLUGIN_DIR


def claude_cmd(prompt: str, args, run_dir: Path, env: dict) -> list:
    """Claude Code in print mode with the plugin; ``env`` is completed in place."""
    plugin = _plugin_dir()
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])  # nninteractive-agent-mcp
    env["NNINTERACTIVE_AGENT_WORKDIR"] = str(run_dir / "ws")
    env["NNINTERACTIVE_AGENT_SNAPSHOTS"] = "1"  # the object's mask at every action, for scoring at commit time
    env["NNINTERACTIVE_SERVER_URL"] = args.server
    env["NNINTERACTIVE_AGENT_PYTHON"] = args.python
    if args.llm_url:  # an open model behind an Anthropic-compatible API (README, "Open models")
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.update(ANTHROPIC_BASE_URL=args.llm_url, ANTHROPIC_API_KEY=env.get("LLM_API_KEY", "dummy"),
                   ANTHROPIC_DEFAULT_OPUS_MODEL=args.model, ANTHROPIC_DEFAULT_SONNET_MODEL=args.model,
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=args.model, ENABLE_TOOL_SEARCH="false",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")  # fmt: skip
    cmd = [
        "claude", "-p", prompt,
        "--plugin-dir", str(plugin), "--add-dir", str(plugin),
        "--model", args.model, "--effort", args.effort,
        "--append-system-prompt",
        INSTRUCTIONS.format(load="load the nninteractive:nninteractive-segmentation skill"),
        "--permission-mode", "dontAsk",
        "--allowedTools", f"{TOOL_PREFIX}*", "Skill", "Read", "ToolSearch", "Agent", "WebSearch",
        # The agent works only through its tools: no shell, no file writes.
        "--disallowedTools", "Bash", "Monitor", "Write", "Edit", "WebFetch",
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
    ]  # fmt: skip
    return cmd + (["--max-budget-usd", str(args.budget)] if args.budget else [])


def codex_cmd(prompt: str, args, run_dir: Path) -> list:
    """``codex exec`` with the MCP server and the skill, configured for this run only. Codex starts MCP
    servers with a minimal environment, so the server gets its settings as arguments and ``env``."""
    skill = _plugin_dir() / "skills" / "nninteractive-segmentation"
    shutil.copytree(skill, run_dir / ".agents" / "skills" / skill.name)  # a repo-scoped skill for this folder
    mcp_args = ["-m", "nninteractive_agent.mcp_server", "--server", args.server, "--python", args.python,
                "--workdir", str(run_dir / "ws")]  # fmt: skip
    env = {"NNINTERACTIVE_AGENT_SNAPSHOTS": "1"}
    for k in ("no_proxy", "NO_PROXY", "NN_INTERACTIVE_API_KEY", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    m = "mcp_servers.nninteractive"
    return [
        "codex", "exec", "--json", "--skip-git-repo-check", "-m", args.model,
        # The shell (Codex reads the skill with it) runs sandboxed in the run folder without network. Where
        # the sandbox cannot start (bwrap without user namespaces), Codex asks to run the command outside it
        # and an automatic reviewer decides. Pinned so that the user's Codex config does not change the run.
        "-s", "workspace-write",
        "-c", 'approval_policy="on-request"', "-c", 'approvals_reviewer="auto_review"',
        "-c", f"model_reasoning_effort={json.dumps(args.effort)}",
        "-c", f"{m}.command={json.dumps(sys.executable)}",
        "-c", f"{m}.args={json.dumps(mcp_args)}",
        "-c", f"{m}.env={{{', '.join(f'{k} = {json.dumps(v)}' for k, v in env.items())}}}",
        "-c", f'{m}.default_tools_approval_mode="approve"',
        "-c", f"{m}.startup_timeout_sec=300", "-c", f"{m}.tool_timeout_sec=1800",
        f"{INSTRUCTIONS.format(load='read the nninteractive-segmentation skill')}\n\n{prompt}",
    ]  # fmt: skip


def _finished_but_idle(transcript: Path, quiet_s: float = 90.0) -> bool:
    """Claude Code emitted its result, no subagent is running or waiting to report, and nothing was
    written for ``quiet_s``: ``claude -p`` sometimes stays alive after its result once subagents ran."""
    if time.time() - transcript.stat().st_mtime < quiet_s:
        return False
    running, result, unprocessed = set(), False, False
    for line in transcript.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        sub = m.get("subtype")
        if m.get("type") == "system" and sub == "task_started":
            running.add(m.get("task_id"))
        elif m.get("type") == "system" and sub == "task_notification":
            running.discard(m.get("task_id"))
            unprocessed = True
        elif m.get("type") == "result":
            result, unprocessed = True, False
    return result and not running and not unprocessed


def run_agent(case: dict, image_path: Path, run_dir: Path, args) -> dict:
    prompt = request(case, image_path, guided=args.track == "guided", describe=not args.no_image_info)
    (run_dir / "request.txt").write_text(prompt + "\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT")
           and not k.startswith("CLAUDE_CODE_")}  # fmt: skip
    cmd = codex_cmd(prompt, args, run_dir) if args.agent == "codex" else claude_cmd(prompt, args, run_dir, env)
    t0, timed_out = time.time(), False
    with open(run_dir / "transcript.jsonl", "w") as f:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            time.sleep(5)
            if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
                timed_out = True
            elif not (args.agent == "claude" and _finished_but_idle(run_dir / "transcript.jsonl")):
                continue
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return {"wall_s": round(time.time() - t0), "timeout": timed_out}


def run_point(case: dict, image_path: Path, run_dir: Path, args) -> dict:
    """One positive point at each target's hint voxel, each target its own object, in order."""
    os.environ["NNINTERACTIVE_AGENT_SNAPSHOTS"] = "1"
    from nninteractive_agent.core.workspace import AgentWorkspace

    t0 = time.time()
    ws = AgentWorkspace(session_spec=args.server, workdir=run_dir / "ws", python_mode="off")
    try:
        ws.load_image(image_path)
        for t in case["targets"]:
            ws.new_object(t["id"])
            ws.point(t["hint"], positive=True)
            ws.commit(notes="one positive point at the hint voxel", overlap="replace")
        ws.save()
        prompts = ws.n_prompts
    finally:
        ws.close()
    return {"wall_s": round(time.time() - t0), "timeout": False, "prompts": prompts}


def run_oracle(case: dict, image_path: Path, run_dir: Path, args) -> dict:
    """Simulated prompts from the reference (oracle.py), ``args.prompts`` per target, each target its
    own object, in order; the target's Dice after every prompt goes to ``trajectories``."""
    os.environ["NNINTERACTIVE_AGENT_SNAPSHOTS"] = "1"
    from nninteractive_agent.core import scale
    from nninteractive_agent.core.workspace import AgentError, AgentWorkspace

    kind = args.track.removeprefix("oracle-")
    t0, trajectories = time.time(), {}
    ws = AgentWorkspace(session_spec=args.server, workdir=run_dir / "ws", python_mode="off")
    try:
        ws.load_image(image_path)
        spacing = ws.image.spacing
        for t in case["targets"]:
            ref = _load(args.data / case["id"] / t["mask"]) > 0
            ws.new_object(t["id"], extent=oracle.extent_for(ref, scale.THRESHOLD_EXTENT))
            curve = []
            for _ in range(args.prompts):
                p = oracle.next_prompt(ws.buffer > 0, ref, spacing, kind)
                if p is None:  # nothing left to correct
                    curve.append(1.0)
                    continue
                try:
                    if p["kind"] == "scribble":
                        ws.scribble(p["points"], positive=p["positive"], axis=p["axis"])
                    else:
                        ws.point(p["points"][0], positive=p["positive"])
                except AgentError:  # a line shorter than one voxel of a downsampled object
                    ws.point(p["points"][len(p["points"]) // 2], positive=p["positive"])
                curve.append(round(oracle.dice(ws.buffer > 0, ref), 4))
            trajectories[t["id"]] = curve
            if ws.buffer.any():
                ws.commit(notes=f"oracle ({kind})", overlap="replace")
            else:
                ws.discard()
        ws.save()
        prompts = ws.n_prompts
    finally:
        ws.close()
    return {"wall_s": round(time.time() - t0), "timeout": False, "prompts": prompts, "trajectories": trajectories}


def _voxtell_name(prompt: str) -> str:
    """VoxTell's file name fragment for a prompt (its ``_safe_prompt_name`` for prompts this short)."""
    return "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in prompt).replace(" ", "_")


def _commit_masks(ws: Path, masks: list) -> None:
    """Record (target id, mask) pairs as commits in the workspace's action-log format, so ``score``
    treats them like an agent's objects (overlapping targets each keep their own mask)."""
    from scipy import ndimage as ndi

    (ws / "snapshots").mkdir(parents=True, exist_ok=True)
    with open(ws / "actions.jsonl", "a") as f:
        for i, (tid, m) in enumerate(masks, 1):
            if not m.any():
                continue
            sl = ndi.find_objects(m.astype(np.uint8))[0]
            crop = m[sl]
            rel = f"snapshots/{i:05d}.npz"
            np.savez_compressed(ws / rel, bbox=np.array([[s.start, s.stop] for s in sl]), shape=np.array(crop.shape),
                                bits=np.packbits(crop))  # fmt: skip
            f.write(json.dumps({"seq": i, "action": "commit", "id": i, "object": tid, "snapshot": rel}) + "\n")


def run_voxtell(case: dict, image_path: Path, run_dir: Path, args) -> dict:
    """VoxTell with each target's request text as its prompt, all targets of the image in one call, or
    ``args.voxtell_group`` at a time. VoxTell keeps every prompt's output on the GPU (17 prompts on a
    whole-body CT need about 25 GB). It decodes each prompt on its own (cross-attention to the image only,
    none between prompts), so splitting changes only the numerics: the masks of an 8-target image changed
    by less than 0.002 Dice."""
    out = run_dir / "voxtell"
    prompts = [t["request"] for t in case["targets"]]
    group = args.voxtell_group or len(prompts)
    t0 = time.time()
    with open(run_dir / "voxtell.log", "w") as log:
        for i in range(0, len(prompts), group):
            cmd = [*shlex.split(args.voxtell), "-i", str(image_path), "-o", str(out), "-p", *prompts[i : i + group]]
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
    stem = image_path.name.removesuffix(".gz").removesuffix(".nii")
    masks = []
    for t, prompt in zip(case["targets"], prompts):
        f = out / f"{stem}_{_voxtell_name(prompt)}.nii.gz"
        m = _load(f) > 0 if f.exists() else np.zeros(case["shape"], bool)
        if m.shape != tuple(case["shape"]):
            raise RuntimeError(f"VoxTell's mask {f.name} has shape {m.shape}, the image {tuple(case['shape'])}")
        masks.append((t["id"], m))
    _commit_masks(run_dir / "ws", masks)
    return {"wall_s": round(time.time() - t0), "timeout": False, "prompts": 0, "text_prompts": prompts,
            "prompts_per_call": min(group, len(prompts))}


# -------------------------------------------------------------------------------------------- scoring
def _load(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _belongs(name: str | None, target_id: str) -> bool:
    return bool(name) and (name == target_id or name.startswith(target_id + "_"))


def _at_commit(ws: Path, shape: tuple) -> dict:
    """{target object name: mask at its object's last commit} from the workspace's action log."""
    log = ws / "actions.jsonl"
    last, names = {}, {}
    for line in log.read_text().splitlines() if log.exists() else []:
        rec = json.loads(line)
        if rec.get("action") == "commit" and rec.get("snapshot"):
            last[rec["id"]], names[rec["id"]] = rec["snapshot"], rec.get("object")
        elif rec.get("action") == "delete_object":
            last.pop(rec.get("id"), None)
    out = {}
    for oid, snap in last.items():
        with np.load(ws / snap) as z:
            bbox, crop_shape = z["bbox"].tolist(), tuple(int(s) for s in z["shape"])
            crop = np.unpackbits(z["bits"], count=int(np.prod(crop_shape))).reshape(crop_shape).astype(bool)
        m = np.zeros(shape, bool)
        m[tuple(slice(lo, hi) for lo, hi in bbox)] = crop
        out[oid] = (names[oid], m)
    return out


def _saved(ws: Path, shape: tuple) -> dict:
    """{object id: (name, mask)} from the agent's last successful save (it may choose the folder)."""
    import re

    log = ws / "log.jsonl"
    for line in reversed(log.read_text().splitlines() if log.exists() else []):
        e = json.loads(line)
        hit = re.search(r"saved \d+ objects to (.+?) \(\+ objects\.json\)", e.get("text") or "")
        if e.get("tool") == "save" and not e.get("error") and hit and Path(hit.group(1)).exists():
            path = Path(hit.group(1))
            break
    else:
        found = sorted((ws / "output").glob("*_instances.*"))
        if not found:
            return {}
        path = found[0]
    lab = _load(path)
    if lab.shape != shape:
        return {}
    objects = json.loads((path.parent / "objects.json").read_text())
    return {int(k): (o["name"], lab == int(k)) for k, o in objects.items()}


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    inter, p, g = int((pred & ref).sum()), int(pred.sum()), int(ref.sum())
    return {"dice": round(2 * inter / max(p + g, 1), 4), "precision": round(inter / max(p, 1), 4),
            "recall": round(inter / max(g, 1), 4)}  # fmt: skip


def _prompts_per_object(ws: Path) -> Counter:
    """{object name: prompts sent for it} from the action log (undone prompts count: they were sent)."""
    log, n = ws / "actions.jsonl", Counter()
    for line in log.read_text().splitlines() if log.exists() else []:
        rec = json.loads(line)
        if rec.get("action") in ("point", "bbox", "scribble"):
            n[rec.get("object")] += 1
    return n


def score(run_dir: Path, case: dict, data_dir: Path) -> dict:
    ws = run_dir / "ws"
    shape = tuple(case["shape"])
    committed, saved, prompts = _at_commit(ws, shape), _saved(ws, shape), _prompts_per_object(ws)
    targets = []
    for t in case["targets"]:
        ref = _load(data_dir / t["mask"]) > 0
        mine = [m for name, m in committed.values() if _belongs(name, t["id"])]
        final = [m for name, m in saved.values() if _belongs(name, t["id"])]
        row = {"id": t["id"], "tier": t["tier"], "objects": len(mine) or len(final)}
        if mine:
            row.update(_metrics(np.any(mine, axis=0), ref), source="commit")
        elif final:
            row.update(_metrics(np.any(final, axis=0), ref), source="saved")
        else:
            row.update(dice=0.0, precision=0.0, recall=0.0, source="missing")
        row["dice_saved"] = _metrics(np.any(final, axis=0), ref)["dice"] if final else None
        row["prompts"] = sum(n for name, n in prompts.items() if _belongs(name, t["id"]))
        targets.append(row)
    return {"targets": targets, "mean_dice": round(float(np.mean([r["dice"] for r in targets])), 4),
            "missing": sum(r["source"] == "missing" for r in targets), **_agent_stats(run_dir)}  # fmt: skip


def _agent_stats(run_dir: Path) -> dict:
    """Tool calls, prompts, images, tokens or cost from the transcript and the tool log."""
    tools, out, usage = Counter(), {}, Counter()
    tr = run_dir / "transcript.jsonl"
    for line in tr.read_text().splitlines() if tr.exists() else []:
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("type") == "assistant":  # Claude Code
            for block in m.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tools[("subagent:" if m.get("parent_tool_use_id") else "") + block["name"].replace(TOOL_PREFIX, "")] += 1
        elif m.get("type") == "result":
            out.update(cost_usd=m.get("total_cost_usd"), turns=m.get("num_turns"))
            usage.update({k: v for k, v in (m.get("usage") or {}).items() if isinstance(v, int)})
        elif m.get("type") == "item.completed" and (m.get("item") or {}).get("type") == "mcp_tool_call":  # Codex
            tools[m["item"].get("tool", "?")] += 1
        elif m.get("type") == "turn.completed":
            usage.update({k: v for k, v in (m.get("usage") or {}).items() if isinstance(v, int)})
    log = run_dir / "ws" / "log.jsonl"
    entries = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    out.update(tool_calls=dict(tools.most_common()), tokens=dict(usage),
               images=sum(e.get("images", 0) for e in entries), tool_errors=sum(bool(e.get("error")) for e in entries),
               prompts=entries[-1].get("total_prompts", 0) if entries else None)  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------- runs
class AgentUnavailable(Exception):
    """The agent could not work (a usage limit, a failed turn): the run says nothing about segmentation."""


def agent_failure(run_dir: Path) -> str | None:
    """Why the agent stopped without finishing its work, from its transcript; None if it worked."""
    tr = run_dir / "transcript.jsonl"
    for line in tr.read_text().splitlines() if tr.exists() else []:
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("type") == "turn.failed":  # codex exec
            return (m.get("error") or {}).get("message") or "turn failed"
        if m.get("type") == "error" and "limit" in str(m.get("message", "")).lower():
            return m["message"]
        if m.get("type") == "result" and m.get("is_error"):  # claude -p
            return str(m.get("result") or m.get("subtype"))
    return None


def retry_wait(message: str, default_min: float = 30) -> float:
    """Seconds until the agent may work again: the reset time a usage-limit message names, else a default."""
    import datetime as dt
    import re

    m = re.search(r"(?:try again at|resets(?: at)?)\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)", message, re.I)
    if not m:
        return default_min * 60
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    now = dt.datetime.now()
    at = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if at <= now:
        at += dt.timedelta(days=1)
    return (at - now).total_seconds() + 120


def config_name(args) -> str:
    if args.track not in AGENT_TRACKS:
        name = args.track + (f"-{args.prompts}" if args.track.startswith("oracle") and args.prompts != 5 else "")
    else:
        name = f"{args.track}-{args.agent}-{args.model.replace('/', '_')}-{args.effort}"
        name += "-noinfo" if args.no_image_info else ""
    return name + (f"-{args.tag}" if args.tag else "")


def _versions(args) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("nninteractive-agent", "nnInteractive", "nninteractive-client"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    if args.track in AGENT_TRACKS:
        cli = "codex" if args.agent == "codex" else "claude"
        try:
            out[cli] = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
        except OSError:
            out[cli] = "not found"
    return out


def _neutral(out_root: Path, image: str, rep: int, image_path: Path) -> tuple[Path, Path]:
    """``--no-image-info``: a working folder and a hard link to the image whose names say nothing about
    the image (the agent sees both paths; ``<image>/rep<k>`` would name it)."""
    folder = out_root / "_neutral" / hashlib.sha256(image.encode()).hexdigest()[:8]
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / ("image" + "".join(image_path.suffixes))
    if not link.exists():
        try:
            os.link(image_path, link)
        except OSError:  # another file system
            shutil.copy2(image_path, link)
    work_dir = folder / f"rep{rep}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    return work_dir, link


def run_one(image: str, rep: int, args, out_root: Path) -> dict:
    data_dir = args.data / image
    case = json.loads((data_dir / "case.json").read_text())
    run_dir = out_root / image / f"rep{rep}"
    if (run_dir / "score.json").exists() and not args.rerun:
        print(f"[{image} rep{rep}] already scored, skipped", flush=True)
        return json.loads((run_dir / "score.json").read_text())
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    agent = args.track in AGENT_TRACKS
    meta = {"image": image, "rep": rep, "track": args.track, "agent": args.agent if agent else None,
            "model": args.model if agent else None, "effort": args.effort if agent else None,
            "llm_url": bool(args.llm_url), "nninteractive": args.server, "python_tool": args.python,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "data_checks": prepare.check(case) or "ok"}  # fmt: skip
    print(f"[{image} rep{rep}] running in {run_dir}", flush=True)
    image_path = (data_dir / case["image"]).resolve()
    work_dir = run_dir
    if agent and args.no_image_info:
        work_dir, image_path = _neutral(out_root, image, rep, image_path)
        meta["image_info"] = False
    runner = {"point": run_point, "voxtell": run_voxtell}.get(args.track, run_agent if agent else run_oracle)
    try:
        meta.update(runner(case, image_path, work_dir, args))
    except Exception as e:  # noqa: BLE001 - a failed run is scored as what it saved (usually nothing)
        meta["error"] = f"{type(e).__name__}: {e}"
    finally:
        if work_dir != run_dir:  # everything back to where scoring and summary look
            for p in work_dir.iterdir():
                shutil.move(str(p), str(run_dir / p.name))
            work_dir.rmdir()
    failure = agent_failure(run_dir) if agent else None
    if failure:
        meta["agent_failure"] = failure
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    if failure:  # no score.json: the next run of this configuration does it again
        print(f"[{image} rep{rep}] the agent could not work: {failure}", flush=True)
        raise AgentUnavailable(failure)
    s = {**meta, **score(run_dir, case, data_dir)}
    if s["prompts"] is None:  # no tool log: the point and oracle tracks count their own prompts
        s["prompts"] = meta.get("prompts", 0)
    (run_dir / "score.json").write_text(json.dumps(s, indent=1))
    miss = f", {s['missing']} missing" if s["missing"] else ""
    limit = " (time limit)" if meta.get("timeout") else ""
    print(f"[{image} rep{rep}] mean Dice {s['mean_dice']:.3f} over {len(s['targets'])} targets{miss}, "
          f"{meta.get('wall_s', 0)} s{limit}", flush=True)
    return s


def wait_while_paused(pause_file: Path | None) -> None:
    """``--pause-file``: before each run (and each retry), wait as long as that file exists."""
    if pause_file and pause_file.exists():
        print(f"paused: {pause_file} exists", flush=True)
        while pause_file.exists():
            time.sleep(60)
        print("resumed", flush=True)


def run(args) -> None:
    cases = sorted(p.parent.name for p in args.data.glob("*/case.json"))
    images = args.images or cases
    missing = sorted(set(images) - set(cases))
    if missing:
        sys.exit(f"not built: {', '.join(missing)} (run prepare.py build first)")
    if args.server != "local" and args.server.startswith("http"):
        host = args.server.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        for var in ("no_proxy", "NO_PROXY"):
            os.environ[var] = ",".join(x for x in (os.environ.get(var), host) if x)
    out_root = args.results / config_name(args)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("command",)}
    cfg["versions"] = _versions(args)
    (out_root / "config.json").write_text(json.dumps(cfg, indent=1))
    jobs = [(image, rep) for rep in range(1, args.repeats + 1) for image in images]

    def attempt(job):
        tries = 0
        while True:
            wait_while_paused(args.pause_file)
            try:
                return run_one(*job, args, out_root)
            except AgentUnavailable as e:
                if args.stop_on_limit:
                    raise
                tries += 1
                if args.max_retries is not None and tries > args.max_retries:
                    print(f"[{job[0]} rep{job[1]}] gave up after {tries} failed attempts (not scored)", flush=True)
                    return None
                wait = retry_wait(str(e))
                print(f"waiting {wait / 60:.0f} min before trying again", flush=True)
                time.sleep(wait)

    try:
        if args.jobs > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(attempt, jobs))
        else:
            for j in jobs:
                attempt(j)
    except AgentUnavailable as e:
        summary([out_root])
        sys.exit(f"stopped, the agent could not work: {e}")
    summary([out_root])


# ------------------------------------------------------------------------------------------- summary
def image_mean(values: dict) -> float:
    """{(image, target): value} -> the mean over images of each image's mean over its targets."""
    by_image = defaultdict(list)
    for (image, _), v in values.items():
        by_image[image].append(v)
    return float(np.mean([np.mean(v) for v in by_image.values()]))


def summary(dirs: list) -> None:
    """Mean Dice per configuration (``image_mean`` of each target's Dice averaged over repeats), per tier
    and per image, and the plain mean over targets. The oracles also get their Dice after each prompt,
    and each agent configuration the oracles' Dice at the agent's own number of prompts per target
    (when oracle results are among ``dirs``)."""
    configs = [d for root in dirs for d in ([root] if (root / "config.json").exists() else sorted(root.glob("*/")))
               if (d / "config.json").exists()]  # fmt: skip
    loaded = [(d, [json.loads(p.read_text()) for p in sorted(d.glob("*/rep*/score.json"))]) for d in configs]
    oracles = {}  # oracle config -> {(image, target): mean Dice after each prompt}
    for cfg_dir, scores in loaded:
        curves = defaultdict(list)
        for s in scores:
            for tid, c in (s.get("trajectories") or {}).items():
                curves[(s["image"], tid)].append(c)
        if curves:
            n = max(len(c) for v in curves.values() for c in v)
            oracles[cfg_dir.name] = {k: [float(np.mean([c[min(i, len(c) - 1)] for c in v])) for i in range(n)]
                                     for k, v in curves.items()}  # fmt: skip
    for cfg_dir, scores in loaded:
        if not scores:
            continue
        per_target = defaultdict(list)  # (image, target) -> Dice per repeat
        tiers, per_rep, used, sent = {}, defaultdict(dict), defaultdict(list), defaultdict(list)
        for s in scores:
            sent[s["image"]].append((s.get("prompts") or 0) / max(len(s["targets"]), 1))  # all prompts of the run
            for t in s["targets"]:
                per_target[(s["image"], t["id"])].append(t["dice"])
                tiers[(s["image"], t["id"])] = t["tier"]
                per_rep[s["rep"]][(s["image"], t["id"])] = t["dice"]
                used[(s["image"], t["id"])].append(t.get("prompts"))
        means = {k: float(np.mean(v)) for k, v in per_target.items()}
        rep_means = [image_mean(v) for v in per_rep.values()]
        out = {
            "config": cfg_dir.name, "runs": len(scores), "n_images": len({i for i, _ in means}), "targets": len(means),
            "mean_dice": round(image_mean(means), 4),
            "mean_dice_over_targets": round(float(np.mean(list(means.values()))), 4),
            "sd_over_repeats": round(float(np.std(rep_means)), 4) if len(rep_means) > 1 else None,
            "tiers": {tier: round(image_mean({k: v for k, v in means.items() if tiers[k] == tier}), 4)
                      for tier in sorted(set(tiers.values()))},
            "images": {im: round(float(np.mean([v for (i, _), v in means.items() if i == im])), 4)
                       for im in sorted({i for i, _ in means})},
            "missing": sum(s["missing"] for s in scores), "timeouts": sum(bool(s.get("timeout")) for s in scores),
            "minutes_per_image": round(float(np.mean([s.get("wall_s", 0) for s in scores])) / 60, 1),
            "prompts_per_target": round(float(np.mean([np.mean(v) for v in sent.values()])), 2),
            "prompts_per_target_over_targets": round(sum(s.get("prompts") or 0 for s in scores) / max(sum(len(s["targets"]) for s in scores), 1), 2),
            "cost_usd": round(sum(s.get("cost_usd") or 0 for s in scores), 2) or None,
            "per_target": {f"{i}/{t}": round(v, 4) for (i, t), v in sorted(means.items())},
        }  # fmt: skip
        if cfg_dir.name in oracles:
            curves = oracles[cfg_dir.name]
            n = max(len(c) for c in curves.values())
            out["dice_after_prompts"] = [round(image_mean({k: c[min(i, len(c) - 1)] for k, c in curves.items()}), 4)
                                         for i in range(n)]  # fmt: skip
        elif all(None not in v for v in used.values()):
            same = {}
            for name, curves in oracles.items():
                if set(means) <= set(curves):
                    # Dice 0 for a target the agent never prompted (it has no object either)
                    at = {k: np.mean([curves[k][min(n, len(curves[k])) - 1] if n else 0.0 for n in v]) for k, v in used.items()}
                    same[name] = round(image_mean(at), 4)
            if same:
                out["oracle_same_prompts"] = same
        (cfg_dir / "summary.json").write_text(json.dumps(out, indent=1))
        sd = f" ± {out['sd_over_repeats']:.3f}" if out["sd_over_repeats"] is not None else ""
        print(f"\n{out['config']}: mean Dice {out['mean_dice']:.3f}{sd} over {out['n_images']} images ({out['targets']} targets; "
              f"over targets {out['mean_dice_over_targets']:.3f}), {out['runs']} runs"
              f" ({out['missing']} targets missing, {out['timeouts']} time limits), {out['minutes_per_image']} min per image, "
              f"{out['prompts_per_target']} prompts per target")  # fmt: skip
        print("  tiers: " + ", ".join(f"{k} {v:.3f}" for k, v in out["tiers"].items()))
        print("  images: " + ", ".join(f"{k} {v:.3f}" for k, v in out["images"].items()))
        if "dice_after_prompts" in out:
            print("  after prompt: " + ", ".join(f"{k + 1}: {v:.3f}" for k, v in enumerate(out["dice_after_prompts"])))
        if "oracle_same_prompts" in out:
            print("  oracles with the agent's prompts per target: "
                  + ", ".join(f"{k} {v:.3f}" for k, v in out["oracle_same_prompts"].items()))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        dirs = [Path(p) for p in sys.argv[2:]] or [HERE / "results"]
        summary(dirs)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=TRACKS, required=True)
    ap.add_argument("--agent", choices=("claude", "codex"), default="claude")
    ap.add_argument("--model", default="opus", help="claude: opus, sonnet, ... or the --llm-url server's model; codex: its model")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--llm-url", default=None, help="claude only: an Anthropic-compatible server (e.g. vLLM)")
    ap.add_argument("--server", default="local", help="nnInteractive: 'local' (this machine, official weights; default) or an nninteractive-server URL")
    ap.add_argument("--prompts", type=int, default=5, help="oracle tracks: prompts per target")
    ap.add_argument("--voxtell", default="voxtell-predict", help="voxtell track: the command that runs VoxTell's "
                    "CLI, e.g. 'conda run -n voxtell voxtell-predict' (VoxTell needs its own environment)")  # fmt: skip
    ap.add_argument("--voxtell-group", type=int, default=None, help="voxtell track: at most this many prompts per "
                    "VoxTell call (default: all targets of an image in one call; fewer need less GPU memory)")  # fmt: skip
    ap.add_argument("--python", default="full", choices=("full", "guarded", "off"), help="the agent's python tool")
    ap.add_argument("--no-image-info", action="store_true", help="agent tracks: say nothing about the image (no "
                    "description; neutral folder and file names); results in <config>-noinfo")  # fmt: skip
    ap.add_argument("--images", nargs="*", default=None, help="image ids (default: all built images)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="images in parallel (each loads nnInteractive when local)")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop an agent after this long (it then saved nothing)")
    ap.add_argument("--budget", type=float, default=None, help="claude: max USD per image (API-key billing only)")
    ap.add_argument("--tag", default="", help="suffix for the results folder")
    ap.add_argument("--rerun", action="store_true", help="run again where a score exists")
    ap.add_argument("--stop-on-limit", action="store_true", help="stop when the agent hits a usage limit "
                    "(default: wait until the reset time the agent names, or 30 min, and go on)")  # fmt: skip
    ap.add_argument("--pause-file", type=Path, default=None, help="wait before each run while this file exists "
                    "(touch it to pause a long benchmark after the current image, remove it to go on)")  # fmt: skip
    ap.add_argument("--max-retries", type=int, default=None, help="give up on an image after this many failed "
                    "retries and go on with the next (it stays unscored; default: retry forever)")  # fmt: skip
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
