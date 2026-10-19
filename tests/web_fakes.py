"""Fakes for the web front end: a workspace on the fake nnInteractive session and a scripted agent
that drives the real tool layer (no Claude, no server). Used by test_web.py and web_dev.py."""

from __future__ import annotations

import asyncio
import itertools

from test_workspace import FakeSession

from nninteractive_agent.core.workspace import AgentWorkspace
from nninteractive_agent.tools import call_tool
from nninteractive_agent.web.agent_sdk import AgentEvent

_ids = itertools.count(1)


def fake_workspace(spec, workdir):
    return AgentWorkspace(session=FakeSession(), workdir=workdir)


class ScriptedAgent:
    """Stands in for AgentSession: one object per request, with a verifier, through call_tool."""

    def __init__(self, ws, on_event, model="opus", effort="high", host_prompt="", delay=0.05):
        self.ws, self.on_event, self.delay = ws, on_event, delay
        self._stop = False

    async def send(self, text: str) -> None:
        self._stop = False
        ws = self.ws
        n = len(ws.objects) + 1
        a0, a1, a2 = ws.image.shape
        z, r = a0 // 2, max(3, min(a1, a2) // 8)
        c1, c2 = a1 // 2 + (n - 1) * r, a2 // 2
        self.on_event(AgentEvent("text", f"Working on **{text}**. I start with an overview."))
        await self._tool("image_info", {})
        await self._tool("new_object", {"name": f"object {n}"})
        box = {"corner1": [z, c1 - r, c2 - r], "corner2": [z, c1 + r, c2 + r]}
        await self._tool("bbox", {**box, "preview": True})
        await self._tool("bbox", box)
        await self._tool(
            "scribble", {"points": [[z, c1 - r // 2, c2], [z, c1 + r // 2, c2]], "positive": False}
        )
        await self._tool("point", {"position": [z, c1, c2 - r // 2]})
        await self._tool("view", {})
        agent_id = f"toolu_agent{next(_ids)}"
        self.on_event(AgentEvent("tool_call", tool="Agent", args={"description": f"verify object {n}"},
                                 data={"id": agent_id, "parent": None}))  # fmt: skip
        self.on_event(AgentEvent("task", data={"task_id": "t" + agent_id, "tool_use_id": agent_id, "status": "started",
                                               "description": f"verify object {n}"}))  # fmt: skip
        await self._tool("stats", {}, parent=agent_id)
        await self._tool("montage", {"axis": 0, "n": 4}, parent=agent_id)
        self.on_event(AgentEvent("task", data={"task_id": "t" + agent_id, "tool_use_id": agent_id, "status": "completed",
                                               "summary": "No errors found. The object looks complete."}))  # fmt: skip
        await self._tool("commit", {"notes": "scripted test object"})
        await self._tool("save", {})
        self.on_event(
            AgentEvent(
                "text",
                f"Done: **object {n}** is committed.\n\n| object | confidence |\n|---|---|\n| object {n} | high |",
            )
        )
        self.on_event(AgentEvent("done", text="success", cost_usd=0.01))

    async def _tool(self, name, args, parent=None):
        if self._stop:
            raise asyncio.CancelledError
        tid = f"toolu_{next(_ids)}"
        self.on_event(AgentEvent("tool_call", tool=("verifier: " if parent else "") + name, args=args,
                                 data={"id": tid, "parent": parent}))  # fmt: skip
        await asyncio.sleep(self.delay)
        out = await asyncio.to_thread(call_tool, self.ws, name, args)
        self.on_event(AgentEvent("tool_result", out.text, name, args, out.images, out.is_error))

    async def interrupt(self):
        self._stop = True

    async def close(self):
        pass


GT_NAMES = {"liver": {1: "liver", 2: "tumor"}, "spleen": {1: "spleen"}}  # label names per case


class GroundTruthAgent:
    """Stands in for AgentSession by committing the dataset's reference labels as objects (for
    work on the 3D view: realistic surfaces without Claude or nnInteractive). The labels file is
    found by the nnU-Net layout: imagesTr/<case>_0000.nii.gz -> labelsTr/<case>.nii.gz (ribs: from
    RibSeg). Binary
    labels with several components (liver tumors) become one object per component."""

    def __init__(self, ws, on_event, **kw):
        self.ws, self.on_event = ws, on_event

    async def send(self, text: str) -> None:
        import re
        from pathlib import Path

        import numpy as np
        import SimpleITK as sitk
        from scipy.ndimage import label as cc

        ws = self.ws
        src = Path(ws.image.source)
        gt_path = src.parent.parent / "labelsTr" / re.sub(r"_0000(\.nii\.gz)$", r"\1", src.name)
        gt_path = Path(str(gt_path).replace("Dataset015_RibFrac", "Dataset018_RibSeg"))  # rib instances
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        names = next((v for k, v in GT_NAMES.items() if k in src.name), {})
        with ws.lock:
            for value in [int(v) for v in np.unique(gt) if v]:
                mask = gt == value
                parts = [mask]
                if value in names and names[value] == "tumor":
                    lab, n = cc(mask)
                    parts = [lab == k for k in range(1, n + 1)]
                for k, part in enumerate(parts, 1):
                    oid = ws._next_id
                    free = part & (ws.instances == 0)
                    ws.instances[free] = oid
                    name = names.get(value, f"label {value}") + (f" {k}" if len(parts) > 1 else "")
                    ws.objects[oid] = {"name": name, "voxels": int(free.sum()), "notes": "reference label",
                                       "prompts": 0}  # fmt: skip
                    ws._next_id += 1
        self.on_event(AgentEvent("tool_call", tool="python", args={"code": f"# reference labels from {gt_path.name}"},
                                 data={"id": f"toolu_{next(_ids)}", "parent": None}))  # fmt: skip
        self.on_event(AgentEvent("tool_result", f"{len(ws.objects)} objects", "python", {}))
        self.on_event(AgentEvent("text", f"Committed the reference labels of **{gt_path.name}**."))
        self.on_event(AgentEvent("done", text="success", cost_usd=0.0))

    async def interrupt(self):
        pass

    async def close(self):
        pass
