"""Browser sessions: one image, workspace and agent conversation per logged-in browser.

Everything runs on the server's asyncio loop; blocking work (image IO, nnInteractive, label
diffs, marching cubes) goes to threads. The page gets JSON events (chat, objects, status) and
binary label patches over one websocket; volumes, surfaces and the agent's images over HTTP.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from ..core.io import LoadedImage, load_image
from ..core.session import model_license
from ..core.render import CURRENT_COLOR, object_color
from ..core.workspace import AgentError, AgentWorkspace
from . import layers
from .agent_sdk import AgentEvent, AgentSession, web_host_prompt
from .llm import Backend, claude_backend

STATE_TOOLS = ("bbox", "scribble", "point", "undo", "new_object", "commit", "discard", "edit_object",
               "delete_object", "python")  # fmt: skip
CLIENT_QUEUE_LIMIT = 2000  # messages; a client this far behind is dropped and resyncs on reconnect


class UserError(Exception):
    """Something the user should be told (shown as a notice in the page)."""


class Client:
    """One open websocket. Messages are queued so a slow browser never blocks the session."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    def put(self, msg) -> None:
        if self.closed:
            return
        if self.queue.qsize() > CLIENT_QUEUE_LIMIT:
            self.closed = True
            self.queue.put_nowait(None)
            return
        self.queue.put_nowait(msg)

    async def pump(self) -> None:
        while True:
            msg = await self.queue.get()
            if msg is None:
                break
            if isinstance(msg, bytes):
                await self.websocket.send_bytes(msg)
            else:
                await self.websocket.send_text(json.dumps(msg, default=str))


class WebSession:
    def __init__(self, sid: str, manager: SessionManager):
        self.sid = sid
        self.mgr = manager
        self.dir = manager.runs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{sid[:6]}"
        self.clients: set = set()
        self.events: list = []  # chat history, replayed to every new client
        self.image: LoadedImage | None = None
        self.image_meta: dict | None = None
        self.ws: AgentWorkspace | None = None
        self.agent: AgentSession | None = None
        self.attach_task: asyncio.Task | None = None
        self.run_task: asyncio.Task | None = None
        self.busy = False
        self.paused = False
        self.status = "Choose an image to start"
        self.model, self.effort = manager.model, manager.effort
        self.last_active = time.time()
        self.layers = layers.LayerSync()
        self.meshes: dict = {}  # key -> {"v", "blob", "color", "name"}
        self._mesh_sig: dict = {}
        self._objects_msg: dict | None = None
        self._pending_calls: list = []
        self._orphans: list = []
        self._running_tasks: set = set()
        self._agent_notes: list = []  # for the agent with the next message (e.g. an uploaded segmentation)
        self._img_count = 0
        self._image_n = 0
        self._t0 = 0.0
        self._prompts0 = 0
        self._dirty = asyncio.Event()
        self._sync_lock = asyncio.Lock()
        self._sync_task = asyncio.create_task(self._sync_loop())

    # ---------------------------------------------------------------- clients
    def touch(self) -> None:
        self.last_active = time.time()

    def _broadcast(self, msg) -> None:
        for c in list(self.clients):
            c.put(msg)

    def _event(self, ev: dict, store: bool = True) -> None:
        ev.setdefault("ts", time.time())
        if store:
            self.events.append(ev)
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.dir / "events.jsonl", "a") as f:
                    f.write(json.dumps(ev, default=str) + "\n")
            except OSError:
                pass
        self._broadcast(ev)

    def _set_status(self, text: str) -> None:
        self.status = text
        self._broadcast(self._state_msg())

    def _state_msg(self) -> dict:
        return {
            "t": "state", "busy": self.busy, "paused": self.paused, "status": self.status, "model": self.model,
            "effort": self.effort, "has_conversation": self.agent is not None,
            "ready": bool(self.ws is not None and self.ws.session is not None), "license": self.mgr.model_license,
        }  # fmt: skip

    async def connect(self, client: Client) -> None:
        """Bring a new page up to date, then let it receive live updates."""
        client.put(
            {
                "t": "hello",
                "image": self.image_meta,
                "server": self.mgr.server_label,
                "license": self.mgr.model_license,
                "agent": self.mgr.llm.describe(),
                "cases": self.mgr.case_list(),
            }
        )
        client.put(self._state_msg())
        for ev in self.events:
            client.put(ev)
        client.put({"t": "replay_done"})
        async with self._sync_lock:
            if self._objects_msg:
                client.put(self._objects_msg)
            for key, m in self.meshes.items():
                client.put(self._mesh_msg(key, m))
            full = await asyncio.to_thread(self._full_patches)
            for p in full:
                client.put(p)
            self.clients.add(client)

    def disconnect(self, client: Client) -> None:
        self.clients.discard(client)
        client.closed = True
        client.queue.put_nowait(None)

    def _full_patches(self) -> list:
        out = []
        for name, arr in self.layers.sent.items():
            header = {"layer": name, "box": [[0, s] for s in arr.shape], "dtype": str(arr.dtype),
                      "shape": list(arr.shape), "full": True, "seq": self.layers.seq}  # fmt: skip
            out.append(layers.encode_patch(header, arr))
        return out

    # ------------------------------------------------------------------ image
    async def open_image(self, path: Path, title: str, tasks: list | None = None) -> None:
        if self.busy:
            raise UserError("The agent is still working. Stop it first.")
        has_seat = self.ws is not None and self.ws.session is not None
        if not has_seat and self.mgr.seats_in_use(exclude=self) >= self.mgr.max_sessions:
            raise UserError(f"All {self.mgr.max_sessions} demo seats are in use. Try again in a few minutes.")
        self.touch()
        self.busy = True
        try:
            self._set_status(f"Loading {title} ...")
            await self._close_workspace()
            img = await asyncio.to_thread(load_image, path)
            if img.data.ndim != 3:
                raise UserError("Only 2D and 3D grayscale or RGB images are supported.")
            self._set_status("Preparing the image for the browser ...")
            key, meta = await asyncio.to_thread(self.mgr.volumes_for, img, path)
            self._image_n += 1
            self.image = img
            self.ws = self.mgr.workspace_factory(self.mgr.server_spec, self.dir / f"image{self._image_n}")
            self.layers.reset()
            self.meshes, self._mesh_sig, self._objects_msg = {}, {}, None
            self.paused = False
            self.image_meta = {
                **meta, "key": key, "title": title, "n": self._image_n, "tasks": list(tasks or []),
                "url": f"api/volume/{key}/image.nii.gz", "url3d": f"api/volume/{key}/image3d.nii.gz",
                "affine": layers.array_to_ras(img).tolist(),
            }  # fmt: skip
            self._event({"t": "image", "image": self.image_meta})
            self.attach_task = asyncio.create_task(self._attach())
        finally:
            self.busy = False
        self._set_status("Sending the image to nnInteractive ...")

    async def _attach(self, reattach: bool = False) -> None:
        ws = self.ws
        try:
            if reattach:
                await asyncio.to_thread(ws.reattach)
            else:
                await asyncio.to_thread(ws.set_image, self.image)
        except Exception as e:  # shown to the user and re-raised; the next message retries
            self._event(
                {"t": "error", "text": f"Could not start an nnInteractive session: {type(e).__name__}: {e}"}
            )
            self._set_status("nnInteractive is not available. Sending a message retries.")
            raise
        self.mgr.model_license = model_license(ws.session)  # what the model in use says, local or remote
        if ws is self.ws:
            self._set_status("Ready. Describe what to segment.")
            self.mark_dirty()

    async def _close_workspace(self) -> None:
        """End the conversation and release the nnInteractive session."""
        if self.agent is not None:
            agent, self.agent = self.agent, None
            try:
                await agent.close()
            except Exception:  # noqa: BLE001, S110 - best effort while closing
                pass
        if self.attach_task is not None and not self.attach_task.done():
            try:
                await self.attach_task
            except Exception:  # noqa: BLE001, S110 - best effort while closing
                pass
        if self.ws is not None:
            await asyncio.to_thread(self.ws.close)

    async def import_objects(self, path: Path, segments: dict, mode: str) -> None:
        """The user's segmentation (a label map file on the image grid) as objects; the agent hears of
        it with the next message."""
        if self.busy:
            raise UserError("The agent is still working. Stop it first.")
        if self.ws is None or self.image is None:
            raise UserError("Open an image first.")
        if self.attach_task is not None and not self.attach_task.done():
            try:
                await self.attach_task
            except Exception as e:  # noqa: BLE001 - shown to the user
                raise UserError(f"nnInteractive is not available: {e}") from e
        self.touch()
        self.busy = True
        try:
            labels = await asyncio.to_thread(load_image, path)
            _check_same_grid(labels, self.image)
            ws = self.ws

            def run():
                with ws.lock:
                    return ws.import_objects(labels.data, segments, mode)

            try:
                res = await asyncio.to_thread(run)
            except AgentError as e:
                raise UserError(str(e)) from e
        finally:
            self.busy = False
        what = "replaced the objects" if mode == "replace" else "added objects"
        self._event({"t": "divider", "text": f"Segmentation uploaded: {what}"})
        self._agent_notes.append(f"[The user uploaded a segmentation, which {res.text.removeprefix('imported a label map: ')}. "
                                 "Use list_objects to see the current objects.]")  # fmt: skip
        self.mark_dirty()
        self._set_status("Segmentation uploaded. Describe what to do with it.")

    # ----------------------------------------------------------------- agent
    async def send(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.busy:
            raise UserError("The agent is still working on the last request.")
        if self.ws is None:
            raise UserError("Open an image first.")
        if self.paused:
            await self.resume()
        self.touch()
        self.busy = True
        self._t0, self._prompts0 = time.time(), self.ws.n_prompts
        self._event({"t": "user", "text": text})
        self._set_status("Starting the agent ...")
        self.run_task = asyncio.create_task(self._run(text))

    async def _run(self, text: str) -> None:
        try:
            if self.attach_task is not None:
                try:
                    await self.attach_task
                except Exception:  # noqa: BLE001 - retry once now
                    self.attach_task = asyncio.create_task(self._attach())
                    await self.attach_task
            if self.agent is None:
                self.agent = self.mgr.agent_factory(self.ws, self._on_agent_event, model=self.model, effort=self.effort,
                                                    host_prompt=web_host_prompt(self.ws.python_mode))  # fmt: skip
            self._set_status("The agent is working ...")
            notes, self._agent_notes = self._agent_notes, []
            await self.agent.send("\n".join([*notes, text]))
        except asyncio.CancelledError:
            self._event({"t": "done", "seconds": time.time() - self._t0, "error": True, "text": "cancelled"})
        except Exception as e:  # noqa: BLE001 - surface everything in the chat
            self._event({"t": "error", "text": f"{type(e).__name__}: {e}"})
            self._event({"t": "done", "seconds": time.time() - self._t0, "error": True, "text": "failed"})
        finally:
            self.busy = False
            self.touch()
            self._flush_orphans()
            self.mark_dirty()
            self._set_status("Done. Send a follow-up or download the result." if self.ws else "")

    async def stop(self) -> None:
        if self.busy and self.agent is not None:
            self._set_status("Stopping ...")
            await self.agent.interrupt()

    async def new_conversation(self) -> None:
        if self.busy:
            raise UserError("Stop the agent first.")
        if self.agent is not None:
            agent, self.agent = self.agent, None
            await agent.close()
        self._event({"t": "divider", "text": "New conversation (objects are kept)"})
        self._set_status("New conversation. The agent sees the objects that are already there.")

    async def settings(self, model: str | None, effort: str | None) -> None:
        llm = self.mgr.llm
        if model in [m["id"] for m in llm.models]:
            self.model = model
        if effort in llm.efforts(self.model):
            self.effort = effort
        self.effort = llm.effort_for(self.model, self.effort)  # a new model may not take the old effort
        if self.agent is not None and not self.busy:
            await self.new_conversation()
        else:
            self._broadcast(self._state_msg())

    async def pause(self) -> None:
        """Free the nnInteractive slot and the agent process after inactivity; objects stay."""
        if self.ws is None or self.paused or self.busy:
            return
        if self.agent is not None:
            agent, self.agent = self.agent, None
            await agent.close()
        try:
            await asyncio.to_thread(self.ws.save)
        except Exception:  # noqa: BLE001, S110 - nothing to save yet
            pass
        await asyncio.to_thread(self.ws.close)
        self.paused = True
        self._event({"t": "divider", "text": f"Paused after {self.mgr.idle_s // 60} min without activity. "
                     "The objects are kept; sending a message continues."})  # fmt: skip
        self._set_status("Paused. Sending a message continues.")

    async def resume(self) -> None:
        if self.mgr.seats_in_use(exclude=self) >= self.mgr.max_sessions:
            raise UserError(f"All {self.mgr.max_sessions} demo seats are in use. Try again in a few minutes.")
        self.paused = False
        self._set_status("Reconnecting to nnInteractive ...")
        self.attach_task = asyncio.create_task(self._attach(reattach=True))
        await self.attach_task

    async def close(self) -> None:
        self._sync_task.cancel()
        if self.run_task is not None and not self.run_task.done():
            self.run_task.cancel()
        await self._close_workspace()
        for c in list(self.clients):
            self.disconnect(c)

    # ------------------------------------------------------------ agent events
    def _on_agent_event(self, ev: AgentEvent) -> None:
        if ev.kind == "text":
            self._event({"t": "text", "text": ev.text})
        elif ev.kind == "tool_call":
            sub = ev.tool.startswith("verifier: ")
            name = ev.tool.removeprefix("verifier: ")
            call = {"t": "call", "id": ev.data.get("id"), "tool": name, "args": ev.args, "parent": ev.data.get("parent"),
                    "sub": sub}  # fmt: skip
            self._event(call)
            if not sub:
                self._set_status(_status_text(name, ev.args))
            orphan = next((o for o in self._orphans if o["tool"] == name and o["args"] == ev.args), None)
            if orphan is not None:
                self._orphans.remove(orphan)
                self._emit_result(orphan, call)
            else:
                self._pending_calls.append(call)
        elif ev.kind == "tool_result":
            res = {"tool": ev.tool, "args": ev.args, "text": ev.text[:6000], "error": ev.is_error,
                   "images": [self._save_image(png) for png in ev.images]}  # fmt: skip
            call = next(
                (c for c in self._pending_calls if c["tool"] == ev.tool and c["args"] == ev.args), None
            )
            if call is not None:
                self._pending_calls.remove(call)
                self._emit_result(res, call)
            else:  # the tool ran before its call message arrived
                self._orphans.append(res)
                asyncio.get_running_loop().call_later(3.0, self._flush_orphans)
            if ev.tool in STATE_TOOLS:
                self.mark_dirty()
        elif ev.kind == "task":
            self._event({"t": "task", **ev.data})
            if ev.data.get("status") == "started":
                self._running_tasks.add(ev.data.get("task_id"))
            else:
                self._running_tasks.discard(ev.data.get("task_id"))
            n = len(self._running_tasks)
            self._set_status(f"{n} verifier{'s are' if n > 1 else ' is'} checking the result ..." if n
                             else "Reading the verifier report ...")  # fmt: skip
        elif ev.kind == "error":
            self._event({"t": "error", "text": ev.text})
        elif ev.kind == "done":
            prompts = (self.ws.n_prompts - self._prompts0) if self.ws else 0
            self._event({"t": "done", "seconds": round(time.time() - self._t0, 1), "cost_usd": ev.cost_usd,
                         "tokens": ev.data.get("tokens"), "error": ev.is_error, "text": ev.text,
                         "prompts": prompts})  # fmt: skip

    def _emit_result(self, res: dict, call: dict | None) -> None:
        self._event({"t": "result", "id": call["id"] if call else None, **res})
        if call is not None and not call.get("sub"):
            focus = _focus_point(res["tool"], res["args"], res["text"])
            if focus is not None and self.image is not None and not self.image.is_2d:
                self._broadcast({"t": "focus", "point": focus})

    def _flush_orphans(self) -> None:
        orphans, self._orphans = self._orphans, []
        for o in orphans:
            self._emit_result(o, None)

    def _save_image(self, png: bytes) -> str:
        self._img_count += 1
        d = self.dir / "images"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{self._img_count:05d}.png"
        (d / name).write_bytes(png)
        # The run in the URL keeps the browser's cache (one day) from showing an earlier run's
        # image with the same number; the server reads the file from the current run either way.
        return f"api/img/{name}?run={self.dir.name}"

    # ------------------------------------------------------- viewer sync
    def mark_dirty(self) -> None:
        self._dirty.set()

    async def _sync_loop(self) -> None:
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            try:
                await self._sync()
            except Exception as e:  # noqa: BLE001 - the viewer must not take the session down
                print(f"[web] sync failed: {type(e).__name__}: {e}")
            await asyncio.sleep(0.25)

    async def _sync(self) -> None:
        ws = self.ws
        if ws is None or ws.image is None or ws.instances is None:
            return
        async with self._sync_lock:
            patches, snap = await asyncio.to_thread(self._snapshot, ws)
            for p in patches:
                self._broadcast(p)
            if snap["objects_msg"] != self._objects_msg:
                self._objects_msg = snap["objects_msg"]
                self._broadcast(self._objects_msg)
        await self._sync_meshes(ws, snap)

    def _snapshot(self, ws: AgentWorkspace) -> tuple:
        with ws.lock:
            patches = self.layers.patches(ws)
            objects = {int(i): dict(o) for i, o in ws.objects.items()}
            cur = ws.current
            cur_voxels = int(np.count_nonzero(ws.buffer)) if cur is not None else 0
            current = None if cur is None else {"name": cur.name, "prompts": len(cur.prompts), "voxels": cur_voxels,
                                                "edit_id": cur.edit_id}  # fmt: skip
        ml = self.image_meta is not None and any(self.image_meta.get("orientation") or [])
        vox_ml = float(np.prod(ws.image.spacing)) / 1000.0
        rows = [{"id": i, "name": o["name"], "voxels": o["voxels"], "notes": o.get("notes", ""), "prompts": o.get("prompts", 0),
                 "color": o.get("color") or object_color(i), "ml": round(o["voxels"] * vox_ml, 2) if ml else None}
                for i, o in sorted(objects.items())]  # fmt: skip
        msg = {"t": "objects", "objects": rows, "current": current, "current_color": CURRENT_COLOR}
        return patches, {"objects_msg": msg, "objects": objects, "current": current}

    async def _sync_meshes(self, ws: AgentWorkspace, snap: dict) -> None:
        if ws.image.is_2d:
            return
        affine = layers.array_to_ras(ws.image)
        colors = {f"o{i}": o.get("color") or object_color(i) for i, o in snap["objects"].items()}
        want = {f"o{i}": (o["name"], o["voxels"], colors[f"o{i}"]) for i, o in snap["objects"].items()}
        cur = snap["current"]
        if cur is not None and cur["voxels"] > 0:
            want["current"] = (cur["name"], cur["voxels"], cur["prompts"])
        for key in [k for k in self.meshes if k not in want]:
            del self.meshes[key]
            self._mesh_sig.pop(key, None)
            self._broadcast({"t": "mesh", "key": key, "remove": True})
        todo = [k for k, sig in want.items() if self._mesh_sig.get(k) != sig]
        if not todo:
            return
        ids = [int(k[1:]) for k in todo if k != "current"]

        def compute():
            with ws.lock:
                crops = layers.object_crops(ws.instances, ids)
                cur_crop = layers.current_crop(ws.buffer) if "current" in todo else None
            out = {f"o{i}": layers.crop_mesh(c, affine, ws.image.spacing) for i, c in crops.items()}
            if "current" in todo:
                out["current"] = layers.crop_mesh(cur_crop, affine, ws.image.spacing)
            return out

        blobs = await asyncio.to_thread(compute)
        if ws is not self.ws:
            return
        for key, blob in blobs.items():
            self._mesh_sig[key] = want[key]
            if blob is None:
                if self.meshes.pop(key, None) is not None:
                    self._broadcast({"t": "mesh", "key": key, "remove": True})
                continue
            m = self.meshes.get(key)
            color = CURRENT_COLOR if key == "current" else colors[key]
            self.meshes[key] = m = {
                "v": (m["v"] + 1) if m else 1,
                "blob": blob,
                "color": color,
                "name": want[key][0],
            }
            self._broadcast(self._mesh_msg(key, m))

    @staticmethod
    def _mesh_msg(key: str, m: dict) -> dict:
        return {"t": "mesh", "key": key, "v": m["v"], "color": m["color"], "name": m["name"],
                "url": f"api/mesh/{key}?v={m['v']}"}  # fmt: skip

    # --------------------------------------------------------------- download
    def download_zip(self) -> Path:
        ws = self.ws
        if ws is None or ws.image is None:
            raise UserError("Nothing to download yet.")
        with ws.lock:
            res = ws.save()
        out_dir = Path(res.data["path"]).parent
        stem = Path(ws.image.source).name.split(".")[0]
        zpath = self.dir / f"{stem}_segmentation.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(out_dir.iterdir()):
                z.write(f, f.name)
        return zpath


class SessionManager:
    """All browser sessions. ``llm`` is the agent's LLM (default: Claude on the subscription).
    ``workspace_factory(spec, workdir)`` and ``agent_factory(ws, on_event, **kw)`` default to the real
    workspace and the LLM's agent; tests pass fakes."""

    def __init__(self, server_spec: str | None, server_label: str, cases: list, max_sessions: int = 10,
                 idle_s: int = 1800, llm: Backend | None = None, root: Path | None = None,
                 workspace_factory=None, agent_factory=None, python_mode: str | None = None):  # fmt: skip
        # The page chooses the image and offers the download, so the agent gets no file paths.
        self.workspace_factory = workspace_factory or (
            lambda spec, workdir: AgentWorkspace(session_spec=spec, workdir=workdir, python_mode=python_mode,
                                                 confine_paths=True)  # fmt: skip
        )
        self.llm = llm or claude_backend()
        self.agent_factory = agent_factory or self.llm.session
        self.server_spec = server_spec
        self.server_label = server_label
        self.model_license: str | None = None  # of the nnInteractive model, once known
        self.cases = cases
        self.max_sessions = max_sessions
        self.idle_s = idle_s
        self.model, self.effort = self.llm.model, self.llm.effort
        self.root = root or Path.home() / ".cache" / "nninteractive_agent" / "web"
        self.runs_dir = self.root / "runs"
        self.volumes_dir = self.root / "volumes"
        self.sessions: dict = {}

    def get(self, sid: str) -> WebSession:
        if sid not in self.sessions:
            self.sessions[sid] = WebSession(sid, self)
        return self.sessions[sid]

    def seats_in_use(self, exclude: WebSession | None = None) -> int:
        return sum(1 for s in self.sessions.values()
                   if s is not exclude and s.ws is not None and (s.ws.session is not None or
                                                                 (s.attach_task and not s.attach_task.done())))  # fmt: skip

    def case_list(self) -> list:
        return [{k: c[k] for k in ("id", "title", "subtitle", "tasks", "modality")} | {"thumb": f"api/cases/{c['id']}/thumb.png"}
                for c in self.cases if c.get("available", True)]  # fmt: skip

    def volumes_for(self, img: LoadedImage, path: Path) -> tuple:
        """(key, meta) of the browser copies of an image, cached by source file."""
        st = path.stat()
        src = f"{path.resolve()}:{st.st_mtime_ns}:{st.st_size}:{layers.VOLUME_FORMAT}"
        key = hashlib.sha1(src.encode()).hexdigest()[:16]
        d = self.volumes_dir / key
        if (d / "meta.json").exists():
            return key, json.loads((d / "meta.json").read_text())
        tmp = self.volumes_dir / f"{key}.tmp{time.time_ns()}"
        meta = layers.write_volumes(img, tmp)
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
        return key, meta

    async def drop(self, sid: str) -> None:
        s = self.sessions.pop(sid, None)
        if s is not None:
            await s.close()

    async def reaper(self) -> None:
        """Pause sessions without activity (frees nnInteractive slots); forget long-dead ones."""
        while True:
            await asyncio.sleep(30)
            now = time.time()
            for sid, s in list(self.sessions.items()):
                idle = now - s.last_active
                if not s.busy and idle > self.idle_s and s.ws is not None and not s.paused:
                    await s.pause()
                if not s.clients and not s.busy and idle > 12 * 3600:
                    await self.drop(sid)

    async def shutdown(self) -> None:
        for sid in list(self.sessions):
            await self.drop(sid)


# ------------------------------------------------------------------- helpers
def _check_same_grid(labels: LoadedImage, image: LoadedImage) -> None:
    """A label map must be on the image's voxel grid (same shape and, from files, same geometry)."""
    if labels.shape != image.shape:
        raise UserError(f"The segmentation has {list(labels.shape)} voxels, the image {list(image.shape)}: "
                        "export it on the image's grid.")  # fmt: skip
    a, b = labels.sitk_reference, image.sitk_reference
    if a is None or b is None:
        return
    tol = 1e-3 * min(b.GetSpacing())
    same = (np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol) and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4))  # fmt: skip
    if not same:
        raise UserError("The segmentation is not on the image's grid (spacing, origin or axis directions differ): "
                        "export it on the image's grid.")  # fmt: skip


def _focus_point(tool: str, args: dict, text: str):
    """The voxel the agent just prompted at or looked at (for the viewer's follow mode)."""
    p = None
    if tool == "scribble" and args.get("points"):
        p = args["points"][0]
    elif tool == "point":
        p = args.get("position")
    elif tool == "bbox":
        c1, c2 = args.get("corner1"), args.get("corner2")
        if c1 and c2 and len(c1) == len(c2) == 3:
            p = [(a + b) / 2 for a, b in zip(c1, c2)]
    elif tool == "view":
        m = re.search(r"slices through \[(\d+), (\d+), (\d+)\]", text)
        p = [int(v) for v in m.groups()] if m else None
    if p is None or len(p) != 3:
        return None
    try:
        return [float(v) for v in p]
    except (TypeError, ValueError):
        return None


def _status_text(tool: str, args: dict) -> str:
    labels = {
        "image_info": "Looking at the image", "view": "Inspecting slices", "montage": "Inspecting a slice montage",
        "projection": "Looking at a projection", "stats": "Checking object statistics", "new_object": "Starting a new object",
        "bbox": "Placing a box", "scribble": "Drawing a scribble", "point": "Placing a point", "undo": "Undoing a prompt",
        "commit": "Committing the object", "discard": "Discarding the object", "edit_object": "Re-opening an object",
        "delete_object": "Deleting an object", "list_objects": "Listing objects", "save": "Saving the result",
        "python": "Running an analysis", "Read": "Reading the skill notes", "Agent": "Starting a verifier",
        "WebSearch": "Searching the web",
    }  # fmt: skip
    text = labels.get(tool, tool)
    if tool == "new_object" and args.get("name"):
        text += f": {args['name']}"
    return text + " ..."
