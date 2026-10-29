"""AgentWorkspace: everything an agent needs to segment an image with nnInteractive.

It owns one image, one nnInteractive session (remote or local) and the objects segmented so
far. nnInteractive segments one object at a time, so the workspace keeps

* the *current* object: the session's target buffer plus the prompts given for it, and
* *committed* objects: an instance map (one id per object) with names and notes.

Every method returns a ``Result`` (text for the agent, optional PNG images, structured data),
so the same calls serve the agent tools and the agent's own Python scripts.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import metrics, pyguard, raster, render, scale
from .io import LoadedImage, default_mask_suffix, describe_orientation, load_image, save_mask
from .render import Overlay, Prompt
from .session import Reconnecting, make_session, model_license, resolve_spec

SOFT_TISSUE_WINDOW = (-160.0, 240.0)


def _looks_like_ct(data: np.ndarray) -> bool:
    step = max(1, data.size // 500_000)
    lo, hi = np.percentile(data.ravel()[::step], (0.5, 99.5))
    return bool(lo < -500 and hi > 150 and np.issubdtype(data.dtype, np.number))


def _drop_components(mask: np.ndarray, keep_largest: bool, min_size: int) -> tuple:
    bbox = metrics.bbox_of(mask)
    sl = tuple(slice(lo, hi) for lo, hi in bbox)
    labels, n = ndi.label(mask[sl], structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.zeros(n + 1, bool)
    if keep_largest:
        keep[int(np.argmax(sizes))] = True
    else:
        keep = sizes >= min_size
        keep[0] = False
    out = np.zeros_like(mask)
    out[sl] = keep[labels]
    return out, int(n - keep[1:].sum()), int(mask.sum() - out.sum())


def _headless_matplotlib() -> None:
    """Agent code draws figures for itself, often in a worker thread; a GUI backend there warns
    ("Starting a Matplotlib GUI outside of the main thread will likely fail") and can crash the
    process later. Use Agg unless pyplot is already in use with another backend (napari)."""
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")


class AgentError(Exception):
    """A usage error the agent can fix (bad coordinates, wrong order of calls, ...)."""


@dataclass
class Result:
    text: str
    images: list = field(default_factory=list)  # PNG bytes
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    __repr__ = __str__


@dataclass
class CurrentObject:
    name: str
    prompts: list = field(default_factory=list)  # render.Prompt
    edit_id: int | None = None  # set when re-opening a committed object
    edit_backup: np.ndarray | None = None
    voxel_history: list = field(default_factory=list)
    previewed_boxes: list = field(default_factory=list)  # (axis, index, lo, hi) of bbox previews
    previewed_lines: list = field(default_factory=list)  # (axis, index, points) of scribble previews
    factors: tuple = scale.FULL  # downsampling per axis of the image nnInteractive segments this object on
    size_note_given: bool = False


class AgentWorkspace:
    def __init__(self, session_spec: str | None = None, workdir=None, session=None, python_mode: str | None = None,
                 confine_paths: bool = False):  # fmt: skip
        """``python_mode``: full, guarded or off (see core/pyguard.py; default $NNINTERACTIVE_AGENT_PYTHON,
        then full). ``confine_paths``: the front end chooses the image and where results go (web demo), so
        ``load_image`` and ``save(out_dir)`` are refused."""
        self._session_spec = session_spec
        self.python_mode = pyguard.resolve_mode(python_mode)
        self._notes: list = []  # for the agent with the next tool result (e.g. a restored session)
        self.confine_paths = confine_paths
        self.session = session
        self.workdir = Path(workdir or Path.cwd() / "runs" / time.strftime("%Y%m%d_%H%M%S")).resolve()
        self.image: LoadedImage | None = None
        self.window: tuple | None = None
        self.buffer: np.ndarray | None = None  # full-resolution mask of the current object
        # Large objects are segmented on a downsampled copy of the image (see core/scale.py). The
        # session then predicts into ``_low`` and ``buffer`` holds the upsampled mask; otherwise
        # ``buffer`` itself is the session's target buffer.
        self._low: np.ndarray | None = None
        self._session_factors: tuple = scale.FULL
        self._low_images: dict = {}  # factors -> downsampled intensities
        self._prev: np.ndarray | None = None  # buffer before the last prompt (for change counts)
        self.instances: np.ndarray | None = None
        self.objects: dict = {}  # id -> {"name", "voxels", "notes", "prompts"}
        self.current: CurrentObject | None = None
        self._next_id = 1
        self._render_count = 0
        self.n_prompts = 0  # all prompts sent, including those from agent Python code
        self._py_namespace: dict | None = None
        self._in_python = False
        self._synced_crc: int | None = None  # buffer checksum after the last workspace change (during python)
        self.lock = threading.RLock()
        # Action log for analysing what the agent did; with snapshots on (benchmark/run.py sets the
        # variable) each entry also keeps the object's mask, so each object can be scored at its commit.
        self._action_seq = 0
        self.snapshots = bool(os.environ.get("NNINTERACTIVE_AGENT_SNAPSHOTS"))

    @property
    def session_spec(self) -> str:
        """Where nnInteractive runs: a server URL or ``local``."""
        return resolve_spec(self._session_spec)

    # ------------------------------------------------------------------ image
    def load_image(self, path, spacing=None, channel=None) -> Result:
        if self.confine_paths:
            raise AgentError("the image is chosen in the front end; load_image is not available here")
        return self.set_image(load_image(path, spacing=spacing, channel=channel))

    def set_image(self, img: LoadedImage) -> Result:
        """Start over with an already loaded image (e.g. taken from a napari layer)."""
        if self.session is None:
            self.session = self._connect()
        t0 = time.time()
        self.session.set_image(img.model_input(), {"spacing": list(img.spacing)})
        self.buffer = np.zeros(img.shape, dtype=np.uint8)
        self.session.set_target_buffer(self.buffer)
        self._prev = np.zeros_like(self.buffer)
        self.instances = np.zeros(img.shape, dtype=np.uint16)
        self.objects, self.current, self._next_id = {}, None, 1
        self._low, self._session_factors, self._low_images = None, scale.FULL, {}
        self.image = img
        self.window = render.auto_window(img.data)
        if _looks_like_ct(img.data):
            self.window = SOFT_TISSUE_WINDOW
        self.workdir.mkdir(parents=True, exist_ok=True)
        info = self.info()
        info.text += f"\n(image sent to nnInteractive in {time.time() - t0:.1f} s)"
        info.text += f"\nnnInteractive model license: {model_license(self.session)} (tell the user once)"
        return info

    def close(self) -> None:
        """Release the nnInteractive session (frees the slot on a shared server). Idempotent."""
        session, self.session = self.session, None
        close = getattr(session, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 - best effort during shutdown
                pass

    def _connect(self):
        """A new session that reconnects by itself when the server drops it."""
        return Reconnecting(make_session(self._session_spec), self._restored_session)

    def _restored_session(self, note: bool = True):
        """A new raw session with the state the old one had: the image at the resolution the open
        object is segmented on, and the open object's mask as its starting point (its prompt history,
        and so undo, is lost). Committed objects live in the workspace and need nothing."""
        img = self._need_image()
        mask = (self.buffer > 0).astype(np.uint8) if self.current is not None else None
        raw = make_session(self._session_spec)
        if self._session_factors == scale.FULL:
            raw.set_image(img.model_input(), {"spacing": list(img.spacing)})
            raw.set_target_buffer(self.buffer)
        else:
            data = self._low_images[self._session_factors]
            raw.set_image(data[None], {"spacing": [s * f for s, f in zip(img.spacing, self._session_factors)]})
            raw.set_target_buffer(self._low)
        if mask is not None and mask.any():
            low = mask if self._low is None else scale.downsample_mask(mask, self._session_factors)
            raw.add_initial_seg_interaction(low, run_prediction=False)
            self.buffer[:] = mask
        else:
            self.buffer[:] = 0
        self._prev[:] = self.buffer
        self._mark_synced()
        if note:
            self._notes.append(
                "NOTE: the nnInteractive server had closed this session (idle timeout or restart). A new session "
                "was opened and the image restored automatically"
                + (f"; the open object '{self.current.name}' continues from its current mask, and its earlier "
                   "prompts can no longer be undone" if self.current is not None else "")
                + ". Committed objects are unaffected. Nothing to do."
            )
        return raw

    def take_notes(self) -> list:
        """Notes for the agent that came up since the last call (and forget them)."""
        notes, self._notes = self._notes, []
        return notes

    def reattach(self) -> Result:
        """Claim a new nnInteractive session for the loaded image after ``close()``. Committed objects
        stay; an open object continues from its current mask (its prompt history is lost)."""
        img = self._need_image()
        self.session = Reconnecting(self._restored_session(note=False), self._restored_session)
        kept = f"; the open object '{self.current.name}' continues from its current mask" if self.current else ""
        return Result(f"new nnInteractive session for {Path(img.source).name}; {len(self.objects)} objects kept{kept}")

    def info(self) -> Result:
        img = self._need_image()
        d = img.data
        step = max(1, d.size // 2_000_000)
        p = np.percentile(d.ravel()[::step], (0.5, 5, 50, 95, 99.5))
        lines = [
            f"image: {Path(img.source).name}",
            f"shape [axis0, axis1, axis2] = {list(img.shape)}"
            + ("  (2D image: axis0 is a dummy axis, always 0)" if img.is_2d else ""),
            f"spacing = {[round(s, 4) for s in img.spacing]}" + (" (from file)" if img.sitk_reference else ""),
            f"dtype = {d.dtype}; intensity percentiles 0.5/5/50/95/99.5% = "
            + ", ".join(metrics.fmt_num(v) for v in p),
            f"default display window = [{metrics.fmt_num(self.window[0])}, {metrics.fmt_num(self.window[1])}]",
        ]
        if not img.is_2d:
            lines += describe_orientation(img.orientation)
        if img.rgb is not None:
            lines.append("color image: views show the original colors; nnInteractive sees the luminance")
        if _looks_like_ct(d):
            lines.append(
                "looks like CT in Hounsfield units. Useful windows: soft tissue [-160, 240] (default), "
                "bone [-400, 1200], lung [-1200, 200]"
            )
        lines += img.notes
        return Result("\n".join(lines), data={"shape": list(img.shape), "spacing": list(img.spacing)})

    # ---------------------------------------------------------------- objects
    def new_object(self, name: str, extent=None) -> Result:
        """Start an object. ``extent``: its estimated size in voxels per axis; an object larger than
        ``scale.TARGET_EXTENT`` is segmented on a downsampled copy of the image."""
        img = self._need_image()
        if self.current is not None and self.buffer.any():
            raise AgentError(
                f"object '{self.current.name}' is still open; commit() or discard() it before starting a new one"
            )
        try:
            factors = scale.factors_for_extent(extent, img.shape)
        except ValueError as e:
            raise AgentError(str(e)) from None
        switched = self._use_factors(factors)
        self._clear_current()
        self.current = CurrentObject(name=str(name), factors=factors)
        self._record("new_object", extent=extent)
        msg = f"new object '{name}' started (id will be #{self._next_id} on commit)"
        if factors != scale.FULL:
            msg += (
                f". It is large, so nnInteractive segments it on a downsampled copy of the image ({scale.describe(factors)}) "
                "and the mask is scaled back up; keep using full-resolution coordinates."
            )
        return Result(msg + switched)

    def _use_factors(self, factors: tuple) -> str:
        """Give the session the image at the resolution an object is segmented on."""
        if factors == self._session_factors:
            return ""
        img, t0 = self.image, time.time()
        if factors == scale.FULL:
            self.session.set_image(img.model_input(), {"spacing": list(img.spacing)})
            self.session.set_target_buffer(self.buffer)
            self._low = None
        else:
            if factors not in self._low_images:
                self._low_images = {factors: scale.downsample_image(img.data, factors)}  # keep one copy
            data = self._low_images[factors]
            self.session.set_image(data[None], {"spacing": [s * f for s, f in zip(img.spacing, factors)]})
            self._low = np.zeros(data.shape, dtype=np.uint8)
            self.session.set_target_buffer(self._low)
        self._session_factors = factors
        what = "full-resolution image" if factors == scale.FULL else f"image at {scale.describe(factors)}"
        return f" ({what} sent to nnInteractive in {time.time() - t0:.1f} s)"

    def _clear_current(self) -> None:
        """No open object: empty session history and masks."""
        self.session.reset_interactions()
        self.buffer[:] = 0
        self._prev[:] = 0
        self._mark_synced()

    def commit(self, notes: str = "", keep_largest: bool = False, min_component_size: int = 0,
               overlap: str | None = None) -> Result:  # fmt: skip
        """Add the open object to the instance map.

        ``keep_largest`` / ``min_component_size`` drop stray connected components first (a
        clean-up step applied to the final mask, outside nnInteractive).

        Every voxel belongs to one object. Where the object overlaps committed ones, ``overlap``
        decides: ``"keep"`` leaves those voxels to them (a leak into a neighbour is cut off),
        ``"replace"`` gives them to this object and the others shrink (a structure inside another:
        lesions in an organ). Without it small overlaps are kept, and an object that would lose
        more than half of itself is not committed until the caller chooses.
        """
        if overlap not in (None, "keep", "replace"):
            raise AgentError("overlap must be 'keep' or 'replace'")
        cur = self._need_current()
        mask = self.buffer > 0
        if not mask.any():
            raise AgentError("the current object is empty; add prompts or discard() it")
        removed_note = ""
        if keep_largest or min_component_size > 0:
            mask, n_removed, vox_removed = _drop_components(mask, keep_largest, int(min_component_size))
            if not mask.any():
                raise AgentError("the clean-up would remove the whole object")
            removed_note = f"\nclean-up removed {n_removed} component(s), {vox_removed:,} voxels"
        oid = cur.edit_id if cur.edit_id is not None else self._next_id
        stats = self._stats(mask, exclude_id=oid)
        n_mask, n_over = int(mask.sum()), sum(stats["overlaps"].values())
        owners = ", ".join(f"#{i} '{self.objects[i]['name']}' ({c:,} voxels)" for i, c in stats["overlaps"].items())
        if overlap is None and n_over > n_mask / 2:
            raise AgentError(
                f"not committed: {n_over:,} of this object's {n_mask:,} voxels already belong to {owners}. Every voxel "
                "belongs to one object, so they would stay there and this object would lose most of itself. If it "
                "lies inside them (a lesion in an organ, a vessel in an organ), commit with overlap='replace': it "
                "takes those voxels and they shrink. If it leaked into them, correct it first (or overlap='keep')."
            )
        if overlap == "keep" and n_over == n_mask:
            raise AgentError(f"not committed: every voxel of this object already belongs to {owners}")
        if overlap == "replace":
            self.instances[mask] = oid
            for i, c in stats["overlaps"].items():
                self.objects[i]["voxels"] = int(np.count_nonzero(self.instances == i))
                given = f"{c:,} voxels given to #{oid} '{cur.name}' (overlap='replace')"
                self.objects[i]["notes"] = f"{self.objects[i]['notes']}; {given}".lstrip("; ")
            free, lost = mask, 0
        else:
            free = mask & (self.instances == 0)
            self.instances[free] = oid
            lost = int(n_mask - free.sum())
        self.objects[oid] = {
            "name": cur.name,
            "voxels": int(free.sum()),
            "notes": notes,
            # a re-opened object keeps counting its earlier prompts
            "prompts": len(cur.prompts) + (self.objects[oid]["prompts"] if cur.edit_id is not None else 0),
        }
        if cur.factors != scale.FULL:
            self.objects[oid]["factors"] = list(cur.factors)
        if cur.edit_id is None:
            self._next_id += 1
        self._record("commit", mask=self.instances == oid, id=oid, notes=notes)
        self._clear_current()
        self.current = None
        msg = f"committed #{oid} '{cur.name}': {int(free.sum()):,} voxels, {len(cur.prompts)} prompts" + removed_note
        if lost:
            msg += (
                f"\nNOTE: {lost:,} voxels overlapped committed objects and were left to them "
                f"({', '.join(f'#{i}' for i in stats['overlaps'])}; overlap='replace' would have taken them)"
            )
        if overlap == "replace" and stats["overlaps"]:
            msg += "\ntook " + ", ".join(
                f"{c:,} voxels from #{i} '{self.objects[i]['name']}' (now {self.objects[i]['voxels']:,})"
                for i, c in stats["overlaps"].items()
            )
            for i in [i for i in stats["overlaps"] if self.objects[i]["voxels"] == 0]:
                msg += f"\n#{i} '{self.objects[i]['name']}' is now empty: delete_object it unless that was a mistake"
        return Result(msg, data={"id": oid})

    def discard(self) -> Result:
        cur = self._need_current()
        self._record("discard")
        self._clear_current()
        if cur.edit_backup is not None:
            self.instances[cur.edit_backup] = cur.edit_id
        self.current = None
        return Result(f"discarded '{cur.name}'" + (f" (#{cur.edit_id} restored)" if cur.edit_id else ""))

    def edit_object(self, object_ref) -> Result:
        """Re-open a committed object; its mask becomes the starting point for more prompts."""
        self._need_image()
        if self.current is not None and self.buffer.any():
            raise AgentError(f"object '{self.current.name}' is still open; commit() or discard() it first")
        oid = self._resolve(object_ref)
        factors = tuple(self.objects[oid].get("factors", scale.FULL))
        switched = self._use_factors(factors)
        mask = self.instances == oid
        self.instances[mask] = 0
        self.current = CurrentObject(name=self.objects[oid]["name"], edit_id=oid, edit_backup=mask, factors=factors)
        self.buffer[:] = mask
        self._send_mask()
        self._record("edit_object", id=oid)
        return Result(
            f"re-opened #{oid} '{self.current.name}' ({int(mask.sum()):,} voxels). Correct it with "
            "scribbles/points, then commit() (keeps id) or discard() (restores the old mask)." + switched
        )

    def _send_mask(self) -> None:
        """Make ``buffer`` the session's starting mask for the open object (clears its prompt history)."""
        mask = (self.buffer > 0).astype(np.uint8)
        if self._low is None:
            self.session.add_initial_seg_interaction(mask, run_prediction=False)
        else:
            self.session.add_initial_seg_interaction(scale.downsample_mask(mask, self._session_factors),
                                                     run_prediction=False)  # fmt: skip
            self.buffer[:] = mask  # keep the full-resolution detail until a prompt changes a region
        self._prev[:] = self.buffer
        self._mark_synced()

    def delete_object(self, object_ref) -> Result:
        oid = self._resolve(object_ref)
        self.instances[self.instances == oid] = 0
        name = self.objects.pop(oid)["name"]
        self._record("delete_object", mask=np.zeros((1, 1, 1), bool), id=oid, object=name)
        return Result(f"deleted #{oid} '{name}'")

    def import_objects(self, labels: np.ndarray, segments: dict | None = None, mode: str = "replace") -> Result:
        """Objects from a label map on the image grid (e.g. the user's existing segmentation): each label
        value becomes an object. ``segments``: label value -> {"name", "notes"}. ``mode``: ``"replace"``
        (the label map becomes all objects) or ``"add"`` (new objects next to the existing ones; they take
        the voxels they cover). An open object is discarded first."""
        img = self._need_image()
        if mode not in ("replace", "add"):
            raise AgentError("mode must be 'replace' or 'add'")
        labels = np.asarray(labels)
        if labels.shape != img.shape:
            raise AgentError(f"the label map has shape {list(labels.shape)}, the image {list(img.shape)}")
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.array_equal(labels, np.round(labels)):
                raise AgentError("the label map must contain integer labels")
            labels = labels.astype(np.int64)
        values = [int(v) for v in np.unique(labels) if v != 0]
        if values and (values[0] < 0 or values[-1] > np.iinfo(np.uint16).max):
            raise AgentError("label values must be between 1 and 65535")
        segments = segments or {}
        discarded = ""
        if self.current is not None:
            discarded = f" (the open object '{self.current.name}' was discarded)"
            self.discard()
        if mode == "replace":
            self.instances[:] = 0
            self.objects, self._next_id = {}, 1
        lut = np.zeros((values[-1] + 1) if values else 1, dtype=np.uint16)
        for v in values:
            seg = segments.get(v) or {}
            oid = self._next_id
            self._next_id += 1
            lut[v] = oid
            self.objects[oid] = {"name": str(seg.get("name") or f"object {v}"), "voxels": 0,
                                 "notes": str(seg.get("notes") or ""), "prompts": 0}  # fmt: skip
        if values:
            mapped = lut[np.clip(labels, 0, values[-1])]
            covered = mapped > 0
            self.instances[covered] = mapped[covered]
        counts = np.bincount(self.instances.ravel(), minlength=max(self.objects, default=0) + 1)
        emptied = []
        for i in list(self.objects):
            self.objects[i]["voxels"] = int(counts[i])
            if counts[i] == 0:  # an existing object the added ones covered completely
                emptied.append(f"#{i} '{self.objects.pop(i)['name']}'")
        new = [i for i in lut[values]] if values else []
        self._record("import_objects", mask=self.instances > 0, mode=mode, ids=[int(i) for i in new])
        what = "replaced all objects with" if mode == "replace" else "added"
        listed = ", ".join(f"#{i} '{self.objects[int(i)]['name']}' ({self.objects[int(i)]['voxels']:,} voxels)"
                           for i in new if int(i) in self.objects)  # fmt: skip
        text = f"imported a label map: {what} {len(new)} object(s){': ' + listed if listed else ''}{discarded}"
        if emptied:
            text += f"; removed {', '.join(emptied)}, now completely covered"
        return Result(text, data={"ids": [int(i) for i in new]})

    def list_objects(self) -> Result:
        lines = [f"#{i} {o['name']}: {o['voxels']:,} voxels, {o['prompts']} prompts" + (f" - {o['notes']}" if o["notes"] else "")
                 for i, o in sorted(self.objects.items())]  # fmt: skip
        if self.current is not None:
            lines.append(
                f"open: '{self.current.name}' ({int(np.count_nonzero(self.buffer)):,} voxels, "
                f"{len(self.current.prompts)} prompts)"
            )
        return Result("\n".join(lines) or "no objects yet")

    # ---------------------------------------------------------------- prompts
    def bbox(self, corner1, corner2, preview: bool = False, require_preview: bool = False) -> Result:
        """Positive 2D box from two inclusive corners in one slice. Only as an object's first prompt.

        ``preview`` draws the box on the image without sending it. ``require_preview`` (the agent
        tool sets it) refuses a box that was not previewed with about the same corners first:
        boxes read off rendered views are often misplaced, boxes computed in code are not.
        """
        cur = self._need_current()
        if cur.prompts:
            raise AgentError(
                "bboxes are only allowed as the first prompt of an object. Correct the result with scribbles "
                "(or points for very small regions), or discard() and start over."
            )
        img = self.image
        c1 = raster.as_point(corner1, img.shape, img.is_2d)
        c2 = raster.as_point(corner2, img.shape, img.is_2d)
        bbox, axis = raster.bbox_from_corners(c1, c2)
        if img.is_2d and axis != 0:
            raise AgentError("in a 2D image both corners have axis0 = 0; give two different corners")
        lo, hi = [b[0] for b in bbox], [b[1] - 1 for b in bbox]
        prompt = Prompt(len(cur.prompts) + 1, "bbox", True, [c1, c2], axis, c1[axis], preview=preview)
        if preview:
            cur.previewed_boxes.append((axis, c1[axis], lo, hi))
            return self._preview_bbox(prompt, lo, hi)
        if require_preview and not any(
            a == axis and k == c1[axis] and max(abs(x - y) for x, y in zip(lo + hi, plo + phi)) <= 2
            for a, k, plo, phi in cur.previewed_boxes
        ):
            raise AgentError(
                "check the box before sending it: call bbox with these corners and preview=true, look at the "
                "image, correct the corners if needed (and preview again), then send it with preview=false"
            )
        if self._low is None:
            changed = self.session.add_bbox_interaction(bbox, include_interaction=True)
        else:
            low = [[scale.to_low_index(lo_, n, m), scale.to_low_index(hi_ - 1, n, m) + 1]
                   for (lo_, hi_), n, m in zip(bbox, img.shape, self._low.shape)]  # fmt: skip
            changed = self._sync_up(self.session.add_bbox_interaction(low, include_interaction=True))
        desc = f"bbox in axis{axis}={c1[axis]}"
        res = self._after_prompt(prompt, changed, desc)
        res.text += "\n" + self._box_fit(axis, c1[axis], lo, hi)
        return res

    def _preview_bbox(self, prompt: Prompt, lo: list, hi: list) -> Result:
        img = self.image
        axis = prompt.axis
        fov = render.fov_around(img.shape, bbox=[[a, b + 1] for a, b in zip(lo, hi)], margin_frac=0.6, min_size=48)
        center = [(a + b) // 2 for a, b in zip(lo, hi)]
        center[axis] = prompt.index
        ov = Overlay(instances=self.instances, names={i: o["name"] for i, o in self.objects.items()},
                     prompts=[prompt], show_objects=True)  # fmt: skip
        rgb = render.render_orthogonal(img, center, fov, ov, self.window, axes=(axis,), px=560)
        ra, ca = raster.plane_axes(axis)
        span = ", ".join(
            f"axis{a} {lo[a]}-{hi[a]} ({hi[a] - lo[a] + 1} voxels, {(hi[a] - lo[a] + 1) * img.spacing[a]:.0f} mm)"
            for a in (ra, ca)
        )
        text = (
            f"PREVIEW, not sent: box in axis{axis}={prompt.index} covering {span}, drawn solid and labelled 'new'. "
            "Check that each edge lies just outside the object (a few voxels) and that the box does not cut the "
            "object anywhere. If it is off, preview corrected corners; if it fits, call bbox with the same corners "
            "and preview=false."
        )
        return self._image_result(rgb, "preview", text)

    def _preview_prompt(self, prompt: Prompt, center: list, desc: str, diag: str) -> Result:
        """A scribble or point drawn over the current mask without sending it."""
        img = self.image
        axes = (prompt.axis,) if prompt.axis is not None else (0, 1, 2)
        if prompt.kind == "scribble":
            # A square view about 1.5 times the line's length: close enough to see whether the line stays
            # inside the object, wide enough to show the object around it.
            pts = np.array(prompt.points)
            ra, ca = raster.plane_axes(prompt.axis)
            size = int(max(48, round(1.5 * (max(np.ptp(pts[:, ra]), np.ptp(pts[:, ca])) + 1))))
            mid = [int(round((pts[:, a].min() + pts[:, a].max()) / 2)) for a in range(3)]
            fov = render.fov_around(img.shape, center=mid, size=size)
        else:
            fov = render.fov_around(img.shape, center=center, size=64)
        center = list(center)
        if prompt.axis is not None:
            center[prompt.axis] = prompt.index
        ov = self._overlay(None, True)
        ov.prompts = ov.prompts + [prompt]
        rgb = render.render_orthogonal(img, center, fov, ov, self.window, axes=axes, px=600)
        text = (
            f"PREVIEW, not sent: {desc}, drawn heavier and labelled 'new' over the current mask (yellow).\n{diag}\n"
            "If it is where it should be, call the same prompt with preview=false; otherwise preview a corrected one."
        )
        return self._image_result(rgb, "preview", text)

    def _box_fit(self, axis: int, index: int, lo: list, hi: list) -> str:
        """How the prediction in the box's slice fits the box: a box read off a view at the wrong
        place shows up as prediction sticking out of it, or as a lot of empty box."""
        from scipy.ndimage import label

        ra, ca = raster.plane_axes(axis)
        m = np.take(self.buffer, index, axis=axis) > 0
        lab, n = label(m)
        inside = np.unique(lab[lo[ra] : hi[ra] + 1, lo[ca] : hi[ca] + 1])
        m = np.isin(lab, inside[inside > 0])  # the parts of the prediction that touch the box
        if not m.any():
            return f"box check (axis{axis}={index}): no prediction inside the box"
        ext = {ra: np.flatnonzero(m.any(1))[[0, -1]], ca: np.flatnonzero(m.any(0))[[0, -1]]}
        out, loose = [], []
        for a in (ra, ca):
            size = hi[a] - lo[a] + 1
            for side, over, slack in (("low", lo[a] - ext[a][0], ext[a][0] - lo[a]),
                                      ("high", ext[a][1] - hi[a], hi[a] - ext[a][1])):  # fmt: skip
                if over > max(3, round(0.08 * size)):
                    out.append(f"{over} voxels past the box at axis{a} ({side} side)")
                elif slack > max(6, round(0.25 * size)):
                    loose.append(f"{slack} voxels of empty box at axis{a} ({side} side)")
        if not out and not loose:
            return f"box check (axis{axis}={index}): the prediction fits the box"
        msg = f"box check (axis{axis}={index}): "
        if out:
            msg += ("WARNING the prediction extends " + "; ".join(out) + ". Either the box cut the object there "
                    "(then discard() and send a corrected box) or the prediction leaks into a neighbour (then "
                    "correct it with a negative scribble). Look at that side. ")  # fmt: skip
        if loose:
            msg += ("The box may be too large or misplaced: " + "; ".join(loose) + " (or the model stopped early "
                    "there). Look at that side.")  # fmt: skip
        return msg.strip()

    def point(self, position, positive: bool = True, preview: bool = False) -> Result:
        cur = self._need_current()
        img = self.image
        p = raster.as_point(position, img.shape, img.is_2d)
        inside = bool(self.buffer[tuple(p)])
        diag = (
            f"point is {'inside' if inside else 'outside'} the current mask; "
            f"intensity there {metrics.fmt_num(float(img.data[tuple(p)]))}"
        )
        if self.instances[tuple(p)]:
            diag += f"; it lies on committed object #{int(self.instances[tuple(p)])}"
        prompt = Prompt(len(cur.prompts) + 1, "point", bool(positive), [p], preview=preview)
        if preview:
            return self._preview_prompt(prompt, p, f"{'positive' if positive else 'negative'} point {p}", diag)
        if self._low is None:
            changed = self.session.add_point_interaction(p, include_interaction=bool(positive))
        else:
            q = scale.to_low_point(p, img.shape, self._low.shape)
            changed = self._sync_up(self.session.add_point_interaction(q, include_interaction=bool(positive)))
        return self._after_prompt(prompt, changed, f"{'positive' if positive else 'negative'} point {p}", diag)

    def scribble(self, points, positive: bool = True, axis: int | None = None, preview: bool = False,
                 require_preview: bool = False) -> Result:  # fmt: skip
        """Polyline through >= 2 points that lie in one slice, drawn 2 px wide.

        ``require_preview`` (the agent tool sets it) refuses an object's first line unless about the
        same line was previewed: lines read off rendered views can leave the object, lines computed
        in code do not."""
        cur = self._need_current()
        img = self.image
        if len(points) < 2:
            raise AgentError("a scribble needs at least 2 points")
        pts = [raster.as_point(p, img.shape, img.is_2d) for p in points]
        last_planes = tuple(q.axis for q in reversed(cur.prompts) if q.axis is not None)[:1]
        slice_axis = (int(np.argmax(img.spacing)),)
        plane_axis, index = raster.infer_plane(pts, axis, img.shape, img.is_2d, prefer=last_planes + slice_axis)
        crop3d, bbox, centerline = raster.scribble_crop(pts, plane_axis, index, img.shape)
        diag = self._line_diagnostics(centerline, bool(positive))
        prompt = Prompt(len(cur.prompts) + 1, "scribble", bool(positive), pts, plane_axis, index, preview=preview)
        if preview:
            cur.previewed_lines.append((plane_axis, index, pts))
            desc = f"{'positive' if positive else 'negative'} scribble in axis{plane_axis}={index}"
            return self._preview_prompt(prompt, [int(round(np.mean([q[a] for q in pts]))) for a in range(3)], desc, diag)
        if require_preview and not cur.prompts and not any(
            a == plane_axis and k == index and len(p) == len(pts)
            and max(abs(x - y) for q, r in zip(p, pts) for x, y in zip(q, r)) <= 2
            for a, k, p in cur.previewed_lines
        ):
            raise AgentError(
                "check the first line of an object before sending it: call scribble with these points and "
                "preview=true, look at the zoomed image (the whole line must stay inside the object), correct the "
                "points if needed (and preview again), then send the same points with preview=false"
            )
        if self._low is None:
            changed = self.session.add_scribble_interaction(
                crop3d, include_interaction=bool(positive), interaction_bbox=bbox
            )
        else:  # drawn 2 px wide on the downsampled grid nnInteractive works on
            shape_low = self._low.shape
            low_pts = [scale.to_low_point(q, img.shape, shape_low) for q in pts]
            if len({tuple(q) for q in low_pts}) == 1:
                raise AgentError(
                    "this scribble is shorter than one voxel of the downsampled image this object is segmented "
                    "on; draw a longer line or use a point"
                )
            low_index = scale.to_low_index(index, img.shape[plane_axis], shape_low[plane_axis])
            crop_low, bbox_low, _ = raster.scribble_crop(low_pts, plane_axis, low_index, shape_low)
            changed = self._sync_up(self.session.add_scribble_interaction(
                crop_low, include_interaction=bool(positive), interaction_bbox=bbox_low
            ))  # fmt: skip
        desc = f"{'positive' if positive else 'negative'} scribble in axis{plane_axis}={index}, {len(centerline)} px long"
        return self._after_prompt(prompt, changed, desc, diag)

    def undo(self) -> Result:
        cur = self._need_current()
        if not getattr(self.session, "supports_undo", False):
            raise AgentError("this nnInteractive server does not support undo")
        if not self.session.undo():
            return Result("nothing to undo (only the most recent prompt can be undone)")
        if self._low is not None:
            self._sync_up(None)
        undone = cur.prompts.pop() if cur.prompts else None
        self._prev[:] = self.buffer
        self._mark_synced()
        self._record("undo")
        stats = self._stats(self.buffer > 0, exclude_id=cur.edit_id)
        return Result(
            f"undid prompt #{undone.step if undone else '?'} ({undone.kind if undone else ''}); object now:\n"
            + metrics.format_stats(stats),
            data={"stats": stats},
        )

    def _after_prompt(self, prompt: Prompt, changed, desc: str, diag: str | None = None) -> Result:
        self.n_prompts += 1
        cur = self.current
        cur.prompts.append(prompt)
        b = tuple(slice(lo, hi) for lo, hi in changed) if changed else (slice(None),) * 3
        new, old = self.buffer[b] > 0, self._prev[b] > 0
        added, removed = int(np.count_nonzero(new & ~old)), int(np.count_nonzero(old & ~new))
        self._prev[b] = self.buffer[b]
        stats = self._stats(self.buffer > 0, exclude_id=cur.edit_id)
        cur.voxel_history.append(stats["voxels"])
        lines = [f"prompt #{prompt.step}: {desc}"]
        if diag:
            lines.append(diag)
        lines.append(f"change: +{added:,} / -{removed:,} voxels")
        if stats["voxels"] == 0:
            lines.append("WARNING: the prediction is empty")
        else:
            lines.append(metrics.format_stats(stats))
            lines += self._size_note(stats)
        self._mark_synced()
        self._record(prompt.kind, prompt=prompt, added=added, removed=removed)
        return Result("\n".join(lines), data={"stats": stats, "added": added, "removed": removed})

    def _sync_up(self, changed_low) -> list:
        """Upsample the session's low-resolution mask into ``buffer`` where it changed; returns
        the full-resolution region (``None`` from the session: everything may have changed)."""
        shape = self.image.shape
        if changed_low:
            region = scale.full_region(changed_low, shape, self._low.shape)
        else:  # the old and the new extent of the object
            old, new = metrics.bbox_of(self.buffer), metrics.bbox_of(self._low)
            if old is not None:
                self.buffer[tuple(slice(lo, hi) for lo, hi in old)] = 0
            if new is None:
                return [[0, s] for s in shape]
            region = scale.full_region(new, shape, self._low.shape)
            if old is not None:
                region = metrics.union_bbox(region, old)
        self.buffer[tuple(slice(lo, hi) for lo, hi in region)] = scale.upsample_mask(self._low, shape, region)
        return region

    def _size_note(self, stats: dict) -> list:
        """Once per object: point out a mask larger than nnInteractive handles well at this scale."""
        cur = self.current
        ext = [hi - lo for lo, hi in stats["bbox"]]
        if cur.size_note_given or max(e / f for e, f in zip(ext, cur.factors)) <= scale.THRESHOLD_EXTENT * 1.15:
            return []
        cur.size_note_given = True
        reduced = " (even at the reduced resolution)" if cur.factors != scale.FULL else ""
        note = (
            f"note: the object spans {ext} voxels. nnInteractive is more accurate on objects up to about "
            f"{scale.THRESHOLD_EXTENT} voxels per axis{reduced}. If this is its real size and not a leak, and it "
            f"is not a thin structure, discard() and start it again with new_object(name, extent={ext})."
        )
        return [note]

    def _line_diagnostics(self, centerline: np.ndarray, positive: bool) -> str:
        idx = tuple(centerline.T)
        inside = float(np.mean(self.buffer[idx] > 0))
        inst = self.instances[idx]
        vals = self.image.data[idx]
        it = metrics.intensity_summary(vals)
        msg = (
            f"line: {inside:.0%} inside the current mask; intensity along the line median "
            f"{metrics.fmt_num(it['median'])} (5-95%: {metrics.fmt_num(it['p5'])} to {metrics.fmt_num(it['p95'])})"
        )
        hit = {int(i): int(c) for i, c in zip(*np.unique(inst[inst > 0], return_counts=True))}
        if hit:
            msg += "; crosses committed " + ", ".join(f"#{i} ({c} px)" for i, c in hit.items())
            if positive:
                msg += " - WARNING: a positive scribble on another object pulls it into this one"
        if not positive and inside == 0 and self.buffer.any():
            msg += (
                " - note: a negative scribble entirely outside the mask removes nothing; to cut away a wrong "
                "region, draw it inside that region"
            )
        return msg

    # ------------------------------------------------------------ inspection
    def stats(self, object_ref=None) -> Result:
        oid, mask = self._mask_for(object_ref)
        if mask is None:
            raise AgentError("no object to describe")
        st = self._stats(mask, exclude_id=oid)
        name = self.objects[oid]["name"] if oid else self.current.name
        text = f"{'#' + str(oid) + ' ' if oid else 'current '}'{name}': " + metrics.format_stats(st)
        if oid is None and len(self.current.voxel_history) >= 2:
            text += "\nvoxels after each prompt: " + ", ".join(f"{v:,}" for v in self.current.voxel_history[-8:])
        return Result(text, data={"stats": st})

    def view(self, center=None, object_ref=None, axes=(0, 1, 2), size=None, window=None, show_objects=True,
             overlay=True) -> Result:  # fmt: skip
        """Orthogonal slices through ``center`` (default: middle of the object).

        ``overlay=False`` shows the bare image (no masks, no prompts) to judge boundaries."""
        img = self._need_image()
        oid, mask = self._mask_for(object_ref, allow_none=True)
        if center is not None:
            c = raster.as_point(center, img.shape, img.is_2d)
        elif mask is not None:
            c = render.object_center(mask)
        else:
            c = [s // 2 for s in img.shape]
        if size is not None:
            fov = render.fov_around(img.shape, center=c, size=size)
        elif mask is not None:
            fov = render.fov_around(img.shape, bbox=metrics.bbox_of(mask))
            fov = [[min(lo, ci), max(hi, ci + 1)] for (lo, hi), ci in zip(fov, c)]
        else:
            fov = render.fov_around(img.shape)
        ov = self._overlay(oid, show_objects) if overlay else Overlay(show_objects=False)
        rgb = render.render_orthogonal(img, c, fov, ov, self._win(window), axes=axes)
        what = f"#{oid} {self.objects[oid]['name']}" if oid else (f"'{self.current.name}'" if self.current else "image")
        text = f"view of {what}: slices through {c}; shown region " + ", ".join(
            f"axis{a} {lo}-{hi - 1}" for a, (lo, hi) in enumerate(fov)
        )
        return self._image_result(rgb, "view", text)

    def montage(self, axis: int = 0, n: int = 9, object_ref=None, slice_range=None, window=None, show_objects=True,
                overlay=True) -> Result:  # fmt: skip
        """``n`` evenly spaced slices along ``axis`` through the object (or ``slice_range``)."""
        img = self._need_image()
        if img.is_2d:
            raise AgentError("montage needs a 3D image; use view() for 2D images")
        oid, mask = self._mask_for(object_ref, allow_none=True)
        bbox = metrics.bbox_of(mask) if mask is not None else None
        if slice_range is not None:
            lo, hi = int(slice_range[0]), int(slice_range[1])
        elif bbox is not None:
            lo, hi = bbox[axis][0], bbox[axis][1] - 1
        else:
            lo, hi = 0, img.shape[axis] - 1
        lo, hi = max(0, lo), min(img.shape[axis] - 1, hi)
        n = max(1, min(int(n), 16, hi - lo + 1))
        indices = np.unique(np.round(np.linspace(lo, hi, n)).astype(int))
        fov = render.fov_around(img.shape, bbox=bbox) if bbox is not None else render.fov_around(img.shape)
        ov = self._overlay(oid, show_objects) if overlay else Overlay(show_objects=False)
        rgb = render.render_montage(img, axis, indices, fov, ov, self._win(window))
        text = f"montage along axis{axis}: slices {', '.join(map(str, indices))}"
        return self._image_result(rgb, "montage", text)

    def overview(self, window=None) -> Result:
        img = self._need_image()
        ov = self._overlay(None, True)
        if img.is_2d:
            rgb = render.render_panel(img, 0, 0, (0, img.shape[1]), (0, img.shape[2]), ov, self._win(window), px=900)
            return self._image_result(rgb, "overview", "whole 2D image")
        c = [s // 2 for s in img.shape]
        full = render.fov_around(img.shape)
        r1 = self._image_result(
            render.render_orthogonal(img, c, full, ov, self._win(window), px=440), "overview",
            f"central slices through {c}",
        )  # fmt: skip
        axis = int(np.argmax(img.spacing)) if max(img.spacing) > 1.5 * min(img.spacing) else 0
        idx = np.unique(np.round(np.linspace(0, img.shape[axis] - 1, 11)[1:-1]).astype(int))
        r2 = self._image_result(
            render.render_montage(img, axis, idx, full, ov, self._win(window), px=300), "overview",
            f"montage along axis{axis}: slices {', '.join(map(str, idx))}",
        )  # fmt: skip
        return Result(r1.text + "\n" + r2.text, images=r1.images + r2.images)

    def render3d(self, object_ref=None, views=None, show_neighbors: bool = True) -> Result:
        """Depth-shaded 3D views of one object (the open one by default) from several directions."""
        img = self._need_image()
        if img.is_2d:
            raise AgentError("render3d needs a 3D image; use view for a 2D image")
        oid, mask = self._mask_for(object_ref)
        if mask is None or not mask.any():
            raise AgentError("no open object with voxels; give object=<id or name> for a committed one")
        name = self.objects[oid]["name"] if oid is not None else self.current.name
        try:
            parsed = [render.parse_view(img, v) for v in (views or render.default_views(img, mask))]
        except ValueError as e:
            raise AgentError(str(e)) from e
        others = None
        if show_neighbors and self.instances.any():
            others = np.where(self.instances == oid, 0, self.instances) if oid is not None else self.instances
        fov = render.fov_around(img.shape, bbox=metrics.bbox_of(mask), margin_frac=0.25, min_size=24)
        sl = tuple(slice(lo, hi) for lo, hi in fov)
        _, n_comp = ndi.label(mask[sl], structure=np.ones((3, 3, 3), bool))
        color = render.object_color(oid) if oid is not None else render.CURRENT_COLOR
        ref = f"#{oid} '{name}'" if oid is not None else f"the open object '{name}'"
        rgb = render.render_3d(img, mask, others, color, parsed, fov, title=f"#{oid} {name}" if oid else name)
        text = (
            f"3D views of {ref} ({int(mask.sum()):,} voxels, {n_comp} connected component(s)), seen from "
            + ", ".join(label for _, _, label in parsed)
            + ". Shaded surface: brighter = facing you and closer. Neighbouring committed objects are faint in "
            "their colors where they lie in front of or beside it. Each panel keeps the axes and tick labels of the "
            "slice view of its plane (a view from the back looks through the object), so positions can be read off "
            "and checked in slices. Look for cut-off ends or lobes, protrusions or bridges into neighbours, stray "
            "pieces and holes; judge where a boundary belongs in slices, not here."
        )
        return self._image_result(rgb, "render3d", text)

    def projection(self, axis: int = 0, mode: str = "max", window=None) -> Result:
        """Projection of the whole image with all committed objects outlined (global check)."""
        img = self._need_image()
        if img.is_2d:
            return self.view(show_objects=True)
        rgb = render.render_projection(img, axis, self._overlay(None, True), self._win(window), mode=mode)
        return self._image_result(rgb, "projection", f"{mode} projection along axis{axis} with all committed objects")

    # ------------------------------------------------------------------ output
    def save(self, out_dir=None) -> Result:
        img = self._need_image()
        if out_dir and self.confine_paths:
            raise AgentError("this front end decides where results go: call save without out_dir")
        out = Path(out_dir) if out_dir else self.workdir / "output"
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(img.source).name.split(".")[0]
        path = save_mask(self.instances, out / f"{stem}_instances{default_mask_suffix(img)}", img)
        (out / "objects.json").write_text(json.dumps({str(k): v for k, v in sorted(self.objects.items())}, indent=2))
        msg = f"saved {len(self.objects)} objects to {path} (+ objects.json)"
        if self.current is not None and self.buffer.any():
            msg += f"\nNOTE: open object '{self.current.name}' is not committed and was not saved"
        return Result(msg, data={"path": str(path)})

    # ----------------------------------------------------------------- python
    def run_python(self, code: str) -> Result:
        """Run agent code with ``ws`` (this workspace), ``np``, ``ndi``, ``skimage`` and ``show(x)``."""
        if self.python_mode == "off":
            raise AgentError("the python tool is switched off (--python off)")
        if self.python_mode == "guarded":
            problems = pyguard.check(code, frozenset({"load_image"}) if self.confine_paths else frozenset())
            if problems:
                raise AgentError("guarded mode refused this code (nothing was run):\n" + "\n".join(problems)
                                 + "\n" + pyguard.GUARDED_RULES)  # fmt: skip
        if self._py_namespace is None:
            import scipy.ndimage as ndi
            import skimage

            _headless_matplotlib()

            self._py_namespace = {"ws": self, "np": np, "ndi": ndi, "skimage": skimage, "Result": Result}
            if self.python_mode == "guarded":
                self._py_namespace["__builtins__"] = pyguard.guarded_builtins()
        images: list = []
        shown_figures: set = set()

        def show(x, title: str | None = None):
            """Attach an image: an RGB/gray array, a matplotlib Figure, or a Result with images."""
            if isinstance(x, Result):
                images.extend(x.images)
                return
            if hasattr(x, "savefig"):
                shown_figures.add(id(x))
                # at most ~1500 px on the long side: huge figures cost tokens and are downscaled anyway
                dpi = min(100.0, 1500.0 / max(x.get_size_inches()))
                buf = io.BytesIO()
                x.savefig(buf, format="png", dpi=dpi)
                images.append(buf.getvalue())
                return
            a = np.asarray(x)
            if a.ndim == 2:
                a = a.astype(np.float32)
                lo, hi = np.percentile(a, (0.5, 99.5))
                a = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
                a = np.stack([a] * 3, -1)
            images.append(render.to_png(a.astype(np.uint8)))

        self._py_namespace["show"] = show
        out = io.StringIO()
        ok = True
        self._in_python = True
        self._mark_synced()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    exec(compile(code, "<agent-python>", "exec"), self._py_namespace)
                except Exception:  # noqa: BLE001 - the traceback goes back to the agent
                    ok = False
                    traceback.print_exc()
        finally:
            self._in_python = False
        n_auto = self._show_pyplot_figures(show, shown_figures)
        text = out.getvalue()
        if len(text) > 12000:
            text = text[:6000] + f"\n... [{len(text) - 12000} characters cut] ...\n" + text[-6000:]
        if not text.strip():
            text = "(no output)" if ok else "(failed without output)"
        if n_auto:
            text += f"\n[{n_auto} matplotlib figure(s) attached automatically; show(fig) does this explicitly]"
        if len(images) > 6:
            text += f"\n[{len(images) - 6} more images were not returned; show() at most 6 per call]"
            images = images[:6]
        changed = self._recount()
        if changed:
            text += "\nobject sizes after this code: " + ", ".join(f"#{i} {v:,} voxels" for i, v in changed)
        if self.current is not None and self._buffer_crc() != self._synced_crc:
            # nnInteractive keeps its own copy of the mask; without this the edit would be ignored
            # (or overwritten) by the next prompt.
            self._send_mask()
            self._record("set_mask", via="python")
            text += (
                f"\nws.buffer was changed by this code: the open object continues from the changed mask "
                f"({int(np.count_nonzero(self.buffer)):,} voxels). nnInteractive no longer remembers the earlier "
                "prompts, so undo() cannot go back past this point."
            )
        return Result(text, images=images)

    def _record(self, action: str, mask=None, prompt: Prompt | None = None, via: str | None = None, **fields) -> None:
        """Append one state-changing action to ``<workdir>/actions.jsonl``: what was done, through
        which path (tool or agent code), and the object's size afterwards (its mask too with
        snapshots on). Never fails the action it records."""
        try:
            self._action_seq += 1
            cur = self.current
            rec = {"seq": self._action_seq, "time": round(time.time(), 2),
                   "via": via or ("python" if self._in_python else "tool"), "action": action,
                   "object": cur.name if cur is not None else fields.pop("object", None)}  # fmt: skip
            if cur is not None and cur.factors != scale.FULL:
                rec["factors"] = list(cur.factors)
            if prompt is not None:
                rec.update(positive=prompt.positive, axis=prompt.axis, index=prompt.index, points=prompt.points)
                if prompt.kind == "scribble":
                    pts = np.asarray(prompt.points, float)
                    rec["length"] = round(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), 1)
            rec.update(fields)
            m = (self.buffer > 0) if mask is None else mask
            rec["voxels"] = int(np.count_nonzero(m))
            if self.snapshots:
                bbox = metrics.bbox_of(m) or [[0, 0]] * 3
                crop = m[tuple(slice(lo, hi) for lo, hi in bbox)]
                rel = f"snapshots/{self._action_seq:05d}.npz"
                (self.workdir / "snapshots").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.workdir / rel, bbox=np.array(bbox), shape=np.array(crop.shape),
                                    bits=np.packbits(crop))  # fmt: skip
                rec["snapshot"] = rel
            with open(self.workdir / "actions.jsonl", "a") as f:
                f.write(json.dumps(rec, default=int) + "\n")
        except Exception:  # noqa: BLE001 - the log must not break the agent's work
            pass

    @staticmethod
    def _show_pyplot_figures(show, shown: set) -> int:
        """Attach pyplot figures the code left open without show() (plt.show() does nothing here),
        so the agent does not decide on images it never saw; then close all of them."""
        import sys

        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None:
            return 0
        n = 0
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if id(fig) not in shown:
                show(fig)
                n += 1
        plt.close("all")
        return n

    def _buffer_crc(self) -> int | None:
        return None if self.buffer is None else zlib.crc32(memoryview(np.ascontiguousarray(self.buffer)).cast("B"))

    def _mark_synced(self) -> None:
        """Remember the buffer as the workspace left it, to detect direct edits by agent code."""
        if self._in_python:
            self._synced_crc = self._buffer_crc()

    def _recount(self) -> list:
        """Update the stored voxel counts after code edited ``instances`` directly; returns the changes."""
        if self.instances is None or not self.objects:
            return []
        counts = np.bincount(self.instances.ravel(), minlength=max(self.objects) + 1)
        changed = []
        editing = self.current.edit_id if self.current is not None else None
        for i, o in sorted(self.objects.items()):
            if i == editing:  # re-opened: out of the instance map until commit/discard
                continue
            n = int(counts[i]) if i < len(counts) else 0
            if n != o["voxels"]:
                o["voxels"] = n
                changed.append((i, n))
        return changed

    # ---------------------------------------------------------------- helpers
    def _need_image(self) -> LoadedImage:
        if self.image is None:
            raise AgentError("no image loaded; call load_image first")
        return self.image

    def _need_current(self) -> CurrentObject:
        self._need_image()
        if self.current is None:
            raise AgentError("no open object; call new_object(name) first (one object per instance)")
        return self.current

    def _resolve(self, ref) -> int:
        if isinstance(ref, str) and ref.lstrip("#").isdigit():
            ref = int(ref.lstrip("#"))
        if isinstance(ref, (int, np.integer)):
            if int(ref) not in self.objects:
                raise AgentError(f"no committed object #{ref}")
            return int(ref)
        matches = [i for i, o in self.objects.items() if o["name"] == ref]
        if len(matches) != 1:
            raise AgentError(f"object name {ref!r} matches {len(matches)} objects; use the id")
        return matches[0]

    def _mask_for(self, ref, allow_none=False):
        """(id or None, mask or None) for an object reference; None/'current' = the open object."""
        if ref in (None, "current"):
            if self.current is not None and self.buffer.any():
                return None, self.buffer > 0
            if allow_none or ref is None:
                return None, None
            raise AgentError("no open object")
        oid = self._resolve(ref)
        return oid, self.instances == oid

    def _stats(self, mask, exclude_id=None) -> dict:
        names = {i: o["name"] for i, o in self.objects.items()}
        return metrics.object_stats(mask, self.image, self.instances, names, exclude_id=exclude_id)

    def _overlay(self, highlight, show_objects) -> Overlay:
        return Overlay(
            current=self.buffer if self.current is not None else None,
            instances=self.instances,
            names={i: o["name"] for i, o in self.objects.items()},
            prompts=list(self.current.prompts) if self.current is not None else [],
            show_objects=show_objects,
            highlight=highlight,
        )

    def _win(self, window):
        if window is None:
            return self.window
        if len(window) != 2 or window[1] <= window[0]:
            raise AgentError("window must be [low, high] with low < high")
        return (float(window[0]), float(window[1]))

    def _image_result(self, rgb: np.ndarray, kind: str, text: str) -> Result:
        png = render.to_png(rgb)
        self._render_count += 1
        path = self.workdir / "renders" / f"{self._render_count:04d}_{kind}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return Result(text, images=[png], data={"path": str(path)})
