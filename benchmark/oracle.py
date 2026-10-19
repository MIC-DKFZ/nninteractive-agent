"""Simulated prompts: a user who knows the reference mask, with the same tool as the agents.

Every prompt goes to the largest error of the current mask: the largest connected region (3D,
26-connected) that the mask misses (a positive prompt) or adds (a negative one). Before the first
prompt the mask is empty, so the first error region is the target itself.

- ``points``: a point at the voxel deepest inside that region (in mm). The first point is the
  guided track's voxel whenever the target is one piece.
- ``scribbles``: a line along the middle of that region in the slice where its cross-section has
  the most pixels (any of the three axes): the longest path through the region's skeleton. A
  region too small for a line gets a point.

An object larger than ``scale.THRESHOLD_EXTENT`` voxels gets its extent passed to ``new_object``,
as the skill tells agents, unless it fills less than ``THIN_FILL`` of its bounding box (a vessel
tree): thin structures stay at full resolution. ``next_prompt`` is pure numpy (no nnInteractive), so
the prompts can be checked without a GPU.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage as ndi

KINDS = ("points", "scribbles")
THIN_FILL = 0.05
LINE_STEP = 3  # pixels between the vertices of a scribble's polyline


def extent_for(ref: np.ndarray, threshold: int) -> list | None:
    """The ``extent`` to pass to ``new_object`` for this reference, or None (full resolution)."""
    idx = np.argwhere(ref)
    ext = (idx.max(0) - idx.min(0) + 1).tolist()
    if max(ext) <= threshold or ref.sum() < THIN_FILL * np.prod(ext):
        return None
    return ext


def _roi(pred: np.ndarray, ref: np.ndarray) -> tuple:
    """Slices around everything the mask or the reference covers: errors lie only there."""
    union = pred | ref
    return ndi.find_objects(union.astype(np.uint8))[0]


def largest_error(pred: np.ndarray, ref: np.ndarray) -> tuple:
    """(region, positive) of the largest error component, as a full-size mask; (None, None) if there
    is no error. A missed region (positive) wins a tie."""
    sl = _roi(pred, ref)
    best, best_n, positive = None, 0, None
    for err, pos in ((ref[sl] & ~pred[sl], True), (pred[sl] & ~ref[sl], False)):
        lab, n = ndi.label(err, structure=np.ones((3, 3, 3), bool))
        if not n:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        k = int(np.argmax(sizes))
        if sizes[k] > best_n:
            best, best_n, positive = lab == k, int(sizes[k]), pos
    if best is None:
        return None, None
    region = np.zeros(pred.shape, bool)
    region[sl] = best
    return region, positive


def deepest_voxel(region: np.ndarray, spacing) -> list:
    """The voxel farthest from the region's border, in mm."""
    sl = ndi.find_objects(region.astype(np.uint8))[0]
    edt = ndi.distance_transform_edt(np.pad(region[sl], 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    p = np.unravel_index(int(np.argmax(edt)), edt.shape)
    return [int(i + s.start) for i, s in zip(p, sl)]


def best_slice(region: np.ndarray) -> tuple:
    """(axis, index) of the slice in which the region has the most pixels."""
    best = (0, 0, -1)
    for axis in range(3):
        if region.shape[axis] == 1:
            continue
        areas = region.sum(axis=tuple(a for a in range(3) if a != axis))
        k = int(np.argmax(areas))
        if areas[k] > best[2]:
            best = (axis, k, int(areas[k]))
    return best[:2]


def _longest_path(pixels: np.ndarray) -> list:
    """The longest of the shortest paths between two pixels of an 8-connected pixel set (two BFS
    passes; exact on trees, close on skeletons with loops)."""
    pts = [tuple(p) for p in pixels.tolist()]
    index = {p: i for i, p in enumerate(pts)}
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    def bfs(start: int) -> tuple:
        prev = {start: None}
        queue, last = deque([start]), start
        while queue:
            last = queue.popleft()
            r, c = pts[last]
            for dr, dc in steps:
                j = index.get((r + dr, c + dc))
                if j is not None and j not in prev:
                    prev[j] = last
                    queue.append(j)
        return last, prev

    far, _ = bfs(0)
    end, prev = bfs(far)
    path, node = [], end
    while node is not None:
        path.append(list(pts[node]))
        node = prev[node]
    return path


def midline(region2d: np.ndarray) -> list:
    """Vertices (row, col) of a polyline along the middle of the largest component of a 2D region;
    a single vertex if the region is too small for a line."""
    from skimage.morphology import skeletonize

    lab, _ = ndi.label(region2d, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == int(np.argmax(sizes))
    path = _longest_path(np.argwhere(skeletonize(comp)))
    if len(path) < 2:
        edt = ndi.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
        return [list(np.unravel_index(int(np.argmax(edt)), edt.shape))]
    keep = list(range(0, len(path), LINE_STEP))
    if keep[-1] != len(path) - 1:
        keep.append(len(path) - 1)
    return [path[k] for k in keep]


def next_prompt(pred: np.ndarray, ref: np.ndarray, spacing, kind: str) -> dict | None:
    """The oracle's next prompt for mask ``pred``: {"kind": "point"|"scribble", "positive", "points",
    "axis"} with points as [axis0, axis1, axis2]; None when the mask equals the reference."""
    if kind not in KINDS:
        raise ValueError(f"oracle kind must be one of {KINDS}, got {kind!r}")
    region, positive = largest_error(pred, ref)
    if region is None:
        return None
    if kind == "points":
        return {"kind": "point", "positive": positive, "points": [deepest_voxel(region, spacing)], "axis": None}  # fmt: skip
    axis, k = best_slice(region)
    vertices = midline(np.take(region, k, axis=axis))
    points = [[int(v) for v in np.insert(np.asarray(rc), axis, k)] for rc in vertices]
    if len(points) == 1:
        return {"kind": "point", "positive": positive, "points": points, "axis": None}
    return {"kind": "scribble", "positive": positive, "points": points, "axis": axis}


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    return 2 * int((pred & ref).sum()) / max(int(pred.sum()) + int(ref.sum()), 1)
