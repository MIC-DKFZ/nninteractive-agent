"""Turning agent coordinates into nnInteractive prompts.

Agents give 3D points ``[axis0, axis1, axis2]``. A scribble is a polyline whose points share
one coordinate (the plane's fixed axis); it is drawn 2 px wide in that plane, which is the
width nnInteractive expects, and handed to the session as a tight crop plus its bounding box.
"""

from __future__ import annotations

import numpy as np
from skimage.draw import line as draw_line

SCRIBBLE_WIDTH = 2


def plane_axes(axis: int) -> tuple:
    """The two in-plane axes (row axis, column axis) of the plane with fixed ``axis``."""
    return tuple(a for a in range(3) if a != axis)


def as_point(p, shape, is_2d: bool) -> list:
    """Validate a point; 2D images also accept [row, col]."""
    p = [float(v) for v in p]
    if len(p) == 2 and is_2d:
        p = [0.0, *p]
    if len(p) != 3:
        raise ValueError(f"points need 3 coordinates [axis0, axis1, axis2], got {p}")
    for k, (v, s) in enumerate(zip(p, shape)):
        if not -0.5 <= v <= s - 0.5:
            raise ValueError(f"point {p} is outside the image on axis{k} (valid 0..{s - 1})")
    return [int(round(v)) for v in p]


def infer_plane(points: list, axis: int | None, shape, is_2d: bool, prefer: tuple = ()) -> tuple:
    """Find the plane (fixed axis, index) that all points lie in.

    A straight line along one axis lies in two planes; then the first axis in ``prefer`` that
    qualifies wins (callers pass e.g. the plane of the previous prompt, then the slice axis),
    else the lowest axis.
    """
    if is_2d:
        if axis not in (None, 0):
            raise ValueError("2D images only have the axis0 = 0 plane")
        return 0, 0
    if axis is not None:
        values = {p[axis] for p in points}
        if len(values) != 1:
            raise ValueError(f"all points must have the same axis{axis} coordinate, got {sorted(values)}")
        return axis, values.pop()
    shared = [a for a in range(3) if len({p[a] for p in points}) == 1]
    if len(shared) == 1:
        return shared[0], points[0][shared[0]]
    if not shared:
        raise ValueError("the points do not lie in one plane: one coordinate must be the same for all points")
    chosen = next((a for a in prefer if a in shared), shared[0])
    return chosen, points[0][chosen]


def polyline_pixels(rc: list) -> np.ndarray:
    """Centerline pixels (N, 2) of a polyline through the given (row, col) vertices."""
    return np.concatenate([seg for seg, _ in _segments(rc)]) if len(rc) > 1 else np.array(rc, dtype=int)


def _segments(rc: list):
    """Per segment: its pixels and whether it runs mostly along rows (vertical on screen)."""
    out = []
    for k, ((r0, c0), (r1, c1)) in enumerate(zip(rc[:-1], rc[1:])):
        rr, cc = draw_line(int(r0), int(c0), int(r1), int(c1))
        pix = np.stack([rr, cc], axis=1)
        out.append((pix if k == 0 else pix[1:], abs(r1 - r0) >= abs(c1 - c0)))
    return out


def scribble_crop(points3d: list, axis: int, index: int, shape, width: int = SCRIBBLE_WIDTH) -> tuple:
    """Rasterize a polyline in plane ``axis = index``, ``width`` px wide.

    The line is thickened perpendicular to each segment's main direction (a vertical-ish segment
    gets extra columns, a horizontal-ish one extra rows), so it is exactly ``width`` px wide and
    never extends beyond its end points.

    Returns ``(crop_3d, bbox, centerline_3d)``: the uint8 crop (singleton along ``axis``),
    its half-open ``[[lo, hi], ...]`` bbox in image coordinates (what
    ``add_scribble_interaction(..., interaction_bbox=bbox)`` takes), and the centerline pixels
    as (N, 3) image coordinates.
    """
    ra, ca = plane_axes(axis)
    rc = [(p[ra], p[ca]) for p in points3d]
    if len({tuple(v) for v in rc}) == 1:
        raise ValueError("the scribble points are all the same; give at least two different points")
    offsets = np.arange(width) - (width - 1) // 2
    thick, center = [], []
    for pix, vertical in _segments(rc):
        center.append(pix)
        for o in offsets:
            thick.append(pix + (np.array([0, o]) if vertical else np.array([o, 0])))
    center = np.concatenate(center)
    pix = np.concatenate(thick)
    keep = (pix[:, 0] >= 0) & (pix[:, 0] < shape[ra]) & (pix[:, 1] >= 0) & (pix[:, 1] < shape[ca])
    pix = pix[keep]
    lo, hi = pix.min(0), pix.max(0) + 1
    crop2d = np.zeros(tuple(hi - lo), dtype=np.uint8)
    crop2d[pix[:, 0] - lo[0], pix[:, 1] - lo[1]] = 1
    bbox = [[0, 0], [0, 0], [0, 0]]
    bbox[axis] = [int(index), int(index) + 1]
    bbox[ra] = [int(lo[0]), int(hi[0])]
    bbox[ca] = [int(lo[1]), int(hi[1])]
    centerline = np.zeros((len(center), 3), dtype=int)
    centerline[:, axis] = index
    centerline[:, ra] = center[:, 0]
    centerline[:, ca] = center[:, 1]
    return np.expand_dims(crop2d, axis), bbox, centerline


def bbox_from_corners(c1: list, c2: list) -> tuple:
    """Half-open bbox from two inclusive corners that share one coordinate (2D box)."""
    flat = [a for a in range(3) if c1[a] == c2[a]]
    if len(flat) != 1:
        raise ValueError(
            "a bbox is a 2D rectangle: the two corners must share exactly one coordinate (the slice), "
            f"got {c1} and {c2}"
        )
    bbox = [[min(a, b), max(a, b) + 1] for a, b in zip(c1, c2)]
    return bbox, flat[0]
