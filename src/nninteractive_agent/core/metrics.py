"""Cheap numeric checks on a segmentation, so the agent only looks at images when needed."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_FULL_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


def bbox_of(mask: np.ndarray):
    """Half-open [[lo, hi], ...] bbox of the nonzero voxels, or None if empty."""
    out = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        nz = np.flatnonzero(np.any(mask, axis=other))
        if nz.size == 0:
            return None
        out.append([int(nz[0]), int(nz[-1]) + 1])
    return out


def crop(arr: np.ndarray, bbox) -> np.ndarray:
    return arr[tuple(slice(lo, hi) for lo, hi in bbox)]


def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [[min(x[0], y[0]), max(x[1], y[1])] for x, y in zip(a, b)]


def expand_bbox(bbox, margin, shape):
    return [[max(0, lo - margin), min(s, hi + margin)] for (lo, hi), s in zip(bbox, shape)]


def component_sizes(mask_crop: np.ndarray) -> list:
    """Sizes of the 26-connected components, largest first."""
    labels, n = ndi.label(mask_crop, structure=_FULL_CONNECTIVITY)
    if n == 0:
        return []
    return sorted((int(s) for s in np.bincount(labels.ravel())[1:]), reverse=True)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = int(np.count_nonzero(a)), int(np.count_nonzero(b))
    if sa + sb == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(a & b)) / (sa + sb)


def slice_profile_warnings(mask_crop: np.ndarray, bbox, max_per_axis: int = 2) -> list:
    """Abrupt changes of the per-slice area along each axis: typical signs of leaks or cut-offs.

    Flags (a) empty slices inside the object's extent (the object is split along that axis) and
    (b) consecutive slices whose areas differ by more than 2x and by a substantial amount.
    """
    warnings = []
    for axis in range(3):
        if mask_crop.shape[axis] < 4:
            continue
        other = tuple(a for a in range(3) if a != axis)
        areas = mask_crop.sum(axis=other).astype(np.int64)
        lo = bbox[axis][0]
        gaps = np.flatnonzero(areas == 0)
        if gaps.size:
            warnings.append(
                f"axis{axis}: empty slices inside the object at {_ranges(gaps + lo)} (object split along axis{axis})"
            )
        big = max(50, 0.15 * areas.max())
        jumps = []
        for i in range(len(areas) - 1):
            a, b = areas[i], areas[i + 1]
            # Objects naturally taper at their ends, so transitions into the first/last slice
            # must be much more abrupt to count.
            ratio = 5 if i == 0 or i + 1 == len(areas) - 1 else 2
            if a > 0 and b > 0 and max(a, b) > ratio * min(a, b) and abs(a - b) > big:
                jumps.append((abs(int(a - b)), i))
        for _, i in sorted(jumps, reverse=True)[:max_per_axis]:
            warnings.append(
                f"axis{axis}: area jumps {int(areas[i])} -> {int(areas[i + 1])} voxels between slices "
                f"{i + lo} and {i + lo + 1}"
            )
    return warnings


def _ranges(idx: np.ndarray) -> str:
    """[3,4,5,9] -> '3-5, 9'"""
    parts, start = [], idx[0]
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur != prev + 1:
            parts.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = cur
    parts.append(f"{start}-{idx[-1]}" if idx[-1] != start else f"{start}")
    return ", ".join(parts)


def intensity_summary(values: np.ndarray, max_samples: int = 200_000) -> dict:
    if values.size == 0:
        return {}
    if values.size > max_samples:
        values = values[:: values.size // max_samples + 1]
    p5, p50, p95 = np.percentile(values.astype(np.float32), (5, 50, 95))
    return {"p5": float(p5), "median": float(p50), "p95": float(p95)}


def fmt_num(v: float) -> str:
    if abs(v) >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.3g}"


def object_stats(mask: np.ndarray, image, instances: np.ndarray, object_names: dict, exclude_id=None) -> dict:
    """Numbers describing one object; ``mask`` is a full-size boolean/uint8 array."""
    bbox = bbox_of(mask)
    if bbox is None:
        return {"voxels": 0}
    m = crop(mask, bbox).astype(bool)
    voxels = int(m.sum())
    sizes = component_sizes(m)
    stats = {
        "voxels": voxels,
        "bbox": bbox,
        "components": len(sizes),
        "component_sizes": sizes[:5],
        "largest_component_fraction": sizes[0] / voxels,
        "intensity": intensity_summary(crop(image.data, bbox)[m]),
        "profile_warnings": slice_profile_warnings(m, bbox) if not image.is_2d else [],
    }
    if image.sitk_reference is not None:
        stats["volume_ml"] = voxels * float(np.prod(image.spacing)) / 1000.0
    border = []
    for axis, ((lo, hi), s) in enumerate(zip(bbox, mask.shape)):
        if s == 1:
            continue
        if lo == 0:
            border.append(f"axis{axis}=0")
        if hi == s:
            border.append(f"axis{axis}={s - 1}")
    stats["touches_border"] = border
    inst = crop(instances, bbox)[m]
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    stats["overlaps"] = {
        int(i): int(c) for i, c in zip(ids, counts) if exclude_id is None or int(i) != int(exclude_id)
    }
    stats["overlap_names"] = {i: object_names.get(i, "?") for i in stats["overlaps"]}
    return stats


def format_stats(stats: dict) -> str:
    if stats.get("voxels", 0) == 0:
        return "empty (0 voxels)"
    lines = []
    size = f"{stats['voxels']:,} voxels"
    if "volume_ml" in stats:
        size += f" ({stats['volume_ml']:.3g} ml)"
    lines.append(size)
    lines.append("bbox (half-open): " + ", ".join(f"axis{a} {lo}-{hi}" for a, (lo, hi) in enumerate(stats["bbox"])))
    cc = stats["components"]
    if cc == 1:
        lines.append("connected components: 1")
    else:
        lines.append(
            f"connected components: {cc} (largest {stats['largest_component_fraction']:.0%}; sizes "
            + ", ".join(f"{s:,}" for s in stats["component_sizes"])
            + ("" if cc <= 5 else ", ...")
            + ")"
        )
    it = stats.get("intensity")
    if it:
        lines.append(
            f"intensity inside: median {fmt_num(it['median'])} (5-95%: {fmt_num(it['p5'])} to {fmt_num(it['p95'])})"
        )
    if stats["touches_border"]:
        lines.append("touches image border at " + ", ".join(stats["touches_border"]))
    if stats["overlaps"]:
        lines.append(
            "OVERLAPS committed objects: "
            + ", ".join(f"#{i} {stats['overlap_names'][i]} ({c:,} voxels)" for i, c in stats["overlaps"].items())
            + " (a leak: correct it; a structure inside them: commit with overlap='replace')"
        )
    for w in stats["profile_warnings"]:
        lines.append("warning: " + w)
    return "\n".join(lines)
