"""Segmenting large objects on a downsampled copy of the image.

nnInteractive predicts 192^3 patches and covers larger objects by zooming out (AutoZoom) and
refining tile by tile, which works less well than predicting an object that fits. For objects
larger than ``THRESHOLD_EXTENT`` voxels the workspace therefore gives the model an image
downsampled so that the object spans about ``TARGET_EXTENT`` voxels, and scales its masks back
up. Both numbers were chosen empirically.

Grid convention: a low-resolution voxel ``j`` along an axis with factor ``f = n_full / n_low``
covers the full-resolution range ``[j * f, (j + 1) * f)`` (voxel edges), so full-resolution
voxel ``i`` falls into ``j = floor((i + 0.5) / f)`` and its centre sits at low-resolution
coordinate ``(i + 0.5) / f - 0.5``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

THRESHOLD_EXTENT = 320  # objects larger than this (voxels, on any axis) are segmented downsampled
TARGET_EXTENT = 256  # ... on an image where they span about this many voxels
MIN_FACTOR = 1.15  # smaller reductions of an axis are not worth it
PER_AXIS = True  # reduce only the axes on which the object is too large (False: one factor for all axes)

FULL = (1.0, 1.0, 1.0)


def factors_for_extent(extent, shape, target: int | None = None, per_axis: bool | None = None,
                       threshold: int | None = None) -> tuple:  # fmt: skip
    """Downsampling factor per axis that makes an object of ``extent`` voxels fit ``target`` when
    it is larger than ``threshold`` (defaults: the module constants, read at call time).

    Axes of size 1 (2D images) are never reduced. Returns ``FULL`` when no reduction is needed.
    """
    if extent is None:
        return FULL
    target = TARGET_EXTENT if target is None else target
    per_axis = PER_AXIS if per_axis is None else per_axis
    threshold = THRESHOLD_EXTENT if threshold is None else threshold
    if len(extent) != 3:
        raise ValueError(f"extent needs 3 values [axis0, axis1, axis2] (voxels), got {extent}")
    if max(float(e) for e, s in zip(extent, shape) if s > 1) <= threshold:
        return FULL
    f = [max(1.0, float(e) / target) if s > 1 else 1.0 for e, s in zip(extent, shape)]
    if not per_axis:
        f = [max(f) if s > 1 else 1.0 for s in shape]
    f = [x if x >= MIN_FACTOR else 1.0 for x in f]
    return FULL if all(x == 1.0 for x in f) else tuple(round(x, 2) for x in f)


def low_shape(shape, factors) -> tuple:
    return tuple(max(1, round(s / f)) for s, f in zip(shape, factors))


def _axis_weights(n_out: int, n_in: int, lo: int, hi: int):
    """For output indices lo..hi-1: the two input neighbours and the weight of the second."""
    x = (np.arange(lo, hi) + 0.5) * (n_in / n_out) - 0.5
    x = np.clip(x, 0, n_in - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (x - i0).astype(np.float32)


def resample_linear(arr: np.ndarray, out_shape, region=None) -> np.ndarray:
    """Separable linear interpolation of ``arr`` onto a grid of ``out_shape`` (voxel centres
    aligned as in the module docstring). ``region`` ([[lo, hi], ...] in output coordinates)
    limits the computation to that part of the output; the result has the region's shape."""
    region = region or [[0, s] for s in out_shape]
    # Only the input range the region needs, as float32.
    plans = [_axis_weights(n_out, n_in, lo, hi) for n_out, n_in, (lo, hi) in zip(out_shape, arr.shape, region)]
    src = tuple(slice(int(i0.min()), int(i1.max()) + 1) for i0, i1, _ in plans)
    out = arr[src].astype(np.float32, copy=False)
    for axis, (i0, i1, w) in enumerate(plans):
        off = src[axis].start
        if out.shape[axis] == len(i0) and np.array_equal(i0 - off, np.arange(len(i0))) and not w.any():
            continue  # identity along this axis
        shape = [1] * out.ndim
        shape[axis] = -1
        w = w.reshape(shape)
        out = np.take(out, i0 - off, axis=axis) * (1 - w) + np.take(out, i1 - off, axis=axis) * w
    return out


def downsample_image(data: np.ndarray, factors) -> np.ndarray:
    """Anti-aliased (Gaussian prefilter, sigma = (f - 1) / 2) linear downsampling."""
    sigma = [max(0.0, (f - 1) / 2) for f in factors]
    smooth = ndi.gaussian_filter(data.astype(np.float32), sigma) if any(sigma) else data.astype(np.float32)
    return resample_linear(smooth, low_shape(data.shape, factors))


def downsample_mask(mask: np.ndarray, factors) -> np.ndarray:
    """Binary mask on the low-resolution grid: a low voxel is set when about half of it is."""
    return (downsample_image(mask > 0, factors) >= 0.5).astype(np.uint8)


def upsample_mask(low: np.ndarray, full_shape, region=None) -> np.ndarray:
    """Binary mask on the full-resolution grid (linear interpolation, threshold 0.5), limited to
    ``region`` if given. Smooth edges instead of the staircase of a nearest-neighbour copy."""
    return (resample_linear(low > 0, full_shape, region) >= 0.5).astype(np.uint8)


def to_low_index(i: int, n_full: int, n_low: int) -> int:
    """The low-resolution voxel that contains full-resolution voxel ``i``."""
    return min(int(np.floor((i + 0.5) * n_low / n_full)), n_low - 1)


def to_low_point(p, full_shape, shape_low) -> list:
    return [to_low_index(int(v), n, m) for v, n, m in zip(p, full_shape, shape_low)]


def full_region(low_region, full_shape, shape_low, margin: int = 1) -> list:
    """Full-resolution range whose values depend on the low-resolution ``low_region`` (plus a
    margin of low voxels for the interpolation)."""
    out = []
    for (lo, hi), n, m in zip(low_region, full_shape, shape_low):
        f = n / m
        out.append([max(0, int(np.floor((lo - margin) * f))), min(n, int(np.ceil((hi + margin) * f)))])
    return out


def describe(factors) -> str:
    return ", ".join(f"axis{a} 1/{f:g}" for a, f in enumerate(factors) if f != 1.0)
