"""What the browser viewer gets from a workspace: the image as NIfTI, label layers as patches,
and object surfaces as triangle meshes.

Workspace arrays are [axis0, axis1, axis2] = the reversed ITK/NIfTI index order, so a C-order
flattening of a workspace array is exactly NIfTI voxel order. Label layers therefore travel as
raw crops that the browser writes straight into its NIfTI-ordered volumes.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np

from ..core import raster
from ..core.io import LoadedImage

MAX_3D_VOXELS = 12_000_000  # the 3D view gets a downsampled copy above this size
VOLUME_FORMAT = 2  # bump when write_volumes changes what it writes (part of the cache key)
MAX_TRIANGLES = 300_000
PROMPT_POS, PROMPT_NEG = 1, 2  # values of the "prompts" layer


# ---------------------------------------------------------------------------- geometry
def geometry(img: LoadedImage) -> tuple:
    """(origin, direction 3x3, spacing) in ITK order for the 3D volume the browser shows."""
    ref = img.sitk_reference
    if ref is not None and ref.GetDimension() == 3:
        return (np.array(ref.GetOrigin(), float), np.array(ref.GetDirection(), float).reshape(3, 3),
                np.array(ref.GetSpacing(), float))  # fmt: skip
    if ref is not None and ref.GetDimension() == 2:
        d = np.eye(3)
        d[:2, :2] = np.array(ref.GetDirection(), float).reshape(2, 2)
        return np.array([*ref.GetOrigin(), 0.0]), d, np.array([*ref.GetSpacing(), 1.0])
    return np.zeros(3), np.eye(3), np.array(img.spacing[::-1], float)


def array_to_ras(img: LoadedImage) -> np.ndarray:
    """4x4 matrix from workspace array indices (axis0, axis1, axis2) to NIfTI world (RAS) mm."""
    origin, direction, spacing = geometry(img)
    flip = np.diag([-1.0, -1.0, 1.0])  # ITK is LPS
    rev = np.eye(3)[::-1]  # ITK index (i, j, k) = (axis2, axis1, axis0)
    m = np.eye(4)
    m[:3, :3] = flip @ direction @ np.diag(spacing) @ rev
    m[:3, 3] = flip @ origin
    return m


def to_sitk(arr: np.ndarray, img: LoadedImage):
    import SimpleITK as sitk

    out = sitk.GetImageFromArray(arr)
    origin, direction, spacing = geometry(img)
    out.SetOrigin(tuple(origin))
    out.SetDirection(tuple(direction.ravel()))
    out.SetSpacing(tuple(spacing))
    return out


def browser_dtype(data: np.ndarray) -> np.dtype:
    """Smallest dtype that keeps the intensities exactly (CT stored as float32/int32 -> int16)."""
    if data.dtype in (np.uint8, np.int16, np.uint16):
        return data.dtype
    step = max(1, data.size // 2_000_000)
    sample = data.ravel()[::step]
    lo, hi = float(data.min()), float(data.max())
    if np.all(np.mod(sample, 1) == 0) and -32768 <= lo and hi <= 32767:
        return np.dtype(np.int16)
    return np.dtype(np.float32)


def shrink_factors(shape: tuple, spacing: tuple, max_voxels: int = MAX_3D_VOXELS) -> list:
    """Per array axis integer factors that bring the volume below ``max_voxels``, coarsening the
    finest axes first so the 3D copy gets closer to isotropic."""
    f = [1, 1, 1]
    while np.prod([s // k for s, k in zip(shape, f)]) > max_voxels:
        cand = [a for a in range(3) if shape[a] // (f[a] + 1) >= 8] or [int(np.argmax(shape))]
        finest = min(spacing[a] * f[a] for a in cand)
        for a in cand:  # coarsen all of the finest axes together (keeps square pixels square)
            if spacing[a] * f[a] <= 1.1 * finest:
                f[a] += 1
    return f


def write_volumes(img: LoadedImage, out_dir) -> dict:
    """image.nii.gz (full resolution, for the slice views), image3d.nii.gz (for the 3D view) and
    empty uint8/uint16 label volumes with the same header."""
    import SimpleITK as sitk

    from ..core.workspace import _looks_like_ct

    out_dir.mkdir(parents=True, exist_ok=True)
    data = img.data
    dtype = browser_dtype(data)
    vol = to_sitk(np.ascontiguousarray(data.astype(dtype, copy=False)), img)
    sitk.WriteImage(vol, str(out_dir / "image.nii.gz"), useCompression=True, compressionLevel=3)
    factors = shrink_factors(img.shape, img.spacing)
    if img.is_2d:
        factors = [1, 1, 1]
    is_ct = bool(_looks_like_ct(data))
    vol3d = sitk.BinShrink(vol, factors[::-1]) if factors != [1, 1, 1] else vol
    if is_ct and not img.is_2d:  # without the patient table, which would hide the body from behind
        arr = sitk.GetArrayFromImage(vol3d)
        axial = next((a for a, o in enumerate(img.orientation) if o in ("S", "I")), 0)
        body = body_mask(arr, [s * f for s, f in zip(img.spacing, factors)], axial)
        arr[~body] = min(-1024, int(arr.min()))
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(vol3d)
        vol3d = out
    sitk.WriteImage(vol3d, str(out_dir / "image3d.nii.gz"), useCompression=True, compressionLevel=3)
    for dt in (np.uint8, np.uint16):  # empty label layers with the image's header, filled by patches
        zeros = to_sitk(np.zeros(img.shape, dt), img)
        sitk.WriteImage(zeros, str(out_dir / f"zeros_{np.dtype(dt).name}.nii.gz"), useCompression=True)
    from ..core.render import auto_window

    meta = {
        "shape": list(img.shape),
        "spacing": [float(s) for s in img.spacing],
        "dtype": str(dtype),
        "is_2d": bool(img.is_2d),
        "is_ct": is_ct,
        "orientation": list(img.orientation),
        "auto_window": [float(v) for v in auto_window(data)],
        "range": [float(data.min()), float(data.max())],
        "shrink3d": factors,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def body_mask(hu: np.ndarray, spacing, axial: int = 0) -> np.ndarray:
    """The patient in a CT: everything above -500 HU, without thin structures (the table) that an
    opening of about 4 mm removes, keeping components of at least 5% of the largest (the body, arms
    beside it), holes filled per axial slice (lungs, bowel gas) and grown back by the opening size."""
    from scipy import ndimage as ndi

    r = [max(1, round(4.0 / s)) for s in spacing]
    grid = np.ogrid[tuple(slice(-k, k + 1) for k in r)]
    ball = sum((g / k) ** 2 for g, k in zip(grid, r)) <= 1
    m = ndi.binary_opening(hu > -500, ball)
    lab, n = ndi.label(m)
    if n == 0:
        return np.ones(hu.shape, bool)
    sizes = np.bincount(lab.ravel())[1:]
    m = np.isin(lab, np.flatnonzero(sizes >= 0.05 * sizes.max()) + 1)
    for k in range(m.shape[axial]):
        sl = [slice(None)] * 3
        sl[axial] = k
        m[tuple(sl)] = ndi.binary_fill_holes(m[tuple(sl)])
    return ndi.binary_dilation(m, ball)


# ------------------------------------------------------------------------ label layers
def prompts_layer(ws) -> np.ndarray:
    """The prompts of the open object, rasterized for display (1 positive, 2 negative)."""
    out = np.zeros(ws.buffer.shape, np.uint8)
    cur = ws.current
    if cur is not None:
        for p in cur.prompts:
            try:
                draw_prompt(out, p, ws.image.spacing)
            except (ValueError, IndexError):
                pass  # a degenerate prompt is not worth breaking the viewer for
    return out


def draw_prompt(out: np.ndarray, p, spacing) -> None:
    value = PROMPT_POS if p.positive else PROMPT_NEG
    shape = out.shape
    if p.kind == "point":
        c = np.round(np.asarray(p.points[0], float)).astype(int)
        r = [max(0, round(2.0 * min(spacing) / s)) for s in spacing]
        sl = tuple(slice(max(0, c[a] - r[a]), min(shape[a], c[a] + r[a] + 1)) for a in range(3))
        grid = np.ogrid[sl]
        d = sum(((g - c[a]) / max(r[a], 0.5)) ** 2 for a, g in enumerate(grid))
        out[sl][d <= 1.0] = value
    elif p.kind == "scribble":
        crop, bbox, _ = raster.scribble_crop(p.points, p.axis, p.index, shape, width=3)
        sl = tuple(slice(lo, hi) for lo, hi in bbox)
        out[sl][crop > 0] = value
    elif p.kind == "bbox":
        c1, c2 = (np.asarray(q, int) for q in p.points)
        lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
        ra, ca = raster.plane_axes(p.axis)
        t = max(1, round(max(shape[ra], shape[ca]) / 320))
        k = int(p.index)
        for a, b in ((ra, ca), (ca, ra)):
            for edge in (lo[a], hi[a]):
                sl = [slice(None)] * 3
                sl[p.axis] = slice(k, k + 1)
                sl[a] = slice(max(0, edge - t // 2), min(shape[a], edge - t // 2 + t))
                sl[b] = slice(lo[b], hi[b] + 1)
                out[tuple(sl)] = value


def changed_box(old: np.ndarray, new: np.ndarray) -> list | None:
    """Half-open [[lo, hi], ...] box around all voxels that differ, or None."""
    ne = old != new
    if not ne.any():
        return None
    box = []
    for ax in range(3):
        idx = np.flatnonzero(ne.any(axis=tuple(a for a in range(3) if a != ax)))
        box.append([int(idx[0]), int(idx[-1]) + 1])
    return box


def encode_patch(header: dict, crop: np.ndarray | None) -> bytes:
    """Binary websocket message: uint32 header length, JSON header, zlib-compressed crop."""
    head = json.dumps(header).encode()
    body = zlib.compress(np.ascontiguousarray(crop).tobytes(), 1) if crop is not None else b""
    return struct.pack("<I", len(head)) + head + body


class LayerSync:
    """Remembers what the browser holds of each label layer and encodes the differences."""

    def __init__(self):
        self.sent: dict = {}
        self.seq = 0

    def reset(self) -> None:
        self.sent = {}

    def current(self, ws) -> dict:
        inst = ws.instances
        dtype = np.uint8 if int(inst.max(initial=0)) < 256 else np.uint16
        return {"objects": inst.astype(dtype, copy=False), "current": np.minimum(ws.buffer, 1).astype(np.uint8),
                "prompts": prompts_layer(ws)}  # fmt: skip

    def patches(self, ws) -> list:
        """Messages that bring the browser up to date (a full copy for layers it lacks)."""
        if ws.image is None or ws.instances is None:
            return []
        out = []
        for name, arr in self.current(ws).items():
            old = self.sent.get(name)
            full = old is None or old.dtype != arr.dtype or old.shape != arr.shape
            box = [[0, s] for s in arr.shape] if full else changed_box(old, arr)
            if box is None:
                continue
            sl = tuple(slice(lo, hi) for lo, hi in box)
            self.seq += 1
            header = {"layer": name, "box": box, "dtype": str(arr.dtype), "shape": list(arr.shape), "full": full,
                      "seq": self.seq}  # fmt: skip
            out.append(encode_patch(header, arr[sl]))
            self.sent[name] = arr.copy()
        return out


# ------------------------------------------------------------------------------ meshes
def _slice_distance(mask: np.ndarray, axis: int, spacing) -> np.ndarray:
    """Per slice along ``axis``: in-plane signed distance (mm) to the outline, negative inside.
    Empty slices get +slice spacing, which rounds the ends off half a slice beyond the last one."""
    from scipy.ndimage import distance_transform_edt as edt

    sampling = [spacing[a] for a in range(3) if a != axis]
    out = np.empty(mask.shape, np.float32)
    for k in range(mask.shape[axis]):
        sl = [slice(None)] * 3
        sl[axis] = k
        m = mask[tuple(sl)]
        if not m.any():
            out[tuple(sl)] = spacing[axis]
        elif m.all():
            out[tuple(sl)] = -spacing[axis]
        else:
            out[tuple(sl)] = edt(~m, sampling=sampling) - edt(m, sampling=sampling)
    return out


def taubin_smooth(xyz: np.ndarray, faces: np.ndarray, iterations: int = 15, lam: float = 0.5, mu: float = -0.53):
    """Volume-preserving mesh smoothing (Taubin 1995): removes the terraces that thick slices
    leave on a marching-cubes surface without shrinking it like plain Laplacian smoothing."""
    from scipy.sparse import coo_matrix, diags

    n = len(xyz)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = coo_matrix((np.ones(len(i), np.float32), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0  # duplicate edges were summed
    deg = np.asarray(adj.sum(1)).ravel()
    avg = diags(1.0 / np.maximum(deg, 1)) @ adj  # neighbour average
    v = xyz.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (avg @ v - v)
        v = v + mu * (avg @ v - v)
    return v


def _signed_volume(xyz: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume of a closed triangle mesh; negative when the triangles face inward."""
    v0, v1, v2 = (xyz[faces[:, k]].astype(np.float64) for k in range(3))
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def object_crops(instances: np.ndarray, ids) -> dict:
    """Per committed object id: (binary crop, offset of the crop), or None without voxels."""
    from scipy.ndimage import find_objects

    ids = [int(i) for i in ids]
    if not ids:
        return {}
    boxes = find_objects(instances, max_label=max(ids))
    out = {}
    for i in ids:
        sl = boxes[i - 1] if i - 1 < len(boxes) else None
        out[i] = None if sl is None else (instances[sl] == i, [s.start for s in sl])
    return out


def current_crop(buffer: np.ndarray):
    from scipy.ndimage import find_objects

    boxes = find_objects(np.minimum(buffer, 1).astype(np.uint8))
    if not boxes or boxes[0] is None:
        return None
    sl = boxes[0]
    return buffer[sl] > 0, [s.start for s in sl]


def crop_mesh(crop, affine: np.ndarray, spacing, max_triangles: int = MAX_TRIANGLES) -> bytes | None:
    """Smoothed iso-surface of a (mask, offset) crop as a little-endian blob: uint32 n_vertices,
    uint32 n_triangles, float32 vertices (RAS mm), uint32 triangle indices.

    With thick slices (coarsest spacing at least 1.5x the finest), the surface between slices
    comes from shape-based interpolation: each slice becomes an in-plane signed distance map,
    which is interpolated (up to 4x) across the slices and smoothed. Interpolating the mask itself
    instead leaves a flat terrace wherever the outline moves between two slices.
    """
    from scipy.ndimage import gaussian_filter, zoom
    from skimage.measure import marching_cubes

    if crop is None or not crop[0].any():
        return None
    mask, offset = crop
    padded = np.pad(mask, 2)
    coarse = int(np.argmax(spacing))
    up = np.ones(3, int)
    up[coarse] = min(4, max(1, round(spacing[coarse] / min(spacing))))
    while np.prod(np.array(padded.shape) * up) > 30e6 and up[coarse] > 1:
        up[coarse] -= 1
    if up[coarse] > 1:
        vol = _slice_distance(padded, coarse, spacing)
        vol = zoom(vol, up, order=1, mode="nearest", grid_mode=False)  # cubic overshoots into sheets
        sigma = [0.6] * 3
        sigma[coarse] = 0.5 * up[coarse]  # rounds off the creases linear interpolation leaves at the slices
        vol = gaussian_filter(vol, sigma)
        level = 0.0
    else:
        vol = gaussian_filter(padded.astype(np.float32), sigma=0.8)
        level = 0.5 if vol.max() > 0.5 else vol.max() / 2
    step = max(1, int(np.ceil((vol.size / 8e6) ** (1 / 3))))
    while True:
        try:
            verts, faces, _, _ = marching_cubes(vol, level=level, step_size=step, allow_degenerate=False)
        except (ValueError, RuntimeError):
            return None
        if len(faces) <= max_triangles or step >= 6:
            break
        step += 1
    # zoom(grid_mode=False) maps the first and last samples onto each other: index i -> i * (n-1)/(N-1)
    n_in = np.array(mask.shape) + 4
    n_out = np.array(vol.shape)
    scale = np.where(n_out > 1, (n_in - 1) / np.maximum(n_out - 1, 1), 1.0)
    verts = verts * scale + (np.asarray(offset, float) - 2)
    xyz = taubin_smooth(verts @ affine[:3, :3].T + affine[:3, 3], faces)
    # Wind the triangles counter-clockwise seen from outside (outward normals), which is what
    # NiiVue's back-face culling expects; otherwise it draws the inside of the far wall. The sign
    # of the enclosed volume tells, whatever the conventions of marching cubes and the affine.
    if _signed_volume(xyz, faces) < 0:
        faces = faces[:, ::-1]
    return (struct.pack("<II", len(xyz), len(faces)) + xyz.astype("<f4").tobytes()
            + faces.astype("<u4").tobytes())  # fmt: skip
