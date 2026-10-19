"""Loading images into the [X, Y, Z] array layout nnInteractive works in, and saving masks.

Every image ends up as a 3D array. A 2D image gets a leading singleton axis (shape
``(1, H, W)``), exactly like napari-nninteractive does, so prompts and masks are always 3D.
Array axes are called axis0/axis1/axis2 everywhere in this package; for files read with
SimpleITK (NIfTI, NRRD, MHA, ...) they are the reversed ITK axes, i.e. axis0 is usually the
slice axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SITK_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".gz")
PIL_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
TIFF_SUFFIXES = (".tif", ".tiff")

# Direction of increasing index for each physical LPS axis: (negative sign, positive sign).
_LPS_LETTERS = (("R", "L"), ("A", "P"), ("I", "S"))
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "I": "S", "S": "I"}


@dataclass
class LoadedImage:
    """An image in nnInteractive's array layout plus what the renderer needs to show it."""

    data: np.ndarray  # [X, Y, Z] intensities fed to the model (grayscale)
    spacing: tuple  # per array axis
    source: str
    rgb: Optional[np.ndarray] = None  # [X, Y, Z, 3] uint8 for display, if the source was color
    # Per array axis, the patient direction an increasing index points to ('L', 'P', 'S', ...),
    # or None when unknown (microscopy, photos, arrays).
    orientation: tuple = (None, None, None)
    sitk_reference: object = None  # SimpleITK image whose geometry masks are saved with
    notes: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        return tuple(int(s) for s in self.data.shape)

    @property
    def is_2d(self) -> bool:
        return self.data.shape[0] == 1

    def model_input(self) -> np.ndarray:
        """4D [1, X, Y, Z] array for ``session.set_image`` (no intensity preprocessing)."""
        return self.data[None]


def _suffix(path: Path) -> str:
    name = path.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb[..., :3].astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb[..., :3]
    if rgb.dtype == np.uint8:
        return rgb
    lo, hi = np.percentile(rgb, (0.5, 99.5))
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def _orientation_from_direction(direction: np.ndarray, ndim: int) -> tuple:
    """Patient direction (LPS letters) of increasing index along each numpy axis."""
    letters = []
    for k in range(ndim):
        vec = direction[:, ndim - 1 - k]  # numpy axis k is ITK index axis ndim-1-k
        j = int(np.argmax(np.abs(vec)))
        letters.append(_LPS_LETTERS[j][1] if vec[j] > 0 else _LPS_LETTERS[j][0])
    return tuple(letters)


def _load_sitk(path: Path) -> LoadedImage:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = tuple(float(s) for s in img.GetSpacing()[::-1])
    dim = img.GetDimension()
    direction = np.array(img.GetDirection(), dtype=float).reshape(dim, dim)
    orientation = _orientation_from_direction(direction, dim) if dim == 3 else (None,) * dim
    rgb = None
    if img.GetNumberOfComponentsPerPixel() in (3, 4):
        rgb = _as_uint8_rgb(arr)
        arr = _to_gray(arr)
    elif img.GetNumberOfComponentsPerPixel() > 1:
        raise ValueError(f"{path.name}: {img.GetNumberOfComponentsPerPixel()}-component images are not supported")
    if arr.ndim == 2:
        arr, spacing, orientation = arr[None], (1.0, *spacing), (None, *orientation)
        rgb = rgb[None] if rgb is not None else None
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected a 2D or 3D image, got shape {arr.shape}")
    return LoadedImage(arr, spacing, str(path), rgb, tuple(orientation), img)


def _from_array(arr: np.ndarray, path: Path, channel: Optional[int]) -> LoadedImage:
    """2D gray, 2D color (H, W, 3/4), 3D stack, or 4D stack with a channel to pick."""
    notes = []
    rgb = None
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
        rgb, arr = _as_uint8_rgb(arr)[None], _to_gray(arr)[None]
        notes.append("color image: the model sees luminance; views show the original colors")
    elif arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 4:
        if channel is None:
            raise ValueError(
                f"{path.name} has shape {arr.shape}; pass `channel` to pick one channel of the "
                f"smallest axis (axis {int(np.argmin(arr.shape))})"
            )
        arr = np.take(arr, channel, axis=int(np.argmin(arr.shape)))
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: unsupported shape {arr.shape}")
    return LoadedImage(arr, (1.0, 1.0, 1.0), str(path), rgb, (None, None, None), None, notes)


def load_image(path, spacing=None, channel: Optional[int] = None) -> LoadedImage:
    """Read an image file. ``spacing`` (per array axis) overrides the file's spacing."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    suffix = _suffix(path)
    if suffix in SITK_SUFFIXES:
        img = _load_sitk(path)
    elif suffix in PIL_SUFFIXES:
        from PIL import Image

        img = _from_array(np.asarray(Image.open(path)), path, channel)
    elif suffix in TIFF_SUFFIXES:
        import tifffile

        img = _from_array(tifffile.imread(path), path, channel)
    elif suffix == ".npy":
        img = _from_array(np.load(path), path, channel)
    elif suffix == ".npz":
        with np.load(path) as f:
            img = _from_array(f["image"] if "image" in f else f[f.files[0]], path, channel)
    else:
        raise ValueError(f"unsupported file type: {path.name}")
    if spacing is not None:
        spacing = tuple(float(s) for s in spacing)
        if len(spacing) == 2 and img.is_2d:
            spacing = (1.0, *spacing)
        if len(spacing) != 3:
            raise ValueError("spacing needs one value per array axis")
        img.spacing = spacing
    if not np.issubdtype(img.data.dtype, np.number) or img.data.dtype == np.bool_:
        img.data = img.data.astype(np.float32)
    return img


def save_mask(mask: np.ndarray, path, image: LoadedImage) -> Path:
    """Save a [X, Y, Z] label array next to the geometry of the image it was drawn on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if mask.max(initial=0) < 256 else np.uint16
    mask = mask.astype(dtype, copy=False)
    suffix = _suffix(path)
    if suffix in SITK_SUFFIXES:
        import SimpleITK as sitk

        ref = image.sitk_reference
        out = sitk.GetImageFromArray(mask[0] if (ref is not None and ref.GetDimension() == 2) else mask)
        if ref is not None:
            out.CopyInformation(ref)
        else:
            out.SetSpacing(tuple(float(s) for s in image.spacing[::-1]))
        sitk.WriteImage(out, str(path), useCompression=True)
    elif suffix in PIL_SUFFIXES:
        from PIL import Image

        Image.fromarray(mask[0] if image.is_2d else mask.max(0)).save(path)
    elif suffix in TIFF_SUFFIXES:
        import tifffile

        tifffile.imwrite(path, mask[0] if image.is_2d else mask, compression="zlib")
    else:
        np.save(path, mask)
    return path


def default_mask_suffix(image: LoadedImage) -> str:
    """File type for saved masks: NIfTI for medical volumes, the source type for 2D images."""
    src = _suffix(Path(image.source))
    if image.sitk_reference is not None:
        return ".nii.gz"
    if image.is_2d and src in PIL_SUFFIXES:
        return ".png"
    return ".tif"


def describe_orientation(orientation: tuple) -> list:
    """Human-readable direction of each axis, e.g. 'index increases toward patient Superior'."""
    names = {"L": "Left", "R": "Right", "A": "Anterior", "P": "Posterior", "S": "Superior", "I": "Inferior"}
    out = []
    for k, d in enumerate(orientation):
        if d is None:
            out.append(f"axis{k}: orientation unknown")
        else:
            out.append(f"axis{k}: {names[_OPPOSITE[d]]} -> {names[d]} (index increases toward patient {names[d]})")
    return out
