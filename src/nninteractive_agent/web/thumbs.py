"""Preview images for the demo-case gallery."""

from __future__ import annotations

import io

import numpy as np

from ..core.io import LoadedImage
from ..core.render import auto_window
from ..core.workspace import SOFT_TISSUE_WINDOW, _looks_like_ct


def thumbnail_png(img: LoadedImage, size: tuple = (480, 300)) -> bytes:
    """A representative slice: coronal for CT, otherwise the middle of the physically thinnest axis,
    shown with its physical aspect ratio on a black canvas of ``size``."""
    from PIL import Image

    data = img.data
    if img.is_2d:
        axis = 0
    elif _looks_like_ct(data) and ("A" in img.orientation or "P" in img.orientation):
        axis = img.orientation.index("A" if "A" in img.orientation else "P")  # coronal
    else:  # the plane with the largest physical area
        axis = int(np.argmin(np.array(data.shape) * np.array(img.spacing)))
    if img.rgb is not None:
        pix = np.take(img.rgb, img.rgb.shape[axis] // 2, axis=axis)
    else:
        sl = np.take(data, data.shape[axis] // 2, axis=axis).astype(np.float32)
        lo, hi = SOFT_TISSUE_WINDOW if _looks_like_ct(data) else auto_window(data)
        pix = (np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    rows, cols = [a for a in range(3) if a != axis]
    if img.orientation[rows] == "S":
        pix = pix[::-1]  # superior at the top
    if img.orientation[cols] == "R":
        pix = pix[:, ::-1]  # radiological convention: patient right on the left
    im = Image.fromarray(np.ascontiguousarray(pix))
    h_mm, w_mm = pix.shape[0] * img.spacing[rows], pix.shape[1] * img.spacing[cols]
    scale = min(size[0] / w_mm, size[1] / h_mm)
    im = im.resize((max(1, int(w_mm * scale)), max(1, int(h_mm * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(im.convert("RGB"), ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "PNG", optimize=True)
    return buf.getvalue()
