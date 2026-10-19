"""Off-screen rendering of slices with segmentation overlays, made for a vision model.

Design choices that matter for an agent reading the images:

* Tick labels are absolute voxel indices, so coordinates for the next prompt can be read
  straight off the image. Each panel names its fixed axis and its row/column axes.
* Voxels are drawn as blocks (nearest neighbour) with physical aspect ratio.
* Medical volumes are shown in radiological convention (superior up, anterior up/left, patient
  right on the viewer's left) and the patient direction is printed at each edge. When that
  flips an axis, tick labels decrease left-to-right or top-to-bottom; the axis label says so.
* ``render_3d`` shows an object's surface as depth-shaded views in the frame of the matching slice
  view (same axes, tick labels and patient letters), so positions can be read off directly.
* The object being edited is a yellow contour with a light fill; committed objects are thin
  colored contours over a fainter fill in their color (so inside and outside are clear even where
  an outline runs along the panel edge), labeled with their id; prompts are green (positive) /
  red (negative) and numbered by step.

Only the matplotlib object API is used (no pyplot), so this works inside napari's process.
"""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass, field

import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from .metrics import bbox_of
from .raster import plane_axes

BG = "#161616"
FG = "#e8e8e8"
CURRENT_COLOR = "#ffd400"
CURRENT_COLOR_RGB = "#00f0ff"  # on color photos yellow vanishes against fat/skin tones
POS_COLOR = "#2bff4a"
NEG_COLOR = "#ff3b3b"
OBJECT_COLORS = [
    "#00e5ff", "#ff4fd8", "#ff9f1a", "#7c8cff", "#9dff00", "#ff6b6b", "#00ffa8", "#c58cff",
    "#ffd8a8", "#4dabf7", "#f783ac", "#94d82d",
]  # fmt: skip
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "I": "S", "S": "I"}
_STROKE = [patheffects.withStroke(linewidth=2.2, foreground="black")]
MAX_EDGE = 1568  # keep composite images within what vision models take without downscaling


def object_color(object_id: int) -> str:
    """The fixed palette for the first objects, then a new hue for every further id (golden-angle
    steps, two brightness levels), so that long series such as 24 ribs do not repeat colours."""
    i = int(object_id) - 1
    if i < len(OBJECT_COLORS):
        return OBJECT_COLORS[i]
    k = i - len(OBJECT_COLORS)
    r, g, b = colorsys.hsv_to_rgb((0.05 + k * 0.618033988749895) % 1.0, 0.72, 1.0 if k % 2 == 0 else 0.8)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


@dataclass
class Prompt:
    """A prompt as the renderer needs it (the workspace keeps these per object)."""

    step: int
    kind: str  # "point" | "scribble" | "bbox"
    positive: bool
    points: list  # point: [p]; scribble: polyline vertices; bbox: [corner1, corner2] (inclusive)
    axis: int | None = None  # plane of scribble/bbox
    index: int | None = None
    preview: bool = False  # a proposed prompt that was not sent (drawn thicker, labelled "new")


@dataclass
class Overlay:
    current: np.ndarray | None = None  # full-size mask of the object being edited
    instances: np.ndarray | None = None  # committed instance map
    names: dict = field(default_factory=dict)
    prompts: list = field(default_factory=list)
    show_objects: bool = True
    highlight: int | None = None  # committed id drawn thicker (when viewing one object)


def auto_window(data: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> tuple:
    step = max(1, data.size // 2_000_000)
    sample = data.ravel()[::step].astype(np.float32)
    lo, hi = np.percentile(sample, (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def _flip(letter, role: str) -> bool:
    """Whether to reverse an axis on screen so medical views follow radiological convention."""
    if letter is None:
        return False
    return letter in ("S", "A") if role == "row" else letter in ("R", "A")


def _nice_step(extent: float, target: int = 7) -> int:
    raw = max(extent / target, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return int(m * mag)
    return int(10 * mag)


def _slice2d(arr: np.ndarray, axis: int, index: int, rows, cols):
    sl = [slice(None)] * 3
    sl[axis] = index
    ra, ca = plane_axes(axis)
    sl[ra] = slice(*rows)
    sl[ca] = slice(*cols)
    return arr[tuple(sl)]


# Opacity of the fill of committed objects (the open object's fill is 0.25).
OBJECT_FILL_ALPHA = 0.15


def render_panel(
    image,
    axis: int,
    index: int,
    rows: tuple,
    cols: tuple,
    overlay: Overlay,
    window: tuple,
    px: int = 480,
    title: str | None = None,
    marks: dict | None = None,
    axis_names: tuple | None = None,
    plane_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """One slice as an RGB array. ``rows``/``cols`` are half-open ranges of the in-plane axes.
    ``plane_rgb`` (rows x cols x 3, floats in [0, 1]) replaces the image slice, e.g. a 3D view.

    ``marks`` optionally gives positions (per axis) of other displayed planes, drawn as small
    arrows at the panel edges so views can be related to each other.
    """
    ra, ca = plane_axes(axis)
    sp_r, sp_c = image.spacing[ra], image.spacing[ca]
    n_r, n_c = rows[1] - rows[0], cols[1] - cols[0]
    phys_h, phys_w = n_r * sp_r, n_c * sp_c
    scale = px / max(phys_h, phys_w)
    img_h, img_w = max(int(round(phys_h * scale)), 60), max(int(round(phys_w * scale)), 60)
    left, right, top, bottom = 58, 14, 30, 44
    dpi = 100
    fig = Figure(figsize=((img_w + left + right) / dpi, (img_h + top + bottom) / dpi), dpi=dpi, facecolor=BG)
    FigureCanvasAgg(fig)
    W, H = img_w + left + right, img_h + top + bottom
    ax = fig.add_axes([left / W, bottom / H, img_w / W, img_h / H])
    ax.set_facecolor("black")
    extent = (cols[0] - 0.5, cols[1] - 0.5, rows[1] - 0.5, rows[0] - 0.5)

    if plane_rgb is not None:
        ax.imshow(plane_rgb, extent=extent, interpolation="nearest", aspect="auto")
    elif image.rgb is not None:
        ax.imshow(_slice2d(image.rgb, axis, index, rows, cols), extent=extent, interpolation="nearest", aspect="auto")
    else:
        ax.imshow(
            _slice2d(image.data, axis, index, rows, cols).astype(np.float32),
            cmap="gray",
            vmin=window[0],
            vmax=window[1],
            extent=extent,
            interpolation="nearest",
            aspect="auto",
        )

    rr = np.arange(rows[0], rows[1])
    cc = np.arange(cols[0], cols[1])

    # Committed objects: a faint fill, thin contours + id label.
    if overlay.instances is not None and overlay.show_objects:
        inst = _slice2d(overlay.instances, axis, index, rows, cols)
        if OBJECT_FILL_ALPHA > 0 and inst.any():
            fill = np.zeros((*inst.shape, 4), dtype=np.float32)
            for oid in np.unique(inst[inst > 0]):
                fill[inst == oid] = to_rgba(object_color(oid), OBJECT_FILL_ALPHA)
            ax.imshow(fill, extent=extent, interpolation="nearest", aspect="auto")
        for oid in np.unique(inst):
            if oid == 0:
                continue
            m = inst == oid
            lw = 2.0 if overlay.highlight == oid else 1.0
            _contour(ax, cc, rr, m, object_color(oid), lw)
            r_c, c_c = np.argwhere(m).mean(0)
            ax.text(
                cols[0] + c_c, rows[0] + r_c, f"#{oid}", color=object_color(oid), fontsize=8, ha="center",
                va="center", path_effects=_STROKE,
            )  # fmt: skip

    # The object being edited: light fill + contour.
    if overlay.current is not None:
        cur = _slice2d(overlay.current, axis, index, rows, cols) > 0
        if cur.any():
            color = CURRENT_COLOR_RGB if image.rgb is not None else CURRENT_COLOR
            fill = np.zeros((*cur.shape, 4), dtype=np.float32)
            fill[cur] = to_rgba(color, 0.25)
            ax.imshow(fill, extent=extent, interpolation="nearest", aspect="auto")
            _contour(ax, cc, rr, _fill_pinholes(cur), color, 1.6)

    for p in overlay.prompts:
        _draw_prompt(ax, p, axis, index, ra, ca, image.shape)

    # Orientation: radiological convention for medical volumes.
    o_r, o_c = image.orientation[ra], image.orientation[ca]
    flip_r, flip_c = _flip(o_r, "row"), _flip(o_c, "col")
    ax.set_xlim(extent[1], extent[0]) if flip_c else ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2]) if flip_r else ax.set_ylim(extent[2], extent[3])

    # About one tick per 65 px of screen, chosen per axis so short axes still get labels.
    step_r = _nice_step(n_r, target=max(3, img_h / 65))
    step_c = _nice_step(n_c, target=max(3, img_w / 65))
    ax.set_xticks(np.arange(int(np.ceil(cols[0] / step_c)) * step_c, cols[1], step_c))
    ax.set_yticks(np.arange(int(np.ceil(rows[0] / step_r)) * step_r, rows[1], step_r))
    ax.tick_params(colors=FG, labelsize=8, length=3)
    ax.grid(True, color="white", alpha=0.18, linewidth=0.5)
    for s in ax.spines.values():
        s.set_color("#555555")
    row_name, col_name = axis_names or (f"axis{ra}", f"axis{ca}")
    ax.set_xlabel(f"{col_name}  (index increases {'<-' if flip_c else '->'})", color=FG, fontsize=8, labelpad=2)
    ax.set_ylabel(f"{row_name}  (index increases {'up' if flip_r else 'down'})", color=FG, fontsize=8, labelpad=2)
    ax.set_title(title or f"axis{axis} = {index}", color=FG, fontsize=10, pad=4)

    if o_r is not None:
        top_l, bot_l = (o_r, _OPPOSITE[o_r]) if flip_r else (_OPPOSITE[o_r], o_r)
        ax.text(0.5, 0.985, top_l, transform=ax.transAxes, color="#ffb000", fontsize=10, ha="center", va="top",
                path_effects=_STROKE)  # fmt: skip
        ax.text(0.5, 0.015, bot_l, transform=ax.transAxes, color="#ffb000", fontsize=10, ha="center", va="bottom",
                path_effects=_STROKE)  # fmt: skip
    if o_c is not None:
        left_l, right_l = (o_c, _OPPOSITE[o_c]) if flip_c else (_OPPOSITE[o_c], o_c)
        ax.text(0.012, 0.5, left_l, transform=ax.transAxes, color="#ffb000", fontsize=10, ha="left", va="center",
                path_effects=_STROKE)  # fmt: skip
        ax.text(0.988, 0.5, right_l, transform=ax.transAxes, color="#ffb000", fontsize=10, ha="right", va="center",
                path_effects=_STROKE)  # fmt: skip

    if marks:
        if ra in marks and rows[0] <= marks[ra] < rows[1]:
            ax.annotate("", xy=(0, marks[ra]), xycoords=("axes fraction", "data"), xytext=(-12, 0),
                        textcoords="offset points", arrowprops=dict(arrowstyle="-|>", color="#4dd2ff", lw=1.2),
                        annotation_clip=False)  # fmt: skip
        if ca in marks and cols[0] <= marks[ca] < cols[1]:
            ax.annotate("", xy=(marks[ca], 0), xycoords=("data", "axes fraction"), xytext=(0, -12),
                        textcoords="offset points", arrowprops=dict(arrowstyle="-|>", color="#4dd2ff", lw=1.2),
                        annotation_clip=False)  # fmt: skip

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _fill_pinholes(mask: np.ndarray) -> np.ndarray:
    """For display only: close holes of a few pixels (specular highlights, noise) so the
    contour shows the object's outline instead of dozens of tiny rings. The fill still shows
    the true mask."""
    from skimage.morphology import remove_small_holes

    return remove_small_holes(mask, max_size=max(4, int(mask.size * 0.0005)))


def _contour(ax, cc, rr, mask, color, lw):
    if mask.all() or not mask.any():
        if mask.all():
            ax.add_patch(Rectangle((cc[0] - 0.5, rr[0] - 0.5), len(cc), len(rr), fill=False, ec=color, lw=lw))
        return
    pad = np.pad(mask.astype(np.float32), 1)
    ax.contour(
        np.arange(cc[0] - 1, cc[-1] + 2), np.arange(rr[0] - 1, rr[-1] + 2), pad, levels=[0.5], colors=[color],
        linewidths=lw,
    )  # fmt: skip


def _draw_prompt(ax, p: Prompt, axis, index, ra, ca, shape):
    color = POS_COLOR if p.positive else NEG_COLOR
    label = dict(color=color, fontsize=8, fontweight="bold", path_effects=_STROKE)
    tag = "new" if p.preview else str(p.step)
    lw = 1.6 if p.preview else 1.0  # previews are drawn heavier
    if p.kind == "point":
        q = p.points[0]
        dist = abs(q[axis] - index)
        if dist <= 3:
            ax.plot([q[ca]], [q[ra]], marker="o", ms=7 if dist == 0 else 5, mfc=color if dist == 0 else "none",
                    mec="black" if dist == 0 else color, mew=1.2 * lw)  # fmt: skip
            ax.text(q[ca] + 1.5, q[ra] - 1.5, tag, **label)
    elif p.kind == "scribble":
        if p.axis == axis and p.index == index:
            pts = np.array(p.points)
            ax.plot(pts[:, ca], pts[:, ra], color=color, lw=2.0 * lw, marker="o", ms=3, solid_capstyle="round")
            ax.text(pts[0, ca] + 1.5, pts[0, ra] - 1.5, tag, **label)
    elif p.kind == "bbox":
        c1, c2 = p.points
        lo = [min(a, b) for a, b in zip(c1, c2)]
        hi = [max(a, b) for a, b in zip(c1, c2)]
        if p.axis == axis and p.index == index:
            ax.add_patch(Rectangle((lo[ca] - 0.5, lo[ra] - 0.5), hi[ca] - lo[ca] + 1, hi[ra] - lo[ra] + 1,
                                   fill=False, ec=color, lw=1.5 * lw, ls="-" if p.preview else "--"))  # fmt: skip
            ax.text(lo[ca], lo[ra] - 1.5, tag, **label)
        elif p.axis != axis and lo[axis] <= index <= hi[axis]:
            # The box's plane crosses this view: show it as a dashed segment.
            if p.axis == ra:
                ax.plot([lo[ca] - 0.5, hi[ca] + 0.5], [p.index, p.index], color=color, lw=1.2, ls="--")
            else:
                ax.plot([p.index, p.index], [lo[ra] - 0.5, hi[ra] + 0.5], color=color, lw=1.2, ls="--")


def stack(panels: list, ncols: int | None = None, gap: int = 6) -> np.ndarray:
    """Arrange RGB panels in a grid (row-major), padding with the background color."""
    ncols = ncols or len(panels)
    nrows = int(np.ceil(len(panels) / ncols))
    bg = np.array([int(BG[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    col_w = [max(p.shape[1] for p in panels[c::ncols]) for c in range(ncols)]
    row_h = [max(p.shape[0] for p in panels[r * ncols : (r + 1) * ncols]) for r in range(nrows)]
    out = np.empty((sum(row_h) + gap * (nrows - 1), sum(col_w) + gap * (ncols - 1), 3), dtype=np.uint8)
    out[:] = bg
    for i, p in enumerate(panels):
        r, c = divmod(i, ncols)
        y, x = sum(row_h[:r]) + gap * r, sum(col_w[:c]) + gap * c
        out[y : y + p.shape[0], x : x + p.shape[1]] = p
    return out


def to_png(rgb: np.ndarray, max_edge: int = MAX_EDGE) -> bytes:
    img = Image.fromarray(rgb)
    if max(img.size) > max_edge:
        f = max_edge / max(img.size)
        img = img.resize((int(img.size[0] * f), int(img.size[1] * f)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def fov_around(shape, bbox=None, center=None, size=None, margin_frac=0.35, min_size=32) -> list:
    """A 3D field of view: the bbox with a margin, or a box of ``size`` around ``center``."""
    if bbox is not None:
        out = []
        for (lo, hi), s in zip(bbox, shape):
            ext = hi - lo
            pad = int(max(np.ceil(ext * margin_frac), (min_size - ext) / 2, 4))
            out.append([max(0, lo - pad), min(s, hi + pad)])
        return out
    if center is not None and size is not None:
        sizes = size if isinstance(size, (list, tuple)) else [size] * 3
        return [[max(0, int(c - z // 2)), min(s, int(c - z // 2) + int(z))] for c, z, s in zip(center, sizes, shape)]
    return [[0, s] for s in shape]


def render_orthogonal(image, center, fov, overlay, window, axes=(0, 1, 2), px=480) -> np.ndarray:
    axes = [a for a in axes if not (image.is_2d and a != 0)]
    marks = {a: center[a] for a in range(3)}
    panels = []
    for a in axes:
        ra, ca = plane_axes(a)
        panels.append(
            render_panel(image, a, int(center[a]), fov[ra], fov[ca], overlay, window, px=px, marks=marks)
        )
    return stack(panels)



# ------------------------------------------------------------------------------ 3D views
_VIEW_NAMES = {"A": "anterior", "P": "posterior", "L": "left", "R": "right", "S": "superior", "I": "inferior"}


def parse_view(image, name: str) -> tuple:
    """(axis, viewer on the high-index side, label) for a patient direction ('A', 'L', ...) or
    'axis<k>-' / 'axis<k>+' (the viewer on the low / high index side of axis k)."""
    name = name.strip()
    if len(name) == 6 and name.startswith("axis") and name[4] in "012" and name[5] in "+-":
        a = int(name[4])
        return a, name[5] == "+", f"axis{a} {'high' if name[5] == '+' else 'low'}-index side"
    letter = name[:1].upper()
    for a, o in enumerate(image.orientation):
        if o is not None and letter in (o, _OPPOSITE[o]):
            return a, letter == o, f"{_VIEW_NAMES[letter]} ({letter})"
    valid = "A, P, L, R, S, I or " if any(image.orientation) else ""
    raise ValueError(f"unknown view {name!r}; use {valid}axis0-, axis0+, axis1-, ... axis2+")


def default_views(image, mask: np.ndarray) -> list:
    """Front, the object's own lateral side and top for medical images, else one view per axis."""
    if not all(image.orientation):
        return ["axis0-", "axis1-", "axis2-"]
    lr = next(a for a, o in enumerate(image.orientation) if o in "RL")
    centre = np.argwhere(mask)[:, lr].mean() if mask.any() else image.shape[lr] / 2
    side = image.orientation[lr] if centre >= image.shape[lr] / 2 else _OPPOSITE[image.orientation[lr]]
    return ["A", side, "S"]


def _first_hit(vol: np.ndarray, axis: int, from_high: bool) -> tuple:
    """Depth (index from the viewer's side) of the first nonzero voxel along ``axis``, the value
    there, and whether there is one; arrays indexed by the two in-plane axes in increasing order."""
    v = np.moveaxis(vol, axis, 0)
    if from_high:
        v = v[::-1]
    nz = v != 0
    depth = nz.argmax(0)
    return depth, np.take_along_axis(v, depth[None], 0)[0], nz.any(0)


def _shade(depth: np.ndarray, hit: np.ndarray, sp_depth: float, sp_r: float, sp_c: float) -> np.ndarray:
    """Brightness of a depth map: Lambert lighting of the surface normals plus a depth cue."""
    from scipy.ndimage import gaussian_filter

    out = np.zeros(depth.shape, np.float32)
    if not hit.any():
        return out
    z = depth.astype(np.float32) * sp_depth
    zmin, zmax = z[hit].min(), z[hit].max()
    # Background as a far plane: silhouettes get steep normals and read as dark outlines. The blur
    # grows with the slice thickness relative to the in-plane spacing, so thick slices do not show
    # as terraces (one ring per slice) when seen from above.
    sigma = float(np.clip(0.8 * sp_depth / min(sp_r, sp_c), 0.8, 4.0))
    zf = gaussian_filter(np.where(hit, z, zmax + 4 * sp_depth), sigma)
    g_r, g_c = np.gradient(zf, sp_r, sp_c)
    n = np.stack([-g_r, -g_c, np.ones_like(zf)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-0.45, -0.35, 1.0]) / np.linalg.norm([-0.45, -0.35, 1.0])
    lam = np.clip(n @ light, 0, 1)
    cue = 1 - 0.4 * (z - zmin) / max(zmax - zmin, 1e-6)
    out[hit] = ((0.18 + 0.82 * lam) * cue)[hit]
    return out


def render_3d(image, target: np.ndarray, others: np.ndarray | None, color, views: list, fov: list, title: str,
              px: int = 420) -> np.ndarray:  # fmt: skip
    """Depth-shaded views of the ``target`` mask, one panel per ``(axis, from_high, label)``, with the
    objects of the label map ``others`` faint in their colors where they lie in front of or beside it.
    Each panel uses the frame of the slice view of its plane (``render_panel``)."""
    sl = tuple(slice(lo, hi) for lo, hi in fov)
    t = target[sl]
    o = others[sl] if others is not None else None
    rgb_t = np.array(to_rgba(color)[:3], np.float32)
    panels = []
    for axis, from_high, label in views:
        ra, ca = plane_axes(axis)
        sp = image.spacing
        d_t, _, hit_t = _first_hit(t.astype(np.uint8), axis, from_high)
        rgb = np.full((*hit_t.shape, 3), 0.07, np.float32)
        rgb[hit_t] = rgb_t * _shade(d_t, hit_t, sp[axis], sp[ra], sp[ca])[hit_t, None]
        if o is not None and o.any():
            d_o, id_o, hit_o = _first_hit(o, axis, from_high)
            shade_o = _shade(d_o, hit_o, sp[axis], sp[ra], sp[ca])
            front = hit_o & (~hit_t | (d_o < d_t))
            for oid in np.unique(id_o[front]):
                m = front & (id_o == oid)
                c = np.array(to_rgba(object_color(int(oid)))[:3], np.float32) * shade_o[m, None]
                alpha = np.where(hit_t[m], 0.4, 0.55)[:, None]  # see-through where the target is behind
                rgb[m] = alpha * c + (1 - alpha) * rgb[m]
        panels.append(render_panel(image, axis, 0, fov[ra], fov[ca], Overlay(), (0, 1), px=px,
                                   title=f"{title}, seen from {label}", plane_rgb=np.clip(rgb, 0, 1)))  # fmt: skip
    return stack(panels)


def render_montage(image, axis, indices, fov, overlay, window, px=300, ncols=None) -> np.ndarray:
    ra, ca = plane_axes(axis)
    panels = [render_panel(image, axis, int(i), fov[ra], fov[ca], overlay, window, px=px) for i in indices]
    ncols = ncols or min(len(panels), 4 if len(panels) > 9 else 3)
    return stack(panels, ncols=ncols)


def render_projection(image, axis, overlay: Overlay, window, mode="max", px=640, fov=None) -> np.ndarray:
    """Intensity projection along ``axis`` with every committed object's projected outline.

    ``fov`` limits the projected range (along ``axis``) and the displayed in-plane region.
    Tick labels stay absolute voxel indices of the two in-plane axes.
    """
    fov = fov or [[0, s] for s in image.shape]
    ra, ca = plane_axes(axis)
    sl = [slice(None)] * 3
    sl[axis] = slice(*fov[axis])
    sl = tuple(sl)
    src = image.rgb if image.rgb is not None else image.data
    sub = src[sl]
    proj = sub.max(axis=axis) if mode == "max" else sub.mean(axis=axis, dtype=np.float32)
    proj_img = type(image)(
        data=np.expand_dims(proj if image.rgb is None else proj.mean(-1), 0),
        spacing=(1.0, image.spacing[ra], image.spacing[ca]),
        source=image.source,
        rgb=np.expand_dims(proj.astype(np.uint8), 0) if image.rgb is not None else None,
        orientation=(None, image.orientation[ra], image.orientation[ca]),
    )
    proj_overlay = Overlay(names=overlay.names, show_objects=True, highlight=overlay.highlight)
    if overlay.instances is not None:
        inst = overlay.instances[sl]
        # Project each object separately so overlapping objects keep their own outline.
        out = np.zeros(inst.shape[:axis] + inst.shape[axis + 1 :], dtype=inst.dtype)
        for oid in np.unique(inst):
            if oid:
                out[np.any(inst == oid, axis=axis) & (out == 0)] = oid
        proj_overlay.instances = out[None]
    if overlay.current is not None:
        proj_overlay.current = np.any(overlay.current[sl], axis=axis)[None]
    if mode == "max" and image.rgb is None:
        window = (window[0], max(window[1], float(np.percentile(proj, 99.5))))
    return render_panel(
        proj_img, 0, 0, tuple(fov[ra]), tuple(fov[ca]), proj_overlay, window, px=px,
        title=f"{mode} projection along axis{axis} (axis{axis} {fov[axis][0]}-{fov[axis][1] - 1})",
        axis_names=(f"axis{ra}", f"axis{ca}"),
    )  # fmt: skip


def object_center(mask: np.ndarray) -> list | None:
    """A point inside the object near its middle: the slice-wise center of the bbox, snapped to the mask."""
    bbox = bbox_of(mask)
    if bbox is None:
        return None
    sub = mask[tuple(slice(lo, hi) for lo, hi in bbox)]
    idx = np.argwhere(sub)
    mid = idx.mean(0)
    best = idx[np.argmin(((idx - mid) ** 2).sum(1))]
    return [int(b + lo) for b, (lo, _) in zip(best, bbox)]
