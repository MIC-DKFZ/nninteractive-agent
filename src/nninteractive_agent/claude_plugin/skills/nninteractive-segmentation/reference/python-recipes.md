# Python recipes for the `python` tool

Only when the `python` tool is available (it can be switched off). If its description says guarded
mode, it runs computations only: imports from numpy, scipy, skimage, matplotlib and computational
standard modules, no files, no `getattr`/`eval`, no attributes starting with `_`; return figures with
`show(fig)`.

The namespace persists between calls. Available: `ws` (workspace), `np`, `ndi` (scipy.ndimage),
`skimage`, `show(x)`.

Useful workspace state:

- `ws.image.data`: intensities `[axis0, axis1, axis2]` (2D images: `[1, H, W]`)
- `ws.image.spacing`, `ws.image.orientation`, `ws.window` (current default display window)
- `ws.buffer`: mask of the open object (uint8). Writing to it replaces the object's mask: the next
  prompt continues from your mask, but nnInteractive forgets the earlier prompts (no undo past it)
- `ws.instances`: committed instance map (uint16), `ws.objects`: `{id: {"name", "voxels", ...}}`

All tool methods exist on `ws` and return `Result` objects (`.text`, `.images`, `.data`):
`ws.new_object(name, extent=None)`, `ws.scribble(points, positive)`, `ws.bbox(c1, c2)`, `ws.point(p, positive)`,
`ws.undo()`, `ws.stats(obj)`, `ws.commit(notes)`, `ws.discard()`, `ws.view(...)`,
`ws.montage(axis, n, ...)`. `r.data["stats"]` holds the numbers after a prompt.
Errors raise `AgentError` with an explanation.

## Find candidate instances

```python
from skimage import filters, measure, morphology
img = ws.image.data.astype(np.float32)
sm = ndi.gaussian_filter(img, 1)
mask = sm > filters.threshold_otsu(sm)
mask = morphology.remove_small_objects(mask, 50)
lab, n = ndi.label(mask)
props = measure.regionprops(lab)
print(n, "candidates")
for p in sorted(props, key=lambda p: -p.area)[:40]:
    print(p.label, p.area, [int(c) for c in p.centroid], p.bbox)
```

Touching objects (nuclei, cells): split with a distance transform + `skimage.segmentation.watershed`
seeded by `skimage.feature.peak_local_max`, or use `skimage.feature.blob_log` for blob centers.
Treat these only as *seeds* for prompts; nnInteractive produces the masks.

Show the candidates with numbers to pick from:

```python
import matplotlib
from matplotlib.figure import Figure
fig = Figure(figsize=(8, 8)); ax = fig.add_subplot()
z = ws.image.shape[0] // 2
ax.imshow(img[z], cmap="gray"); ax.contour(lab[z] > 0, [0.5], colors="y", linewidths=0.6)
for p in props:
    c = p.centroid
    if abs(c[0] - z) < 3:
        ax.text(c[2], c[1], str(p.label), color="cyan", fontsize=7)
show(fig)
```

## Segment many instances in a loop

```python
results = {}
for i, (name, seed) in enumerate(targets.items()):     # targets: name -> [a0, a1, a2]
    ws.new_object(name)
    r = ws.point(seed, True)          # tiny instances; larger ones: ws.scribble([...], True) through the middle,
                                      # or ws.bbox(c1, c2) from a candidate mask where the box holds little else
    s = r.data["stats"]
    results[name] = (s["voxels"], s.get("components"), s.get("overlaps"))
    ws.commit("auto: one point")
for k, v in results.items():
    print(k, v)
```

Then compare sizes between siblings, look for objects with several components or overlaps, and
inspect those first (`ws.edit_object(id)` to correct a committed one).

## Per-object table

```python
for oid, o in ws.objects.items():
    m = ws.instances == oid
    print(oid, o["name"], int(m.sum()))
```
