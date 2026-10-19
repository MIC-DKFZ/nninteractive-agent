---
name: nninteractive-segmentation
description: Segment structures in 2D or 3D images (CT, MRI, microscopy, electron microscopy, photos) from a text request, using nnInteractive prompts (scribbles, boxes, points) through the nninteractive tools, and verify and correct the results. Use whenever asked to segment, delineate, outline, label or count structures in an image file.
---

# Segmenting with nnInteractive

You drive nnInteractive, a promptable 3D segmentation model, through the nninteractive tools.
You decide *what* to segment and *where* to prompt; the model produces the mask. Your job is to
choose good prompts, check the results critically, and correct them until they are right.

## How nnInteractive behaves

- **One object at a time.** Each object is a single structure instance. Segmenting "the rib cage"
  means one object per rib; "all nuclei" means one object per nucleus. Start each with
  `new_object(name)`, finish with `commit`.
- **One object per voxel.** Committed objects never overlap. To segment something inside an
  object you already committed (lesions in the liver, vessels in an organ, gas in the stomach),
  commit it with `overlap="replace"`: it takes those voxels and the outer object shrinks around
  it. Without that, the committed object keeps them, and `commit` refuses when the new object
  would lose most of itself. A new object that overlaps a neighbour it should not touch is a
  leak: correct it.
- **2D prompts, 3D result.** A prompt lives in one slice, but the prediction is a full 3D mask.
  Always check slices other than the one you prompted in.
- **Prompts accumulate.** Each new prompt refines the current mask; the model sees its previous
  prediction and all earlier prompts of this object. A bad prompt keeps influencing later
  predictions: `undo` it right away if a prompt made things worse.
- **Prompt types: scribbles preferred, boxes only where they fit, points rarely.**
  - `scribble`: a polyline in one slice, drawn 2 px wide by the tool. The default first prompt
    and the prompt for corrections. **Long lines work best**: run the line along the
    middle of the region it describes and let it cover most of the region's visible extent,
    with enough points to follow its curves. **The whole line must stay inside that region**:
    positive = inside the target (or inside the part that is missing), negative = inside the
    wrongly included region. It may come close to a boundary but must not cross it.
    `preview=true` draws the line on a zoomed view of its slice without sending it. **The first
    line of every object must be previewed** (the tool refuses otherwise): check that it stays
    inside the object along its whole length, correct and preview again if needed, then send
    the same points with `preview=false`. Preview corrections too when you are not sure.
  - `bbox`: a 2D box (two inclusive corners in one slice), only as an object's first prompt
    and only where a box encloses the object without much else: a compact object whose box in
    that slice contains little besides it. Not for elongated structures that are not aligned
    with the image axes, not for convoluted shapes, and not where the box would also take in a
    lot of neighboring structures you do not want: use a scribble there. A box also tells
    nnInteractive the object's extent, which helps for small objects with little contrast to
    their surroundings. Boxes read off a view are easily off by several voxels, so the tool
    makes you preview each box (`preview=true`); send it only once every edge sits just outside
    the object. A box that cut the object is fixed by `discard()` and a corrected box.
  - `point`: only where a line does not fit: a tiny object, or a tiny missed bit or false blob
    (a few voxels across). Place it in the middle of the region.
  - Lasso is not available.
- **Large objects.** nnInteractive is most accurate on objects up to about 320 voxels along
  every axis. For a larger object, pass your estimate of its size to `new_object`
  (`extent=[a0, a1, a2]` in voxels): it is then segmented on a downsampled copy of the image
  and the mask is scaled back up. You keep working in full-resolution coordinates. Thin
  structures (vessel trees, thin layers such as skin) lose detail when downsampled: segment
  them at full resolution.
- **You decide what belongs to the object.** nnInteractive has no fixed definition of a
  structure: whether a liver includes its tumors, or a vessel its wall, follows from your
  prompts. Decide what to include, prompt accordingly (e.g. a positive scribble through a tumor
  that belongs to the organ), and state the decision in your report.
- Do not preprocess intensities. The display window only changes what you see.

## Coordinates and views

- All coordinates are voxel indices `[axis0, axis1, axis2]`. Read them from the **tick labels**:
  every panel shows absolute indices, names its fixed axis in the title (`axis0 = 42`) and its
  row/column axes in the axis labels. In a panel `axis0 = 42` with rows axis1 and columns axis2,
  the pixel at row 210, column 130 is `[42, 210, 130]`.
- Medical volumes are shown in radiological convention (patient right on the viewer's left,
  anterior/superior up). This flips some axes on screen; the axis label then says
  `index increases <-` or `up`. Trust the tick numbers, not the screen direction. Orange letters
  at the panel edges give the patient direction (R/L, A/P, S/I).
- 2D images have shape `[1, H, W]`: use `[0, row, col]` (or `[row, col]`).
- In views, yellow contour + fill = the open object; thin colored contours over a faint fill,
  labeled `#id` = committed objects; green/red = positive/negative prompts with their step
  number; blue arrows at the panel edges = where the other panels cut.
- `view` defaults to the open object with a margin. Zoom in with `center` + `size` (edge length
  in voxels) before placing a correction near small details: precise coordinates need a view in
  which a voxel is several pixels wide. Use `window` for contrast (e.g. CT bone `[-400, 1200]`),
  and `overlay: false` to see the bare image when the mask hides the boundary you need to judge.

## Workflow

1. **Understand the request.** Which structures, how many instances, what counts as one object?
   If the request is ambiguous, choose the most plausible reading and state it in your report.
   Only if you really do not know what a structure looks like in this kind of image, a short web
   search is fine; otherwise rely on what you know.
2. **Orient.** `load_image` returns intensity statistics, orientation and overview images. Find
   the targets. For more than a handful of instances, first write down a list of targets with an
   approximate location each (seed point and rough extent) before segmenting anything. When the
   targets are many or small and you have the `python` tool, use it for candidate detection
   (thresholds, connected components, blob detection; see `reference/python-recipes.md`);
   without it, find them on `projection` and `montage` views. Note each target's
   approximate extent in voxels: targets larger than about 320 voxels along an axis need
   `extent` in `new_object`. If
   `reference/domains.md` has a section for this kind of image (CT, MRI, fluorescence
   microscopy, electron microscopy, photographs), read it now.
3. **Segment each target:**
   1. `new_object` with a descriptive name (and `extent` for a large object).
   2. Initial prompt: a long positive `scribble` along the middle of the object, covering most
      of its visible extent. Before choosing its points, view that slice zoomed so the object
      fills most of the panel (`view` with `center` and `size` about 1.2 times the object). Draw it in the plane where the object is best visible and easiest
      to trace: for a compact object the slice where it is largest and clearest, the line
      along its longest diameter; for a tubular structure (vessel, nerve, airway, rib) a plane
      that runs along its length, even if another plane has the finer resolution. Structures
      packed against neighbors of the same intensity (e.g. one muscle among others) start on a
      cross-section, the line through the middle of the cross-section, even when they are
      elongated: a line along their length leaks into the neighbors. Use a `bbox` instead only
      where it fits (see Prompt types); a tiny object (a few voxels across) starts with a `point`.
   3. Read the returned numbers (every prompt returns them): size, components, intensity inside,
      overlaps with committed objects, slice-area warnings, and for a box the **box check**: if
      the prediction sticks out of the box or leaves part of it empty, look at that side first.
      Then look: `render3d` (3D images) shows the whole shape in one image: a cut-off end or
      lobe, a protrusion or bridge into a neighbour, stray pieces. It shows no intensities,
      so confirm what you see there in slices. `montage` along the object's longest axis
      checks the boundary slice by slice; `view` for orthogonal slices.
      **Look for missing parts first.** A result that looks clean is often too small: on the
      benchmark, most accepted results that fell short were under-segmented after a single
      prompt (whole parts of a muscle, the rim of a lesion, its first and last slices). In the
      montage, compare the mask with the visible structure in every slice: does it cover the
      full cross-section, and does it reach the slices where the structure begins and ends?
   4. Correct the errors from largest to smallest, one prompt at a time, re-checking after each
      (see the playbook below). Corrections may go in any plane: for each error, use the
      orientation and slice in which that error is most clearly visible. Place each correction
      in the middle of the error: a scribble along its center line that covers most of its
      visible extent. If a correction made the result worse, `undo` it. When a correction
      scribble or point has to sit close to a boundary or another structure, check it first
      with `preview=true` (it is drawn over the current mask and reports how much of the line
      lies inside it).
   5. Check the ends of every object along its longest axis in zoomed views, especially for
      elongated structures (ribs, vessels, bones, tubes, muscles): results most often stop short
      there. A missing part gets a positive scribble along its middle (see the playbook). If you found candidates with classical methods,
      compare each final mask with its candidate region (low coverage = missing parts).
      Stop when nothing visible is wrong. Ignore 1-2 voxel boundary differences: focus on leaks,
      missing parts, wrong extra pieces and merged neighbors. After about 8 corrections without
      convergence, commit the best state with a note describing the problem, and move on.
   6. `commit`, with notes on anything uncertain. If a few clearly wrong specks remain that are
      not worth more prompts, `commit` can drop them (`keep_largest` or `min_component_size`).
4. **Check globally.** `projection` (or `view` with all objects) to see that every target is
   there, ids match names, and neighbors do not overlap. `list_objects` for sizes: an object much
   larger or smaller than its siblings is suspicious. With more than a few objects, let verifier
   subagents check them in parallel where your host allows it (see Efficiency). Fix with
   `edit_object`.
5. **Save and report.** `save`, then summarize what was segmented, how confident you are per
   object, and anything a human should review. Once per conversation, also tell the user the
   license of the nnInteractive weights that `load_image` reported: the official weights are
   CC BY-NC-SA 4.0, which allows non-commercial use only.

## Correction playbook

| What you see | What to do |
|---|---|
| Leak into a neighboring structure | Negative scribble along the middle of the leaked part, covering most of it, in the slice where it is largest. For a leak through a narrow bridge, put it on the leaked side close to the bridge. |
| Part of the target missing | Positive scribble along the middle of the missing part, covering most of it, in the slice where the gap is largest. If the object is cut off at its top/bottom, use a slice from another axis that runs through the missing part. |
| Small false blob, disconnected | Negative point (or short negative scribble) in it. |
| Merged with another instance of the same kind | Negative scribble inside the other instance. Make sure the other instance gets its own object. |
| Correction removed or added far too much | `undo`, then retry with the line placed more centrally in the error, clear of the region you want to keep. |
| Prompt changed nothing | The line was probably not where you thought: zoom in and re-read the coordinates. Check the line diagnostics in the result. |
| Completely wrong structure | `discard` and start over with a better initial prompt. |

The scribble result reports what fraction of the line lies inside the current mask and the
intensities along it. Use this as a sanity check: a negative correction should lie inside the
mask, a positive correction mostly outside it, and the intensities along a positive line should
resemble the target's.

## Efficiency

- Numbers before pictures. Every prompt returns statistics; render only when you need to decide
  where to prompt or to confirm a result. Prefer one `montage` over several `view` calls.
- If you have the `python` tool, batch repetitive work with it: it runs inside the workspace and has the same
  methods as the tools (`ws.new_object`, `ws.scribble`, `ws.bbox`, `ws.point`, `ws.commit`, `ws.stats`, ...).
  For many similar instances, segment them in a loop, collect their statistics, and inspect only
  the outliers plus a few spot checks.
- Keep images small: zoomed views of the region you are working on, not the whole volume.
- Look at images in parallel:
  - Request independent images (views of different objects or regions, montages along
    different axes) in the same message, not one per turn.
  - To check more than about 4 objects, split them into batches of 3-6. Verifier subagents can
    check the batches in parallel, but only subagents that share your nninteractive workspace
    see your objects (in Claude Code: the `nninteractive-verifier` agent). Start one per
    batch, all in the same message so they run in parallel. Give each the object ids, what
    each object should be, and anything specific to watch for. Verifiers only look: they
    return each object's errors (type, size, a point inside the error, the best plane to
    correct it). You make the corrections, then re-check those objects (yourself, or with
    another verifier round). Hosts that start a separate nninteractive server for each
    subagent (the Codex CLI, for example) give it an empty workspace: there, as without
    subagents, check the batches yourself, one `montage` per batch.

## References

- `reference/python-recipes.md`: candidate detection, loops over many instances, custom figures
  (with the `python` tool).
- `reference/domains.md`: notes for CT, MRI, microscopy, electron microscopy and photographs.
