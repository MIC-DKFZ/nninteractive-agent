# Domain notes

## CT (Hounsfield units)

- Windows: soft tissue `[-160, 240]` (default), bone `[-400, 1200]`, lung `[-1200, 200]`.
- Slices are often much thicker than the in-plane spacing (e.g. 5 mm vs 0.8 mm), so objects look
  blocky in the other planes. Still choose each prompt's plane by where the object (or the error)
  is best visible and easiest to trace, and check the result in all planes.
- Organs end gradually; the first and last slices are small. A tiny isolated piece one slice
  beyond the end is usually a false positive.
- Ribs: 12 per side (sometimes 11 or 13), numbered from the top. Each rib is its own object. They
  are thin and curved, so use a long positive scribble following the rib (several points) in the
  slice that shows the longest stretch of it, often an axial slice where the rib is a curved arc.
  A rib runs from its head, which touches the vertebral body
  (costovertebral joint), to its anterior end at the costal cartilage: include head and neck up to
  the vertebra, but not the vertebra itself. The 11th and 12th ribs are short and end freely in the
  flank muscles; check their tips explicitly. Check with a bone window and a projection along the
  anterior-posterior axis; neighboring ribs must not merge.
- Vessels and airways are tree-shaped: scribble along their length in a plane that shows a long
  stretch of them, segment them in parts if needed and check branch ends. Expect more corrections
  than for compact organs, and flag thin branches for review.

## MRI

- Intensities have no absolute meaning; use the default percentile window and adjust per image.
- Contrast between neighboring tissues can be low: verify boundaries in all three planes.

## Fluorescence microscopy (nuclei, cells)

- Often anisotropic (axial spacing larger than lateral): the lateral planes usually show the
  nuclei best; corrections go wherever the error is most visible.
- Many similar instances: detect seeds in Python (blob detection or watershed on a smoothed image),
  then segment each instance in a loop: a point for a nucleus a few voxels across, otherwise a
  short scribble through its middle or a box computed from its candidate mask. Touching nuclei are the main
  failure mode: check for merged instances (an object about twice the typical size, or two
  intensity peaks) and separate them with a negative point in the other nucleus.

## Electron microscopy

- Membranes are dark lines; organelles (mitochondria) are textured regions bounded by membranes.
  Scribbles must not cross membranes. Expect many instances.

## Photographs and other 2D color images

- The model sees luminance only; the views show color. Objects separated only by color may be
  harder; use more prompts at such boundaries.
- Shape is `[1, H, W]`: use `[0, row, col]` or `[row, col]`; there is no 3D check to do, but look
  closely at thin parts and at boundaries with low contrast.
