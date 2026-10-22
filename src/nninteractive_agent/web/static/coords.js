// Workspace index <-> NiiVue voxel. Workspace arrays are [axis0, axis1, axis2] = reversed NIfTI
// (i, j, k), but NiiVue's voxel coordinates (mm2vox, convertVox2Frac, the vox of a location event)
// index its RAS-reoriented copy of the volume: RAS axis r is native axis |permRAS[r]| - 1, flipped
// where permRAS[r] < 0 (what NiiVue's getValue undoes). Converting with that permutation is exact,
// so a mentioned location is the voxel NiiVue shows and reads the value from, for any orientation.

/** Workspace index [axis0, axis1, axis2] of NiiVue voxel ``vox`` (as in a location event) of ``img``. */
export function workspaceIndex(img, vox) {
  const ijk = [0, 0, 0]
  img.permRAS.forEach((s, r) => {
    const n = Math.abs(s) - 1
    const v = Math.round(vox[r])
    ijk[n] = s > 0 ? v : img.hdr.dims[n + 1] - 1 - v
  })
  return [ijk[2], ijk[1], ijk[0]]
}

/** NiiVue crosshair position (fraction) of the centre of workspace voxel ``p``. */
export function workspaceFrac(img, p) {
  const ijk = [p[2], p[1], p[0]]
  const vox = img.permRAS.map((s) => {
    const n = Math.abs(s) - 1
    return s > 0 ? ijk[n] : img.hdr.dims[n + 1] - 1 - ijk[n]
  })
  return img.convertVox2Frac(vox)
}
