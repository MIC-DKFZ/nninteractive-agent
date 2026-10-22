// Runs static/coords.js on NiiVue's own NVImage for the volumes written by test_viewer_coords.py.
// Every voxel holds its flat workspace index + 1, so NiiVue's readout value names the voxel it shows.
//   node viewer_coords.mjs <static dir as ES modules> <cases.json>
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

globalThis.CustomEvent ??= class extends Event { constructor(t, o) { super(t, o); this.detail = o?.detail } } // Node < 19
const [staticDir, casesPath] = process.argv.slice(2)
const { NVImage } = await import(pathToFileURL(`${staticDir}/vendor/niivue-0.69.0.esm.js`))
const { workspaceFrac, workspaceIndex } = await import(pathToFileURL(`${staticDir}/coords.js`))

const apply = (m, v) => [0, 1, 2].map((r) => m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3])
const failures = []
for (const c of JSON.parse(readFileSync(casesPath, 'utf8'))) {
  const buf = readFileSync(c.file)
  const img = await NVImage.new(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), 'image.nii.gz')
  const A = img.hdr.affine
  const [n0, n1, n2] = c.shape
  const shown = (mm) => { const v = img.mm2vox(mm); return img.getValue(v[0], v[1], v[2]) - 1 } // what NiiVue displays
  const fail = (what, p, got) => failures.length < 20 && failures.push(`${c.name}: ${what} at ${p} gave ${got}`)
  for (let a0 = 0; a0 < n0; a0++) for (let a1 = 0; a1 < n1; a1++) for (let a2 = 0; a2 < n2; a2++) {
    const p = [a0, a1, a2]
    const flat = (a0 * n1 + a1) * n2 + a2
    // clicks inside the voxel name it; clicks on its faces may name a neighbour, but always the one
    // whose value NiiVue shows
    for (const d of [[0, 0, 0], [0.4, -0.4, 0.3], [-0.3, 0.4, -0.4], [0.5, -0.5, 0.5], [-0.4999, 0.4999, -0.5]]) {
      const ijk = [a2 + d[0], a1 + d[1], a0 + d[2]]
      const mm = [0, 1, 2].map((r) => A[r][0] * ijk[0] + A[r][1] * ijk[1] + A[r][2] * ijk[2] + A[r][3])
      const q = workspaceIndex(img, img.mm2vox(mm)) // a location event's vox
      const inside = d.every((x) => Math.abs(x) < 0.45)
      if (inside && q.join() !== p.join()) fail('workspaceIndex', p, q)
      const inImage = q.every((v, k) => v >= 0 && v < c.shape[k]) // else the viewer says "Outside the image"
      if (inImage && shown(mm) !== (q[0] * n1 + q[1]) * n2 + q[2]) fail('readout value vs index', q, shown(mm))
    }
    const f = img.convertFrac2MM(workspaceFrac(img, p), true) // crosshair placed by focusVoxel
    const q = workspaceIndex(img, img.mm2vox([f[0], f[1], f[2]]))
    if (q.join() !== p.join() || shown([f[0], f[1], f[2]]) !== flat) fail('workspaceFrac', p, q)
    const M = c.array_to_ras // focusObject: a surface's centre (world mm from layers.array_to_ras)
    const o = img.convertFrac2MM(img.convertMM2Frac(apply(M, p), true), true) // as viewer.focusObject
    const qo = workspaceIndex(img, img.mm2vox([o[0], o[1], o[2]]))
    if (qo.join() !== p.join()) fail('focusObject', p, qo)
  }
}
console.log(failures.length ? failures.join('\n') : 'ok')
process.exit(failures.length ? 1 : 0)
