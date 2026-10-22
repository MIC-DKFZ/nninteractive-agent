// A synthetic 3D scene for browser_checks.py transparency: a cube of uniform material in air with a
// sphere in its centre, rendered by static/render3d.js on its own canvas over the page. Injected as a
// module into a page of the web demo (same origin); sets window.r3scene.
import { Render3D } from '/static/render3d.js'

const canvas = document.createElement('canvas')
canvas.style.cssText = 'position:fixed;left:0;top:0;width:400px;height:400px;z-index:1000'
document.body.appendChild(canvas)
const r3 = new Render3D(canvas)

function sphere(c, r, n = 48) {
  const pos = [], tris = []
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    const th = Math.PI * i / n, ph = 2 * Math.PI * j / n
    pos.push(c[0] + r * Math.sin(th) * Math.cos(ph), c[1] + r * Math.sin(th) * Math.sin(ph), c[2] + r * Math.cos(th))
  }
  const P = (k) => pos.slice(3 * k, 3 * k + 3)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const a = i * (n + 1) + j, b = a + n + 1
    for (const t of [[a, b, b + 1], [a, b + 1, a + 1]]) {
      const [p0, p1, p2] = t.map(P)
      const u = p1.map((v, k) => v - p0[k]), w = p2.map((v, k) => v - p0[k])
      const nrm = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
      if (Math.hypot(...nrm) < 1e-9) continue // the poles
      const out = nrm[0] * (p0[0] - c[0]) + nrm[1] * (p0[1] - c[1]) + nrm[2] * (p0[2] - c[2])
      tris.push(...(out > 0 ? t : [t[0], t[2], t[1]])) // counter-clockwise from outside
    }
  }
  return [new Float32Array(pos), new Uint32Array(tris)]
}

const N = 48
const data = new Float32Array(N * N * N)
for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  data[i + N * (j + N * k)] = [i, j, k].every((v) => v >= 4 && v < N - 4) ? 0.4 : 0
}
r3.setVolume({ dims: [N, N, N], data, affine: [2, 0, 0, -N, 0, 2, 0, -N, 0, 0, 2, -N, 0, 0, 0, 1], lo: 0, hi: 1 })
const [spos, stris] = sphere([0, 0, 0], 16)

/** The material's opacity per 2 mm is ``alpha``; ``mode``: 'shaded' or 'mip'. */
function preset(alpha, mode) {
  const n = 256, tf = new Float32Array(n * 4), material = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    const v = i / (n - 1)
    tf.set([0.75, 0.75, 0.8, v > 0.25 && v < 0.6 ? alpha : 0], 4 * i)
  }
  return { mode, tf, material, refLen: 2, step: 0.5, light: [0.3, 0.75, 0.3, 40], grad: [0.002, 0.02] }
}

function frame() {
  r3.anim = null
  r3.render()
  const gl = r3.gl, [w, h] = r3.size
  const px = new Uint8Array(w * h * 4)
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px) // same task as render: the buffer is intact
  return { px, w, h }
}

// mean colour change (0..255) over the sphere's centre between two frames
function change(a, b) {
  let s = 0, n = 0
  for (let y = 0; y < a.h; y++) for (let x = 0; x < a.w; x++) {
    if (Math.hypot(x - a.w / 2, y - a.h / 2) > 0.08 * a.w) continue
    const i = 4 * (y * a.w + x)
    for (let c = 0; c < 3; c++) s += Math.abs(a.px[i + c] - b.px[i + c])
    n += 3
  }
  return s / n
}

/** How visible the sphere is through the volume as the uncommitted object ('current') and as a
 * committed one, and with no volume at all. */
function run(alpha, surfaceOpacity = 1, mode = 'shaded') {
  r3.resize()
  r3.setPreset(preset(alpha, mode))
  r3.surfaceOpacity = surfaceOpacity
  r3.clearMeshes()
  const none = frame()
  r3.setMesh('current', spos, stris, [255, 212, 0])
  const current = change(none, frame())
  r3.removeMesh('current', false)
  r3.setMesh('o1', spos, stris, [255, 212, 0])
  const committed = change(none, frame())
  r3.setPreset(null)
  const bare = frame()
  r3.removeMesh('o1', false)
  const bare0 = frame()
  return { current, committed, bare: change(bare0, bare) }
}

window.r3scene = { run }
