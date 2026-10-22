// Slice views (one NiiVue canvas, custom tile layout) and a 3D view (second canvas, our own
// renderer in render3d.js: volume rendering and object surfaces composited by depth).
//
// Workspace arrays are [axis0, axis1, axis2] = reversed NIfTI (i, j, k), so a crop of a workspace
// array in C order is a contiguous run of NIfTI voxel order and can be copied straight into img.
// NiiVue's own voxel coordinates index its RAS-reoriented copy: convert positions with coords.js.
import { workspaceFrac, workspaceIndex } from './coords.js'
import { Render3D } from './render3d.js'
import { DRAG_MODE, Niivue } from './vendor/niivue-0.69.0.esm.js'

const BACK = [0, 0, 0, 1]
const CT_WINDOWS = [
  ['Soft tissue', -160, 240], ['Liver', -15, 135], ['Lung', -1350, 150], ['Bone', -500, 1300], ['Brain', 0, 80],
]
const CURRENT_LUT = { R: [0, 255], G: [0, 212], B: [0, 0], A: [0, 255], I: [0, 1] }
const PROMPT_LUT = { R: [0, 43, 255], G: [0, 255, 59], B: [0, 74, 59], A: [0, 255, 255], I: [0, 1, 2] }
const FILL_OPACITY = 0.45
// A half-filled circle (the usual window/level symbol) as the cursor of the window tool.
const WINDOW_CURSOR = `url("data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle cx="10" cy="10" r="7.5" fill="#111" stroke="#fff" stroke-width="1.5"/><path d="M10 2.5a7.5 7.5 0 0 1 0 15z" fill="#fff"/></svg>',
)}") 10 10, crosshair`
const CURRENT_OPACITY = 0.5
const TILE_NAMES = { 0: 'Axial', 1: 'Coronal', 2: 'Sagittal', 4: '3D' }
// Tile rectangles [left, top, width, height] (fractions of the viewport) per layout.
const SIDE = 0.68
function heroLayout(hero) {
  const others = [0, 1, 2, 4].filter((t) => t !== hero)
  const out = { [hero]: [0, 0, SIDE, 1] }
  others.forEach((t, i) => { out[t] = [SIDE, i / 3, 1 - SIDE, 1 / 3] })
  return out
}
const LAYOUTS = {
  grid: { 0: [0, 0, 0.5, 0.5], 1: [0.5, 0, 0.5, 0.5], 2: [0, 0.5, 0.5, 0.5], 4: [0.5, 0.5, 0.5, 0.5] },
  axial: heroLayout(0), coronal: heroLayout(1), sagittal: heroLayout(2), '3d': heroLayout(4),
  single: { 0: [0, 0, 1, 1] },
}

function hexRgb(hex) {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255]
}

async function inflate(buf, format = 'deflate') {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream(format))
  return new Response(stream).arrayBuffer()
}

/** The parts of a (gzipped) NIfTI-1 file the 3D view needs: size, voxels and voxel -> RAS mm. */
async function parseNifti(buf) {
  const b = new Uint8Array(buf, 0, 2)
  if (b[0] === 0x1f && b[1] === 0x8b) buf = await inflate(buf, 'gzip')
  const dv = new DataView(buf)
  const le = dv.getInt32(0, true) === 348
  const i16 = (o) => dv.getInt16(o, le), f32 = (o) => dv.getFloat32(o, le)
  const dims = [i16(42), i16(44), i16(46)].map((d) => Math.max(1, d))
  const types = { 2: Uint8Array, 4: Int16Array, 8: Int32Array, 16: Float32Array, 64: Float64Array, 256: Int8Array, 512: Uint16Array, 768: Uint32Array }
  const T = types[i16(70)]
  if (!T) throw new Error(`NIfTI data type ${i16(70)} is not supported`)
  const n = dims[0] * dims[1] * dims[2]
  const off = Math.round(f32(108))
  let data = off % T.BYTES_PER_ELEMENT || !le ? null : new T(buf, off, n)
  if (!data) { // unaligned or big-endian: copy element by element
    data = new Float32Array(n)
    const sz = T.BYTES_PER_ELEMENT, signed = T === Int16Array || T === Int8Array || T === Int32Array
    for (let i = 0; i < n; i++) {
      const o = off + i * sz
      data[i] = T === Float32Array ? dv.getFloat32(o, le) : T === Float64Array ? dv.getFloat64(o, le)
        : sz === 1 ? (signed ? dv.getInt8(o) : dv.getUint8(o)) : sz === 2 ? (signed ? dv.getInt16(o, le) : dv.getUint16(o, le))
          : (signed ? dv.getInt32(o, le) : dv.getUint32(o, le))
    }
  }
  const slope = f32(112), inter = f32(116)
  if (slope && (slope !== 1 || inter)) data = Float32Array.from(data, (v) => v * slope + inter)
  let affine
  if (i16(254) > 0) {
    affine = [...Array(12)].map((_, k) => f32(280 + 4 * k)).concat([0, 0, 0, 1])
  } else { // qform
    const [qb, qc, qd] = [f32(256), f32(260), f32(264)]
    const qa = Math.sqrt(Math.max(0, 1 - qb * qb - qc * qc - qd * qd))
    const qfac = f32(76) < 0 ? -1 : 1
    const [dx, dy, dz] = [f32(80), f32(84), f32(88) * qfac]
    const R = [
      [qa * qa + qb * qb - qc * qc - qd * qd, 2 * (qb * qc - qa * qd), 2 * (qb * qd + qa * qc)],
      [2 * (qb * qc + qa * qd), qa * qa + qc * qc - qb * qb - qd * qd, 2 * (qc * qd - qa * qb)],
      [2 * (qb * qd - qa * qc), 2 * (qc * qd + qa * qb), qa * qa + qd * qd - qc * qc - qb * qb],
    ]
    const t = [f32(268), f32(272), f32(276)]
    affine = [0, 1, 2].flatMap((r) => [R[r][0] * dx, R[r][1] * dy, R[r][2] * dz, t[r]]).concat([0, 0, 0, 1])
  }
  return { dims, data, affine }
}

// Transfer functions of the 3D view. Points: [intensity, [r, g, b] (sRGB 0..1), opacity per mm (CT)
// or per voxel]; material points: [intensity, shading]. Linear in between.
const SKIN = [0.93, 0.76, 0.66]
const GLASS = [0.8, 0.87, 0.96]
const BONE_TF = [[150, [0.62, 0.42, 0.32], 0], [220, [0.85, 0.7, 0.58], 0.04], [350, [0.93, 0.87, 0.76], 0.35],
  [700, [0.98, 0.95, 0.88], 0.7], [1500, [1, 0.99, 0.96], 0.9]]
const CT_LO = -1024, CT_HI = 3071

function table(points, lo, hi, n, value) {
  const out = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    const x = lo + ((hi - lo) * i) / (n - 1)
    let k = points.findIndex((p) => p[0] > x)
    if (k < 0) k = points.length
    const a = points[Math.max(0, k - 1)], b = points[Math.min(points.length - 1, k)]
    const t = b[0] > a[0] ? Math.min(1, Math.max(0, (x - a[0]) / (b[0] - a[0]))) : 0
    out.set(value(a, b, t), 4 * i)
  }
  return out
}

function preset(points, material, lo, hi, extra = {}) {
  const lerp = (u, v, t) => u + (v - u) * t
  const tf = table(points, lo, hi, 1024, (a, b, t) => [0, 1, 2].map((c) => lerp(a[1][c], b[1][c], t)).concat(lerp(a[2], b[2], t)))
  const mat = table(material, lo, hi, 1024, (a, b, t) => [lerp(a[1], b[1], t), 0, 0, 0])
  const norm = (x) => (x - lo) / (hi - lo)
  const glass = extra.glass && { ...extra.glass, iso: norm(extra.glass.iso), air: norm(extra.glass.air) }
  return { mode: 'shaded', tf, material: mat, ...extra, glass }
}

/** Name -> 3D view preset (null: no volume) for an image; ``voxel``: mean voxel size (mm). */
function volumePresets(meta, voxel) {
  if (meta.is_ct) {
    const g = [0.004, 0.025] // gradient magnitudes per mm in the 4095-HU range: soft tissue noise .. edges
    const light = [0.08, 1.0, 0.3, 40] // linear light: a small ambient term already looks bright
    return {
      range: [CT_LO, CT_HI],
      presets: {
        Bones: preset([[CT_LO, [0, 0, 0], 0], ...BONE_TF], [[CT_LO, 1]], CT_LO, CT_HI, { refLen: 1, grad: g, light }),
        'Bones + skin': preset([[CT_LO, [0, 0, 0], 0], ...BONE_TF], [[CT_LO, 1]], CT_LO, CT_HI,
          { refLen: 1, grad: g, light, glass: { rgb: GLASS, alpha: 0.55, iso: -400, air: -990 } }),
        Skin: preset([[CT_LO, SKIN, 0], [-600, SKIN, 0], [-350, SKIN, 0.8]], [[CT_LO, 1]], CT_LO, CT_HI,
          { refLen: 1, grad: g, light: [0.1, 0.95, 0.15, 24] }),
        Off: null,
      },
    }
  }
  const [lo, hi] = meta.range
  const [w0, w1] = meta.auto_window
  const c = (w1 - w0) / (hi - lo || 1) / voxel // a window's worth of contrast per mm, normalized
  const start = w0 + 0.15 * (w1 - w0)
  return {
    range: [lo, hi],
    presets: {
      Translucent: preset([[lo, [0.5, 0.5, 0.5], 0], [start, [0.55, 0.57, 0.6], 0], [w1, [1, 1, 1], 0.12]], [[lo, 0.8]], lo, hi,
        { grad: [0.05 * c, 0.4 * c], light: [0.12, 0.9, 0.25, 30] }),
      MIP: { ...preset([[lo, [0, 0, 0], 0], [w0, [0, 0, 0], 0], [w1, [1, 1, 1], 1]], [[lo, 0]], lo, hi), mode: 'mip', step: 1 },
      Off: null,
    },
  }
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`)
  const total = Number(resp.headers.get('Content-Length')) || 0
  const reader = resp.body.getReader()
  const chunks = []
  let got = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    got += value.length
    onProgress?.(got, total)
  }
  const out = new Uint8Array(got)
  let off = 0
  for (const c of chunks) { out.set(c, off); off += c.length }
  return out.buffer
}

export class Viewer {
  constructor(els, callbacks = {}) {
    this.els = els
    this.cb = callbacks
    this.meta = null
    this.layout = 'grid'
    this.overlayMode = 'fill'
    this.surfaceOpacity = 1
    this.objects = []
    this.hidden = new Set()
    this.meshes = new Map() // key -> { mesh, v, color, center }
    this.vol = {}
    this.queue = []
    this.ready = false
    this.loadToken = 0
    this.lastUser = 0
    this._glPending = false
    this.tool = 'navigate'
    this.drag = null
    // While the user holds a mouse button (or scrolls) nothing touches the GPU except drawing:
    // label refreshes and surface uploads wait in here until the interaction ends.
    this.deferred = []
    this.overlaysDirty = false
    this.pointerDown2d = false
    this.wheelUntil = 0
    this.mouse = null // last pointer position over the slice canvas (CSS px)
  }

  async init() {
    const common = { backColor: BACK, show3Dcrosshair: false, isColorbar: false, dragAndDropEnabled: false, loadingText: '' }
    this.nv2 = new Niivue({
      ...common, isRadiologicalConvention: true, crosshairColor: [0.35, 0.66, 1, 0.75], crosshairWidth: 1,
      multiplanarPadPixels: 2, isOrientationTextVisible: true,
    })
    await this.nv2.attachToCanvas(this.els.canvas2d)
    this.nv2.onLocationChange = (loc) => this._onLocation(loc)
    this.r3 = new Render3D(this.els.canvas3d)
    this._installInteraction()
    new ResizeObserver(() => this._resize()).observe(this.els.viewport)
  }

  // ------------------------------------------------------------------ image
  async load(meta, onProgress) {
    const token = ++this.loadToken
    this.ready = false
    this.queue = []
    this.meta = meta
    this.objects = []
    this.hidden.clear()
    this.r3.clearMeshes()
    this.r3.clearVolume()
    this.meshes.clear()
    for (const v of [...this.nv2.volumes]) this.nv2.removeVolume(v)
    this.vol = {}

    const [a, b] = [0.75, 0.25] // progress share of the two downloads
    const main = await fetchWithProgress(meta.url, (got, total) => onProgress?.(a * (total ? got / total : 0), got, total))
    if (token !== this.loadToken) return false
    onProgress?.(a, 0, 0, 'Building the slice views ...')
    await this.nv2.loadFromArrayBuffer(main, 'image.nii.gz')
    this.vol.image = this.nv2.volumes[0]
    await this._addLabelLayers('uint8')
    if (token !== this.loadToken) return false

    this.presets3d = {}
    if (!meta.is_2d) {
      const buf = await fetchWithProgress(meta.url3d, (got, total) => onProgress?.(a + b * (total ? got / total : 0), got, total))
      if (token !== this.loadToken) return false
      onProgress?.(1, 0, 0, 'Building the 3D view ...')
      const nii = await parseNifti(buf)
      const A = nii.affine
      const voxel = Math.cbrt(Math.abs(A[0] * (A[5] * A[10] - A[6] * A[9]) - A[1] * (A[4] * A[10] - A[6] * A[8]) + A[2] * (A[4] * A[9] - A[5] * A[8])))
      const { range, presets } = volumePresets(meta, voxel || 1)
      this.presets3d = presets
      performance.mark('nni-3d-volume-start')
      this.r3.setVolume({ ...nii, lo: range[0], hi: range[1] })
      performance.measure('nni-3d-volume', 'nni-3d-volume-start')
    }

    this.windows = meta.is_ct
      ? CT_WINDOWS.map(([n, lo, hi]) => ({ name: n, lo, hi }))
      : [{ name: 'Auto', lo: meta.auto_window[0], hi: meta.auto_window[1] }]
    this.windows.push({ name: 'Full range', lo: meta.range[0], hi: meta.range[1] })
    this.setWindow(0)
    this.volumePresets = Object.keys(this.presets3d)
    this.setVolume3d(this.volumePresets[0])
    this.setLayout(meta.is_2d ? 'single' : this.layout === 'single' ? 'grid' : this.layout)
    this.reset3d()
    this.ready = true
    const q = this.queue
    this.queue = []
    for (const job of q) await job()
    return true
  }

  async _addLabelLayers(objDtype) {
    const tpl = (dt) => `${this.meta.url.replace(/image\.nii\.gz$/, '')}zeros_${dt}.nii.gz`
    const keep = { current: this.vol.current?.img, prompts: this.vol.prompts?.img }
    for (const v of [this.vol.objects, this.vol.current, this.vol.prompts]) if (v) this.nv2.removeVolume(v)
    for (const [name, dt] of [['objects', objDtype], ['current', 'uint8'], ['prompts', 'uint8']]) {
      const buf = await (await fetch(tpl(dt))).arrayBuffer()
      await this.nv2.loadFromArrayBuffer(buf, `${name}.nii.gz`)
      const v = this.nv2.volumes[this.nv2.volumes.length - 1]
      v.name = name
      this.vol[name] = v
    }
    for (const name of ['current', 'prompts']) if (keep[name]) this.vol[name].img.set(keep[name])
    this.vol.current.setColormapLabel(CURRENT_LUT)
    this.vol.prompts.setColormapLabel(PROMPT_LUT)
    this.objDtype = objDtype
    this._applyObjectLut()
    this._applyOverlayMode()
  }

  // ---------------------------------------------------------------- patches
  applyPatch(buf) {
    const job = () => this._applyPatch(buf)
    if (!this.ready) this.queue.push(job)
    else job()
  }

  async _applyPatch(buf) {
    const dv = new DataView(buf)
    const hlen = dv.getUint32(0, true)
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)))
    const raw = await inflate(buf.slice(4 + hlen))
    if (!this.meta || header.shape.join() !== this.meta.shape.join()) return
    if (header.layer === 'objects' && header.dtype !== this.objDtype) {
      await this._addLabelLayers(header.dtype)
    }
    const vol = this.vol[header.layer]
    if (!vol) return
    const data = header.dtype === 'uint16' ? new Uint16Array(raw) : new Uint8Array(raw)
    const [n0, n1, n2] = header.shape
    const [[s0, e0], [s1, e1], [s2, e2]] = header.box
    const img = vol.img
    if (header.full) {
      img.set(data)
    } else {
      const w = e2 - s2
      let k = 0
      for (let a0 = s0; a0 < e0; a0++) {
        for (let a1 = s1; a1 < e1; a1++) {
          img.set(data.subarray(k, k + w), (a0 * n1 + a1) * n2 + s2)
          k += w
        }
      }
    }
    void n0
    this._scheduleGL()
  }

  _scheduleGL() {
    this.overlaysDirty = true
    if (this._glPending) return
    this._glPending = true
    setTimeout(() => {
      this._glPending = false
      this._flushOverlays()
    }, 120)
  }

  get interacting() {
    return this.pointerDown2d || !!this.drag3d || Date.now() < this.wheelUntil
  }

  _flushOverlays() {
    if (!this.overlaysDirty || this.interacting || !this.ready) return // _endInteraction flushes
    this.overlaysDirty = false
    performance.mark('nni-overlays-start')
    this._refreshOverlays()
    performance.measure('nni-overlays', 'nni-overlays-start')
    this._readoutFromCrosshair()
  }

  // The label layers only: NiiVue's updateGLVolume would also re-upload the image volume.
  // All overlays are redone because NiiVue composites them into one texture in order.
  _refreshOverlays() {
    const nv = this.nv2
    nv.refreshColormaps()
    for (let i = 1; i < nv.volumes.length; i++) if (nv.volumes[i].toRAS) nv.refreshLayers(nv.volumes[i], i)
    nv.drawScene()
  }

  _deferOr(job) {
    if (this.interacting) this.deferred.push(job)
    else job()
  }

  _endInteraction() {
    if (this.interacting) return
    const jobs = this.deferred
    this.deferred = []
    for (const job of jobs) job()
    this._flushOverlays()
  }

  // ---------------------------------------------------------------- objects
  setObjects(objects) {
    this.objects = objects
    if (this.ready) {
      this._applyObjectLut()
      this._scheduleGL()
    }
  }

  _applyObjectLut() {
    if (!this.vol.objects) return
    const lut = { R: [0], G: [0], B: [0], A: [0], I: [0], labels: [''] }
    for (const o of this.objects) {
      const [r, g, b] = hexRgb(o.color)
      lut.R.push(r); lut.G.push(g); lut.B.push(b); lut.I.push(o.id)
      lut.A.push(this.hidden.has(o.id) ? 0 : 255)
      lut.labels.push(o.name)
    }
    if (lut.I.length === 1) { lut.R.push(0); lut.G.push(0); lut.B.push(0); lut.A.push(0); lut.I.push(1); lut.labels.push('') }
    this.vol.objects.setColormapLabel(lut)
  }

  setHidden(id, hidden) {
    if (hidden) this.hidden.add(id)
    else this.hidden.delete(id)
    this._applyObjectLut()
    this._scheduleGL()
    this.r3.setMeshVisible(`o${id}`, !hidden)
  }

  setOverlayMode(mode) {
    this.overlayMode = mode
    if (this.ready) {
      this._applyOverlayMode()
      this._scheduleGL()
    }
  }

  _applyOverlayMode() {
    const show = this.overlayMode !== 'off'
    this.nv2.overlayOutlineWidth = this.overlayMode === 'outline' ? 2 : 0
    const outline = this.overlayMode === 'outline'
    if (this.vol.objects) this.vol.objects.opacity = show ? (outline ? 1 : FILL_OPACITY) : 0
    if (this.vol.current) this.vol.current.opacity = show ? (outline ? 1 : CURRENT_OPACITY) : 0
    if (this.vol.prompts) this.vol.prompts.opacity = show ? 1 : 0
  }

  // ----------------------------------------------------------------- meshes
  async setMesh(msg) {
    const token = this.loadToken
    const job = async () => {
      const resp = await fetch(msg.url)
      if (resp.status !== 200 || token !== this.loadToken) return
      const buf = await resp.arrayBuffer()
      const [nv, nf] = new Uint32Array(buf, 0, 2)
      const pts = new Float32Array(buf.slice(8, 8 + nv * 12))
      const tris = new Uint32Array(buf.slice(8 + nv * 12, 8 + nv * 12 + nf * 12))
      const prev = this.meshes.get(msg.key)
      if (prev && prev.v > msg.v) return
      const visible = !(msg.key.startsWith('o') && this.hidden.has(Number(msg.key.slice(1))))
      this.r3.setMesh(msg.key, pts, tris, hexRgb(msg.color), 1, visible)
      let cx = 0, cy = 0, cz = 0
      const step = Math.max(1, Math.floor(nv / 2000))
      let n = 0
      for (let i = 0; i < nv; i += step) { cx += pts[3 * i]; cy += pts[3 * i + 1]; cz += pts[3 * i + 2]; n++ }
      this.meshes.set(msg.key, { v: msg.v, center: [cx / n, cy / n, cz / n] })
    }
    if (!this.ready) this.queue.push(job)
    else this._deferOr(job)
  }

  removeMesh(key) {
    this._deferOr(() => {
      this.r3.removeMesh(key)
      this.meshes.delete(key)
    })
  }

  setSurfaceOpacity(v) {
    this.surfaceOpacity = v
    this.r3.surfaceOpacity = v
    this.r3.draw()
  }

  // ------------------------------------------------------------ display
  windowNames() { return (this.windows || []).map((w) => w.name) }

  setWindow(i) {
    const w = this.windows?.[i]
    if (!w || !this.vol.image) return
    this._bakeWindow(w.lo, w.hi)
    this.cb.onWindowChange?.({ preset: i, lo: w.lo, hi: w.hi })
  }

  setVolume3d(name) {
    if (!(name in (this.presets3d || {}))) return
    this.volume3d = name
    this.r3.setPreset(this.presets3d[name])
  }

  reset3d() {
    this.r3.resetView()
  }

  /** Back to the unzoomed, centred slice views, the default window and the default 3D view. */
  resetView() {
    this.nv2.scene.pan2Dxyzmm = [0, 0, 0, 1]
    this.nv2.opts.crosshairWidth = 1
    this.setWindow(0)
    this.reset3d()
  }

  // ------------------------------------------------------- mouse and keys
  // Left button: the selected tool (navigate, window, scroll, pan, zoom). Always available: wheel =
  // slices, Ctrl/Cmd + wheel or pinch = zoom, right-drag = window/level, middle-drag or Shift +
  // drag = pan, double-click = enlarge the view. A click without dragging always moves the
  // crosshair. Window/level, slice scrolling and zoom are ours; NiiVue does crosshair and pan.
  _installInteraction() {
    const pane2 = this.els.canvas2d.parentElement
    this._applyMouseConfig()
    pane2.addEventListener('mousedown', (e) => this._onDown(e), { capture: true })
    window.addEventListener('mousemove', (e) => this._onMove(e), { capture: true })
    window.addEventListener('mouseup', (e) => this._onUp(e), { capture: true })
    // Any button held in the slice views counts as interacting (also NiiVue's own crosshair and
    // pan drags), as does scrolling until 400 ms after the last wheel event.
    pane2.addEventListener('mousedown', () => { this.pointerDown2d = true }, { capture: true })
    pane2.addEventListener('wheel', () => {
      this.wheelUntil = Date.now() + 400
      clearTimeout(this._wheelTimer)
      this._wheelTimer = setTimeout(() => this._endInteraction(), 450)
    }, { capture: true, passive: true })
    pane2.addEventListener('contextmenu', (e) => e.preventDefault())
    pane2.addEventListener('mousemove', (e) => { this.mouse = this._canvasXY(e) })
    pane2.addEventListener('mouseleave', () => { this.mouse = null })
    // Capture phase: runs before NiiVue's handlers (its wheel scrolls slices, its double-click
    // resets brightness/contrast, which we do not want).
    pane2.addEventListener('wheel', (e) => this._onWheel(e), { passive: false, capture: true })
    this.els.canvas3d.parentElement.addEventListener('wheel', (e) => this._onWheel3d(e), { passive: false, capture: true })
    // 3D view (render3d.js): drag rotates (about 300 degrees per view width) or pans.
    const pane3 = this.els.canvas3d.parentElement
    pane3.addEventListener('mousedown', (e) => this._onDown3d(e), { capture: true })
    pane3.addEventListener('contextmenu', (e) => e.preventDefault())
    this.els.canvas3d.style.cursor = 'grab'
    this.drag3d = null
    this.els.viewport.addEventListener('dblclick', (e) => { e.stopPropagation(); this._onDblClick(e) }, { capture: true })
    this._updateCursor()
  }

  setTool(tool) {
    this.tool = tool
    if (!this.nv2) return // applied in _installInteraction
    this._applyMouseConfig()
    this._updateCursor()
  }

  _applyMouseConfig() {
    // (Set after construction: NiiVue's constructor ignores options whose default is undefined.)
    this.nv2.setMouseEventConfig({
      leftButton: { primary: this.tool === 'pan' ? DRAG_MODE.pan : DRAG_MODE.crosshair, withShift: DRAG_MODE.pan },
      rightButton: DRAG_MODE.none, // window/level, handled in _onDown
      centerButton: DRAG_MODE.pan,
    })
  }

  _updateCursor() {
    const cursors = { navigate: 'crosshair', window: WINDOW_CURSOR, scroll: 'ns-resize', pan: 'grab', zoom: 'zoom-in' }
    this.els.canvas2d.style.cursor = this.drag?.kind === 'pan' ? 'grabbing' : cursors[this.tool] || 'default'
  }

  _canvasXY(e) {
    const r = this.els.canvas2d.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  _tileAt(x, y) {
    const dpr = this.nv2.uiData?.dpr || window.devicePixelRatio || 1
    const i = this.nv2.tileIndex(x * dpr, y * dpr)
    return i >= 0 ? this.nv2.screenSlices[i] : null
  }

  _onDown(e) {
    if (!this.ready) return
    this.lastUser = Date.now()
    const [x, y] = this._canvasXY(e)
    let kind = null
    if (e.button === 2) kind = 'window'
    else if (e.button === 0 && !e.shiftKey && ['window', 'scroll', 'zoom'].includes(this.tool)) kind = this.tool
    else if (e.button === 0 && !e.shiftKey && this.tool === 'pan') kind = 'pan' // NiiVue pans; we only detect clicks
    if (!kind) return
    if (kind !== 'pan') {
      e.preventDefault()
      e.stopPropagation() // NiiVue does not see this drag
    }
    const v = this.vol.image
    this.drag = { kind, x0: x, y0: y, last: [x, y], moved: false, acc: 0, width0: v.cal_max - v.cal_min }
    this._updateCursor()
  }

  // Left drag rotates (a drag across the whole 3D view turns it by about 300 degrees); right,
  // middle or Shift + left drag pans.
  _onDown3d(e) {
    if (!this.ready) return
    e.preventDefault()
    e.stopPropagation()
    const pan = e.button === 1 || e.button === 2 || (e.button === 0 && e.shiftKey)
    if (!pan && e.button !== 0) return
    const w = this.els.canvas3d.getBoundingClientRect().width || 1
    this.drag3d = { pan, x0: e.clientX, y0: e.clientY, last: [e.clientX, e.clientY], az0: this.r3.azimuth, el0: this.r3.elevation,
      k: Math.min(0.8, Math.max(0.25, 300 / w)) }  // fmt: skip
    this.els.canvas3d.style.cursor = pan ? 'move' : 'grabbing'
  }

  _onMove(e) {
    if (this.drag3d || (this.drag && this.drag.kind !== 'pan')) e.stopPropagation() // NiiVue: hands off
    const r = this.drag3d
    if (r) {
      if (r.pan) {
        this.r3.panBy(e.clientX - r.last[0], e.clientY - r.last[1])
        r.last = [e.clientX, e.clientY]
        return
      }
      r.az = r.az0 + (e.clientX - r.x0) * r.k
      r.el = Math.min(90, Math.max(-90, r.el0 + (e.clientY - r.y0) * r.k))
      this.r3.setView(r.az, r.el) // drawn at the next animation frame
      this.els.pane3d.dataset.view = `${Math.round(r.az)},${Math.round(r.el)}` // for tests
      return
    }
    const d = this.drag
    if (!d) return
    const [x, y] = this._canvasXY(e)
    const dx = x - d.last[0], dy = y - d.last[1]
    d.last = [x, y]
    if (!d.moved && Math.hypot(x - d.x0, y - d.y0) < 3) return
    d.moved = true
    this.lastUser = Date.now()
    if (d.kind === 'window') this._windowBy(dx, dy, d.width0)
    else if (d.kind === 'zoom') this._zoomAt(d.x0, d.y0, Math.exp(-dy * 0.01))
    else if (d.kind === 'scroll') {
      const tile = this._tileAt(d.x0, d.y0)
      if (!tile) return
      const step = Math.min(8, Math.max(1.5, 400 / this._slicesAlong(tile.axCorSag)))
      d.acc += dy
      while (Math.abs(d.acc) >= step) {
        this.nv2.sliceScroll2D(d.acc > 0 ? 0.01 : -0.01, d.x0, d.y0)
        d.acc -= Math.sign(d.acc) * step
      }
    }
  }

  _onUp(e) {
    const ours = this.drag3d || (this.drag && this.drag.kind !== 'pan')
    if (ours) e.stopPropagation()
    this.pointerDown2d = false
    if (this.drag3d) {
      this.drag3d = null
      this.els.canvas3d.style.cursor = 'grab'
      this._endInteraction()
      return
    }
    // NiiVue finishes its own drags in its mouseup handler, which runs after this one.
    setTimeout(() => this._endInteraction(), 0)
    const d = this.drag
    if (!d) return
    this.drag = null
    this._updateCursor()
    if (!d.moved) {
      this.nv2.mouseClick(d.x0, d.y0) // a click moves the crosshair, whatever the tool
      this.nv2.drawScene()
      this._readoutFromCrosshair()
    } else if (d.kind === 'window') {
      this._finishWindow()
    }
  }

  // Drag up: brighter, right: less contrast (wider window). Speed relative to the window width at
  // drag start, so narrow windows stay fine-grained.
  //
  // Applying a window makes NiiVue re-upload and recolour the whole volume (100-300 ms for a
  // 512x512x227 CT), far too slow to follow the mouse. While dragging, the slices drawn with the
  // current ("baked") window are therefore remapped with an SVG filter: target gray =
  // a * baked gray + c, which is exact as long as the target window lies inside the baked one.
  // Nothing is sent to the GPU while the button is held; mouse-up bakes the final window and
  // removes the filter.
  _windowBy(dx, dy, width0) {
    const v = this.vol.image
    const t = this._wlTarget || { lo: v.cal_min, hi: v.cal_max }
    const g = Math.max(width0, 1e-6) / 250
    const c = (t.lo + t.hi) / 2 + dy * g
    const w = Math.max(g, t.hi - t.lo + 2 * dx * g)
    this._wlTarget = { lo: c - w / 2, hi: c + w / 2 }
    if (!this._wlFrame) this._wlFrame = requestAnimationFrame(() => this._previewWindow())
  }

  _bakeWindow(lo, hi) {
    const v = this.vol.image
    performance.mark('nni-window-bake-start')
    v.cal_min = lo
    v.cal_max = hi
    this.nv2.refreshLayers(v, 0)
    this.nv2.drawScene()
    performance.measure('nni-window-bake', 'nni-window-bake-start')
  }

  _previewWindow() {
    this._wlFrame = null
    const t = this._wlTarget
    const v = this.vol.image
    if (!t || !v) return
    performance.mark('nni-window-preview-start')
    // Preview only: no GPU work until the button is released. Intensities outside the window
    // the slices were drawn with stay saturated in the preview until then.
    const lo0 = v.cal_min, hi0 = v.cal_max
    const w = t.hi - t.lo
    const a = (hi0 - lo0) / w, c = (lo0 - t.lo) / w
    // 256-entry table; entry 0 stays 0 so the empty background (and voxels below the baked
    // window) stays black instead of turning gray.
    const table = Array.from({ length: 256 }, (_, k) => (k === 0 ? 0 : Math.min(1, Math.max(0, (a * k) / 255 + c))).toFixed(4)).join(' ')
    for (const f of document.querySelectorAll('#wlFilter feFuncR, #wlFilter feFuncG, #wlFilter feFuncB')) f.setAttribute('tableValues', table)
    this.els.canvas2d.style.filter = 'url(#wlFilter)'
    performance.measure('nni-window-preview', 'nni-window-preview-start')
    this.cb.onWindowChange?.({ preset: null, lo: t.lo, hi: t.hi, live: true })
  }

  _finishWindow() {
    if (this._wlFrame) cancelAnimationFrame(this._wlFrame)
    this._wlFrame = null
    const t = this._wlTarget
    this._wlTarget = null
    if (t) this._bakeWindow(t.lo, t.hi)
    this.els.canvas2d.style.filter = '' // same task as the bake: no frame shows both or neither
    const v = this.vol.image
    this.cb.onWindowChange?.({ preset: null, lo: v.cal_min, hi: v.cal_max })
  }

  _slicesAlong(axCorSag) {
    const letters = [['S', 'I'], ['A', 'P'], ['L', 'R']][axCorSag] || []
    const k = (this.meta.orientation || []).findIndex((o) => letters.includes(o))
    return this.meta.shape[k >= 0 ? k : 0]
  }

  /** One slice forward (+1) or back (-1) in the view under the pointer, else the large/axial view. */
  stepSlice(n) {
    if (!this.ready || this.meta.is_2d) return
    let xy = this.mouse && this._tileAt(...this.mouse) ? this.mouse : null
    if (!xy) {
      const t = { axial: 0, coronal: 1, sagittal: 2 }[this.layout] ?? 0
      const [l, tp, w, h] = LAYOUTS[this.layout][t] || LAYOUTS.grid[0]
      const r = this.els.canvas2d.getBoundingClientRect()
      xy = [(l + w / 2) * r.width, (tp + h / 2) * r.height]
    }
    this.lastUser = Date.now()
    for (let i = 0; i < Math.abs(n); i++) this.nv2.sliceScroll2D(n > 0 ? 0.01 : -0.01, xy[0], xy[1])
  }

  // In 3D every wheel event zooms, towards the pointer, in the same direction as the slice views.
  _onWheel3d(e) {
    if (!this.ready) return
    e.preventDefault()
    e.stopPropagation()
    const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY
    const r = this.els.canvas3d.getBoundingClientRect()
    this.r3.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-dy * 0.0025))
  }

  // Ctrl/Cmd + wheel (also what a trackpad pinch sends) zooms around the pointer; a plain wheel
  // keeps NiiVue's behaviour of changing the slice.
  _onWheel(e) {
    this.lastUser = Date.now()
    if (!(e.ctrlKey || e.metaKey) || !this.ready) return
    e.preventDefault()
    e.stopPropagation()
    const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY // lines -> pixels
    this._zoomAt(...this._canvasXY(e), Math.exp(-dy * 0.0025))
  }

  /** Zoom the slice views by ``factor``, keeping the point under (x, y) (CSS px) in place. */
  _zoomAt(x, y, factor) {
    const nv = this.nv2
    const dpr = nv.uiData?.dpr || window.devicePixelRatio || 1
    if (nv.tileIndex(x * dpr, y * dpr) < 0) return
    let mm = nv.screenXY2mm(x * dpr, y * dpr)
    if (!mm || !Number.isFinite(mm[0])) mm = nv.frac2mm(nv.scene.crosshairPos)
    const pan = nv.scene.pan2Dxyzmm
    const zoom = Math.min(30, Math.max(0.6, pan[3] * factor))
    const change = pan[3] - zoom
    nv.scene.pan2Dxyzmm = [pan[0] + change * mm[0], pan[1] + change * mm[1], pan[2] + change * mm[2], zoom]
    nv.opts.crosshairWidth = Math.min(1, 1 / zoom) // in voxels: keep the line thin on screen
    nv.drawScene()
  }

  setLayout(mode) {
    if (this.meta?.is_2d) mode = 'single'
    this.layout = mode
    const L = LAYOUTS[mode]
    const tiles = Object.entries(L).filter(([t]) => t !== '4').map(([t, pos]) => ({ sliceType: Number(t), position: pos }))
    this.nv2.setCustomLayout(tiles)
    const p3 = L[4]
    const pane = this.els.pane3d
    pane.hidden = !p3
    if (p3) {
      Object.assign(pane.style, { left: `${p3[0] * 100}%`, top: `${p3[1] * 100}%`, width: `${p3[2] * 100}%`, height: `${p3[3] * 100}%` })
    }
    this._renderTileLabels()
    requestAnimationFrame(() => this._resize())
  }

  _renderTileLabels() {
    const L = LAYOUTS[this.layout]
    this.els.tileLabels.innerHTML = ''
    for (const [t, pos] of Object.entries(L)) {
      const el = document.createElement('div')
      el.className = 'tile-label'
      el.textContent = TILE_NAMES[t]
      el.title = this.layout === 'grid' ? 'Enlarge (or double-click the view)' : 'Back to the grid'
      Object.assign(el.style, { left: `${pos[0] * 100}%`, top: `${pos[1] * 100}%` })
      el.addEventListener('click', () => this.cb.onLayoutRequest?.(this._toggleTarget(Number(t))))
      this.els.tileLabels.appendChild(el)
    }
  }

  _toggleTarget(t) {
    const name = { 0: 'axial', 1: 'coronal', 2: 'sagittal', 4: '3d' }[t]
    return this.layout === name ? 'grid' : name
  }

  _onDblClick(e) {
    if (this.meta?.is_2d) return
    const r = this.els.viewport.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height
    for (const [t, [l, tp, w, h]] of Object.entries(LAYOUTS[this.layout])) {
      if (x >= l && x < l + w && y >= tp && y < tp + h) {
        this.cb.onLayoutRequest?.(this._toggleTarget(Number(t)))
        return
      }
    }
  }

  _resize() {
    this.nv2?.resizeListener()
    this.r3?.resize()
  }

  // -------------------------------------------------------- crosshair / focus
  focusVoxel(p) {
    if (!this.ready || !this.vol.image) return
    this.nv2.scene.crosshairPos = workspaceFrac(this.vol.image, p.map((v) => Math.round(v)))
    this.nv2.drawScene()
    this._readoutFromCrosshair()
  }

  focusObject(id) {
    const m = this.meshes.get(`o${id}`)
    if (m && this.vol.image) {
      this.nv2.scene.crosshairPos = this.vol.image.convertMM2Frac(m.center, true) // slice mm, as the readout
      this.nv2.drawScene()
      this._readoutFromCrosshair()
      return
    }
    // no surface (2D image): centre of the label voxels
    const img = this.vol.objects?.img
    if (!img) return
    const [, n1, n2] = this.meta.shape
    let s0 = 0, s1 = 0, s2 = 0, n = 0
    for (let i = 0; i < img.length; i++) {
      if (img[i] === id) {
        const a2 = i % n2, r = (i - a2) / n2, a1 = r % n1, a0 = (r - a1) / n1
        s0 += a0; s1 += a1; s2 += a2; n++
      }
    }
    if (n) this.focusVoxel([s0 / n, s1 / n, s2 / n])
  }

  userIsBrowsing(ms = 12000) { return Date.now() - this.lastUser < ms }

  _readoutFromCrosshair() {
    if (this.ready) this.nv2.createOnLocationChange()
  }

  _onLocation(loc) {
    if (!this.meta || !loc.values?.length) return
    const p = workspaceIndex(this.vol.image, loc.values[0].vox)
    const value = loc.values[0].value
    const objIdx = this.nv2.volumes.indexOf(this.vol.objects)
    const id = objIdx >= 0 ? Math.round(loc.values[objIdx]?.value ?? 0) : 0
    const obj = this.objects.find((o) => o.id === id)
    this.cb.onLocation?.({ point: p, value, object: obj || null, inside: p.every((v, k) => v >= 0 && v < this.meta.shape[k]) })
  }
}
