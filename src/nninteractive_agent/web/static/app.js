import { hydrateIcons, icon } from './icons.js'
import { Viewer } from './viewer.js'
import { Chat } from './chat.js'

const $ = (id) => document.getElementById(id)
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const state = {
  image: null, cases: [], busy: false, ready: false, paused: false, status: '', model: 'opus', effort: 'high',
  replaying: false, connected: false, objects: [], current: null, location: null, t0: 0, userMsgs: 0, loading: false,
}

hydrateIcons()

// ------------------------------------------------------------------ toasts
function toast(text, isError = false) {
  const t = document.createElement('div')
  t.className = `toast${isError ? ' err' : ''}`
  t.textContent = text
  $('toasts').appendChild(t)
  setTimeout(() => t.remove(), isError ? 8000 : 4500)
}

// ------------------------------------------------------------------ viewer
const viewer = new Viewer(
  { viewport: $('viewport'), canvas2d: $('canvas2d'), canvas3d: $('canvas3d'), pane3d: $('pane3d'), tileLabels: $('tileLabels') },
  {
    onLocation: (loc) => {
      state.location = loc.inside ? loc.point : null
      $('mentionBtn').disabled = !state.location
      if (!loc.inside) { $('readout').textContent = 'Outside the image'; return }
      const [a0, a1, a2] = loc.point
      const unit = state.image?.is_ct ? ' HU' : ''
      const v = Number.isInteger(loc.value) ? loc.value : Number(loc.value).toPrecision(4)
      const obj = loc.object ? ` · #${loc.object.id} ${loc.object.name}` : ''
      $('readout').textContent = `axis0 ${a0} · axis1 ${a1} · axis2 ${a2} · ${v}${unit}${obj}`
    },
    onLayoutRequest: (mode) => setLayout(mode),
    onWindowChange: (w) => showWindow(w),
  },
)

// The window dropdown follows presets and shows the values after a window/level drag.
function showWindow({ preset, lo, hi }) {
  const sel = $('windowSel')
  let custom = sel.querySelector('option[value=custom]')
  if (preset !== null && preset !== undefined) {
    custom?.remove()
    sel.value = String(preset)
    return
  }
  if (!custom) {
    custom = document.createElement('option')
    custom.value = 'custom'
    sel.appendChild(custom)
  }
  const f = (v) => (Math.abs(v) >= 100 || Number.isInteger(v) ? Math.round(v) : Number(v.toPrecision(3)))
  custom.textContent = `Custom (W ${f(hi - lo)} / L ${f((hi + lo) / 2)})`
  sel.value = 'custom'
}

function setTool(tool) {
  viewer.setTool(tool)
  for (const b of $('toolSeg').children) b.classList.toggle('on', b.dataset.value === tool)
}
for (const b of $('toolSeg').children) b.addEventListener('click', () => setTool(b.dataset.value))
setTool('navigate')
// Without WebGL2 (hardware acceleration off, remote desktop, old browsers) the viewer cannot start;
// opening an image then says so instead of loading forever.
const NO_WEBGL = 'This browser cannot show the image: WebGL2 is not available. Turn on hardware acceleration '
  + 'in the browser settings, or use a current Chrome, Edge, Firefox or Safari (not through remote desktop).'
let viewerError = null
const viewerInit = viewer.init().catch((e) => { viewerError = e; console.error('The viewer could not start:', e) })
window.nniViewer = viewer // for the browser checks in tests/

function setLayout(mode) {
  viewer.setLayout(mode)
  for (const b of $('layoutSeg').children) b.classList.toggle('on', b.dataset.value === mode)
}
for (const b of $('layoutSeg').children) b.addEventListener('click', () => setLayout(b.dataset.value))

function showLoading(text, frac = null) {
  $('loading').hidden = false
  $('loading').classList.remove('error')
  $('loadingText').textContent = text
  $('loadingBar').style.width = frac === null ? '0%' : `${Math.round(frac * 100)}%`
}
const hideLoading = () => { $('loading').hidden = true }
function showLoadingError(text) {
  showLoading(text)
  $('loading').classList.add('error')
}

function showGallery(show) {
  $('gallery').hidden = !show
  $('backToImageBtn').hidden = !(show && state.image)
}

async function loadImage(meta) {
  state.image = meta
  state.userMsgs = 0
  state.loading = true
  showGallery(false)
  $('viewer').hidden = false
  updateImageChip()
  renderSuggestions()
  showLoading('Downloading the image ...', 0)
  try {
    await viewerInit
    // NiiVue: "Unable to initialize WebGL2 ..."; the 3D view: "WebGL2 is not available" or a shader error
    if (viewerError) {
      showLoadingError(/webgl2/i.test(String(viewerError.message)) ? NO_WEBGL : `The image viewer could not start: ${viewerError.message}`)
      return
    }
    const ok = await viewer.load(meta, (frac, got, total, text) => {
      const mb = (x) => (x / 1e6).toFixed(0)
      showLoading(text || (total ? `Downloading the image ... ${mb(got)} of ${mb(total)} MB` : 'Downloading the image ...'), frac)
    })
    if (!ok) return
  } catch (e) {
    showLoadingError(`Could not display the image: ${e.message}`)
    return
  } finally {
    state.loading = false
  }
  hideLoading()
  $('windowSel').innerHTML = viewer.windowNames().map((n, i) => `<option value="${i}">${n}</option>`).join('')
  $('volume3dSel').innerHTML = (viewer.volumePresets || []).map((n) => `<option>${n}</option>`).join('')
  for (const c of document.querySelectorAll('.ctl-3d')) c.hidden = meta.is_2d
  $('layoutSeg').hidden = meta.is_2d
  setLayout(viewer.layout)
  viewer.setObjects(state.objects)
  updateComposer()
}

$('windowSel').addEventListener('change', (e) => { if (e.target.value !== 'custom') viewer.setWindow(Number(e.target.value)) })
$('overlaySel').addEventListener('change', (e) => viewer.setOverlayMode(e.target.value))
$('volume3dSel').addEventListener('change', (e) => viewer.setVolume3d(e.target.value))
$('surfaceOpacity').addEventListener('input', (e) => viewer.setSurfaceOpacity(Number(e.target.value)))
$('resetViewBtn').addEventListener('click', () => viewer.resetView())
$('mentionBtn').addEventListener('click', () => {
  if (!state.location) return
  const [a0, a1, a2] = state.location
  insertAtCursor(`[location axis0=${a0}, axis1=${a1}, axis2=${a2}] `)
})

function updateImageChip() {
  const m = state.image
  $('imageChip').hidden = !m
  if (!m) return
  $('imageTitle').textContent = m.title
  const sp = m.spacing.map((s) => (s >= 10 ? s.toFixed(0) : s.toFixed(2).replace(/\.?0+$/, '')))
  $('imageMeta').textContent = `${m.shape.join(' × ')} · ${sp.join(' × ')}${m.orientation.some(Boolean) ? ' mm' : ''}`
}
$('changeImageBtn').addEventListener('click', () => showGallery(true))
$('backToImageBtn').addEventListener('click', () => showGallery(false))

// ----------------------------------------------------------------- gallery
function renderCards() {
  const cards = $('cards')
  cards.innerHTML = ''
  for (const c of state.cases) {
    const b = document.createElement('button')
    b.className = 'card'
    b.innerHTML = `<div class="thumb"></div><div class="body"><div class="title"><span></span><span class="badge"></span></div><div class="sub"></div></div>`
    b.querySelector('.thumb').style.backgroundImage = `url("${c.thumb}")`
    b.querySelector('.title span').textContent = c.title
    b.querySelector('.badge').textContent = c.modality
    b.querySelector('.sub').textContent = c.subtitle
    b.addEventListener('click', () => openCase(c, b))
    cards.appendChild(b)
  }
  const up = document.createElement('label')
  up.className = 'card upload-card'
  up.innerHTML = `<span class="big-icon">${icon('upload')}</span><div class="title">Upload an image</div>
    <div class="hint">NIfTI, NRRD, MHA, TIFF stack, PNG or JPG. Drop a file here or click to choose.</div>
    <input type="file" hidden accept=".nii,.gz,.nrrd,.mha,.mhd,.tif,.tiff,.png,.jpg,.jpeg">`
  const input = up.querySelector('input')
  input.addEventListener('change', () => input.files[0] && uploadFile(input.files[0]))
  up.addEventListener('dragover', (e) => { e.preventDefault(); up.classList.add('drag') })
  up.addEventListener('dragleave', () => up.classList.remove('drag'))
  up.addEventListener('drop', (e) => {
    e.preventDefault()
    up.classList.remove('drag')
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0])
  })
  cards.appendChild(up)
}

async function openCase(c, btn) {
  if (state.busy) { toast('The agent is still working. Stop it first.'); return }
  btn.disabled = true
  try {
    const r = await fetch('api/open', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case: c.id }) })
    if (!r.ok) toast((await r.json().catch(() => ({}))).detail || `Could not open the image (HTTP ${r.status})`, true)
  } finally {
    btn.disabled = false
  }
}

function uploadFile(file) {
  if (state.busy) { toast('The agent is still working. Stop it first.'); return }
  const xhr = new XMLHttpRequest()
  const form = new FormData()
  form.append('file', file)
  showGallery(false)
  $('viewer').hidden = false
  showLoading(`Uploading ${file.name} ...`, 0)
  xhr.upload.onprogress = (e) => e.lengthComputable && showLoading(`Uploading ${file.name} ... ${Math.round((100 * e.loaded) / e.total)}%`, e.loaded / e.total)
  xhr.onload = () => {
    if (xhr.status >= 300) {
      let msg = `Upload failed (HTTP ${xhr.status})`
      try { msg = JSON.parse(xhr.responseText).detail || msg } catch { /* not JSON */ }
      toast(msg, true)
      if (!state.image) { $('viewer').hidden = true; showGallery(true) }
      hideLoading()
    } else {
      showLoading('Preparing the image ...', 1)
    }
  }
  xhr.onerror = () => { toast('Upload failed', true); hideLoading() }
  xhr.open('POST', 'api/upload')
  xhr.send(form)
}

// ----------------------------------------------------------------- objects
function fmtVolume(o) {
  if (o.ml === null || o.ml === undefined) return `${o.voxels.toLocaleString()} vox`
  if (o.ml >= 1000) return `${(o.ml / 1000).toFixed(2)} L`
  if (o.ml >= 10) return `${o.ml.toFixed(0)} mL`
  return `${o.ml.toFixed(o.ml >= 1 ? 1 : 2)} mL`
}

function renderObjects() {
  const list = $('objects')
  list.innerHTML = ''
  const cur = state.current
  if (cur) {
    const row = document.createElement('div')
    row.className = 'obj current'
    row.innerHTML = `<span class="sw"></span><span class="name"></span><span class="vol"></span><span></span><span></span>`
    row.querySelector('.name').textContent = `Working on: ${cur.name}`
    row.querySelector('.vol').textContent = `${cur.prompts} prompt${cur.prompts === 1 ? '' : 's'}`
    list.appendChild(row)
  }
  for (const o of state.objects) {
    const hidden = viewer.hidden.has(o.id)
    const row = document.createElement('div')
    row.className = `obj${hidden ? ' hidden-obj' : ''}`
    row.innerHTML = `<span class="sw" title="Show or hide"></span><span class="name"><span class="id"></span><span class="nm"></span><span class="note"></span></span>
      <span class="vol"></span><button class="icon-btn mini" title="Show or hide">${icon(hidden ? 'eyeOff' : 'eye')}</button>
      <button class="icon-btn mini" title="Jump to this object">${icon('target')}</button>`
    row.querySelector('.sw').style.background = o.color
    row.querySelector('.id').textContent = `#${o.id}`
    row.querySelector('.nm').textContent = o.name
    row.querySelector('.note').textContent = o.notes || ''
    row.querySelector('.note').hidden = !o.notes
    row.title = o.notes ? `${o.name}: ${o.notes}` : o.name
    row.querySelector('.vol').textContent = fmtVolume(o)
    const toggle = () => { viewer.setHidden(o.id, !viewer.hidden.has(o.id)); renderObjects() }
    row.querySelector('.sw').addEventListener('click', toggle)
    const [eye, go] = row.querySelectorAll('button')
    eye.addEventListener('click', toggle)
    go.addEventListener('click', () => viewer.focusObject(o.id))
    list.appendChild(row)
  }
  if (!cur && !state.objects.length) list.innerHTML = '<div class="empty">No objects yet.</div>'
  $('objCount').textContent = state.objects.length
  $('downloadBtn').disabled = !state.objects.length
}
$('downloadBtn').addEventListener('click', () => { location.href = 'api/download' })

// -------------------------------------------------------------------- chat
const chat = new Chat($('chat'), { onImageClick: (i) => openLightbox(i) })
let lbIndex = 0
function openLightbox(i) {
  lbIndex = Math.max(0, Math.min(i, chat.images.length - 1))
  const im = chat.images[lbIndex]
  if (!im) return
  $('lbImg').src = im.url
  $('lbCap').textContent = `${im.caption}  (${lbIndex + 1} of ${chat.images.length})`
  $('lightbox').hidden = false
}
$('lbClose').addEventListener('click', () => { $('lightbox').hidden = true })
$('lbPrev').addEventListener('click', () => openLightbox(lbIndex - 1))
$('lbNext').addEventListener('click', () => openLightbox(lbIndex + 1))
$('lightbox').addEventListener('click', (e) => { if (e.target.id === 'lightbox') $('lightbox').hidden = true })

function renderSuggestions() {
  const box = $('suggestions')
  box.innerHTML = ''
  if (!state.image || state.userMsgs > 0 || state.busy) return
  for (const t of state.image.tasks || []) {
    const b = document.createElement('button')
    b.className = 'suggestion'
    b.textContent = t
    b.addEventListener('click', () => { $('input').value = t; autosize(); $('input').focus() })
    box.appendChild(b)
  }
}

// ---------------------------------------------------------------- composer
const input = $('input')
function autosize() {
  input.style.height = 'auto'
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`
  updateComposer()
}
input.addEventListener('input', autosize)
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    submit()
  }
})
function insertAtCursor(text) {
  const s = input.selectionStart ?? input.value.length, e = input.selectionEnd ?? s
  input.value = input.value.slice(0, s) + text + input.value.slice(e)
  input.selectionStart = input.selectionEnd = s + text.length
  input.focus()
  autosize()
}
function submit() {
  if (state.busy) return
  const text = input.value.trim()
  if (!text || !state.image) return
  send({ cmd: 'send', text })
  input.value = ''
  autosize()
}
$('sendBtn').addEventListener('click', () => (state.busy ? send({ cmd: 'stop' }) : submit()))
$('newConvBtn').addEventListener('click', () => send({ cmd: 'new_conversation' }))

function updateComposer() {
  const can = state.connected && !!state.image
  input.disabled = !can
  input.placeholder = !state.image ? 'Open an image first' : state.busy ? 'The agent is working. You can type your next message.' : 'Describe what to segment ...'
  if (state.busy) input.disabled = false
  const btn = $('sendBtn')
  btn.classList.toggle('stop', state.busy)
  btn.innerHTML = icon(state.busy ? 'stop' : 'send')
  btn.title = state.busy ? 'Stop the agent' : 'Send (Enter)'
  btn.disabled = !state.connected || (!state.busy && (!can || !input.value.trim()))
  $('newConvBtn').disabled = state.busy || !state.connected
  for (const b of document.querySelectorAll('#modelSeg button, #modelSeg select, #effortSeg button')) b.disabled = state.busy
}

// ---------------------------------------------------------------- status
let timer = null
function renderStatus() {
  const line = $('statusline')
  const show = state.busy || (state.image && !state.ready && !state.paused)
  line.hidden = !show
  $('statusText').textContent = state.status
  if (state.busy && !timer) {
    timer = setInterval(() => {
      const s = Math.max(0, Math.round(Date.now() / 1000 - state.t0))
      $('elapsed').textContent = state.t0 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : ''
    }, 1000)
  } else if (!state.busy && timer) {
    clearInterval(timer)
    timer = null
    $('elapsed').textContent = ''
  }
  const dot = $('serverDot')
  dot.className = `dot ${!state.connected ? 'err' : state.ready ? (state.busy ? 'busy' : 'ok') : state.image && !state.paused ? 'busy' : ''}`
  // The nnInteractive backend and the license of its model weights, e.g. "local GPU · CC BY-NC-SA 4.0"
  if (state.server) $('serverLabel').textContent = state.license ? `${state.server} · ${state.license}` : state.server
  $('serverChip').title = (!state.connected ? 'Disconnected from the demo server' : state.ready ? 'nnInteractive session active'
    : state.paused ? 'Paused: the nnInteractive session was released' : 'No nnInteractive session yet')
    + (state.license ? `\nnnInteractive model license: ${state.license}` : '')
}

// ---------------------------------------------------------------- settings
for (const [btn, pop] of [['settingsBtn', 'settingsPop'], ['controlsBtn', 'controlsPop']]) {
  $(btn).addEventListener('click', (e) => {
    e.stopPropagation()
    const open = $(pop).hidden
    for (const p of ['settingsPop', 'controlsPop']) $(p).hidden = true
    $(pop).hidden = !open
  })
}
document.addEventListener('click', (e) => {
  for (const p of ['settingsPop', 'controlsPop']) if (!$(p).contains(e.target)) $(p).hidden = true
})
// The server lists its LLM's models and efforts; a server without the list runs Claude.
const EFFORT_LABELS = { low: 'Low', medium: 'Medium', high: 'High', xhigh: 'X-High', max: 'Max' }
const CLAUDE_EFFORTS = ['low', 'medium', 'high', 'xhigh']
const DEFAULT_AGENT = { label: '', models: [{ id: 'opus', label: 'Opus', efforts: CLAUDE_EFFORTS }, { id: 'sonnet', label: 'Sonnet', efforts: CLAUDE_EFFORTS }] }
function setting(key, value) { if (value && state[key] !== value) send({ cmd: 'settings', [key]: value }) }
for (const [seg, key] of [['modelSeg', 'model'], ['effortSeg', 'effort']]) {
  $(seg).addEventListener('click', (e) => { const b = e.target.closest('button[data-value]'); if (b) setting(key, b.dataset.value) })
}
$('modelSeg').addEventListener('change', (e) => { if (e.target.matches('select')) setting('model', e.target.value) })
$('agentChip').addEventListener('click', (e) => {
  e.stopPropagation()
  $('controlsPop').hidden = true
  $('settingsPop').hidden = !$('settingsPop').hidden
})
function renderSettings() {
  const agent = state.agent || DEFAULT_AGENT
  $('agentLabel').textContent = agent.label ? `Runs on ${agent.label}` : ''
  $('agentLabel').hidden = !agent.label
  const model = agent.models.find((m) => m.id === state.model) || agent.models[0]
  // The header says which LLM the agent runs on: "Claude Opus · High", "GPT-6.1-Sol · High", "Qwen3.8-27B · Medium"
  const effort = EFFORT_LABELS[state.effort] || state.effort
  const modelName = `${(agent.name || 'claude').startsWith('claude') ? 'Claude ' : ''}${model?.label || state.model}`
  $('agentChipLabel').textContent = `${modelName} · ${effort}`
  $('agentChip').title = `The agent runs on ${agent.label || 'Claude'}: model ${model?.label || state.model}, effort ${effort}. Click to change.`
  const seg = (items, on) => items.map(([v, label]) => `<button data-value="${esc(v)}"${v === on ? ' class="on"' : ''}${state.busy ? ' disabled' : ''}>${esc(label)}</button>`).join('')
  const models = agent.models.map((m) => [m.id, m.label])
  $('modelSeg').classList.toggle('seg', models.length <= 3)
  $('modelSeg').innerHTML = models.length <= 3 ? seg(models, model?.id)
    : `<select${state.busy ? ' disabled' : ''}>${models.map(([v, label]) => `<option value="${esc(v)}"${v === model?.id ? ' selected' : ''}>${esc(label)}</option>`).join('')}</select>`
  $('effortSeg').innerHTML = seg((model?.efforts || []).map((e) => [e, EFFORT_LABELS[e] || e]), state.effort)
}

// ---------------------------------------------------------------- splitter
{
  const sp = $('splitter')
  sp.addEventListener('pointerdown', (e) => {
    sp.setPointerCapture(e.pointerId)
    sp.classList.add('drag')
    const move = (ev) => {
      const w = Math.max(340, Math.min(window.innerWidth * 0.6, window.innerWidth - ev.clientX))
      document.documentElement.style.setProperty('--side-w', `${w}px`)
    }
    const up = () => { sp.classList.remove('drag'); sp.removeEventListener('pointermove', move); sp.removeEventListener('pointerup', up) }
    sp.addEventListener('pointermove', move)
    sp.addEventListener('pointerup', up)
  })
}

// ---------------------------------------------------------------- keyboard
document.addEventListener('keydown', (e) => {
  if (!$('lightbox').hidden) {
    if (e.key === 'Escape') $('lightbox').hidden = true
    if (e.key === 'ArrowLeft') openLightbox(lbIndex - 1)
    if (e.key === 'ArrowRight') openLightbox(lbIndex + 1)
    return
  }
  if (e.target.closest('textarea, input, select') || e.ctrlKey || e.metaKey || e.altKey || !state.image) return
  const mode = { 1: 'grid', 2: 'axial', 3: 'coronal', 4: 'sagittal', 5: '3d' }[e.key]
  if (mode && !state.image.is_2d) setLayout(mode)
  const tool = { c: 'navigate', escape: 'navigate', w: 'window', s: 'scroll', p: 'pan', z: 'zoom' }[e.key.toLowerCase()]
  if (tool) setTool(tool)
  const step = { ArrowUp: -1, ArrowDown: 1, PageUp: -10, PageDown: 10 }[e.key]
  if (step) { e.preventDefault(); viewer.stepSlice(step) }
  if (e.key.toLowerCase() === 'r') viewer.resetView()
})

// --------------------------------------------------------------- websocket
let sock = null
let retry = 0
function send(cmd) {
  if (sock?.readyState === WebSocket.OPEN) sock.send(JSON.stringify(cmd))
  else toast('Not connected to the demo server', true)
}
function connect() {
  const url = new URL('api/ws', location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  sock = new WebSocket(url)
  sock.binaryType = 'arraybuffer'
  sock.onopen = () => { retry = 0; state.connected = true; renderStatus(); updateComposer() }
  sock.onmessage = (m) => (typeof m.data === 'string' ? onEvent(JSON.parse(m.data)) : viewer.applyPatch(m.data))
  sock.onclose = (e) => {
    state.connected = false
    renderStatus()
    updateComposer()
    if (e.code === 4401) { location.href = 'login'; return }
    setTimeout(connect, Math.min(10000, 600 * 2 ** retry++))
  }
}

function onEvent(ev) {
  switch (ev.t) {
    case 'hello':
      state.cases = ev.cases || []
      state.agent = ev.agent || null
      state.server = ev.server
      state.license = ev.license || null
      renderStatus()
      renderSettings()
      renderCards()
      chat.reset()
      state.replaying = true
      state.userMsgs = 0
      if (ev.image && (!state.image || state.image.key !== ev.image.key || state.image.n !== ev.image.n)) loadImage(ev.image)
      else if (!ev.image) { state.image = null; $('viewer').hidden = true; showGallery(true); updateImageChip() }
      break
    case 'replay_done':
      state.replaying = false
      renderSuggestions()
      break
    case 'state':
      Object.assign(state, { busy: ev.busy, ready: ev.ready, paused: ev.paused, status: ev.status, model: ev.model, effort: ev.effort })
      if (ev.license) state.license = ev.license
      renderStatus()
      renderSettings()
      updateComposer()
      renderSuggestions()
      break
    case 'image':
      chat.add(ev)
      state.objects = []
      state.current = null
      renderObjects()
      if (!state.replaying) loadImage(ev.image)
      break
    case 'user':
      state.userMsgs++
      if (!state.replaying) state.t0 = ev.ts
      chat.add(ev)
      renderSuggestions()
      break
    case 'objects':
      state.objects = ev.objects
      state.current = ev.current
      renderObjects()
      viewer.setObjects(ev.objects)
      break
    case 'mesh':
      if (ev.remove) viewer.removeMesh(ev.key)
      else viewer.setMesh(ev)
      break
    case 'focus':
      if ($('followToggle').checked) {
        if (viewer.userIsBrowsing()) {
          const t = $('followToggle').parentElement
          t.classList.add('paused')
          t.title = 'Paused while you browse the image'
          clearTimeout(t._timer)
          t._timer = setTimeout(() => { t.classList.remove('paused'); t.title = 'Move the slices to where the agent is working' }, 4000)
        } else {
          viewer.focusVoxel(ev.point)
        }
      }
      break
    case 'notice':
      toast(ev.text)
      break
    default:
      chat.add(ev)
  }
}

renderObjects()
renderStatus()
updateComposer()
connect()
