// The conversation: user messages, agent text, the agent's tool steps (with the images it looked
// at), verifier subagents and per-request summaries.
import { marked } from './vendor/marked-18.0.13.esm.js'
import DOMPurify from './vendor/purify-3.4.15.es.mjs'
import { icon } from './icons.js'

const TOOL_ICONS = {
  image_info: 'image', view: 'eye', montage: 'layers', projection: 'layers', stats: 'bars', list_objects: 'bars',
  new_object: 'plus', bbox: 'square', scribble: 'pen', point: 'dot', undo: 'undo', commit: 'check', discard: 'trash',
  delete_object: 'trash', edit_object: 'pen', save: 'save', python: 'code', Read: 'file',
}

const fmt = (p) => (Array.isArray(p) ? `[${p.map((v) => Math.round(v)).join(', ')}]` : '')
const ref = (o) => (o === undefined || o === null ? 'the open object' : typeof o === 'number' ? `#${o}` : `“${o}”`)

export function stepLabel(tool, a = {}) {
  if (a.preview && ['bbox', 'scribble', 'point'].includes(tool)) return `Check before sending: ${stepLabel(tool, { ...a, preview: false })}`
  switch (tool) {
    case 'image_info': return 'Image overview'
    case 'view': return a.center ? `View slices at ${fmt(a.center)}` : `View ${ref(a.object)}`
    case 'montage': return `Montage along axis${a.axis ?? 0}${a.object !== undefined ? ` of ${ref(a.object)}` : ''}`
    case 'projection': return `${(a.mode || 'max').replace(/^./, (c) => c.toUpperCase())} projection along axis${a.axis ?? 0}`
    case 'stats': return `Statistics of ${ref(a.object)}`
    case 'new_object': return `New object “${a.name ?? ''}”`
    case 'bbox': return `Box ${fmt(a.corner1)} to ${fmt(a.corner2)}`
    case 'scribble': return `${a.positive === false ? 'Negative' : 'Positive'} scribble (${(a.points || []).length} points)`
    case 'point': return `${a.positive === false ? 'Negative' : 'Positive'} point at ${fmt(a.position)}`
    case 'undo': return 'Undo the last prompt'
    case 'commit': return 'Commit the object'
    case 'discard': return 'Discard the object'
    case 'edit_object': return `Re-open ${ref(a.object)}`
    case 'delete_object': return `Delete ${ref(a.object)}`
    case 'list_objects': return 'List the objects'
    case 'save': return 'Save the result'
    case 'python': return 'Python analysis'
    case 'Read': return `Read ${String(a.file_path || '').split('/').pop()}`
    default: return tool
  }
}

function resultSummary(text, isError) {
  if (!text) return ''
  if (isError) return text.slice(0, 500)
  const lines = text.split('\n')
  const keep = lines.slice(1).filter((l) => /^(change:|WARNING|NOTE|warning|OVERLAPS|committed|clean-up)/.test(l)).slice(0, 3)
  return [lines[0], ...keep].join('\n').slice(0, 400)
}

function md(text) {
  return DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: false }))
}

function el(tag, cls, html) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (html !== undefined) e.innerHTML = html
  return e
}

function duration(s) {
  if (s < 60) return `${Math.round(s)} s`
  const m = Math.floor(s / 60)
  return `${m} min ${Math.round(s - 60 * m)} s`
}

export class Chat {
  constructor(root, { onImageClick } = {}) {
    this.root = root
    this.onImageClick = onImageClick
    // Follow new content unless the user scrolled up to read; scrolling back down re-enables it.
    this.follow = true
    root.addEventListener('scroll', () => {
      this.follow = root.scrollHeight - root.scrollTop - root.clientHeight < 48
    }, { passive: true })
    new MutationObserver(() => this._stick()).observe(root, { childList: true, subtree: true })
    this.reset()
  }

  /** Scroll to the newest content (e.g. after the user sent a message). */
  scrollToEnd() {
    this.follow = true
    this._stick()
  }

  reset() {
    this.root.querySelectorAll(':scope > :not(.chat-empty)').forEach((n) => n.remove())
    this.empty = this.root.querySelector('.chat-empty')
    this.group = null
    this.steps = new Map() // call id -> step
    this.cards = new Map() // Agent tool_use id -> verifier card
    this.pendingByTool = new Map() // tool -> [step] without result (fallback matching)
    this.images = []
    this.count = 0
    this._showEmpty()
  }

  _showEmpty() { this.empty.hidden = this.count > 0 }

  _append(node, parent = this.root) {
    parent.appendChild(node)
    if (parent === this.root) { this.count++; this._showEmpty() }
    return node
  }

  add(ev) {
    switch (ev.t) {
      case 'user': this.group = null; this._append(el('div', 'msg-user', '')).textContent = ev.text; this.scrollToEnd(); break
      case 'text': this.group = null; this._append(el('div', 'msg-agent', md(ev.text))); break
      case 'call': this._call(ev); break
      case 'result': this._result(ev); break
      case 'task': this._task(ev); break
      case 'done': this._done(ev); break
      case 'error': this.group = null; this._append(el('div', 'msg-error', '')).textContent = ev.text; break
      case 'divider': this.group = null; this._append(el('div', 'divider', '')).textContent = ev.text; break
      case 'image': this.group = null; this._append(el('div', 'divider', '')).textContent = `Opened ${ev.image.title}`; break
    }
  }

  // --------------------------------------------------------------- steps
  _newGroup(parent) {
    const g = el('div', 'steps')
    g.innerHTML = `<button class="steps-head"><span class="chev">${icon('chevronRight')}</span><span class="n"></span><span class="last"></span></button>
      <div class="thumbs"></div><div class="step-list"></div>`
    g.querySelector('.steps-head').addEventListener('click', () => g.classList.toggle('open'))
    g.n = 0
    this._append(g, parent)
    return g
  }

  _call(ev) {
    if (ev.tool === 'Agent' && !ev.sub) {
      this.group = null
      this._card(ev.id, ev.args?.description || 'checking the result')
      return
    }
    let group
    if (ev.sub && ev.parent) {
      group = this._card(ev.parent, 'checking the result')
    } else {
      if (!this.group) this.group = this._newGroup()
      group = this.group
    }
    const step = el('div', 'step pending')
    step.innerHTML = `<span class="ic">${icon(TOOL_ICONS[ev.tool] || 'dot')}</span><span class="what"></span>`
    step.querySelector('.what').textContent = stepLabel(ev.tool, ev.args)
    if (ev.tool === 'python' && ev.args?.code) {
      const pre = el('pre', 'code')
      pre.textContent = ev.args.code
      step.appendChild(pre)
    }
    step.group = group
    step.label = stepLabel(ev.tool, ev.args)
    group.querySelector('.step-list').appendChild(step)
    group.n++
    this._updateHead(group, step.label)
    if (ev.id) this.steps.set(ev.id, step)
    if (!this.pendingByTool.has(ev.tool)) this.pendingByTool.set(ev.tool, [])
    this.pendingByTool.get(ev.tool).push(step)
    if (ev.tool === 'Read') step.classList.remove('pending')
    this._stick()
  }

  _updateHead(group, last) {
    if (group.classList.contains('verifier')) {
      group.querySelector('.count-steps').textContent = `${group.n} step${group.n === 1 ? '' : 's'}`
      return
    }
    group.querySelector('.n').textContent = `${group.n} step${group.n === 1 ? '' : 's'}`
    group.querySelector('.last').textContent = last
  }

  _result(ev) {
    let step = ev.id ? this.steps.get(ev.id) : null
    const pend = this.pendingByTool.get(ev.tool) || []
    if (!step) step = pend[0]
    if (!step) return
    const i = pend.indexOf(step)
    if (i >= 0) pend.splice(i, 1)
    step.classList.remove('pending')
    if (ev.error) step.classList.add('err')
    const summary = resultSummary(ev.text, ev.error)
    if (summary) {
      const res = el('div', 'res')
      res.textContent = summary
      step.appendChild(res)
    }
    if (ev.images?.length) {
      const strip = el('div', 'thumbs')
      for (const url of ev.images) {
        strip.appendChild(this._thumb(url, step.label))
        step.group.querySelector(':scope > .thumbs').appendChild(this._thumb(url, step.label))
      }
      step.appendChild(strip)
    }
    this._stick()
  }

  _thumb(url, caption) {
    let idx = this.images.findIndex((x) => x.url === url)
    if (idx < 0) { this.images.push({ url, caption }); idx = this.images.length - 1 }
    const img = new Image()
    img.loading = 'lazy'
    img.src = url
    img.alt = caption
    img.title = caption
    img.addEventListener('click', (e) => { e.stopPropagation(); this.onImageClick?.(idx) })
    img.addEventListener('load', () => this._stick(), { once: true })
    return img
  }

  _stick() {
    if (this.follow) this.root.scrollTop = this.root.scrollHeight
  }

  // ------------------------------------------------------------ verifiers
  _card(id, description) {
    let card = this.cards.get(id)
    if (card) return card
    card = el('div', 'verifier')
    card.innerHTML = `<button class="steps-head"><span class="chev">${icon('chevronRight')}</span><span class="badge">Verifier</span>
      <span class="last"></span><span class="count-steps"></span><span class="state running">checking</span></button>
      <div class="thumbs"></div><div class="vsum"></div><div class="step-list"></div>`
    card.querySelector('.last').textContent = description
    card.querySelector('.steps-head').addEventListener('click', () => card.classList.toggle('open'))
    card.n = 0
    this.cards.set(id, card)
    this._append(card)
    return card
  }

  _task(ev) {
    const card = this._card(ev.tool_use_id || ev.task_id, ev.description || 'checking the result')
    if (ev.description) card.querySelector('.last').textContent = ev.description
    const state = card.querySelector('.state')
    if (ev.status === 'started') { state.className = 'state running'; state.textContent = 'checking' }
    else if (ev.status) {
      state.className = `state ${ev.status}`
      state.textContent = { completed: 'done', failed: 'failed', stopped: 'stopped' }[ev.status] || ev.status
      if (ev.summary) card.querySelector('.vsum').innerHTML = md(ev.summary)
    }
  }

  _done(ev) {
    this.group = null
    for (const card of this.cards.values()) {
      const s = card.querySelector('.state')
      if (s.classList.contains('running')) { s.className = 'state stopped'; s.textContent = 'ended' }
    }
    for (const list of this.pendingByTool.values()) {
      for (const st of list) st.classList.remove('pending')
      list.length = 0
    }
    const bits = [`${ev.error ? (ev.text === 'cancelled' || ev.text === 'interrupted' ? 'Stopped' : 'Ended with an error') : 'Finished'} after ${duration(ev.seconds || 0)}`]
    if (ev.prompts) bits.push(`${ev.prompts} prompt${ev.prompts === 1 ? '' : 's'}`)
    if (ev.cost_usd) bits.push(`≈ $${ev.cost_usd.toFixed(2)} at API prices`)
    else if (ev.tokens) bits.push(`${ev.tokens >= 1e6 ? `${(ev.tokens / 1e6).toFixed(1)}M` : `${Math.max(1, Math.round(ev.tokens / 1e3))}k`} tokens`)
    const d = el('div', `msg-done${ev.error ? ' err' : ''}`, `${icon(ev.error ? 'alert' : 'check')}<span></span>`)
    d.querySelector('span').textContent = bits.join(' · ')
    this._append(d)
  }
}
