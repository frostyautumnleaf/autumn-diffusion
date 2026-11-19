// Option panel: renders the declarative sections of a tab into the left rail.
//
// Values are read from and written to the per-tab form in state.js. The panel
// rebuilds itself only when the *shape* of the panel changes (a section became
// visible, a dropdown lost an option), never on a plain keystroke, so typing
// keeps focus.

import { state, TOPICS } from './state.js';
import { sectionsFor, snapFrames as snapVideoFrames } from './sections.js';
import { buildField, getPath, setPath, fieldDefault } from './fields.js';
import { icon } from './icons.js';
import { attachTooltip, menu, toast } from './components.js';
import { el, clear, clamp, copyText } from './util.js';

// Form context -------------------------------------------------------------
/** Read a form value, falling back to the server defaults for that tab. */
function readValue(kind, key) {
  const direct = getPath(state.form(kind), key);
  if (direct !== undefined && direct !== null) return direct;
  const defaults = (state.settings.defaults && state.settings.defaults[kind]) || {};
  const fromDefaults = getPath(defaults, key);
  if (fromDefaults !== undefined) return fromDefaults;
  const section = state.settings[kind] || {};
  const fromSection = getPath(section, key);
  return fromSection === undefined ? '' : fromSection;
}

export function makeFormCtx(kind, extra = {}) {
  const ctx = {
    kind,
    get: (key) => readValue(kind, key),
    set: (key, value) => state.setFormValue(kind, key, value),
    refresh: () => extra.onRefresh && extra.onRefresh(),
    options: {
      samplers: state.models.samplers && state.models.samplers.length
        ? state.models.samplers : ['euler', 'euler_a', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_3m_sde', 'dpmpp_sde', 'lms', 'heun', 'uni_pc', 'ddim', 'ddpm'],
      schedulers: state.models.schedulers && state.models.schedulers.length
        ? state.models.schedulers : ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'beta'],
      image_files: (state.models.image_files || []).map((f) => f.name || f),
      video_files: (state.models.video_files || []).map((f) => f.name || f),
      audio_files: (state.models.audio_files || []).map((f) => f.name || f),
      attachments: (state.form(kind).attachments || []).map((a) => a.name || a),
      interpolation_models: state.models.interpolation_models || [],
      seedvr_models: state.models.seedvr_models || [],
      seedvr_vae_models: state.models.seedvr_vae_models || [],
      upscale_models: state.models.upscale_models || [],
    },

    /** Set the canvas from an aspect-ratio button, keeping the model multiple. */
    setSize: (width, height) => {
      const multiple = kind === 'video' ? multipleForFamily(readValue(kind, 'family')) : 8;
      const fit = (value) => Math.max(multiple, Math.round(value / multiple) * multiple);
      state.setForm(kind, { width: fit(width), height: fit(height) });
    },

    /** Convert a duration in seconds into a frame count on the model grid. */
    setDuration: (seconds) => {
      const family = readValue(kind, 'family') || 'wan22';
      const fps = Number(readValue(kind, 'fps')) || 24;
      state.setForm(kind, { frames: snapVideoFrames(family, Math.round(seconds * fps)) });
    },
  };
  return ctx;
}

function multipleForFamily(family) {
  const table = { wan22: 16, hunyuan: 16, hunyuan15: 16, ltx2: 32, minimax_h3: 32, svd: 8, generic: 16 };
  return table[family] || 16;
}

// Panel --------------------------------------------------------------------
export class Panel {
  constructor(root, kind, { onRefresh } = {}) {
    this.root = root;
    this.kind = kind;
    this.onRefresh = onRefresh;
    this.query = '';
    this.shape = '';
    this.ctx = makeFormCtx(kind, { onRefresh: () => this.render() });
    this._onForm = (kind) => {
      if (kind && kind !== this.kind) return;
      this.syncShape();
    };
    state.on(TOPICS.form, this._onForm);
    state.on(TOPICS.models, () => this.syncShape());
    state.on(TOPICS.settings, () => this.syncShape());
  }

  setKind(kind) {
    if (kind === this.kind) return;
    this.kind = kind;
    this.ctx = makeFormCtx(kind, { onRefresh: () => this.render() });
    this.render();
  }

  setQuery(query) {
    const next = (query || '').trim().toLowerCase();
    if (next === this.query) return;
    this.query = next;
    this.render();
  }

  /** Recompute the visible shape and rebuild only when something changed. */
  syncShape() {
    const shape = this.shapeOf();
    if (shape === this.shape) { this.updateHints(); return; }
    this.render();
  }

  shapeOf() {
    const parts = [];
    for (const section of sectionsFor(this.kind)) {
      if (section.when && !section.when(this.ctx)) continue;
      parts.push(section.id);
      for (const field of section.fields) {
        if (field.when && !field.when(this.ctx)) continue;
        parts.push(`${section.id}:${field.key}:${field.type}`);
        if (typeof field.options === 'function') {
          parts.push(JSON.stringify(field.options(this.ctx) || []).slice(0, 300));
        }
      }
    }
    return parts.join('|');
  }

  isOpen(section) {
    if (this.query) return true;
    const saved = state.ui.sections[section.id];
    if (saved !== undefined) return !!saved;
    const index = sectionsFor(this.kind).findIndex((item) => item.id === section.id);
    return index < 2;
  }

  render() {
    this.shape = this.shapeOf();
    const active = document.activeElement;
    const restoreKey = active && active.dataset ? active.dataset.field : null;
    const restoreStart = active && typeof active.selectionStart === 'number' ? active.selectionStart : null;
    const scrollTop = this.root.scrollTop;

    clear(this.root);
    const list = sectionsFor(this.kind).filter((section) => {
      if (section.when && !section.when(this.ctx)) return false;
      if (!this.query) return true;
      return this.matches(section);
    });
    if (!list.length) {
      this.root.appendChild(el('div', { class: 'empty' }, [
        el('div', { class: 'empty-icon', html: icon('search', 22) }),
        el('div', { text: this.query ? 'No setting matches that search.' : 'Nothing to configure here yet.' }),
      ]));
      return;
    }
    for (const section of list) this.root.appendChild(this.renderSection(section));
    this.root.scrollTop = scrollTop;
    if (restoreKey) {
      const node = this.root.querySelector(`[data-field=\"${restoreKey}\"] input, [data-field=\"${restoreKey}\"] textarea, [data-field=\"${restoreKey}\"] select`);
      if (node) {
        node.focus();
        if (restoreStart !== null && node.setSelectionRange) {
          try { node.setSelectionRange(restoreStart, restoreStart); } catch (err) { /* type=number */ }
        }
      }
    }
  }

  matches(section) {
    const q = this.query;
    if (section.label.toLowerCase().includes(q)) return true;
    const hint = typeof section.hint === 'function' ? section.hint(this.ctx) : section.hint;
    if (hint && String(hint).toLowerCase().includes(q)) return true;
    return section.fields.some((field) => {
      if (field.when && !field.when(this.ctx)) return false;
      const label = String(field.label || '').toLowerCase();
      return label.includes(q) || String(field.key).toLowerCase().includes(q);
    });
  }

  renderSection(section) {
    const open = this.isOpen(section);
    const node = el('section', { class: `section ${open ? 'open' : ''}`, dataset: { section: section.id } });
    const hint = typeof section.hint === 'function' ? section.hint(this.ctx) : section.hint;
    const head = el('div', {
      class: 'section-head',
      onclick: () => {
        const next = !node.classList.contains('open');
        node.classList.toggle('open', next);
        state.ui.sections = { ...state.ui.sections, [section.id]: next };
        state.persistUI();
      },
    }, [
      el('span', { class: 'section-icon', html: icon(section.icon || 'sliders_h', 15) }),
      el('span', { class: 'section-label', text: section.label }),
      el('span', { class: 'section-count', text: hint ? String(hint) : '' }),
      el('span', { class: 'chev', html: icon('chevron_right', 14) }),
    ]);
    if (!hint) head.querySelector('.section-count').remove();
    const label = head.querySelector('.section-label');
    if (section.tip) {
      label.classList.add('has-tip');
      attachTooltip(label, { title: section.label, text: section.tip });
    }
    // Right-click a section to reset it, or to copy what it currently holds.
    head.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      menu(null, [
        { label: 'Reset this section to defaults', icon: 'refresh',
          run: () => resetSection(this, section) },
        { label: 'Copy these settings as JSON', icon: 'copy',
          run: () => copySection(this, section) },
      ], { at: { x: event.clientX, y: event.clientY } });
    });

    const body = el('div', { class: 'section-body' });
    for (const field of section.fields) {
      const built = buildField(field, this.ctx);
      if (built) body.appendChild(built);
    }
    node.appendChild(head);
    node.appendChild(body);
    return node;
  }

  /** Refresh the small value read-outs in the section headers. */
  updateHints() {
    for (const node of this.root.querySelectorAll('.section[data-section]')) {
      const section = sectionsFor(this.kind).find((item) => item.id === node.dataset.section);
      const head = node.querySelector('.section-head');
      if (!section || !head) continue;
      const hint = typeof section.hint === 'function' ? section.hint(this.ctx) : section.hint;
      let badge = head.querySelector('.section-count');
      if (hint) {
        if (!badge) {
          badge = el('span', { class: 'section-count' });
          head.querySelector('.chev').before(badge);
        }
        badge.textContent = String(hint);
      } else if (badge) {
        badge.remove();
      }
    }
  }

  destroy() {
    state.off(TOPICS.form, this._onForm);
  }
}

/** Write the stored default back over every field a section owns. */
function resetSection(panel, section) {
  let touched = 0;
  for (const field of section.fields) {
    if (!field.key || String(field.key).startsWith('_')) continue;
    if (field.when && !field.when(panel.ctx)) continue;
    const def = fieldDefault(field, panel.kind);
    if (def === undefined) continue;
    let copy = def;
    try { copy = structuredClone(def); } catch (err) { copy = def; }
    state.setFormValue(panel.kind, field.key, copy);
    touched += 1;
  }
  panel.render();
  toast(touched ? `${section.label}: ${touched} setting(s) reset` : 'Nothing to reset',
    { kind: 'ok', timeout: 1800 });
}

async function copySection(panel, section) {
  const values = {};
  for (const field of section.fields) {
    if (!field.key || String(field.key).startsWith('_')) continue;
    values[field.key] = panel.ctx.get(field.key);
  }
  try {
    await copyText(JSON.stringify(values, null, 2));
    toast(`${section.label} copied as JSON`, { kind: 'ok', timeout: 1800 });
  } catch (err) {
    toast(err.message, { kind: 'err' });
  }
}

// Helpers shared with the action bar ---------------------------------------
export function canvasLabel(kind, values) {
  const width = values.width || (kind === 'video' ? 1280 : 1024);
  const height = values.height || (kind === 'video' ? 720 : 1024);
  return `${width}×${height}`;
}

export function clampCanvas(value, multiple = 8) {
  return clamp(Math.round(value / multiple) * multiple, multiple, 8192);
}

export { readValue };
