// Field factory: turns declarative specs into the controls used by every tab.
// Values live in the tab form; every edit writes through to state.setFormValue.

import { el, clear, clamp, debounce, fileName, baseName, matchScore, mediaKind } from './util.js';
import { icon } from './icons.js';
import { state, TOPICS } from './state.js';
import { menu, promptDialog, toast } from './components.js';

// Path helpers -------------------------------------------------------------
export function getPath(source, path) {
  if (!path.includes('.')) return source[path];
  let node = source;
  for (const part of path.split('.')) {
    if (node === null || node === undefined) return undefined;
    node = node[part];
  }
  return node;
}

export function setPath(target, path, value) {
  const parts = path.split('.');
  let node = target;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i];
    if (typeof node[key] !== 'object' || node[key] === null || Array.isArray(node[key])) node[key] = {};
    node = node[key];
  }
  node[parts[parts.length - 1]] = value;
}

// Popover ------------------------------------------------------------------
export function popover(anchor, build, { width = 320, align = 'left' } = {}) {
  closePopovers();
  const node = el('div', { class: 'popover', style: { width: `${width}px` } });
  const api = { node, close: () => { cleanup(); node.remove(); } };
  build(api, node);
  document.body.appendChild(node);
  const rect = anchor.getBoundingClientRect();
  const box = node.getBoundingClientRect();
  let left = align === 'right' ? rect.right - box.width : rect.left;
  left = Math.max(8, Math.min(left, window.innerWidth - box.width - 8));
  let top = rect.bottom + 4;
  if (top + box.height > window.innerHeight - 8) top = Math.max(8, rect.top - box.height - 4);
  node.style.left = `${left}px`;
  node.style.top = `${top}px`;
  node.dataset.popover = '1';
  const onDown = (event) => {
    if (!node.contains(event.target) && !anchor.contains(event.target)) api.close();
  };
  const onKey = (event) => { if (event.key === 'Escape') api.close(); };
  setTimeout(() => {
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    window.addEventListener('scroll', api.close, true);
  }, 0);
  node._cleanup = () => {
    document.removeEventListener('pointerdown', onDown);
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('scroll', api.close, true);
  };
  function cleanup() { if (node._cleanup) node._cleanup(); }
  return api;
}

export function closePopovers() {
  document.querySelectorAll('.popover[data-popover]').forEach((node) => {
    if (node._cleanup) node._cleanup();
    node.remove();
  });
}

// Combobox -----------------------------------------------------------------
/** Searchable single-value picker used for models, VAEs, LoRAs and files. */
class Combo {
  constructor(spec, ctx) {
    this.spec = spec;
    this.ctx = ctx;
    this.value = '';
    this.filter = '';
    this.opened = null;
    this.input = el('input', {
      class: 'input combo-input',
      placeholder: spec.placeholder || spec.emptyLabel || 'None',
      readonly: false,
      autocomplete: 'off',
      spellcheck: 'false',
    });
    this.chevron = el('button', {
      class: 'icon-btn combo-toggle', tabindex: '-1',
      html: icon('chevron_down', 13),
      onclick: (event) => { event.preventDefault(); this.toggle(); },
    });
    this.node = el('div', { class: 'combo' }, [this.input, this.chevron]);
    this.input.addEventListener('focus', () => this.toggle(true));
    this.input.addEventListener('input', () => {
      this.filter = this.input.value;
      if (this.opened) this.render();
      else this.toggle(true);
      if (this.spec.freeEntry) this.commit(this.input.value);
    });
    this.input.addEventListener('keydown', (event) => this.onKey(event));
    this.input.addEventListener('blur', () => {
      setTimeout(() => {
        if (!this.opened && !this.spec.freeEntry) this.input.value = this.value || '';
      }, 120);
    });
  }

  options() {
    const raw = typeof this.spec.options === 'function'
      ? this.spec.options(this.ctx)
      : (Array.isArray(this.spec.options)
        ? this.spec.options
        : (this.ctx.options && this.ctx.options[this.spec.options]) || state.models.models[this.spec.optionsCategory] || []);
    return raw.map((item) => (typeof item === 'string' ? { value: item, label: item } : item));
  }

  filtered() {
    const all = this.options();
    if (!this.filter) return all.slice(0, 240);
    return all
      .map((item) => ({ item, score: matchScore(this.filter, item.value) }))
      .filter((entry) => entry.score >= 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 240)
      .map((entry) => entry.item);
  }

  setValue(value) {
    this.value = value === null || value === undefined ? '' : String(value);
    if (document.activeElement !== this.input) this.input.value = this.value;
  }

  commit(value) {
    const next = value === null || value === undefined ? '' : String(value);
    if (next === this.value) return;
    this.value = next;
    this.ctx.set(this.spec.key, next);
  }

  toggle(force) {
    if (this.opened) { this.opened.close(); this.opened = null; return; }
    if (force === false) return;
    this.filter = this.spec.freeEntry ? '' : '';
    this.opened = popover(this.node, (api) => {
      this.render(api);
    }, { width: Math.max(260, this.node.getBoundingClientRect().width + 40) });
    this.opened.node.classList.add('combo-list-open');
    setTimeout(() => this.input.focus(), 10);
  }

  render(api = this.opened) {
    if (!api) return;
    const list = api.node.querySelector('.combo-items') || el('div', { class: 'combo-items' });
    if (!api.node.querySelector('.combo-items')) {
      const search = el('input', {
        class: 'input sm', placeholder: this.spec.searchHint || 'Filter…',
        oninput: (event) => { this.filter = event.target.value; this.render(api); },
        onkeydown: (event) => {
          if (event.key === 'Enter') { const items = this.filtered(); if (items[0]) this.pick(items[0].value, api); }
        },
      });
      search.value = this.filter;
      api.node.appendChild(el('div', { class: 'row', style: { padding: '2px 3px 6px' } }, [search]));
      api.node.appendChild(list);
    }
    clear(list);
    const items = this.filtered();
    const empty = this.spec.emptyLabel
      ? [{ value: '', label: this.spec.emptyLabel }] : [];
    for (const item of empty.concat(items)) {
      const selected = item.value === this.value;
      list.appendChild(el('div', {
        class: `menu-item ${selected ? 'sel' : ''}`,
        style: { display: 'flex' },
        onclick: () => this.pick(item.value, api),
      }, [
        el('span', { class: 'grow trunc', text: item.label || item.value || '—' }),
        selected ? el('span', { class: 'acc-t', html: icon('check', 13) }) : null,
        item.hint ? el('span', { class: 'tiny faint mono', text: item.hint }) : null,
      ]));
    }
    if (!items.length) {
      list.appendChild(el('div', { class: 'tiny faint', style: { padding: '8px' } }, 'Nothing matches that filter.'));
    }
    list.style.maxHeight = '262px';
    list.style.overflow = 'auto';
  }

  pick(value, api = this.opened) {
    this.commit(value);
    this.input.value = value || '';
    if (api) api.close();
    if (this.opened) this.opened = null;
  }

  onKey(event) {
    if (event.key === 'Escape') { if (this.opened) { this.opened.close(); this.opened = null; } }
    if (event.key === 'Enter') {
      const items = this.filtered();
      this.pick((items[0] && items[0].value) || this.input.value);
      event.preventDefault();
    }
    if (event.key === 'ArrowDown' && !this.opened) { this.toggle(true); event.preventDefault(); }
  }
}

// Controls -----------------------------------------------------------------
function labelRow(spec) {
  const hint = typeof spec.hint === 'function' ? spec.hint(spec) : spec.hint;
  const row = el('div', { class: 'field-top' }, [
    el('span', { class: 'field-label', text: spec.label }),
    hint ? el('span', { class: 'field-hint', text: String(hint) }) : null,
  ]);
  return row;
}

function wrap(spec, control, extra) {
  const node = el('div', { class: 'field', dataset: { key: spec.key || '', type: spec.type } }, [
    spec.label === null ? null : labelRow(spec),
    control,
    spec.help ? el('div', { class: 'field-help', text: spec.help }) : null,
    extra || null,
  ]);
  return node;
}

const CONTROLS = {
  text(spec, ctx) {
    const input = el('input', {
      class: 'input', placeholder: spec.placeholder || '', value: ctx.get(spec.key) ?? '',
      oninput: debounce((event) => ctx.set(spec.key, event.target.value), 260),
    });
    return wrap(spec, input, null);
  },

  area(spec, ctx) {
    const input = el('textarea', {
      class: 'textarea', placeholder: spec.placeholder || '', rows: spec.rows || 4,
      oninput: debounce((event) => ctx.set(spec.key, event.target.value), 300),
    });
    input.value = ctx.get(spec.key) ?? '';
    return wrap(spec, input);
  },

  code(spec, ctx) {
    const input = el('textarea', {
      class: 'textarea', spellcheck: 'false', rows: spec.rows || 8,
      style: { fontFamily: 'var(--mono)', fontSize: '11px' },
      oninput: debounce((event) => ctx.set(spec.key, event.target.value), 400),
    });
    input.value = ctx.get(spec.key) ?? '';
    return wrap(spec, input);
  },

  number(spec, ctx) {
    const value = numValue(ctx.get(spec.key), spec);
    const slider = el('input', {
      type: 'range', min: spec.min ?? 0, max: spec.max ?? 100,
      step: spec.step ?? 1, value,
      style: { '--pct': pctOf(value, spec) },
    });
    const box = el('input', {
      class: 'input', type: 'number', min: spec.min, max: spec.max, step: spec.step ?? 'any',
      value,
    });
    const push = debounce((next) => ctx.set(spec.key, next), 220);
    const sync = (next, from) => {
      const parsed = numValue(next, spec);
      if (from !== 'slider') slider.value = parsed;
      if (from !== 'box') box.value = parsed;
      slider.style.setProperty('--pct', pctOf(parsed, spec));
    };
    slider.addEventListener('input', () => { sync(slider.value, 'slider'); push(Number(slider.value)); });
    box.addEventListener('input', () => { sync(box.value, 'box'); push(Number(box.value)); });
    if (spec.slider === false) {
      return wrap(spec, el('div', { class: 'num-row wide' }, [box, unitLabel(spec)]));
    }
    return wrap(spec, el('div', { class: 'num-row' }, [slider, box, unitLabel(spec)]));
  },

  /** Percent-bound number: stored as 0..1, edited as 0..100. */
  percent(spec, ctx) {
    const scale = spec.scale || 100;
    const readValue = () => {
      const raw = Number(ctx.get(spec.key));
      const base = Number.isFinite(raw) ? raw : (spec.default ?? 0);
      return Math.round(base * scale * 100) / 100;
    };
    const slider = el('input', {
      type: 'range', min: 0, max: scale, step: spec.step ?? 1, value: readValue(),
      style: { '--pct': `${readValue()}%` },
    });
    const box = el('input', {
      class: 'input', type: 'number', min: 0, max: scale, step: spec.step ?? 1,
      value: readValue(),
    });
    const push = debounce((next) => ctx.set(spec.key, next / scale), 220);
    const sync = (value, from) => {
      const parsed = clamp(Number(value) || 0, 0, scale);
      if (from !== 'slider') slider.value = parsed;
      if (from !== 'box') box.value = parsed;
      slider.style.setProperty('--pct', `${parsed}%`);
    };
    slider.addEventListener('input', () => { sync(slider.value, 'slider'); push(Number(slider.value)); });
    box.addEventListener('input', () => { sync(box.value, 'box'); push(Number(box.value)); });
    if (spec.slider === false) return wrap(spec, el('div', { class: 'num-row' }, [box, unitLabel(spec)]));
    return wrap(spec, el('div', { class: 'num-row' }, [slider, box, unitLabel(spec)]));
  },

  select(spec, ctx) {
    const select = el('select', {
      class: 'select', onchange: (event) => ctx.set(spec.key, event.target.value),
    });
    const options = resolveOptions(spec, ctx);
    if (spec.emptyLabel) select.appendChild(el('option', { value: '', text: spec.emptyLabel }));
    for (const item of options) {
      select.appendChild(el('option', { value: item.value, text: item.label || item.value }));
    }
    select.value = ctx.get(spec.key) ?? '';
    if (select.selectedIndex < 0) select.selectedIndex = 0;
    return wrap(spec, select);
  },

  combo(spec, ctx) {
    const combo = new Combo(spec, ctx);
    combo.setValue(ctx.get(spec.key));
    return wrap(spec, combo.node);
  },

  toggle(spec, ctx) {
    const input = el('input', { type: 'checkbox', checked: !!ctx.get(spec.key) });
    const node = el('label', { class: 'switch' }, [
      input,
      el('span', { class: 'track' }, [el('span', { class: 'knob' })]),
    ]);
    input.addEventListener('change', () => ctx.set(spec.key, input.checked));
    if (spec.label === null) return node;
    return wrap(spec, node);
  },

  check(spec, ctx) {
    const input = el('input', { type: 'checkbox', checked: !!ctx.get(spec.key) });
    const node = el('label', { class: 'check' }, [
      input, el('span', { class: 'box', html: icon('check', 11) }),
      el('span', { class: 'txt', text: spec.text || spec.label || '' }),
    ]);
    input.addEventListener('change', () => ctx.set(spec.key, input.checked));
    return el('div', { class: 'field' }, [node]);
  },

  segmented(spec, ctx) {
    const group = el('div', { class: `segmented ${spec.grow ? 'grow' : ''}` });
    const options = resolveOptions(spec, ctx);
    const buttons = [];
    const refresh = () => {
      const value = ctx.get(spec.key);
      buttons.forEach((button, index) => {
        button.classList.toggle('on', String(options[index].value) === String(value));
      });
    };
    options.forEach((item, index) => {
      const button = el('button', {
        type: 'button', text: item.label || item.value,
        title: item.hint || '',
        onclick: () => { ctx.set(spec.key, item.value); refresh(); },
      });
      buttons.push(button);
      group.appendChild(button);
    });
    refresh();
    return wrap(spec, group);
  },

  chips(spec, ctx) {
    const current = () => toArray(ctx.get(spec.key));
    const box = el('div', { class: 'chips' });
    const render = () => {
      clear(box);
      for (const item of resolveOptions(spec, ctx)) {
        const on = current().includes(item.value);
        box.appendChild(el('div', {
          class: `chip ${on ? 'on' : ''}`,
          onclick: () => {
            const list = current();
            const next = on ? list.filter((v) => v !== item.value) : list.concat(item.value);
            ctx.set(spec.key, next);
            render();
          },
        }, [item.icon ? el('span', { html: icon(item.icon, 12) }) : null, item.label || item.value]));
      }
    };
    render();
    return wrap(spec, box);
  },

  tags(spec, ctx) {
    const box = el('div', { class: 'chips' });
    const input = el('input', {
      class: 'input sm', placeholder: spec.placeholder || 'Add and press Enter',
      onkeydown: (event) => {
        if (event.key !== 'Enter') return;
        const value = input.value.trim();
        if (!value) return;
        input.value = '';
        ctx.set(spec.key, toArray(ctx.get(spec.key)).concat(value));
        render();
      },
    });
    const render = () => {
      const list = toArray(ctx.get(spec.key));
      clear(box);
      for (const [index, value] of list.entries()) {
        box.appendChild(el('div', { class: 'chip on' }, [
          el('span', { text: value }),
          el('span', {
            class: 'x', html: icon('x', 10),
            onclick: () => { ctx.set(spec.key, list.filter((_, i) => i !== index)); render(); },
          }),
        ]));
      }
    };
    render();
    return wrap(spec, el('div', { class: 'col', style: { gap: '6px' } }, [box, input]));
  },

  /** LoRA stack: model, model strength, clip strength. */
  loras(spec, ctx) {
    return listEditor(spec, ctx, {
      addLabel: 'Add LoRA',
      newItem: () => ({ name: '', strength_model: 1.0, strength_clip: 1.0, model_only: false, enabled: true }),
      title: (item, index) => item.name ? baseName(item.name) : `LoRA ${index + 1}`,
      emptyText: 'No LoRAs in the stack.',
      fields: [
        { key: 'name', type: 'combo', optionsCategory: 'loras', emptyLabel: 'Choose a LoRA…', span: 2, label: 'Model' },
        { key: 'strength_model', type: 'number', label: 'Model weight', min: -2, max: 3, step: 0.05 },
        { key: 'strength_clip', type: 'number', label: 'CLIP weight', min: -2, max: 3, step: 0.05 },
        { key: 'model_only', type: 'check', label: null, text: 'Model only (no CLIP)', span: 2 },
      ],
    });
  },

  /** Textual-inversion embeddings, applied as prompt tokens. */
  embeddings(spec, ctx) {
    return listEditor(spec, ctx, {
      addLabel: 'Add embedding',
      newItem: () => ({ name: '', strength: 1.0, enabled: true }),
      title: (item, index) => item.name ? baseName(item.name) : `Embedding ${index + 1}`,
      emptyText: 'No embeddings added.',
      fields: [
        { key: 'name', type: 'combo', optionsCategory: 'embeddings', emptyLabel: 'Choose an embedding…', span: 2, label: 'Embedding' },
        { key: 'strength', type: 'number', label: 'Token strength', min: 0, max: 2, step: 0.05 },
      ],
    });
  },

  controlnets(spec, ctx) {
    return listEditor(spec, ctx, {
      addLabel: 'Add ControlNet',
      newItem: () => ({ name: '', image: '', strength: 1.0, start_percent: 0, end_percent: 1, enabled: true }),
      title: (item, index) => item.name ? baseName(item.name) : `ControlNet ${index + 1}`,
      emptyText: 'No ControlNets applied.',
      fields: [
        { key: 'name', type: 'combo', optionsCategory: 'controlnet', emptyLabel: 'Choose a ControlNet…', span: 2, label: 'Model' },
        { key: 'image', type: 'combo', options: 'attachments', emptyLabel: 'No control image', span: 2, label: 'Control image' },
        { key: 'strength', type: 'number', label: 'Strength', min: 0, max: 2.5, step: 0.05 },
        { key: 'start_percent', type: 'percent', label: 'Start', unit: '%' },
        { key: 'end_percent', type: 'percent', label: 'End', unit: '%' },
      ],
    });
  },

  /** Prompt-guidance guides (MiniMax H3 style regions, objects, camera moves). */
  guides(spec, ctx) {
    return listEditor(spec, ctx, {
      addLabel: 'Add guide',
      newItem: () => ({ type: 'image', image: '', strength: 1.0, start_frame: 0, end_frame: -1 }),
      title: (item, index) => `Guide ${index + 1}`,
      fields: [
        { key: 'type', type: 'select', options: [
          { value: 'image', label: 'Image reference' },
          { value: 'subject', label: 'Subject' },
          { value: 'camera', label: 'Camera move' },
          { value: 'style', label: 'Style' }] },
        { key: 'image', type: 'combo', options: 'attachments', emptyLabel: 'No attachment' },
        { key: 'text', type: 'text', placeholder: 'Camera pans left…' },
        { key: 'strength', type: 'number', label: 'Strength', min: 0, max: 2, step: 0.05 },
        { key: 'start_frame', type: 'number', label: 'From frame', min: 0, max: 100000, step: 1 },
        { key: 'end_frame', type: 'number', label: 'To frame (-1 = all)', min: -1, max: 100000, step: 1 },
      ],
    });
  },

  references(spec, ctx) {
    return listEditor(spec, ctx, {
      addLabel: 'Add reference image',
      newItem: () => ({ image: '', size: 'match', strength: 1.0 }),
      title: (item, index) => `Reference ${index + 1}`,
      fields: [
        { key: 'image', type: 'combo', options: 'attachments', emptyLabel: 'No attachment', span: 2 },
        { key: 'size', type: 'select', options: [
          { value: 'match', label: 'Match target' }, { value: 'square', label: 'Square' },
          { value: 'keep', label: 'Original' }] },
        { key: 'strength', type: 'number', label: 'Strength', min: 0, max: 2, step: 0.05 },
      ],
    });
  },

  /** Regional prompting canvas: named rectangles with weight and prompt. */
  regions(spec, ctx) {
    return regionEditor(spec, ctx);
  },
  /** Raw key/value rows, for environment variables and headers. */
  pairs(spec, ctx) {
    return listEditor(spec, ctx, {
      addLabel: spec.addLabel || 'Add entry',
      newItem: () => ({ key: '', value: '' }),
      title: (item, index) => item.key || `Entry ${index + 1}`,
      fields: [
        { key: 'key', type: 'text', placeholder: 'NAME' },
        { key: 'value', type: 'text', placeholder: 'value' },
      ],
    });
  },

  buttons(spec, ctx) {
    const row = el('div', { class: 'row', style: { gap: '6px', flexWrap: 'wrap' } });
    const items = typeof spec.items === 'function' ? spec.items(ctx) : (spec.items || []);
    for (const item of items) {
      row.appendChild(el('button', {
        class: `btn sm ${item.kind || 'ghost'}`,
        html: `${item.icon ? icon(item.icon, 13) : ''}<span>${item.label}</span>`,
        onclick: () => item.run(ctx),
      }));
    }
    return spec.label === null ? row : wrap(spec, row);
  },

  note(spec) {
    return el('div', { class: 'field tiny faint' }, [
      spec.title ? el('div', { class: 'sub-label', text: spec.title }) : null,
      el('div', { text: spec.text }),
    ]);
  },

  heading(spec) {
    return el('div', { class: 'sub-label', text: spec.label });
  },

  /** Free text args, shown monospaced. */
  args(spec, ctx) {
    const input = el('textarea', {
      class: 'textarea', rows: spec.rows || 3, spellcheck: 'false',
      style: { fontFamily: 'var(--mono)', fontSize: '11px' },
      placeholder: spec.placeholder || '--lowvram --preview-method auto',
      oninput: debounce((event) => ctx.set(spec.key, event.target.value), 400),
    });
    input.value = ctx.get(spec.key) ?? '';
    return wrap(spec, input);
  },

  /** File drop target: uploads straight into a ComfyUI model folder. */
  filepick(spec, ctx) {
    const target = el('div', { class: 'assets-drop', text: spec.text || 'Drop a file here to upload' });
    const value = el('div', { class: 'tiny mono faint trunc', text: ctx.get(spec.key) || '' });
    wireUpload(target, async (file) => {
      const { api } = await import('./api.js');
      const data = await api.upload(file, { category: spec.category || 'input' });
      ctx.set(spec.key, data.file.name);
      value.textContent = data.file.name;
      toast(`Uploaded ${file.name}`, { kind: 'ok' });
      state.emit(TOPICS.models);
    });
    return wrap(spec, el('div', { class: 'col', style: { gap: '5px' } }, [target, value]));
  },
};

// Generic list editor ------------------------------------------------------
function listEditor(spec, ctx, { addLabel, newItem, title, fields }) {
  const box = el('div', { class: 'rows' });
  const items = () => {
    const list = ctx.get(spec.key);
    return Array.isArray(list) ? list : [];
  };
  const write = (list) => { ctx.set(spec.key, list); render(); };

  const render = () => {
    clear(box);
    const list = items();
    if (!list.length) {
      box.appendChild(el('div', { class: 'empty-rows', text: spec.emptyText || 'Nothing added yet.' }));
    }
    list.forEach((item, index) => {
      const rows = el('div', { class: 'field-grid' });
      for (const field of fields) {
        const cell = buildField({ ...field, key: `${spec.key}.${index}.${field.key}`, label: field.label || field.key }, {
          ...ctx,
          get: () => getPath(item, field.key),
          set: (_key, value) => {
            const next = items().slice();
            next[index] = { ...next[index] };
            setPath(next[index], field.key, value);
            ctx.set(spec.key, next);
          },
        });
        if (cell) {
          cell.classList.add('field', 'tight');
          if (field.span === 2) cell.style.gridColumn = '1 / -1';
          rows.appendChild(cell);
        }
      }
      box.appendChild(el('div', { class: 'row-item' }, [
        el('div', { class: 'row-item-head' }, [
          el('span', { class: 'row-item-title', text: title(item, index) }),
          el('span', { class: 'grow' }),
          el('button', {
            class: 'icon-btn', style: { width: '22px', height: '22px' },
            html: icon('copy', 12), title: 'Duplicate',
            onclick: () => { const next = items().slice(); next.splice(index + 1, 0, { ...item }); write(next); },
          }),
          el('button', {
            class: 'icon-btn danger', style: { width: '22px', height: '22px' },
            html: icon('x', 12), title: 'Remove',
            onclick: () => write(items().filter((_, i) => i !== index)),
          }),
        ]),
        rows,
      ]));
    });
  };

  const add = () => { write(items().concat(newItem())); };
  render();
  return wrap({ ...spec, label: null }, el('div', { class: 'col', style: { gap: '7px' } }, [
    el('div', { class: 'row' }, [
      spec.label ? el('span', { class: 'field-label', text: spec.label }) : null,
      el('span', { class: 'grow' }),
      el('button', {
        class: 'btn sm ghost', html: `${icon('plus', 12)}<span>${addLabel}</span>`, onclick: add,
      }),
    ]),
    box,
  ]));
}

// Regional prompting -------------------------------------------------------
function regionEditor(spec, ctx) {
  const box = el('div', { class: 'rows' });
  const getRegions = () => {
    const value = ctx.get(spec.key) || [];
    return Array.isArray(value) ? value : [];
  };
  const write = (list) => { ctx.set(spec.key, list); render(); };

  const render = () => {
    const list = getRegions();
    clear(box);
    if (!list.length) {
      box.appendChild(el('div', { class: 'empty-rows' }, [
        el('div', { text: 'No regions yet. Add one, then drag the handles on the canvas.' }),
      ]));
    }
    list.forEach((region, index) => {
      const grid = el('div', { class: 'field-grid' });
      const push = (patch) => {
        const next = getRegions().slice();
        next[index] = { ...next[index], ...patch };
        ctx.set(spec.key, next);
      };
      const cell = (field) => {
        const node = buildField({ ...field, key: `${spec.key}.${index}.${field.key}` }, {
          ...ctx, get: () => getPath(region, field.key), set: (_k, value) => push({ [field.key]: value }),
        });
        if (node) {
          node.classList.add('field');
          if (field.span === 2) node.style.gridColumn = '1 / -1';
          grid.appendChild(node);
        }
      };
      cell({ key: 'prompt', type: 'text', label: 'Prompt', span: 2, placeholder: 'contents of this region' });
      cell({ key: 'label', type: 'text', label: 'Label', placeholder: 'optional name' });
      cell({ key: 'x', type: 'percent', label: 'X', unit: '%', slider: false });
      cell({ key: 'y', type: 'percent', label: 'Y', unit: '%', slider: false });
      cell({ key: 'w', type: 'percent', label: 'Width', unit: '%', slider: false });
      cell({ key: 'h', type: 'percent', label: 'Height', unit: '%', slider: false });
      cell({ key: 'strength', type: 'number', label: 'Strength', min: 0, max: 3, step: 0.05 });
      cell({ key: 'mode', type: 'select', label: 'Blend', options: [
        { value: '', label: 'Default' },
        { value: 'area', label: 'Area (additive)' },
        { value: 'mask', label: 'Masked rectangle' },
        { value: 'concat', label: 'Concat chain' },
      ] });
      cell({ key: 'negative', type: 'text', label: 'Region negative', span: 2,
        placeholder: 'things to keep out of this region' });
      box.appendChild(el('div', { class: 'row-item' }, [
        el('div', { class: 'row-item-head' }, [
          el('span', { class: 'row-item-title', text: region.label || `Region ${index + 1}` }),
          el('span', { class: 'grow' }),
          el('button', {
            class: 'icon-btn', style: { width: '22px', height: '22px' }, html: icon('copy', 12),
            title: 'Duplicate',
            onclick: () => write(getRegions().slice(0, index + 1).concat({ ...region }, getRegions().slice(index + 1))),
          }),
          el('button', {
            class: 'icon-btn danger', style: { width: '22px', height: '22px' }, html: icon('x', 12),
            title: 'Remove',
            onclick: () => write(getRegions().filter((_, i) => i !== index)),
          }),
        ]),
        grid,
      ]));
    });
  };

  render();
  return wrap({ ...spec, label: null }, el('div', { class: 'col', style: { gap: '7px' } }, [
    el('div', { class: 'row' }, [
      el('span', { class: 'grow' }),
      el('button', {
        class: 'btn sm ghost', html: `${icon('plus', 12)}<span>Add region</span>`,
        onclick: () => {
          const list = getRegions();
          const size = 40;
          write(list.concat({
            label: `Region ${list.length + 1}`, prompt: '', negative: '',
            x: ((list.length % 3) * 0.22) + 0.05, y: ((list.length % 2) * 0.22) + 0.1,
            w: 0.4, h: 0.4, strength: 1.0, mode: '', enabled: true,
          }));
        },
      }),
    ]),
    box,
  ]));
}

// Helpers ------------------------------------------------------------------
function numValue(value, spec) {
  const parsed = Number(value);
  const base = Number.isFinite(parsed) ? parsed : (spec.default ?? spec.min ?? 0);
  if (spec.min !== undefined && base < spec.min) return spec.min;
  if (spec.max !== undefined && base > spec.max) return spec.max;
  return base;
}

function pctOf(value, spec) {
  const min = spec.min ?? 0;
  const max = spec.max ?? 100;
  const ratio = max === min ? 0 : (Number(value) - min) / (max - min);
  return `${clamp(ratio, 0, 1) * 100}%`;
}

function unitLabel(spec) {
  return spec.unit ? el('span', { class: 'tiny faint nowrap', text: spec.unit }) : null;
}

function toArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined || value === '') return [];
  return [value];
}

function resolveOptions(spec, ctx) {
  let raw = null;
  if (typeof spec.options === 'function') raw = spec.options(ctx);
  else if (Array.isArray(spec.options)) raw = spec.options;
  else if (typeof spec.options === 'string') raw = (ctx.options && ctx.options[spec.options]) || [];
  else if (spec.optionsCategory) raw = state.models.models[spec.optionsCategory] || [];
  return (raw || []).map((item) => (typeof item === 'string' ? { value: item, label: item } : item));
}

/** Build one field from a spec. Returns null when the spec is filtered out. */
export function buildField(spec, ctx) {
  if (spec.when && !spec.when(ctx)) return null;
  const resolved = typeof spec.hint === 'function' ? String(spec.hint(ctx) ?? '') : spec.hint;
  spec = { ...spec, hint: resolved };
  const control = CONTROLS[spec.type] || CONTROLS.text;
  const node = control(spec, ctx);
  if (spec.id) node.id = spec.id;
  if (spec.class) node.classList.add(spec.class);
  node.dataset.field = spec.key || spec.type;
  if (spec.search !== false && spec.label) node.dataset.search = String(spec.label).toLowerCase();
  return node;
}

export function buildFields(specs, ctx) {
  const out = [];
  for (const spec of specs) {
    const node = buildField(spec, ctx);
    if (node) out.push(node);
  }
  return out;
}

// Upload plumbing ----------------------------------------------------------
export function wireUpload(node, onFile, { accept } = {}) {
  let dragging = 0;
  const hasFiles = (event) => Array.from(event.dataTransfer ? event.dataTransfer.types : []).includes('Files');
  node.addEventListener('dragenter', (event) => {
    if (!hasFiles(event)) return;
    dragging += 1;
    node.classList.add('drag');
  });
  node.addEventListener('dragover', (event) => { if (hasFiles(event)) { event.preventDefault(); node.classList.add('drag'); } });
  node.addEventListener('dragleave', () => {
    dragging = Math.max(0, dragging - 1);
    if (!dragging) node.classList.remove('drag');
  });
  node.addEventListener('drop', async (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragging = 0;
    node.classList.remove('drag');
    const files = Array.from(event.dataTransfer.files);
    for (const file of files) {
      if (accept && !accept(file)) continue;
      try { await onFile(file); } catch (err) { toast(`${file.name}: ${err.message}`, { kind: 'err' }); }
    }
  });
}

export { Combo, listEditor };
