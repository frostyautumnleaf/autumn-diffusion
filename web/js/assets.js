// Asset browser: the ComfyUI model, input, output, wildcard and music folders,
// reachable from the dashboard so files never have to be moved by hand.
//
// The server exposes one listing endpoint per (root, category, subfolder) and
// the same upload endpoint the prompt box uses, so dropping a LoRA on this view
// puts it in `engine/ComfyUI/models/loras` and the model dropdowns pick it up on
// the next refresh.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, menu, modal, confirmDialog, promptDialog, lightbox } from './components.js';
import { wireUpload } from './fields.js';
import { el, clear, bytes, dateTime, baseName, copyText, debounce, fileName, plural } from './util.js';

const ROOTS = [
  { id: 'models', label: 'Models', icon: 'layers' },
  { id: 'input', label: 'Input', icon: 'image' },
  { id: 'output', label: 'Output', icon: 'frame' },
  { id: 'wildcards', label: 'Wildcards', icon: 'shuffle' },
  { id: 'music', label: 'Music', icon: 'music' },
];

const TEXT_EXT = new Set(['.txt', '.json', '.md', '.csv', '.yaml', '.yml', '.py']);

const KIND_ICON = {
  model: 'box', image: 'image', video: 'film', audio: 'music', text: 'file', file: 'file',
};

/** Which form fields a model can be dropped into, per tab. */
const USE_KEYS = {
  checkpoints: { image: ['checkpoint'], video: ['checkpoint'] },
  diffusion_models: { image: ['unet'], video: ['model'] },
  unet: { image: ['unet'], video: ['model'] },
  vae: { image: ['vae'], video: ['vae'] },
  vae_approx: { image: ['vae'], video: ['vae'] },
  text_encoders: { image: ['clip'], video: ['clip'] },
  clip: { image: ['clip'], video: ['clip'] },
  upscale_models: { image: ['upscale_model'], video: ['upscale_model'] },
  seedvr2: { video: ['seedvr.dit_model'] },
  frame_interpolation: { video: ['interpolate.model'] },
};

function refreshModelLists() {
  window.dispatchEvent(new CustomEvent('autumn:refresh'));
}

export class Assets {
  constructor(root) {
    this.root = root;
    this.rootId = 'models';
    this.category = state.ui.assetsCategory || 'checkpoints';
    this.subfolder = state.ui.assetsSubfolder || '';
    this.query = '';
    this.view = 'grid';
    this.selected = null;
    this.categories = [];
    this.listing = { dirs: [], files: [], exists: false, path: '' };
    this.busy = false;

    this.render();
    this.loadCategories();
    this.loadList();

    state.on(TOPICS.ui, (patch) => {
      if (patch && patch.assetsCategory && patch.assetsCategory !== this.category) {
        this.category = patch.assetsCategory;
        this.subfolder = '';
        this.selected = null;
        this.loadList();
      }
    });
  }

  // ------------------------------------------------------------------ chrome
  render() {
    clear(this.root);
    this.root.classList.add('assets');
    this.rail = el('div', { class: 'assets-rail scroll' });
    this.main = el('div', { class: 'assets-main' });
    this.detail = el('div', { class: 'assets-detail' });
    this.root.append(this.rail, this.main, this.detail);
    this.renderRail();
    this.renderDetail();
  }

  renderRail() {
    clear(this.rail);
    this.rail.appendChild(el('div', { class: 'rail' }, [
      el('div', { class: 'rail-title', text: 'Folders' }),
      ...ROOTS.map((item) => el('div', {
        class: `cat-item ${this.rootId === item.id ? 'on' : ''}`,
        dataset: { root: item.id },
        onclick: () => this.setRoot(item.id),
      }, [
        el('span', { class: 'section-icon', html: icon(item.icon, 14) }),
        el('span', { class: 'grow trunc', text: item.label }),
      ])),
    ]));
    this.rail.appendChild(el('div', { class: 'sep' }));

    if (this.rootId === 'models') {
      this.rail.appendChild(el('div', { class: 'rail-title', text: 'Categories' }));
      this.categoryList = el('div', { class: 'col' });
      this.rail.appendChild(this.categoryList);
      this.paintCategories();
    } else {
      const path = (this.listing.path || '').replace(String(state.paths.root || ''), '').replace(/^\/+/, '');
      this.rail.appendChild(el('div', {
        class: 'tiny faint', style: { padding: '4px 8px', wordBreak: 'break-all' },
        text: path || `${this.rootId} folder`,
      }));
    }
  }

  paintCategories() {
    if (!this.categoryList) return;
    clear(this.categoryList);
    for (const item of this.categories) {
      this.categoryList.appendChild(el('div', {
        class: `cat-item ${this.category === item.key ? 'on' : ''}`,
        title: `${item.files} file${item.files === 1 ? '' : 's'} · ${item.folder}/`,
        onclick: () => {
          this.category = item.key;
          this.subfolder = '';
          this.selected = null;
          state.setUI({ assetsCategory: item.key });
          this.paintCategories();
          this.loadList();
        },
      }, [
        el('span', { class: 'section-icon', html: icon(categoryIcon(item.key), 14) }),
        el('span', { class: 'grow trunc', text: item.label }),
        el('span', { class: 'n mono tiny faint', text: item.files ? String(item.files) : '' }),
      ]));
    }
  }

  renderBar() {
    const bar = el('div', { class: 'assets-bar' });
    const crumbs = this.subfolder ? this.subfolder.split('/') : [];
    if (crumbs.length) {
      bar.appendChild(el('button', {
        class: 'icon-btn', title: 'Up one folder', html: icon('arrow_up', 14),
        onclick: () => {
          this.subfolder = crumbs.slice(0, -1).join('/');
          this.selected = null;
          this.loadList();
        },
      }));
    }
    bar.appendChild(el('span', {
      class: 'stage-title trunc',
      text: this.rootId === 'models'
        ? labelFor(this.category, this.categories)
        : (ROOTS.find((item) => item.id === this.rootId) || {}).label || '',
    }));
    if (crumbs.length) {
      bar.appendChild(el('span', { class: 'tiny faint mono trunc', text: crumbs.join(' / ') }));
    }
    bar.appendChild(el('span', { class: 'grow' }));

    bar.appendChild(el('input', {
      class: 'input sm', style: { width: '150px' }, placeholder: 'Filter…',
      spellcheck: 'false',
      oninput: debounce((event) => {
        this.query = event.target.value.trim().toLowerCase();
        this.renderList();
      }, 180),
    }));

    const action = (name, title, run) => el('button', {
      class: 'icon-btn', title, 'aria-label': title, html: icon(name, 14), onclick: run,
    });
    bar.appendChild(action('upload', 'Upload files here', () => this.pickFiles()));
    bar.appendChild(action('download', 'Fetch a model from a URL', () => this.fetchDialog()));
    if (this.rootId === 'models') {
      bar.appendChild(action('plus', 'New folder', () => this.newFolder()));
    }
    bar.appendChild(action(this.view === 'grid' ? 'list' : 'grid',
      this.view === 'grid' ? 'List view' : 'Grid view', () => {
        this.view = this.view === 'grid' ? 'list' : 'grid';
        this.renderMain();
      }));
    bar.appendChild(action('refresh', 'Re-read the folder', () => {
      this.loadCategories();
      this.loadList();
    }));
    return bar;
  }

  renderMain() {
    clear(this.main);
    this.main.appendChild(this.renderBar());

    this.drop = el('div', { class: 'assets-drop' }, [
      el('span', { html: icon('upload', 16) }),
      el('span', { text: ` Drop files here to add them to ${this.targetLabel()}` }),
    ]);
    this.main.appendChild(this.drop);
    wireUpload(this.drop, (file) => this.upload(file));
    wireUpload(this.main, (file) => this.upload(file));

    this.list = el('div', { class: 'assets-list' });
    this.main.appendChild(this.list);
    this.renderList();
    this.main.appendChild(this.renderFoot());
  }

  /** Footer: the folder on disk, what is in it, and whether it can be written. */
  renderFoot() {
    const files = this.listing.files || [];
    const dirs = this.listing.dirs || [];
    const size = this.listing.total_bytes
      || files.reduce((sum, file) => sum + (file.size || 0), 0);
    const path = this.listing.path || '';
    const readOnly = this.listing.exists && this.listing.writable === false;
    const parts = [
      files.length || dirs.length ? `${files.length} ${plural(files.length, 'file')}` : 'empty',
      dirs.length ? `${dirs.length} ${plural(dirs.length, 'folder')}` : '',
      size ? bytes(size) : '',
    ].filter(Boolean);
    return el('div', { class: 'assets-foot' }, [
      el('span', { class: 'mono tiny faint trunc', title: path, text: path }),
      el('span', { class: 'grow' }),
      el('span', { class: 'tiny faint', text: parts.join(' · ') }),
      readOnly
        ? el('span', {
          class: 'tiny', style: { color: 'var(--warn)' },
          title: 'The folder cannot be written by this process', text: 'read-only',
        })
        : null,
    ]);
  }

  targetLabel() {
    if (this.rootId === 'models') {
      const item = this.categories.find((entry) => entry.key === this.category);
      return (item ? item.folder : this.category) + (this.subfolder ? `/${this.subfolder}` : '');
    }
    return this.rootId + (this.subfolder ? `/${this.subfolder}` : '');
  }

  renderList() {
    if (!this.list) return;
    clear(this.list);
    const dirs = (this.listing.dirs || []).filter((item) => !this.query
      || item.name.toLowerCase().includes(this.query));
    const files = (this.listing.files || []).filter((item) => !this.query
      || item.name.toLowerCase().includes(this.query));

    if (!this.listing.exists) {
      this.list.appendChild(el('div', { class: 'empty' }, [
        el('b', { class: 'e-ic', html: icon('folder', 26) }),
        el('div', { text: 'That folder does not exist yet. Drop a file in it to create it.' }),
      ]));
      return;
    }
    if (!dirs.length && !files.length) {
      this.list.appendChild(el('div', { class: 'empty' }, [
        el('b', { class: 'e-ic', html: icon(this.query ? 'search' : 'folder', 26) }),
        el('div', { text: this.query ? 'Nothing matches that filter.' : 'Empty. Drop files here.' }),
      ]));
      return;
    }

    if (this.view === 'grid') {
      const grid = el('div', { class: 'card-grid' });
      for (const item of dirs) grid.appendChild(this.dirCard(item));
      for (const item of files) grid.appendChild(this.fileCard(item));
      this.list.appendChild(grid);
    } else {
      const table = el('table', { class: 'table' });
      table.appendChild(el('thead', {}, [el('tr', {}, [
        el('th', { text: 'Name' }),
        el('th', { class: 'num', text: 'Size' }),
        el('th', { class: 'num', text: 'Modified' }),
      ])]));
      const body = el('tbody', {});
      for (const item of dirs) body.appendChild(this.dirRow(item));
      for (const item of files) body.appendChild(this.fileRow(item));
      table.appendChild(body);
      this.list.appendChild(table);
    }
  }

  dirCard(item) {
    return el('div', {
      class: 'card',
      ondblclick: () => this.openFolder(item),
      onclick: () => this.select(item),
      oncontextmenu: (event) => {
        event.preventDefault();
        this.entryMenu(event.currentTarget, item, { x: event.clientX, y: event.clientY });
      },
    }, [
      el('div', { class: 'card-title' }, [
        el('span', { class: 'section-icon', html: icon('folder', 14) }),
        el('span', { class: 'trunc', text: item.name }),
      ]),
      el('div', { class: 'tiny faint', text: `${item.children || 0} files` }),
    ]);
  }

  dirRow(item) {
    return el('tr', {
      class: this.selected && this.selected.path === item.path ? 'sel' : '',
      ondblclick: () => this.openFolder(item),
      onclick: () => this.select(item),
      oncontextmenu: (event) => {
        event.preventDefault();
        this.entryMenu(event.currentTarget, item, { x: event.clientX, y: event.clientY });
      },
    }, [
      el('td', { class: 'name' }, [
        el('span', { class: 'section-icon', html: icon('folder', 13) }),
        el('span', { text: item.name }),
      ]),
      el('td', { class: 'num faint', text: `${item.children || 0}` }),
      el('td', { class: 'num faint', text: dateTime(item.modified) }),
    ]);
  }

  fileCard(item) {
    const thumb = item.url && ['image', 'video', 'audio'].includes(item.kind)
      ? el('div', {
        class: 'file-preview',
        style: { minHeight: '86px', marginBottom: '8px' },
        html: item.kind === 'video'
          ? `<video src="${item.url}" muted preload="metadata"></video>`
          : item.kind === 'audio'
            ? `<span style="padding:12px;color:var(--text-mute)">${icon('music', 22)}</span>`
            : `<img src="${item.url}" alt="" loading="lazy">`,
      })
      : null;
    return el('div', {
      class: `card ${this.selected && this.selected.path === item.path ? 'sel' : ''}`,
      onclick: () => this.select(item),
      ondblclick: () => this.previewMedia(item),
      oncontextmenu: (event) => {
        event.preventDefault();
        this.entryMenu(event.currentTarget, item, { x: event.clientX, y: event.clientY });
      },
    }, [
      thumb,
      el('div', { class: 'card-title' }, [
        el('span', { class: 'section-icon', html: icon(KIND_ICON[item.kind] || 'file', 13) }),
        el('span', { class: 'trunc', text: item.name }),
      ]),
      el('div', { class: 'tiny faint mono', text: item.size ? bytes(item.size) : '—' }),
    ]);
  }

  fileRow(item) {
    return el('tr', {
      class: this.selected && this.selected.path === item.path ? 'sel' : '',
      onclick: () => this.select(item),
      ondblclick: () => this.previewMedia(item),
      oncontextmenu: (event) => {
        event.preventDefault();
        this.entryMenu(event.currentTarget, item, { x: event.clientX, y: event.clientY });
      },
    }, [
      el('td', { class: 'name' }, [
        el('span', { class: 'section-icon', html: icon(KIND_ICON[item.kind] || 'file', 13) }),
        el('span', { text: item.name }),
      ]),
      el('td', { class: 'num', text: item.size ? bytes(item.size) : '—' }),
      el('td', { class: 'num faint', text: dateTime(item.modified) }),
    ]);
  }

  // ----------------------------------------------------------------- detail
  async select(item) {
    this.selected = item;
    this.renderList();
    this.renderDetail();
    if (item && !item.is_dir && item.kind === 'model') {
      try {
        const data = await api.assetAction('stat', {
          category: this.category, path: item.path, root: this.rootId });
        if (this.selected === item) this.stat = data.stat;
      } catch (err) {
        this.stat = null;
      }
      if (this.selected === item) this.renderDetail();
    }
  }

  renderDetail() {
    clear(this.detail);
    const item = this.selected;
    if (!item) {
      this.detail.appendChild(el('div', { class: 'empty' }, [
        el('b', { class: 'e-ic', html: icon('info', 24) }),
        el('div', { text: 'Pick a file to see what it is and where it goes.' }),
      ]));
      return;
    }
    if (item.is_dir) {
      this.detail.appendChild(el('div', { class: 'card-title' }, [
        el('span', { class: 'section-icon', html: icon('folder', 14) }),
        el('span', { class: 'trunc', text: item.name }),
      ]));
      this.detail.appendChild(el('div', { class: 'tiny faint', text: `${item.children || 0} files inside` }));
      this.detail.appendChild(el('div', { class: 'sep' }));
      this.detail.appendChild(el('button', {
        class: 'btn sm block', html: `${icon('arrow_right', 12)}<span>Open</span>`,
        onclick: () => this.openFolder(item),
      }));
      this.detail.appendChild(el('button', {
        class: 'btn sm block ghost', html: `${icon('save', 12)}<span>Rename folder</span>`,
        onclick: () => this.rename(item),
      }));
      this.detail.appendChild(el('button', {
        class: 'btn sm block danger', html: `${icon('trash', 12)}<span>Delete folder</span>`,
        onclick: () => this.remove(item),
      }));
      return;
    }

    if (item.url && item.kind === 'image') {
      this.detail.appendChild(el('div', { class: 'file-preview' }, [
        el('img', { src: item.url, alt: item.name, loading: 'lazy',
          onclick: () => this.previewMedia(item) }),
      ]));
    } else if (item.url && item.kind === 'video') {
      this.detail.appendChild(el('div', { class: 'file-preview' }, [
        el('video', { src: item.url, controls: true, muted: true, preload: 'metadata' }),
      ]));
    } else if (item.url && item.kind === 'audio') {
      this.detail.appendChild(el('div', { class: 'file-preview', style: { display: 'block', padding: '10px' } }, [
        el('audio', { src: item.url, controls: true, preload: 'metadata', style: { width: '100%' } }),
      ]));
    }

    this.detail.appendChild(el('div', { class: 'card-title' }, [
      el('span', { class: 'section-icon', html: icon(KIND_ICON[item.kind] || 'file', 14) }),
      el('span', { class: 'trunc', text: item.name }),
    ]));
    this.detail.appendChild(el('dl', { class: 'kv' }, [
      el('dt', { text: 'Size' }), el('dd', { text: item.size ? bytes(item.size, 2) : '—' }),
      el('dt', { text: 'Modified' }), el('dd', { text: dateTime(item.modified) }),
      el('dt', { text: 'Type' }), el('dd', { text: (item.ext || '').replace('.', '') || item.kind }),
      el('dt', { text: 'Folder' }), el('dd', { class: 'mono', text: this.targetLabel() }),
    ]));

    const meta = this.stat && this.stat.safetensors && !this.stat.safetensors.error
      ? this.stat.safetensors : null;
    if (meta) {
      const tensors = Object.entries(meta.tensors || {}).slice(0, 6);
      const rows = [];
      if (meta.total_dtype) rows.push(el('dt', { text: 'Dtype' }), el('dd', { text: String(meta.total_dtype) }));
      for (const [name, spec] of tensors) {
        rows.push(el('dt', { text: name.length > 22 ? `${name.slice(0, 21)}…` : name }),
          el('dd', { class: 'mono', text: `${(spec.dtype || '').replace('F', '')} ${(spec.shape || []).join('×')}` }));
      }
      if (rows.length) {
        this.detail.appendChild(el('div', { class: 'sep' }));
        this.detail.appendChild(el('div', { class: 'sub-label', text: 'Safetensors' }));
        this.detail.appendChild(el('dl', { class: 'kv' }, rows));
      }
    }

    this.detail.appendChild(el('div', { class: 'sep' }));
    const use = this.useTargets(item);
    for (const entry of use) {
      this.detail.appendChild(el('button', {
        class: 'btn sm block primary',
        html: `${icon('wand', 12)}<span>Use in ${entry.label}</span>`,
        onclick: () => this.useHere(item, entry),
      }));
    }
    if (TEXT_EXT.has(item.ext)) {
      this.detail.appendChild(el('button', {
        class: 'btn sm block', html: `${icon('code', 12)}<span>Edit text</span>`,
        onclick: () => this.editText(item),
      }));
    }
    if (item.url) {
      this.detail.appendChild(el('button', {
        class: 'btn sm block', html: `${icon('download', 12)}<span>Download</span>`,
        onclick: () => this.download(item),
      }));
    }
    this.detail.appendChild(el('button', {
      class: 'btn sm block ghost', html: `${icon('copy', 12)}<span>Copy path</span>`,
      onclick: async () => {
        await copyText(item.rel || item.path);
        toast('Path copied', { kind: 'ok', timeout: 1400 });
      },
    }));
    this.detail.appendChild(el('button', {
      class: 'btn sm block ghost', html: `${icon('save', 12)}<span>Rename</span>`,
      onclick: () => this.rename(item),
    }));
    this.detail.appendChild(el('button', {
      class: 'btn sm block ghost', html: `${icon('branch', 12)}<span>Move to…</span>`,
      onclick: () => this.move(item),
    }));
    this.detail.appendChild(el('button', {
      class: 'btn sm block danger', html: `${icon('trash', 12)}<span>Delete</span>`,
      onclick: () => this.remove(item),
    }));
  }

  useTargets(item) {
    const keys = USE_KEYS[this.category] || {};
    const out = [];
    if (keys.image) out.push({ tab: 'image', fields: keys.image, label: 'Image tab' });
    if (keys.video) out.push({ tab: 'video', fields: keys.video, label: 'Video tab' });
    if (this.category === 'loras') out.push({ tab: 'image', lora: true, label: 'Image LoRAs' },
      { tab: 'video', lora: true, label: 'Video LoRAs' });
    if (this.category === 'embeddings') out.push({ tab: 'image', embedding: true, label: 'Image embeddings' },
      { tab: 'video', embedding: true, label: 'Video embeddings' });
    if (this.category === 'controlnet') out.push({ tab: 'image', controlnet: true, label: 'Image ControlNet' },
      { tab: 'video', controlnet: true, label: 'Video ControlNet' });
    return out;
  }

  useHere(item, entry) {
    const name = item.path || item.name;
    const form = state.form(entry.tab);
    if (entry.lora) {
      state.setForm(entry.tab, {
        loras: (form.loras || []).concat([{ name, strength_model: 1, strength_clip: 1, model_only: false, enabled: true }]),
      });
    } else if (entry.embedding) {
      state.setForm(entry.tab, {
        embeddings: (form.embeddings || []).concat([{ name, strength: 1, enabled: true }]),
      });
    } else if (entry.controlnet) {
      state.setForm(entry.tab, {
        controlnets: (form.controlnets || []).concat(
          [{ name, image: '', strength: 1, start_percent: 0, end_percent: 1, enabled: true }]),
      });
    } else {
      const patch = {};
      for (const field of entry.fields) patch[field] = name;
      state.setForm(entry.tab, patch);
    }
    state.setUI({ tab: entry.tab });
    toast(`${baseName(name)} set in the ${entry.tab} tab`, { kind: 'ok' });
  }

  // ---------------------------------------------------------------- actions
  setRoot(rootId) {
    if (this.rootId === rootId) return;
    this.rootId = rootId;
    this.subfolder = '';
    this.selected = null;
    this.stat = null;
    if (rootId === 'models') this.category = state.ui.assetsCategory || 'checkpoints';
    this.renderRail();
    this.renderMain();
    this.loadList();
  }

  openFolder(item) {
    this.subfolder = item.path;
    this.selected = null;
    this.loadList();
  }

  pickFiles() {
    const input = el('input', { type: 'file', multiple: true, style: { display: 'none' } });
    document.body.appendChild(input);
    input.addEventListener('change', async () => {
      const files = Array.from(input.files || []);
      input.remove();
      for (const file of files) await this.upload(file);
    });
    input.click();
  }

  async upload(file) {
    const category = this.rootId === 'models' ? this.category : '';
    const label = toast(`Uploading ${file.name}…`, { kind: 'info', timeout: 0 });
    try {
      await api.upload(file, { category: category || 'input', subfolder: this.subfolder });
      label.dismiss();
      toast(`${file.name} added`, { kind: 'ok', timeout: 2200 });
      refreshModelLists();
      this.loadCategories();
      this.loadList();
    } catch (err) {
      label.dismiss();
      toast(`${file.name}: ${err.message}`, { kind: 'err', timeout: 6000 });
    }
  }

  fetchDialog() {
    let url = '';
    let name = '';
    modal({
      title: 'Fetch a model',
      icon: 'download',
      width: 540,
      body: (node) => {
        node.appendChild(el('div', { class: 'field' }, [
          el('div', { class: 'field-top' }, [el('span', { class: 'field-label', text: 'URL' })]),
          el('input', {
            class: 'input', placeholder: 'https://huggingface.co/…/resolve/main/model.safetensors',
            oninput: (event) => { url = event.target.value; },
          }),
          el('div', { class: 'field-help', text: 'Hugging Face resolve links and direct file URLs both work. '
            + 'Private repos need the token in the URL.' }),
        ]));
        node.appendChild(el('div', { class: 'field' }, [
          el('div', { class: 'field-top' }, [el('span', { class: 'field-label', text: 'File name (optional)' })]),
          el('input', {
            class: 'input', placeholder: 'taken from the URL',
            oninput: (event) => { name = event.target.value; },
          }),
        ]));
        node.appendChild(el('div', { class: 'field-help',
          text: `Saving into ${this.targetLabel()}` }));
      },
      actions: [
        { label: 'Cancel', kind: 'ghost', run: (close) => close(null) },
        {
          label: 'Download', kind: 'primary', primary: true,
          run: async (close) => {
            if (!url.trim()) { toast('A URL is required', { kind: 'warn' }); return; }
            close(null);
            const label = toast('Downloading…', { kind: 'info', timeout: 0 });
            try {
              const result = await api.assetAction('download', {
                url: url.trim(), name: name.trim(),
                category: this.rootId === 'models' ? this.category : 'checkpoints',
                subfolder: this.rootId === 'models' ? this.subfolder : '',
              });
              label.dismiss();
              toast(`Saved ${fileName(result.path || url)}`, { kind: 'ok' });
              refreshModelLists();
              this.loadCategories();
              this.loadList();
            } catch (err) {
              label.dismiss();
              toast(`Download failed: ${err.message}`, { kind: 'err', timeout: 8000 });
            }
          },
        },
      ],
    });
  }

  async newFolder() {
    const name = await promptDialog({ title: 'New folder', label: 'Name', placeholder: 'my-style-collection' });
    if (!name) return;
    try {
      await api.assetAction('mkdir', {
        category: this.category, name, path: this.subfolder, root: 'models' });
      toast('Folder created', { kind: 'ok', timeout: 1600 });
      this.loadList();
      this.loadCategories();
    } catch (err) {
      toast(`Could not create the folder: ${err.message}`, { kind: 'err' });
    }
  }

  async rename(item) {
    const next = await promptDialog({ title: 'Rename', label: 'New name', value: item.name });
    if (!next || next === item.name) return;
    try {
      await api.assetAction('rename', {
        category: this.category, path: item.path, new_name: next, root: this.rootId });
      toast('Renamed', { kind: 'ok', timeout: 1600 });
      this.selected = null;
      this.loadList();
      refreshModelLists();
    } catch (err) {
      toast(`Rename failed: ${err.message}`, { kind: 'err' });
    }
  }

  async move(item) {
    const target = await chooseCategory(this.categories, this.rootId, this.category);
    if (!target) return;
    try {
      await api.assetAction('move', {
        category: this.category, path: item.path, to_category: target.category,
        to_subfolder: target.subfolder || '', root: this.rootId });
      toast('Moved', { kind: 'ok', timeout: 1600 });
      this.selected = null;
      this.loadList();
      this.loadCategories();
      refreshModelLists();
    } catch (err) {
      toast(`Move failed: ${err.message}`, { kind: 'err' });
    }
  }

  async remove(item) {
    const ok2 = await confirmDialog({
      title: item.is_dir ? 'Delete folder' : 'Delete file',
      message: `${item.name} moves to ${this.targetLabel()}/.trash. It is not removed from disk, `
        + 'and ComfyUI stops seeing it straight away.',
      confirmLabel: 'Delete',
    });
    if (!ok2) return;
    try {
      await api.deleteAsset(this.category, item.path, this.rootId, false);
      toast(`${item.name} moved to trash`, { kind: 'ok', timeout: 2000 });
      this.selected = null;
      this.stat = null;
      this.loadList();
      this.loadCategories();
      refreshModelLists();
    } catch (err) {
      toast(`Delete failed: ${err.message}`, { kind: 'err' });
    }
  }

  async editText(item) {
    let text = '';
    try {
      const data = await api.assetAction('text', {
        category: this.category, path: item.path, root: this.rootId });
      text = (data.text && data.text.text) || '';
    } catch (err) {
      toast(`Could not read the file: ${err.message}`, { kind: 'err' });
      return;
    }
    let box = null;
    modal({
      title: item.name,
      icon: 'code',
      width: 720,
      body: (node) => {
        box = el('textarea', {
          class: 'textarea', rows: 18, spellcheck: 'false',
          style: { fontFamily: 'var(--mono)', fontSize: '11.5px' },
        });
        box.value = text;
        node.appendChild(box);
        if (this.category === 'embeddings' || item.ext === '.txt') {
          node.appendChild(el('div', { class: 'field-help',
            text: 'Wildcards: one entry per line, blank lines ignored, # starts a comment.' }));
        }
      },
      actions: [
        { label: 'Cancel', kind: 'ghost', run: (close) => close(null) },
        {
          label: 'Save', kind: 'primary', primary: true,
          run: async (close) => {
            try {
              await api.writeAsset(this.category, item.path, box.value, this.rootId);
              close(null);
              toast('Saved', { kind: 'ok', timeout: 1600 });
              refreshModelLists();
            } catch (err) {
              toast(`Save failed: ${err.message}`, { kind: 'err' });
            }
          },
        },
      ],
    });
  }

  previewMedia(item) {
    if (!item.url) return;
    if (item.kind === 'text') {
      this.editText(item);
      return;
    }
    if (!['image', 'video', 'audio'].includes(item.kind)) return;
    lightbox({
      title: item.name,
      subtitle: this.targetLabel(),
      url: item.url,
      kind: item.kind === 'video' ? 'video' : 'image',
      meta: [['size', bytes(item.size, 2)], ['modified', dateTime(item.modified)]],
    });
  }

  download(item) {
    const anchor = el('a', { href: item.url, download: item.name });
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  entryMenu(anchor, item, at = null) {
    menu(anchor, [
      { label: item.is_dir ? 'Open' : 'Preview', icon: item.is_dir ? 'arrow_right' : 'expand',
        disabled: item.is_dir ? false : !item.url,
        run: () => (item.is_dir ? this.openFolder(item) : this.previewMedia(item)) },
      { label: 'Rename', icon: 'save', run: () => this.rename(item) },
      { label: 'Move to…', icon: 'branch', run: () => this.move(item) },
      { label: 'Copy path', icon: 'copy', run: async () => {
        await copyText(item.rel || item.path);
        toast('Path copied', { kind: 'ok', timeout: 1400 });
      } },
      { sep: true },
      { label: 'Delete', icon: 'trash', danger: true, run: () => this.remove(item) },
    ], at ? { at } : {});
  }

  // ------------------------------------------------------------------- data
  async loadCategories() {
    try {
      const data = await api.categories();
      this.categories = data.categories || [];
      this.paintCategories();
    } catch (err) {
      this.categories = [];
    }
  }

  async loadList() {
    try {
      const data = await api.assets(
        this.rootId === 'models' ? this.category : 'other', this.subfolder, this.rootId);
      this.listing = data;
      this.stat = null;
      this.renderRail();
      this.renderMain();
    } catch (err) {
      this.listing = { dirs: [], files: [], exists: false, path: '' };
      this.renderMain();
    }
  }
}

// Category picker used by "Move to…" ---------------------------------------
function chooseCategory(categories, rootId, current) {
  return new Promise((resolve) => {
    const choice = { category: current, subfolder: '' };
    const options = categories.filter((item) => item.key !== current);
    let sub = null;
    modal({
      title: 'Move to another folder',
      icon: 'branch',
      width: 460,
      body: (node) => {
        const select = el('select', { class: 'select' }, [
          el('option', { value: current, text: `${labelFor(current, categories)} (current)` }),
          ...options.map((item) => el('option', { value: item.key, text: item.label })),
        ]);
        select.onchange = (event) => { choice.category = event.target.value; };
        sub = el('input', { class: 'input', placeholder: 'subfolder (optional)',
          oninput: (event) => { choice.subfolder = event.target.value; } });
        node.appendChild(el('div', { class: 'field-grid' }, [
          el('div', { class: 'field' }, [
            el('div', { class: 'field-top' }, [el('span', { class: 'field-label', text: 'Category' })]),
            select,
          ]),
          el('div', { class: 'field' }, [
            el('div', { class: 'field-top' }, [el('span', { class: 'field-label', text: 'Subfolder' })]),
            sub,
          ]),
        ]));
      },
      actions: [
        { label: 'Cancel', kind: 'ghost', run: (close) => close(null) },
        { label: 'Move', kind: 'primary', primary: true, run: (close) => close(choice) },
      ],
    });
  });
}

function labelFor(key, categories) {
  const item = categories.find((entry) => entry.key === key);
  return item ? item.label : key.replace(/_/g, ' ');
}

function categoryIcon(key) {
  if (key === 'loras') return 'layers';
  if (key === 'vae' || key === 'vae_approx') return 'frame';
  if (key === 'controlnet') return 'mask';
  if (key === 'embeddings') return 'target';
  if (key === 'upscale_models') return 'zoom_in';
  if (key === 'text_encoders' || key === 'clip') return 'code';
  if (key === 'seedvr2') return 'sparkles';
  if (key === 'audio_encoders') return 'mic';
  return 'box';
}

export default Assets;
