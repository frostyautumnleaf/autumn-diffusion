// Settings tab: engine control, ComfyUI tuning, live preview, optional
// installs (SeedVR2, KJNodes/taeh3), custom nodes, music engine, interface,
// paths, system meters and maintenance.
//
// Every field writes straight into the server settings document. Values are
// merged optimistically into `state.settings`, then flushed to
// POST /api/settings as a nested patch after a short idle, so a slider drag
// costs one request rather than one per pixel.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { buildField, getPath } from './fields.js';
import { icon } from './icons.js';
import { toast, confirmDialog, modal, menu } from './components.js';
import { el, clear, bytes, seconds as secs, timeAgo, copyText, debounce, clamp } from './util.js';

const ACCENTS = [
  { value: 'amber', label: 'Amber', color: '#f0a24b' },
  { value: 'ember', label: 'Ember', color: '#ef6d48' },
  { value: 'moss', label: 'Moss', color: '#7cc47f' },
  { value: 'violet', label: 'Violet', color: '#a78bfa' },
  { value: 'cyan', label: 'Cyan', color: '#5cc8d7' },
];

const PREVIEW_METHODS = [
  { value: 'latent2rgb', label: 'latent2rgb (fast, approximate)' },
  { value: 'taesd', label: 'taesd (SD decoder)' },
  { value: 'taehv', label: 'taehv (HV decoder)' },
  { value: 'auto', label: 'auto' },
  { value: 'none', label: 'off' },
];

const VRAM_MODES = [
  { value: '', label: 'auto' },
  { value: 'gpu-only', label: '--gpu-only' },
  { value: 'highvram', label: '--highvram' },
  { value: 'lowvram', label: '--lowvram' },
  { value: 'novram', label: '--novram' },
];

const ATTENTION = [
  { value: '', label: 'auto' },
  { value: 'split', label: 'split' },
  { value: 'quad', label: 'quad' },
  { value: 'math', label: 'math' },
  { value: 'pytorch', label: 'pytorch' },
  { value: 'xformers', label: 'xformers' },
  { value: 'sage', label: 'sage' },
];

const DTYPES = [
  { value: '', label: 'auto' },
  { value: 'bf16', label: 'bf16' },
  { value: 'fp16', label: 'fp16' },
  { value: 'fp32', label: 'fp32' },
];

const CACHE_MODES = [
  { value: '', label: 'default' },
  { value: 'lru', label: 'LRU cache' },
  { value: 'classic', label: 'classic cache' },
  { value: 'ram', label: 'RAM cache' },
  { value: 'none', label: 'no cache' },
];

const SEEDVR_URL = 'https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler';
const KJNODES_URL = 'https://github.com/kijai/ComfyUI-KJNodes';

// Block registry -----------------------------------------------------------
const BLOCKS = [
  { id: 'engine', label: 'ComfyUI engine', icon: 'power' },
  { id: 'tuning', label: 'Engine tuning', icon: 'cpu' },
  { id: 'preview', label: 'Live preview', icon: 'eye' },
  { id: 'seedvr', label: 'SeedVR2 upscaler', icon: 'sparkles' },
  { id: 'nodes', label: 'Custom nodes', icon: 'layers' },
  { id: 'music', label: 'Music engine', icon: 'music', when: () => state.features.music },
  { id: 'features', label: 'Features and installs', icon: 'rocket' },
  { id: 'interface', label: 'Interface', icon: 'palette' },
  { id: 'paths', label: 'Paths and storage', icon: 'folder' },
  { id: 'system', label: 'System', icon: 'gauge' },
  { id: 'maintenance', label: 'Maintenance', icon: 'flask' },
];

// Small builders -----------------------------------------------------------
function pill(kind, text) {
  return el('span', { class: `pill ${kind || ''}` }, [el('span', { class: 'dot' }), el('span', { text })]);
}

function statCell(label, value, unit = '') {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'k', text: label }),
    el('div', { class: 'v' }, [String(value), unit ? el('span', { class: 'u', text: ` ${unit}` }) : null]),
  ]);
}

function bar(pct, kind = '') {
  return el('div', { class: `bar ${kind}` }, [el('i', { style: { width: `${clamp(pct, 0, 100)}%` } })]);
}

function kv(pairs) {
  const list = el('dl', { class: 'kv' });
  for (const [key, value] of pairs) {
    if (value === null || value === undefined || value === '') continue;
    list.appendChild(el('dt', { text: key }));
    list.appendChild(el('dd', { text: String(value) }));
  }
  return list;
}

function button(label, { iconName = '', kind = '', run, title = '', disabled = false } = {}) {
  return el('button', {
    class: `btn sm ${kind}`.trim(),
    disabled,
    title: title || label,
    html: `${iconName ? icon(iconName, 13) : ''}<span>${label}</span>`,
    onclick: (event) => run(event),
  });
}

function note(text, kind = '') {
  return el('div', {
    class: `tiny ${kind || 'faint'}`,
    style: { padding: '7px 0' },
    text,
  });
}

// Settings view ------------------------------------------------------------
export class SettingsView {
  constructor(root, { nav = null } = {}) {
    this.root = root;
    this.nav = nav;
    this.args = [];
    this.loaded = false;
    this.pending = {};
    this.restartPending = false;
    this.shapes = new Map();

    this.ctx = {
      kind: 'settings',
      get: (key) => getPath(state.settings, key),
      set: (key, value) => this.write(key, value),
      refresh: () => this.render(),
      options: {
        samplers: state.models.samplers,
        schedulers: state.models.schedulers,
        seedvr_models: state.models.seedvr_models || [],
        seedvr_vae_models: state.models.seedvr_vae_models || [],
      },
    };

    this.flush = debounce(() => this.save(), 420);
    this.render();
    this.loadEngine();

    state.on(TOPICS.settings, () => this.render());
    state.on(TOPICS.state, () => this.render());
    state.on(TOPICS.ui, () => this.render());
  }

  // ------------------------------------------------------------------ state
  async loadEngine() {
    try {
      const data = await api.engine();
      this.args = data.args || [];
      if (data.engine) state.engine = { ...state.engine, ...data.engine };
      if (data.nodes) state.nodes = data.nodes;
      if (data.enhancements) state.enhancements = { ...state.enhancements, ...data.enhancements };
      if (data.capabilities) state.capabilities = data.capabilities;
      this.loaded = true;
      this.render();
    } catch (err) {
      this.loaded = true;
      this.render();
    }
  }

  write(key, value) {
    const patch = {};
    (function set(target, parts, val) {
      const head = parts[0];
      if (parts.length === 1) target[head] = val;
      else {
        target[head] = target[head] && typeof target[head] === 'object' ? target[head] : {};
        set(target[head], parts.slice(1), val);
      }
    })(patch, String(key).split('.'), value);

    state.setSettings(patch);
    for (const [section, body] of Object.entries(patch)) {
      this.pending[section] = deepMerge(this.pending[section] || {}, body);
    }
    if (key === 'app.reduced_motion') {
      document.body.dataset.reducedMotion = value ? '1' : '0';
    }
    this.flush();
  }

  async save() {
    const patch = this.pending;
    this.pending = {};
    if (!Object.keys(patch).length) return;
    try {
      const data = await api.saveSettings(patch);
      state.setSettings(data.settings);
      if (data.problems) state.problems = data.problems;
      if (data.restart_required) {
        this.restartPending = true;
        this.loadEngine();
      }
      this.render();
    } catch (err) {
      toast(`Settings not saved: ${err.message}`, { kind: 'err', timeout: 8000 });
      this.render();
    }
  }

  // ----------------------------------------------------------------- render
  render() {
    const active = document.activeElement;
    const restoreKey = active && active.dataset ? active.dataset.field : null;
    const restoreStart = active && typeof active.selectionStart === 'number' ? active.selectionStart : null;
    const scrollTop = this.root.scrollTop;

    clear(this.root);
    const inner = el('div', { class: 'settings-inner' });
    inner.appendChild(this.renderIntro());
    for (const block of BLOCKS) {
      if (block.when && !block.when()) continue;
      inner.appendChild(this.renderBlock(block));
    }
    this.root.appendChild(inner);
    this.renderNav();

    this.root.scrollTop = scrollTop;
    if (restoreKey && document.activeElement !== active) {
      const node = this.root.querySelector(`[data-field="${restoreKey}"] input, [data-field="${restoreKey}"] textarea, [data-field="${restoreKey}"] select`);
      if (node) {
        node.focus();
        if (restoreStart !== null && node.setSelectionRange) {
          try { node.setSelectionRange(restoreStart, restoreStart); } catch (err) { /* number input */ }
        }
      }
    }
  }

  renderIntro() {
    const problems = state.problems || [];
    return el('div', { class: 'settings-intro' }, [
      el('div', { class: 'row' }, [
        el('div', {}, [
          el('div', { class: 's-title', text: 'Autumn Diffusion' }),
          el('div', {
            class: 'tiny faint',
            text: `Dashboard ${state.version || 'dev'} · everything lives inside the program folder`,
          }),
        ]),
        el('span', { class: 'grow' }),
        button('Console', { iconName: 'terminal', run: () => state.setUI({ logsOpen: true }) }),
      ]),
      problems.length
        ? el('div', { class: 'alert-box' }, problems.map((problem) => el('div', { text: problem })))
        : null,
      this.restartPending
        ? el('div', { class: 'alert-box' }, [
          el('div', { text: 'Engine settings changed. Restart ComfyUI to apply them.' }),
          button('Restart engine', {
            iconName: 'refresh', kind: 'primary',
            run: () => this.engineAction('restart', {}, 'Engine restarted'),
          }),
        ])
        : null,
    ]);
  }

  renderBlock(block) {
    const node = el('section', { class: 'settings-block', id: `settings-block-${block.id}` });
    const head = el('h3', {}, [
      el('span', { class: 'section-icon', html: icon(block.icon || 'sliders_h', 15) }),
      el('span', { text: block.label }),
    ]);
    const actions = el('span', { class: 'grow' });
    head.appendChild(actions);
    node.appendChild(head);
    const body = el('div', { class: 'block-body' });
    const renderer = this[`block_${block.id}`];
    if (renderer) body.appendChild(renderer.call(this, actions));
    node.appendChild(body);
    return node;
  }

  renderNav() {
    if (!this.nav) return;
    clear(this.nav);
    const list = el('div', { class: 'nav-list' });
    for (const block of BLOCKS) {
      if (block.when && !block.when()) continue;
      list.appendChild(el('button', {
        class: 'nav-item',
        html: `${icon(block.icon || 'sliders_h', 14)}<span>${block.label}</span>`,
        onclick: () => {
          const target = this.root.querySelector(`#settings-block-${block.id}`);
          if (target) this.root.scrollTo({ top: target.offsetTop - 8, behavior: 'smooth' });
        },
      }));
    }
    this.nav.appendChild(list);
  }

  // ------------------------------------------------------------------ blocks
  block_engine(actions) {
    const e = state.engine || {};
    const box = el('div', { class: 'col', style: { gap: '10px' } });

    actions.appendChild(button('Refresh', {
      iconName: 'refresh', kind: 'ghost',
      run: () => { this.loadEngine(); toast('Engine info refreshed', { kind: 'ok', timeout: 1600 }); },
    }));

    if (!e.installed) {
      box.appendChild(el('div', { class: 'card' }, [
        el('div', { class: 'card-title', html: `${icon('warning', 14)}<span>ComfyUI is not installed yet</span>` }),
        el('div', {
          class: 'tiny faint',
          text: 'The installer clones comfy-org/ComfyUI into engine/ComfyUI and installs its '
            + 'requirements into the program venv. Nothing is installed system-wide.',
        }),
        el('div', { class: 'row', style: { marginTop: '10px' } }, [
          button('Install ComfyUI', {
            iconName: 'download', kind: 'primary',
            run: () => this.engineAction('install', {}, 'ComfyUI installed'),
          }),
        ]),
      ]));
      return box;
    }

    const running = !!e.running;
    const online = !!(e.online || e.connected);
    box.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'row', style: { gap: '10px', marginBottom: '10px' } }, [
        pill(running ? (online ? 'ok' : 'warn') : '', running ? (online ? 'Engine online' : 'Starting') : 'Engine stopped'),
        el('span', { class: 'mono tiny faint', text: e.url || '' }),
        el('span', { class: 'grow' }),
        e.demo ? pill('warn', 'demo mode') : null,
      ]),
      el('div', { class: 'row', style: { gap: '6px', flexWrap: 'wrap' } }, [
        running
          ? button('Stop', {
            iconName: 'stop',
            run: () => this.engineAction('stop', {}, 'Engine stopped'),
          })
          : button('Start', {
            iconName: 'power', kind: 'primary',
            run: () => this.engineAction('start', {}, 'Engine starting'),
          }),
        button('Restart', {
          iconName: 'refresh',
          run: () => this.engineAction('restart', {}, 'Engine restarted'),
        }),
        online ? button('Interrupt', {
          iconName: 'x',
          run: () => this.engineAction('interrupt', {}, 'Current job interrupted'),
        }) : null,
        online ? button('Free VRAM', {
          iconName: 'memory',
          run: () => this.freeVram(),
        }) : null,
        online ? button('Reload nodes', {
          iconName: 'layers',
          run: () => this.engineAction('refresh_nodes', {}, 'Node list reloaded'),
        }) : null,
        el('a', {
          class: 'btn sm ghost',
          href: e.url || '#',
          target: '_blank',
          rel: 'noreferrer',
          html: `${icon('external', 13)}<span>Open engine UI</span>`,
        }),
      ]),
      el('div', { class: 'sep' }),
      kv([
        ['Version', e.version || 'unknown'],
        ['Git ref', e.git_ref || ''],
        ['Latest', e.latest_version || 'not checked'],
        ['Process', e.pid ? `pid ${e.pid}` : 'not running'],
        ['Up for', e.started_at ? secs((Date.now() / 1000) - e.started_at) : '—'],
        ['Nodes', `${(state.nodes || []).length} custom node packs`],
        ['Last error', e.last_error || ''],
      ]),
    ]));

    // Version / update card -------------------------------------------------
    const checkedAt = Number(getPath(state.settings, 'update.checked_at') || 0);
    const updateRow = el('div', { class: 'row', style: { gap: '6px', flexWrap: 'wrap', marginTop: '10px' } }, [
      e.update_available
        ? button(`Update to ${e.latest_version}`, {
          iconName: 'arrow_down', kind: 'primary',
          run: () => this.updateComfyui(),
        })
        : button('Check for updates', {
          iconName: 'refresh',
          run: () => this.checkUpdate(),
        }),
      button('Update now', {
        iconName: 'download', kind: e.update_available ? '' : 'ghost',
        disabled: !e.update_available,
        run: () => this.updateComfyui(),
      }),
    ]);
    box.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'card-title', html: `${icon('github', 14)}<span>ComfyUI updates</span>` }),
      el('div', { class: 'stat-grid' }, [
        statCell('Installed', e.version || '—'),
        statCell('Latest', e.latest_version || (e.update_available ? e.latest_version : 'unknown')),
        statCell('Checked', checkedAt ? timeAgo(checkedAt) : 'never'),
      ]),
      e.update_available
        ? el('div', { class: 'alert-box', style: { marginTop: '10px' } }, [
          el('div', { text: `ComfyUI ${e.latest_version} is available (running ${e.version}).` }),
        ])
        : el('div', { class: 'tiny faint', style: { marginTop: '10px' }, text: 'Up to date with the checked release.' }),
      updateRow,
    ]));

    if (this.args && this.args.length) {
      box.appendChild(el('div', { class: 'card' }, [
        el('div', { class: 'card-title', html: `${icon('code', 14)}<span>Launch command</span>` }),
        el('div', { class: 'args-box', text: `python main.py ${this.args.join(' ')}` }),
        el('div', { class: 'row', style: { marginTop: '8px' } }, [
          button('Copy command', {
            iconName: 'copy', kind: 'ghost',
            run: () => copyText(`python main.py ${this.args.join(' ')}`)
              .then((ok2) => ok2 && toast('Command copied', { kind: 'ok', timeout: 1400 })),
          }),
        ]),
      ]));
    }
    return box;
  }

  block_tuning(actions) {
    const box = el('div', { class: 'card' });
    actions.appendChild(button('Reset', {
      iconName: 'refresh', kind: 'ghost',
      run: () => this.resetSection('comfyui'),
    }));

    const specs = [
      { key: 'comfyui.enabled', type: 'toggle', label: 'Engine enabled', help: 'Off stops Autumn from ever launching ComfyUI.' },
      { key: 'comfyui.auto_start', type: 'toggle', label: 'Start automatically', help: 'Launch the engine when the dashboard opens a job.' },
      { key: 'comfyui.port', type: 'number', label: 'Engine port', min: 1024, max: 65535, step: 1 },
      { key: 'comfyui.listen', type: 'select', label: 'Bind address', options: [
        { value: '127.0.0.1', label: '127.0.0.1 (this machine)' },
        { value: '0.0.0.0', label: '0.0.0.0 (LAN)' },
      ], help: 'The engine stays reachable only from the dashboard; 0.0.0.0 is for debugging.' },
      { key: 'comfyui.preview_method', type: 'select', label: 'Preview method', options: PREVIEW_METHODS },
      { key: 'comfyui.preview_size', type: 'number', label: 'Preview size', min: 64, max: 1024, step: 16 },
      { key: 'comfyui.vram_mode', type: 'select', label: 'VRAM strategy', options: VRAM_MODES },
      { key: 'comfyui.reserve_vram', type: 'number', label: 'Reserve VRAM (GB)', min: 0, max: 24, step: 0.5, help: 'Leave this much memory free for the desktop.' },
      { key: 'comfyui.high_ram', type: 'toggle', label: 'Prefer pagefile over model reloads (--high-ram)' },
      { key: 'comfyui.cpu', type: 'toggle', label: 'Run on CPU (--cpu)', help: 'Slow. Useful only for testing graphs.' },
      { key: 'comfyui.attention', type: 'select', label: 'Cross-attention', options: ATTENTION },
      { key: 'comfyui.unet_dtype', type: 'select', label: 'UNet dtype', options: DTYPES },
      { key: 'comfyui.vae_dtype', type: 'select', label: 'VAE dtype', options: DTYPES },
      { key: 'comfyui.text_encoder_dtype', type: 'select', label: 'Text encoder dtype', options: DTYPES },
      { key: 'comfyui.cache', type: 'select', label: 'Object cache', options: CACHE_MODES },
      { key: 'comfyui.deterministic', type: 'toggle', label: 'Deterministic algorithms (--deterministic)' },
      { key: 'comfyui.disable_metadata', type: 'toggle', label: 'Disable PNG metadata (--disable-metadata)' },
      { key: 'comfyui.multi_user', type: 'toggle', label: 'Multi-user input structure (--multi-user)' },
      { key: 'comfyui.enable_manager', type: 'toggle', label: 'Expose ComfyUI Manager (--enable-manager)', help: 'Off keeps node installs inside this dashboard.' },
      { key: 'comfyui.max_upload_size', type: 'text', label: 'Max upload size (MB)' },
      { key: 'comfyui.fast', type: 'text', label: '--fast mode', placeholder: 'bwd or triton', help: 'Empty disables the flag.' },
      { key: 'comfyui.extra_args', type: 'args', label: 'Extra arguments', rows: 3, placeholder: '--disable-smart-memory --asyncio-mode' },
      {
        key: 'comfyui.extra_model_paths', type: 'tags', label: 'Extra model path configs',
        placeholder: 'Add a .json or folder path, press Enter',
      },
    ];
    box.appendChild(el('div', { class: 'grid-2' }, specs.map((spec) => buildField(spec, this.ctx))));
    box.appendChild(el('div', { class: 'sep' }));
    box.appendChild(el('div', { class: 'sub-label', text: 'Environment variables for the engine process' }));
    box.appendChild(this.envField());
    return box;
  }

  block_preview(actions) {
    const box = el('div', { class: 'card' });
    const enh = state.enhancements || {};
    actions.appendChild(button('Reset', {
      iconName: 'refresh', kind: 'ghost', run: () => this.resetSection('preview'),
    }));

    const specs = [
      { key: 'preview.live', type: 'toggle', label: 'Live preview while generating' },
      { key: 'preview.max_resolution', type: 'number', label: 'Max preview edge', min: 128, max: 2048, step: 64 },
      { key: 'preview.jpeg_quality', type: 'number', label: 'Preview JPEG quality', min: 40, max: 100, step: 1 },
      { key: 'preview.min_interval_ms', type: 'number', label: 'Minimum frame interval (ms)', min: 0, max: 2000, step: 10 },
    ];
    box.appendChild(el('div', { class: 'grid-2' }, specs.map((spec) => buildField(spec, this.ctx))));

    box.appendChild(el('div', { class: 'sep' }));
    box.appendChild(el('div', { class: 'card-title', html: `${icon('sparkles', 14)}<span>High quality MiniMax H3 preview</span>` }));
    box.appendChild(el('div', {
      class: 'tiny faint',
      text: 'MiniMax H3 latents decode badly through latent2rgb. When this is on, H3 jobs route the '
        + 'preview through KJNodes ModelPreviewOverride with taeh3.safetensors, which is the H3 '
        + 'temporal autoencoder. Other model families keep using the fast decoder.',
    }));
    box.appendChild(el('div', { class: 'row', style: { gap: '8px', marginTop: '8px' } }, [
      pill(enh.kjnodes && enh.taeh3 ? 'ok' : '', `KJNodes ${enh.kjnodes ? 'installed' : 'missing'}`),
      pill(enh.kjnodes && enh.taeh3 ? 'ok' : '', `taeh3 ${enh.taeh3 ? 'installed' : 'missing'}`),
    ]));
    box.appendChild(el('div', { class: 'row', style: { gap: '6px', marginTop: '10px', flexWrap: 'wrap' } }, [
      button(enh.kjnodes && enh.taeh3 ? 'Reinstall KJNodes + taeh3' : 'Install KJNodes + taeh3', {
        iconName: 'download', kind: enh.kjnodes && enh.taeh3 ? 'ghost' : 'primary',
        run: () => this.installEnhancement('kjnodes', 'KJNodes and taeh3', { taeh3: true }),
      }),
      button('taeh3 only', {
        iconName: 'box',
        run: () => this.installEnhancement('taeh3', 'taeh3.safetensors'),
      }),
    ]));
    box.appendChild(buildField({
      key: 'preview.hq_minimax_h3', type: 'toggle', label: 'Use the H3 autoencoder preview for MiniMax H3 jobs',
    }, this.ctx));
    box.appendChild(buildField({
      key: 'preview.taeh3_file', type: 'text', label: 'taeh3 file', help: 'models/vae_approx inside the engine.',
    }, this.ctx));
    return box;
  }

  block_seedvr(actions) {
    const box = el('div', { class: 'col', style: { gap: '10px' } });
    const installed = !!state.enhancements.seedvr2;
    const wanted = !!state.features.seedvr;

    actions.appendChild(button(installed ? 'Uninstall' : 'Install', {
      iconName: installed ? 'trash' : 'download',
      kind: installed ? 'danger' : 'primary',
      run: () => (installed ? this.uninstallSeedvr() : this.installSeedvr()),
    }));

    if (!installed) {
      box.appendChild(el('div', { class: 'card' }, [
        el('div', { class: 'card-title', html: `${icon('sparkles', 14)}<span>SeedVR2 is not installed</span>` }),
        el('div', {
          class: 'tiny faint',
          text: 'SeedVR2 restores detail while upscaling video. Installing clones the node pack into '
            + 'the engine and downloads the DiT and VAE weights (several GB). The Video tab gains a '
            + 'SeedVR2 section as soon as it finishes.',
        }),
        el('div', { class: 'row', style: { marginTop: '10px' } }, [
          button('Install SeedVR2', {
            iconName: 'download', kind: 'primary',
            run: () => this.installSeedvr(),
          }),
          el('a', {
            class: 'btn sm ghost', href: SEEDVR_URL, target: '_blank', rel: 'noreferrer',
            html: `${icon('github', 13)}<span>Repository</span>`,
          }),
        ]),
      ]));
      return box;
    }

    box.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'row', style: { marginBottom: '8px' } }, [
        pill('ok', 'Node pack installed'),
        el('span', { class: 'grow' }),
        wanted ? null : el('span', { class: 'tiny faint', text: 'Switched off for new jobs' }),
      ]),
      el('div', { class: 'tiny faint', text: 'These are the defaults a new video job starts with. '
        + 'Per-job values live in the SeedVR2 section of the Video tab.' }),
    ]));

    const d = 'defaults.video.seedvr.';
    const specs = [
      { key: `${d}enabled`, type: 'toggle', label: 'Upscale every new video by default' },
      { key: `${d}dit_model`, type: 'combo', label: 'DiT model', options: 'seedvr_models', emptyLabel: 'engine default' },
      { key: `${d}vae_model`, type: 'combo', label: 'VAE model', options: 'seedvr_vae_models' },
      { key: `${d}resolution`, type: 'number', label: 'Target resolution', min: 256, max: 4320, step: 8 },
      { key: `${d}max_resolution`, type: 'number', label: 'Max resolution cap (0 = none)', min: 0, max: 8192, step: 8 },
      { key: `${d}batch_size`, type: 'number', label: 'Frames per pass', min: 1, max: 64, step: 1 },
      { key: `${d}uniform_batch_size`, type: 'toggle', label: 'Uniform batch size' },
      { key: `${d}temporal_overlap`, type: 'number', label: 'Temporal overlap', min: 0, max: 64, step: 1 },
      { key: `${d}prepend_frames`, type: 'number', label: 'Prepend frames', min: 0, max: 64, step: 1 },
      { key: `${d}color_correction`, type: 'select', label: 'Colour correction', options: [
        'none', 'wavelet', 'repaint'].map((value) => ({ value, label: value })) },
      { key: `${d}input_noise_scale`, type: 'number', label: 'Input noise scale', min: 0, max: 1, step: 0.01 },
      { key: `${d}latent_noise_scale`, type: 'number', label: 'Latent noise scale', min: 0, max: 1, step: 0.01 },
      { key: `${d}device`, type: 'text', label: 'Device', placeholder: 'cuda:0' },
      { key: `${d}offload_device`, type: 'text', label: 'Offload device', placeholder: 'cpu' },
      { key: `${d}blocks_to_swap`, type: 'number', label: 'Blocks to swap', min: 0, max: 60, step: 1 },
      { key: `${d}cache_model`, type: 'toggle', label: 'Keep the model loaded between jobs' },
      { key: `${d}attention_mode`, type: 'select', label: 'Attention', options: [
        'sdpa', 'flash2', 'flash3', 'xformers', 'math', 'sage'].map((value) => ({ value, label: value })) },
      { key: `${d}encode_tiled`, type: 'toggle', label: 'Tiled encode' },
      { key: `${d}encode_tile_size`, type: 'number', label: 'Encode tile', min: 256, max: 4096, step: 64 },
      { key: `${d}decode_tiled`, type: 'toggle', label: 'Tiled decode' },
      { key: `${d}decode_tile_size`, type: 'number', label: 'Decode tile', min: 256, max: 4096, step: 64 },
      { key: `${d}enable_debug`, type: 'toggle', label: 'Verbose node logging' },
    ];
    box.appendChild(el('div', { class: 'card' }, el('div', { class: 'grid-2' },
      specs.map((spec) => buildField(spec, this.ctx)))));
    return box;
  }

  block_nodes(actions) {
    const box = el('div', { class: 'col', style: { gap: '10px' } });
    const nodes = state.nodes || [];
    actions.appendChild(button('Add node pack', {
      iconName: 'plus', kind: 'primary', run: () => this.installNodeDialog(),
    }));

    if (!nodes.length) {
      box.appendChild(el('div', { class: 'empty' }, [
        el('div', { class: 'empty-icon', html: icon('layers', 22) }),
        el('div', { text: 'No custom node packs installed.' }),
      ]));
    } else {
      const table = el('table', { class: 'table' });
      table.appendChild(el('thead', {}, el('tr', {}, [
        el('th', { text: 'Node pack' }), el('th', { text: 'Ref' }),
        el('th', { text: 'Origin' }), el('th', { text: '' }),
      ])));
      const body = el('tbody');
      for (const entry of nodes) {
        body.appendChild(el('tr', {}, [
          el('td', {}, [
            el('div', { class: 'row', style: { gap: '7px' } }, [
              el('span', { class: 'trunc', text: entry.name }),
              entry.enabled ? null : pill('warn', 'disabled'),
            ]),
          ]),
          el('td', { class: 'mono tiny', text: entry.ref || '' }),
          el('td', { class: 'tiny trunc', style: { maxWidth: '260px' } }, [
            entry.repo
              ? el('a', { href: entry.repo, target: '_blank', rel: 'noreferrer', text: entry.repo.replace(/^https:\/\/github\.com\//, '') })
              : el('span', { class: 'faint', text: 'local' }),
          ]),
          el('td', {}, el('div', { class: 'row', style: { gap: '4px', justifyContent: 'flex-end' } }, [
            el('button', {
              class: 'icon-btn', title: entry.enabled ? 'Disable' : 'Enable',
              html: icon(entry.enabled ? 'eye' : 'eye_off', 13),
              onclick: () => this.toggleNode(entry),
            }),
            el('button', {
              class: 'icon-btn danger', title: 'Remove from disk',
              html: icon('trash', 13),
              onclick: () => this.removeNode(entry),
            }),
          ])),
        ]));
      }
      table.appendChild(body);
      box.appendChild(el('div', { class: 'card', style: { padding: '0' } }, table));
    }
    box.appendChild(note(`Custom nodes live in engine/ComfyUI/custom_nodes. ${nodes.length} pack(s) found.`));
    return box;
  }

  block_music(actions) {
    const box = el('div', { class: 'col', style: { gap: '10px' } });
    const info = state.music || {};
    const installed = !!(info.installed || info.stub);

    if (info.running) {
      actions.appendChild(button('Unload worker', {
        iconName: 'stop', kind: 'ghost',
        run: async () => {
          try { await api.musicAction('unload'); toast('Music worker unloaded', { kind: 'ok' }); } catch (err) { toast(err.message, { kind: 'err' }); }
          this.render();
        },
      }));
    }

    box.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'row', style: { marginBottom: '10px' } }, [
        pill(installed ? (info.running ? 'ok' : '') : 'warn',
          installed ? (info.running ? 'Worker online' : 'Worker idle') : 'YuE not installed'),
        info.stub ? pill('warn', 'synthetic stub') : null,
        el('span', { class: 'grow' }),
        installed ? null : button('Install YuE', {
          iconName: 'download', kind: 'primary',
          run: () => this.installMusic(),
        }),
      ]),
      kv([
        ['Repository', info.yue_dir || ''],
        ['Weights present', info.yue_present ? 'yes' : 'no'],
        ['Interpreter', info.interpreter || ''],
        ['Output', info.output || ''],
        ['Last load', info.load ? `${secs(info.load.seconds || 0)} · ${timeAgo(info.load.at || 0)}` : ''],
        ['Last error', info.last_error || ''],
      ]),
    ]));

    const specs = [
      { key: 'music.model', type: 'text', label: 'YuE model id', help: 'Downloaded into models/huggingface inside the program folder.' },
      { key: 'music.vae', type: 'text', label: 'YuE VAE id' },
      { key: 'music.device', type: 'select', label: 'Device', options: [
        { value: 'auto', label: 'auto' }, { value: 'cuda', label: 'cuda' }, { value: 'cpu', label: 'cpu' }] },
      { key: 'music.low_vram', type: 'toggle', label: 'Low VRAM mode (stage the weights)' },
      { key: 'music.cache_dir', type: 'text', label: 'Weight cache folder', help: 'Empty keeps models/huggingface inside the program folder.' },
      { key: 'music.token', type: 'text', label: 'Hugging Face token (private weights)' },
      { key: 'music.stub', type: 'toggle', label: 'Synthetic music engine', help: 'Generates a tone instead of audio. Useful for testing without weights.' },
      { key: 'music.stub_seconds', type: 'number', label: 'Stub length (s)', min: 1, max: 300, step: 1 },
    ];
    box.appendChild(el('div', { class: 'card' }, el('div', { class: 'grid-2' },
      specs.map((spec) => buildField(spec, this.ctx)))));
    return box;
  }

  block_features(actions) {
    const box = el('div', { class: 'col', style: { gap: '10px' } });
    const features = state.features || {};
    const choices = [
      {
        key: 'image_video', icon: 'image', title: 'Image and video generation',
        desc: 'ComfyUI backend, regional prompting, ControlNets, LoRAs and the video model families.',
        size: 'about 6 GB of Python packages',
      },
      {
        key: 'music', icon: 'music', title: 'YuE music generation',
        desc: 'Separate worker process with its own venv, streamed waveforms and FLAC output.',
        size: 'about 9 GB plus model weights',
      },
      {
        key: 'seedvr', icon: 'sparkles', title: 'SeedVR2 video upscaler',
        desc: 'Optional custom node. Adds the SeedVR2 section to the Video tab.',
        size: 'several GB of weights',
      },
    ];

    const grid = el('div', { class: 'opt-grid' });
    for (const choice of choices) {
      const on = !!features[choice.key];
      grid.appendChild(el('div', {
        class: `opt ${on ? 'on' : ''}`,
        onclick: () => this.setFeature(choice.key, !on),
      }, [
        el('div', { class: 'o-top' }, [
          el('span', { class: 'o-ic', html: icon(choice.icon, 18) }),
          el('span', { class: 'o-title', text: choice.title }),
          el('span', { class: 'grow' }),
          el('span', { class: 'pill' }, [el('span', { class: 'dot' }), el('span', { text: on ? 'on' : 'off' })]),
        ]),
        el('div', { class: 'o-desc', text: choice.desc }),
        el('div', { class: 'o-desc', style: { marginTop: '6px', color: 'var(--text-faint)' }, text: choice.size }),
      ]));
    }
    box.appendChild(grid);
    box.appendChild(el('div', { class: 'row', style: { gap: '6px', marginTop: '10px', flexWrap: 'wrap' } }, [
      button('Run the installer for the selected features', {
        iconName: 'rocket', kind: 'primary',
        run: () => this.runInstaller(),
      }),
      button('Open launch configuration', {
        iconName: 'settings', kind: 'ghost',
        run: () => window.dispatchEvent(new CustomEvent('autumn:setup')),
      }),
    ]));
    return box;
  }

  block_interface(actions) {
    const box = el('div', { class: 'card' });
    actions.appendChild(button('Reset', {
      iconName: 'refresh', kind: 'ghost', run: () => this.resetSection('app'),
    }));

    box.appendChild(el('div', { class: 'grid-2' }, [
      buildField({
        key: 'app.theme', type: 'segmented', label: 'Theme', grow: true,
        options: [{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }],
      }, this.ctx),
      buildField({
        key: 'app.accent', type: 'select', label: 'Accent', options: ACCENTS.map((a) => ({ value: a.value, label: a.label })),
      }, this.ctx),
      buildField({ key: 'app.history_visible', type: 'toggle', label: 'Show the history block' }, this.ctx),
      buildField({
        key: 'app.history_height', type: 'number', label: 'History height (px)', min: 120, max: 720, step: 8,
      }, this.ctx),
      buildField({ key: 'app.sidebar_visible', type: 'toggle', label: 'Show the session sidebar' }, this.ctx),
      buildField({ key: 'app.compact_sections', type: 'toggle', label: 'Start option sections collapsed' }, this.ctx),
      buildField({ key: 'app.autosave_forms', type: 'toggle', label: 'Remember the form between sessions' }, this.ctx),
      buildField({ key: 'app.reduced_motion', type: 'toggle', label: 'Reduce animation' }, this.ctx),
      buildField({ key: 'app.host', type: 'select', label: 'Dashboard bind address', options: [
        { value: '127.0.0.1', label: '127.0.0.1 (local only)' },
        { value: '0.0.0.0', label: '0.0.0.0 (LAN, run-web mode)' },
      ] }, this.ctx),
      buildField({ key: 'app.port', type: 'number', label: 'Dashboard port', min: 1024, max: 65535, step: 1 }, this.ctx),
      buildField({ key: 'app.open_browser', type: 'toggle', label: 'Open a browser when starting natively' }, this.ctx),
      buildField({ key: 'app.confirm_deletes', type: 'toggle', label: 'Confirm before deleting files' }, this.ctx),
    ]));

    const swatches = el('div', { class: 'row', style: { gap: '8px', marginTop: '12px', flexWrap: 'wrap' } });
    for (const accent of ACCENTS) {
      const on = (getPath(state.settings, 'app.accent') || 'amber') === accent.value;
      swatches.appendChild(el('button', {
        class: `accent-dot ${on ? 'on' : ''}`,
        title: accent.label,
        style: { background: accent.color },
        onclick: () => this.write('app.accent', accent.value),
      }));
    }
    box.appendChild(el('div', { class: 'sep' }));
    box.appendChild(el('div', { class: 'sub-label', text: 'Accent preview' }));
    box.appendChild(swatches);
    return box;
  }

  block_paths(actions) {
    const box = el('div', { class: 'card' });
    const paths = state.paths || {};
    const install = state.install || {};
    actions.appendChild(button('Copy all', {
      iconName: 'copy', kind: 'ghost',
      run: () => copyText(JSON.stringify(paths, null, 2)).then(() => toast('Paths copied', { kind: 'ok' })),
    }));
    box.appendChild(kv([
      ['Program folder', paths.root || ''],
      ['ComfyUI', paths.comfyui || ''],
      ['Models', paths.models || ''],
      ['Engine output', paths.output || ''],
      ['Music output', paths.music || ''],
      ['Wildcards', (install.wildcards && install.wildcards.path) || 'data/wildcards'],
      ['App venv', (install.venv && install.venv.app_path) || ''],
      ['Music venv', (install.venv && install.venv.music_path) || ''],
      ['Config', 'data/config.json'],
      ['History', 'data/history.sqlite3'],
    ]));
    box.appendChild(el('div', { class: 'sep' }));
    box.appendChild(el('div', {
      class: 'tiny faint',
      text: 'Every folder sits inside the program directory. Nothing is written to your home '
        + 'folder and no package is installed outside the two venvs.',
    }));
    const disks = (state.system && state.system.disks) || [];
    if (disks.length) {
      const rows = el('div', { class: 'col', style: { gap: '10px', marginTop: '12px' } });
      for (const disk of disks) {
        rows.appendChild(el('div', {}, [
          el('div', { class: 'row', style: { marginBottom: '4px' } }, [
            el('span', { class: 'tiny', text: disk.label }),
            el('span', { class: 'grow' }),
            el('span', { class: 'mono tiny faint', text: `${bytes(disk.free)} free of ${bytes(disk.total)}` }),
          ]),
          bar(disk.percent, disk.percent > 90 ? 'err' : disk.percent > 80 ? 'warn' : ''),
        ]));
      }
      box.appendChild(rows);
    }
    return box;
  }

  block_system(actions) {
    const box = el('div', { class: 'col', style: { gap: '10px' } });
    const system = state.system || {};
    const host = system.host || {};
    const memory = system.memory || {};
    const process = system.process || {};
    const gpus = system.gpus || [];

    actions.appendChild(button('Refresh', {
      iconName: 'refresh', kind: 'ghost',
      run: async () => {
        try {
          const data = await api.system();
          state.system = data.system;
          if (data.engine) state.engine = { ...state.engine, ...data.engine };
          if (data.queue) state.queue = data.queue;
          this.render();
        } catch (err) { toast(err.message, { kind: 'err' }); }
      },
    }));

    box.appendChild(el('div', { class: 'stat-grid' }, [
      statCell('Memory', `${Math.round(memory.percent || 0)}`, '% used'),
      statCell('Process RSS', bytes(process.rss || 0)),
      statCell('Threads', String(process.threads || 0)),
      statCell('Uptime', secs(process.uptime || 0), ''),
    ]));

    if (gpus.length) {
      const cards = el('div', { class: 'card-grid' });
      for (const gpu of gpus) {
        cards.appendChild(el('div', { class: 'card' }, [
          el('div', { class: 'card-title', html: `${icon('gpu', 14)}<span>${gpu.name || 'GPU'}</span>` }),
          el('div', { class: 'row', style: { marginBottom: '6px' } }, [
            el('span', { class: 'mono tiny faint', text: `${bytes(gpu.memory_used || 0)} / ${bytes(gpu.memory_total || 0)}` }),
            el('span', { class: 'grow' }),
            el('span', { class: 'mono tiny', text: `${Math.round(gpu.percent || 0)}%` }),
          ]),
          bar(gpu.percent || 0, (gpu.percent || 0) > 92 ? 'err' : ''),
          el('div', { class: 'tiny faint', style: { marginTop: '8px' }, text: `driver ${gpu.driver || '—'} · ${gpu.source || 'local'}` }),
        ]));
      }
      box.appendChild(cards);
    } else {
      box.appendChild(el('div', { class: 'card' }, [
        el('div', { class: 'card-title', html: `${icon('gpu', 14)}<span>No GPU reported</span>` }),
        el('div', { class: 'tiny faint', text: 'nvidia-smi returned nothing. Generation still works on CPU, slowly.' }),
      ]));
    }

    box.appendChild(el('div', { class: 'card' }, [
      el('div', { class: 'card-title', html: `${icon('monitor', 14)}<span>Host</span>` }),
      kv([
        ['Hostname', host.hostname || ''],
        ['System', `${host.os || ''} ${host.os_version || ''}`],
        ['Machine', host.machine || ''],
        ['Python', host.python || ''],
        ['Interpreter', host.executable || ''],
        ['Virtual env', host.virtual_env ? 'active' : 'not active'],
        ['Addresses', (host.addresses || []).join(', ')],
        ['Autumn', state.version || ''],
      ]),
    ]));
    return box;
  }

  block_maintenance() {
    const box = el('div', { class: 'card' }, el('div', { class: 'row', style: { gap: '6px', flexWrap: 'wrap' } }, [
      button('Clear history…', {
        iconName: 'history',
        run: (event) => this.clearHistoryMenu(event.currentTarget),
      }),
      button('Clear engine queue', {
        iconName: 'stop_circle',
        run: async () => {
          try { await api.clearQueue(); toast('Queue cleared', { kind: 'ok' }); } catch (err) { toast(err.message, { kind: 'err' }); }
        },
      }),
      button('Re-index models', {
        iconName: 'refresh',
        run: async () => {
          try {
            await api.assetAction('refresh', {});
            const data = await api.models();
            state.setModels(data);
            toast('Model folders re-indexed', { kind: 'ok' });
          } catch (err) { toast(err.message, { kind: 'err' }); }
        },
      }),
      button('Reset a settings section…', {
        iconName: 'sliders',
        run: (event) => this.resetSectionMenu(event.currentTarget),
      }),
      button('Export diagnostics', {
        iconName: 'code',
        run: () => this.exportDiagnostics(),
      }),
      button('Console', {
        iconName: 'terminal', kind: 'ghost',
        run: () => state.setUI({ logsOpen: true }),
      }),
    ]));
    box.appendChild(el('div', { class: 'sep' }));
    box.appendChild(el('div', {
      class: 'tiny faint',
      text: 'Clearing history removes the job records and their thumbnails. Files on disk stay '
        + 'until you delete them in the Assets tab.',
    }));
    return box;
  }

  // ----------------------------------------------------------------- custom
  envField() {
    const current = getPath(state.settings, 'comfyui.env') || {};
    const text = Object.entries(current).map(([key, value]) => `${key}=${value}`).join('\n');
    const area = el('textarea', {
      class: 'textarea', rows: 4, spellcheck: 'false',
      placeholder: 'CUDA_VISIBLE_DEVICES=0\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
      style: { fontFamily: 'var(--mono)', fontSize: '11px' },
      dataset: { field: 'comfyui.env' },
    });
    area.value = text;
    area.addEventListener('input', debounce(() => {
      const out = {};
      for (const line of area.value.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const index = trimmed.indexOf('=');
        if (index <= 0) continue;
        out[trimmed.slice(0, index).trim()] = trimmed.slice(index + 1).trim();
      }
      this.write('comfyui.env', out);
    }, 500));
    return area;
  }

  // ---------------------------------------------------------------- actions
  async engineAction(action, body, message) {
    try {
      const data = await api.engineAction(action, body);
      if (data.engine) state.engine = { ...state.engine, ...data.engine };
      state.emit(TOPICS.state, 'engine');
      if (message) toast(message, { kind: 'ok', timeout: 2600 });
      this.restartPending = false;
      this.loadEngine();
      return data;
    } catch (err) {
      toast(`${action}: ${err.message}`, { kind: 'err', timeout: 9000 });
      throw err;
    }
  }

  async checkUpdate() {
    try {
      const data = await api.engineAction('check_update', {});
      const latest = data.latest || '';
      if (state.engine) state.engine = { ...state.engine, latest_version: latest };
      if (latest && latest !== (state.engine && state.engine.version)) {
        toast(`ComfyUI ${latest} is available`, { kind: 'warn', timeout: 6000 });
      } else {
        toast('ComfyUI is up to date', { kind: 'ok', timeout: 2600 });
      }
      this.render();
    } catch (err) {
      toast(`Update check failed: ${err.message}`, { kind: 'err', timeout: 8000 });
    }
  }

  async updateComfyui() {
    const e = state.engine || {};
    const okToGo = await confirmDialog({
      title: 'Update ComfyUI',
      message: `ComfyUI updates from ${e.version || 'unknown'} to ${e.latest_version || 'the latest release'}. `
        + 'The engine restarts and reloads its node list. Jobs in flight are lost. Continue?',
      confirmLabel: 'Update and restart',
    });
    if (!okToGo) return;
    toast('Updating ComfyUI…', { kind: 'info', timeout: 120000 });
    try {
      await api.engineAction('update', {});
      toast('ComfyUI updated', { kind: 'ok', timeout: 5000 });
      window.dispatchEvent(new CustomEvent('autumn:refresh'));
    } catch (err) {
      toast(`Update failed: ${err.message}`, { kind: 'err', timeout: 9000 });
    }
    this.loadEngine();
  }

  freeVram() {
    const unload = el('input', { type: 'checkbox', checked: true });
    const deep = el('input', { type: 'checkbox', checked: false });
    modal({
      title: 'Free VRAM',
      icon: 'memory',
      width: 460,
      body: el('div', { class: 'col', style: { gap: '10px' } }, [
        el('label', { class: 'check' }, [unload, el('span', { class: 'box', html: icon('check', 11) }),
          el('span', { class: 'txt', text: 'Unload loaded models' })]),
        el('label', { class: 'check' }, [deep, el('span', { class: 'box', html: icon('check', 11) }),
          el('span', { class: 'txt', text: 'Deep clean (also clears the cache and gc)' })]),
      ]),
      actions: [{
        label: 'Free memory', kind: 'primary', run: async (close) => {
          close();
          try {
            await api.engineAction('free', { unload_models: unload.checked, deep_clean: deep.checked });
            toast('VRAM released', { kind: 'ok' });
          } catch (err) { toast(err.message, { kind: 'err' }); }
        },
      }],
    });
  }

  async installEnhancement(name, label, extra = {}) {
    toast(`Installing ${label}…`, { kind: 'info', timeout: 90000 });
    try {
      const data = await api.setEnhancement(name, true, extra);
      if (data.enhancements) state.enhancements = { ...state.enhancements, ...data.enhancements };
      toast(`${label} installed`, { kind: 'ok', timeout: 5000 });
      window.dispatchEvent(new CustomEvent('autumn:refresh'));
    } catch (err) {
      toast(`${label}: ${err.message}`, { kind: 'err', timeout: 9000 });
    }
    this.render();
  }

  async installSeedvr() {
    const okToGo = await confirmDialog({
      title: 'Install SeedVR2',
      message: 'SeedVR2 clones a ComfyUI custom node into the engine and downloads a DiT and a VAE '
        + '(several GB). The engine reloads its node list afterwards. Continue?',
      confirmLabel: 'Install SeedVR2',
    });
    if (!okToGo) return;
    await this.installEnhancement('seedvr2', 'SeedVR2');
  }

  async uninstallSeedvr() {
    const okToGo = await confirmDialog({
      title: 'Turn SeedVR2 off',
      message: 'The SeedVR2 section is hidden again and the node pack is not loaded into new jobs. '
        + 'Files already downloaded stay on disk.',
      confirmLabel: 'Turn off',
    });
    if (!okToGo) return;
    try {
      const data = await api.setEnhancement('seedvr2', false);
      if (data.enhancements) state.enhancements = { ...state.enhancements, ...data.enhancements };
      toast('SeedVR2 turned off', { kind: 'ok' });
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
    this.render();
  }

  async installMusic() {
    const okToGo = await confirmDialog({
      title: 'Install the YuE music engine',
      message: 'This clones the YuE repository into the program folder, creates a second venv for '
        + 'its dependencies and downloads the model weights on first run. Continue?',
      confirmLabel: 'Install YuE',
    });
    if (!okToGo) return;
    try {
      await api.musicAction('install');
      toast('Music install started — watch the console', { kind: 'info', timeout: 8000 });
      state.setUI({ logsOpen: true });
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  runInstaller() {
    const features = { ...state.features };
    Promise.resolve(confirmDialog({
      title: 'Run the installer',
      message: `Installs ${features.image_video ? 'ComfyUI, ' : ''}${features.music ? 'YuE, ' : ''}`
        + `${features.seedvr ? 'SeedVR2' : ''} into the program folder using the local venvs. `
        + 'Progress appears in the console drawer.',
      confirmLabel: 'Install',
    })).then(async (go) => {
      if (!go) return;
      try {
        await api.startInstall(features);
        state.setUI({ logsOpen: true });
        toast('Installer started', { kind: 'ok' });
      } catch (err) {
        toast(err.message, { kind: 'err' });
      }
    });
  }

  async setFeature(key, value) {
    try {
      const data = await api.saveSettings({ features: { [key]: value } });
      state.setSettings(data.settings);
      toast(`${key} ${value ? 'on' : 'off'}`, { kind: 'ok', timeout: 2000 });
      window.dispatchEvent(new CustomEvent('autumn:features'));
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
    this.render();
  }

  // Node pack dialogs ------------------------------------------------------
  installNodeDialog() {
    const url = el('input', { class: 'input', placeholder: 'https://github.com/owner/ComfyUI-SomeNodes' });
    const name = el('input', { class: 'input', placeholder: 'optional folder name' });
    const reqs = el('input', { type: 'checkbox', checked: true });
    modal({
      title: 'Add a custom node pack',
      icon: 'layers',
      width: 560,
      body: el('div', { class: 'col', style: { gap: '10px' } }, [
        el('div', { class: 'sub-label', text: 'GitHub URL' }),
        url,
        el('div', { class: 'sub-label', text: 'Folder name' }),
        name,
        el('label', { class: 'check' }, [reqs, el('span', { class: 'box', html: icon('check', 11) }),
          el('span', { class: 'txt', text: 'Install its requirements into the program venv' })]),
        el('div', { class: 'tiny faint', text: 'Cloned into engine/ComfyUI/custom_nodes. Reload the node list when it finishes.' }),
      ]),
      actions: [{
        label: 'Install', kind: 'primary', run: async (close) => {
          const value = url.value.trim();
          if (!value.startsWith('https://github.com/')) {
            toast('Enter a https://github.com/… URL', { kind: 'warn' });
            return;
          }
          close();
          toast('Installing node pack…', { kind: 'info', timeout: 90000 });
          try {
            await api.installNode(value, name.value.trim() || null, reqs.checked);
            toast('Node pack installed', { kind: 'ok', timeout: 5000 });
            await api.engineAction('refresh_nodes', {});
            window.dispatchEvent(new CustomEvent('autumn:refresh'));
          } catch (err) {
            toast(`Node install failed: ${err.message}`, { kind: 'err', timeout: 9000 });
          }
          this.loadEngine();
        },
      }],
    });
  }

  async toggleNode(entry) {
    try {
      await api.toggleNode(entry.name, !entry.enabled);
      toast(`${entry.name} ${entry.enabled ? 'enabled' : 'disabled'}`, { kind: 'ok', timeout: 2400 });
      window.dispatchEvent(new CustomEvent('autumn:refresh'));
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
    this.loadEngine();
  }

  async removeNode(entry) {
    const okToGo = await confirmDialog({
      title: `Remove ${entry.name}`,
      message: 'The node pack folder is deleted from engine/ComfyUI/custom_nodes. Python packages it '
        + 'installed stay in the venv.',
      confirmLabel: 'Remove',
    });
    if (!okToGo) return;
    try {
      await api.removeNode(entry.name);
      toast(`${entry.name} removed`, { kind: 'ok' });
      window.dispatchEvent(new CustomEvent('autumn:refresh'));
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
    this.loadEngine();
  }

  // Maintenance ------------------------------------------------------------
  clearHistoryMenu(anchor) {
    menu(anchor, [
      { label: 'Clear everything', icon: 'trash', run: () => this.clearHistory('') },
      { label: 'Images only', icon: 'image', run: () => this.clearHistory('image') },
      { label: 'Videos only', icon: 'video', run: () => this.clearHistory('video') },
      { label: 'Music only', icon: 'music', run: () => this.clearHistory('music') },
      { sep: true },
      { label: 'Also clear the engine history', icon: 'terminal', run: () => this.clearHistory('', true) },
    ]);
  }

  async clearHistory(kind, engine = false) {
    const okToGo = await confirmDialog({
      title: 'Clear history',
      message: `Remove ${kind ? `${kind} ` : ''}job records from the history database? Output files are not deleted.`,
      confirmLabel: 'Clear',
    });
    if (!okToGo) return;
    try {
      const data = await api.clearHistory(kind || undefined, engine || undefined);
      toast(`${data.removed || 0} record(s) cleared`, { kind: 'ok' });
      window.dispatchEvent(new CustomEvent('autumn:refresh'));
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  resetSectionMenu(anchor) {
    const sections = [
      { key: 'comfyui', label: 'ComfyUI tuning' },
      { key: 'preview', label: 'Live preview' },
      { key: 'app', label: 'Interface' },
      { key: 'music', label: 'Music defaults' },
      { key: 'defaults', label: 'Generation defaults' },
    ];
    menu(anchor, sections.map((section) => ({
      label: section.label,
      icon: 'refresh',
      run: () => this.resetSection(section.key),
    })));
  }

  async resetSection(section) {
    const okToGo = await confirmDialog({
      title: `Reset ${section}`,
      message: 'Every value in this section returns to its default. Other sections are untouched.',
      confirmLabel: 'Reset section',
    });
    if (!okToGo) return;
    try {
      const data = await api.resetSection(section);
      state.setSettings(data.settings);
      toast(`${section} reset`, { kind: 'ok' });
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
    this.render();
  }

  async exportDiagnostics() {
    const payload = {
      version: state.version,
      demo: state.demo,
      features: state.features,
      engine: state.engine,
      enhancements: state.enhancements,
      nodes: (state.nodes || []).map((node) => ({ name: node.name, ref: node.ref, enabled: node.enabled })),
      system: state.system,
      settings: redact(state.settings),
      problems: state.problems,
      at: new Date().toISOString(),
    };
    const ok2 = await copyText(JSON.stringify(payload, null, 2));
    toast(ok2 ? 'Diagnostics copied to the clipboard' : 'Copy blocked by the browser',
      { kind: ok2 ? 'ok' : 'warn', timeout: 3000 });
  }
}

// Helpers ------------------------------------------------------------------
function deepMerge(target, source) {
  const out = { ...target };
  for (const [key, value] of Object.entries(source || {})) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      out[key] = deepMerge(out[key] || {}, value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function redact(settings) {
  const copy = JSON.parse(JSON.stringify(settings || {}));
  if (copy.music && copy.music.token) copy.music.token = '***';
  if (copy.comfyui && copy.comfyui.env) {
    for (const key of Object.keys(copy.comfyui.env)) {
      if (/token|key|secret/i.test(key)) copy.comfyui.env[key] = '***';
    }
  }
  return copy;
}

export { BLOCKS as SETTINGS_BLOCKS, SEEDVR_URL, KJNODES_URL };
