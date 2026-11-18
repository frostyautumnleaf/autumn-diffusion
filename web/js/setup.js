// Launch configuration: the first-run wizard.
//
// It asks three questions (what to install, where to keep it, which ports), then
// streams the installer over the `install` WebSocket events. The same screen is
// reachable later from Settings, so a user who skipped it can still add YuE.

import { state } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast } from './components.js';
import { el, clear, bytes, clockTime } from './util.js';

const ENGINES = [
  {
    id: 'image_video',
    title: 'Image & video',
    icon: 'image',
    desc: 'ComfyUI as the engine: SDXL, Flux, Qwen-Image, Wan, Hunyuan, LTX and friends. '
      + 'Cloned into engine/ComfyUI with its own virtualenv.',
    size: '~3.5 GB of wheels',
  },
  {
    id: 'music',
    title: 'Music (YuE)',
    icon: 'music',
    desc: 'YuE full-length music with vocals and lyrics. Runs in a second virtualenv so its '
      + 'torch build cannot clash with ComfyUI.',
    size: '~4.5 GB of wheels',
  },
  {
    id: 'seedvr',
    title: 'SeedVR2 upscaler',
    icon: 'sparkles',
    desc: 'Diffusion video / image restorer. Adds a section to the video tab. Can also be '
      + 'installed later with one button.',
    size: '~7 GB of models',
  },
];

export class SetupWizard {
  constructor(root) {
    this.root = root;
    this.step = 0;
    this.chosen = {
      image_video: true,
      music: false,
      seedvr: false,
    };
    this.app = {
      theme: state.ui.theme || 'dark',
      accent: state.ui.accent || 'amber',
      port: (state.settings.app && state.settings.app.port) || 8190,
      comfy_port: (state.settings.comfyui && state.settings.comfyui.port) || 8188,
      open_browser: true,
    };
    this.installing = false;
    this.done = false;
    this.lines = [];
    this.progress = 0;
    this.visible = false;
    this.onDone = null;
  }

  /** Show the wizard when the server says configuration is missing. */
  sync() {
    const shouldShow = state.setupRequired && !this.done;
    if (shouldShow && !this.visible) {
      this.visible = true;
      this.render();
    } else if (!shouldShow && this.visible) {
      this.visible = false;
      clear(this.root);
    } else if (this.visible) {
      this.render();
    }
  }

  // ------------------------------------------------------------------ steps
  features() {
    return {
      image_video: this.chosen.image_video,
      music: this.chosen.music,
      seedvr: this.chosen.seedvr && this.chosen.image_video,
    };
  }

  needsInstall() {
    const features = this.features();
    const install = state.install || {};
    if (features.image_video && !state.engineInstalled) return true;
    if (features.music && !(state.music && (state.music.installed || state.music.stub))) return true;
    if (features.seedvr && !(state.enhancements && state.enhancements.seedvr2)) return true;
    void install;
    return false;
  }

  render() {
    clear(this.root);
    const card = el('div', { class: 'wizard-card' });
    const node = el('div', { class: 'wizard' }, [card]);
    this.root.appendChild(node);

    if (this.step === 0) this.renderEngines(card);
    else if (this.step === 1) this.renderSystem(card);
    else this.renderInstall(card);
  }

  head(card, title, lead) {
    card.appendChild(el('div', { class: 'row', style: { gap: '10px', marginBottom: '10px' } }, [
      el('span', { class: 'brand-mark', style: { width: '32px', height: '32px' },
        html: icon('leaf', 18) }),
      el('div', {}, [
        el('div', { class: 'brand-name', text: 'Autumn Diffusion' }),
        el('div', { class: 'tiny faint', text: `Launch configuration · step ${this.step + 1} of 3` }),
      ]),
    ]));
    // Segmented step bar: one filled segment per completed step.
    card.appendChild(el('div', { class: 'steps' },
      [0, 1, 2].map((i) => el('span', { class: `seg ${i <= this.step ? 'on' : ''}` }))));
    card.appendChild(el('h1', { text: title }));
    if (lead) card.appendChild(el('p', { class: 'lead', text: lead }));
  }

  renderEngines(card) {
    this.head(card, 'What should this machine generate?',
      'Everything is installed inside the program folder and into its own Python '
      + 'virtualenvs. Nothing touches the system Python.');

    const grid = el('div', { class: 'opt-grid' });
    for (const engine of ENGINES) {
      const on = this.chosen[engine.id];
      grid.appendChild(el('div', {
        class: `opt ${on ? 'on' : ''}`,
        onclick: () => {
          this.chosen[engine.id] = !on;
          if (engine.id === 'image_video' && !this.chosen.image_video) this.chosen.seedvr = false;
          this.render();
        },
      }, [
        el('div', { class: 'o-top' }, [
          el('span', { class: 'o-ic', html: icon(engine.icon, 17) }),
          el('span', { class: 'grow' }),
          el('span', { class: `pill ${on ? 'ok' : ''}`, html: on
            ? `<span class="dot"></span>selected` : 'off' }),
        ]),
        el('div', { class: 'o-title', text: engine.title }),
        el('div', { class: 'o-desc', text: engine.desc }),
        el('div', { class: 'tiny faint mono', text: engine.size }),
      ]));
    }
    card.appendChild(grid);

    card.appendChild(el('div', { class: 'row', style: { gap: '8px', marginTop: '12px' } }, [
      el('button', {
        class: 'btn sm ghost', html: `${icon('image', 12)}<span>Image & video only</span>`,
        onclick: () => { this.chosen = { image_video: true, music: false, seedvr: false }; this.render(); },
      }),
      el('button', {
        class: 'btn sm ghost', html: `${icon('layers', 12)}<span>Everything</span>`,
        onclick: () => { this.chosen = { image_video: true, music: true, seedvr: true }; this.render(); },
      }),
      el('span', { class: 'grow' }),
    ]));

    this.foot(card, {
      nextLabel: this.needsInstall() ? 'Start the install' : 'Continue',
      next: () => { this.step = 1; this.render(); },
      skip: true,
    });
  }

  renderSystem(card) {
    this.head(card, 'Where it goes, and which ports',
      'Change these only if something clashes. The dashboard stays on 127.0.0.1 unless '
      + 'you start it with run-web.sh.');

    const system = state.system || {};
    const host = system.host || {};
    const gpu = (system.gpus && system.gpus[0]) || {};
    const paths = state.paths || {};
    const disks = system.disks || [];
    const stats = disks.find((disk) => disk.label === 'program') || disks[0] || {};
    const vram = gpu.memory_total || gpu.total || 0;

    card.appendChild(el('dl', { class: 'kv' }, [
      el('dt', { text: 'Program folder' }),
      el('dd', { class: 'mono', text: breakablePath(paths.root) }),
      el('dt', { text: 'ComfyUI' }), el('dd', { class: 'mono', text: paths.comfyui || 'engine/ComfyUI' }),
      el('dt', { text: 'Models' }), el('dd', { class: 'mono', text: paths.models || 'engine/ComfyUI/models' }),
      el('dt', { text: 'Outputs' }), el('dd', { class: 'mono', text: paths.output || 'engine/ComfyUI/output' }),
      el('dt', { text: 'Free on this disk' }),
      el('dd', { text: stats.free ? `${bytes(stats.free)} free of ${bytes(stats.total || 0)}` : '—' }),
      el('dt', { text: 'Python' }),
      el('dd', { class: 'mono', text: host.python ? `${host.python} · ${breakablePath(host.executable)}` : '—' }),
      el('dt', { text: 'GPU' }),
      el('dd', { text: gpu.name ? `${gpu.name}${vram ? ` · ${bytes(vram, 0)}` : ''}` : 'not detected' }),
    ]));

    card.appendChild(el('div', { class: 'field-grid' }, [
      el('div', { class: 'field' }, [
        el('div', { class: 'field-top' }, [
          el('span', { class: 'field-label', text: 'Dashboard port' }),
          el('span', { class: 'field-hint', text: 'http://127.0.0.1' }),
        ]),
        el('input', {
          class: 'input', type: 'number', value: this.app.port,
          oninput: (event) => { this.app.port = Number(event.target.value) || 8190; },
        }),
      ]),
      el('div', { class: 'field' }, [
        el('div', { class: 'field-top' }, [
          el('span', { class: 'field-label', text: 'ComfyUI port' }),
          el('span', { class: 'field-hint', text: 'engine' }),
        ]),
        el('input', {
          class: 'input', type: 'number', value: this.app.comfy_port,
          oninput: (event) => { this.app.comfy_port = Number(event.target.value) || 8188; },
        }),
      ]),
    ]));

    card.appendChild(el('label', { class: 'check' }, [
      el('input', {
        type: 'checkbox', checked: this.app.open_browser,
        onchange: (event) => { this.app.open_browser = event.target.checked; },
      }),
      el('span', { class: 'box', html: icon('check', 11) }),
      el('span', { class: 'txt', text: 'Open a browser tab when the dashboard is ready' }),
    ]));

    this.foot(card, {
      back: () => { this.step = 0; this.render(); },
      nextLabel: this.needsInstall() ? 'Install and finish' : 'Finish',
      next: () => this.submit(),
    });
  }

  renderInstall(card) {
    this.head(card, this.done ? 'Ready' : 'Installing',
      this.done
        ? 'Everything is installed and the engines are configured. Open the dashboard to start generating.'
        : 'This downloads Python wheels and clones the engines. It takes a while the first '
          + 'time; the log below is live.');

    card.appendChild(el('div', { class: 'meter wide', style: { marginBottom: '12px' } }, [
      el('div', { class: 'track' }, [el('div', {
        class: 'fill', style: { width: `${Math.max(2, this.progress)}%` },
      })]),
      el('span', { class: 'num', text: `${Math.round(this.progress)}%` }),
    ]));

    const log = el('div', { class: 'log-lines', style: { maxHeight: '240px', borderRadius: '10px' } },
      this.lines.map((line) => el('div', { class: 'log-line' }, [
        el('span', { class: 't', text: clockTime(line.at) }),
        el('span', { class: 'm', text: line.message }),
      ])));
    card.appendChild(log);
    requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; });

    if (this.done) {
      card.appendChild(el('div', { class: 'row', style: { marginTop: '14px' } }, [
        el('span', { class: 'pill ok', html: '<span class="dot"></span>ready' }),
        el('span', { class: 'grow' }),
        el('button', {
          class: 'btn primary', html: `${icon('arrow_right', 13)}<span>Open the dashboard</span>`,
          onclick: () => this.close(),
        }),
      ]));
      return;
    }

    card.appendChild(el('div', { class: 'row', style: { marginTop: '14px' } }, [
      el('span', { class: 'tiny faint', text: 'The dashboard opens by itself when this finishes.' }),
      el('span', { class: 'grow' }),
      el('button', {
        class: 'btn ghost', text: 'Run in background', title: 'The install keeps going; the dashboard opens when it ends',
        onclick: () => this.close(),
      }),
    ]));
  }

  foot(card, { back = null, next = null, nextLabel = 'Continue', skip = false } = {}) {
    const row = el('div', { class: 'row', style: { marginTop: '18px', gap: '8px' } });
    if (back) {
      row.appendChild(el('button', { class: 'btn ghost', text: 'Back', onclick: back }));
    }
    if (skip) {
      row.appendChild(el('button', {
        class: 'btn ghost', text: 'Skip, install later',
        onclick: () => this.submit({ install: false }),
      }));
    }
    row.appendChild(el('span', { class: 'grow' }));
    if (next) {
      row.appendChild(el('button', {
        class: 'btn primary', html: `${icon('arrow_right', 13)}<span>${nextLabel}</span>`,
        onclick: next,
      }));
    }
    card.appendChild(row);
  }

  // ----------------------------------------------------------------- submit
  async submit({ install = true } = {}) {
    const wanted = install && this.needsInstall();
    try {
      const data = await api.setup(this.features(), {
        theme: this.app.theme,
        accent: this.app.accent,
        port: this.app.port,
        open_browser: this.app.open_browser,
      }, wanted);
      state.setSettings(data.settings);
      if (data.features) state.features = { ...state.features, ...data.features };
      if (wanted) {
        this.installing = true;
        this.step = 2;
        this.render();
      } else {
        this.done = true;
        this.close();
      }
    } catch (err) {
      toast(`Could not save the configuration: ${err.message}`, { kind: 'err', timeout: 7000 });
    }
  }

  /** Fed by the `install` WebSocket events while the installer runs. */
  pushInstall(data) {
    if (!this.visible) return;
    const message = data.message || data.stage || 'working';
    if (!this.lines.length || this.lines[this.lines.length - 1].message !== message) {
      this.lines.push({ message, at: data.at || Date.now() / 1000 });
      if (this.lines.length > 200) this.lines.shift();
    }
    if (typeof data.percent === 'number') this.progress = data.percent;
    if (this.step === 2) this.render();
  }

  installFinished(data) {
    if (!this.visible) return;
    this.progress = 100;
    this.done = true;
    this.installing = false;
    if (data && data.ok === false) {
      toast(`Install stopped: ${data.error || 'see the log'}`, { kind: 'err', timeout: 9000 });
    } else {
      this.lines.push({ message: 'installation complete', at: Date.now() / 1000 });
    }
    this.step = 2;
    this.render();
  }

  close() {
    this.visible = false;
    clear(this.root);
    if (this.onDone) this.onDone();
  }
}

/**
 * Zero-width breaks after each "/" so a long folder path wraps on a separator
 * instead of splitting a word in half.
 */
export function breakablePath(value) {
  return String(value || '—').replace(/\//g, '/\u200b');
}

export function estimateText(features, system) {
  const parts = [];
  if (features.image_video) parts.push('ComfyUI');
  if (features.music) parts.push('YuE');
  if (features.seedvr) parts.push('SeedVR2');
  return parts.join(' + ') || 'nothing';
}

export default SetupWizard;
