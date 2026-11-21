// Application shell: top bar, tabs, layout, the WebSocket router and the
// resizers. Every other module renders into a host element this file owns.
//
// Routing rules are deliberately boring: server events update `state`, `state`
// notifies its subscribers, and this file only re-renders the shell when the
// tab, the engine or the queue changed.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { Socket } from './ws.js';
import { icon } from './icons.js';
import { toast, menu, modal } from './components.js';
import { Panel } from './panel.js';
import { Composer, missingPrimary } from './composer.js';
import { Stage } from './preview.js';
import { MusicStage } from './musicstage.js';
import { History } from './history.js';
import { Sidebar } from './sidebar.js';
import { LogDrawer, engineEntry } from './logs.js';
import { Assets } from './assets.js';
import { SettingsView } from './settings.js';
import { SetupWizard } from './setup.js';
import { initVolume, mountVolumeControl } from './volume.js';
import { el, clear, $, clamp, debounce, num, bytes, seconds, fileName } from './util.js';

const TABS = [
  { id: 'image', label: 'Image Generation', short: 'Image', icon: 'image', feature: 'image_video' },
  { id: 'video', label: 'Video Generation', short: 'Video', icon: 'video', feature: 'image_video' },
  { id: 'music', label: 'Music Generation', short: 'Music', icon: 'music', feature: 'music' },
  { id: 'assets', label: 'Model browser', short: 'Assets', icon: 'folder', feature: null },
  { id: 'settings', label: 'Settings', short: 'Settings', icon: 'sliders_h', feature: null },
];

const GEN_TABS = new Set(['image', 'video', 'music']);

const LIMITS = {
  rail: { min: 300, max: 620, css: '--rail-w', key: 'railWidth' },
  side: { min: 210, max: 460, css: '--side-w', key: 'sideWidth' },
  history: { min: 34, max: 620, css: '--history-h', key: 'historyHeight' },
};

class App {
  constructor() {
    this.nodes = {
      tabs: $('#tabs'),
      enginePill: $('#engine-pill'),
      queuePill: $('#queue-pill'),
      meter: $('#vram-meter'),
      rail: $('#rail'),
      railHead: $('#rail-head'),
      railTitle: $('#rail-title'),
      railSearch: $('#rail-search'),
      railScroll: $('#rail-scroll'),
      railNav: $('#rail-nav'),
      genView: $('#gen-view'),
      stageHost: $('#stage-host'),
      musicHost: $('#music-host'),
      composerHost: $('#composer-host'),
      actionbar: $('#actionbar'),
      logHost: $('#log-drawer'),
      assetsView: $('#assets-view'),
      settingsView: $('#settings-view'),
      sidebar: $('#sidebar'),
      history: $('#history'),
      setupHost: $('#setup-host'),
    };

    this.stage = null;
    this.musicStage = null;
    this.panel = null;
    this.composers = new Map();
    this.history = null;
    this.sidebar = null;
    this.logs = null;
    this.assets = null;
    this.settingsView = null;
    this.setup = null;
    this.socket = null;
    this.busy = false;
    this._rendered = null;
    this._actionKey = null;
    this.modelsLoaded = false;

    applyTheme();
    initVolume();
    this.renderChrome();
    this.bindGlobalEvents();
    this.buildViews();
    this.setTab(state.ui.tab || 'image', { silent: true });
    this.boot();
  }

  // ------------------------------------------------------------------ boot
  async boot() {
    try {
      const data = await api.state();
      state.applySnapshot(data);
      state.connected = true;
    } catch (err) {
      toast(`Cannot reach the dashboard server: ${err.message}`, { kind: 'err', timeout: 8000 });
    }
    this.renderChrome();
    this.setTab(state.ui.tab || 'image', { silent: true });
    if (this.setup) this.setup.sync();
    await this.loadCatalog();
    this.connect();
    this.startPolling();
  }

  async loadCatalog() {
    if (state.setupRequired) return;
    try {
      const [models, wildcards, presets] = await Promise.all([
        api.models().catch(() => null),
        api.wildcards().catch(() => null),
        api.presets().catch(() => null),
      ]);
      if (models) state.setModels(models);
      if (wildcards) state.setWildcards(wildcards.entries, wildcards.folders);
      if (presets) state.setPresets(presets.presets || presets.entries || []);
      this.modelsLoaded = true;
      this.applyModelDefaults();
    } catch (err) {
      this.modelsLoaded = false;
    }
  }

  /**
   * Choose a model for a form that never had one.
   *
   * Only untouched fields are filled: a user who cleared the model on purpose
   * (to run a checkpoint instead, or none at all) is left alone.
   */
  applyModelDefaults() {
    const list = (category) => (state.models && state.models.models && state.models.models[category]) || [];
    const adopt = (kind, key, options) => {
      if (!options.length) return;
      const form = state.form(kind);
      if (form[key] !== undefined) return;
      if (state.formValue(kind, key)) return;
      state.setFormValue(kind, key, options[0]);
    };
    adopt('image', 'checkpoint', list('checkpoints'));
    adopt('image', 'vae', list('vae'));
    adopt('video', 'model', [...list('diffusion_models'), ...list('unet')]);
    adopt('video', 'clip', list('text_encoders'));
    adopt('video', 'vae', list('vae'));
  }

  async refreshState() {
    try {
      const data = await api.state();
      state.applySnapshot(data);
      if (this.setup) this.setup.sync();
    } catch (err) {
      toast(`Refresh failed: ${err.message}`, { kind: 'err', timeout: 5000 });
    }
  }

  // ------------------------------------------------------------- view setup
  buildViews() {
    if (state.features.image_video && !this.stage) this.stage = new Stage(this.nodes.stageHost);
    if (state.features.music && !this.musicStage) this.musicStage = new MusicStage(this.nodes.musicHost);
    if (!this.panel) {
      this.panel = new Panel(this.nodes.railScroll, state.ui.tab, {
        onRefresh: () => this.refreshState().then(() => this.loadCatalog()),
      });
    }
    if (!this.history) this.history = new History(this.nodes.history);
    if (!this.sidebar) this.sidebar = new Sidebar(this.nodes.sidebar);
    if (!this.logs) this.logs = new LogDrawer(this.nodes.logHost);
    if (!this.setup) this.setup = new SetupWizard(this.nodes.setupHost);
  }

  composerFor(kind) {
    let entry = this.composers.get(kind);
    if (!entry) {
      const host = el('div.composer-host');
      entry = { host, instance: new Composer(host, kind) };
      this.composers.set(kind, entry);
    }
    return entry;
  }

  // ------------------------------------------------------------------- tabs
  availableTabs() {
    return TABS.filter((tab) => !tab.feature || state.features[tab.feature] || tab.id === 'settings');
  }

  renderTabs() {
    clear(this.nodes.tabs);
    for (const tab of TABS) {
      const enabled = !state.featuresKnown || !tab.feature || state.features[tab.feature];
      const button = el('button', {
        class: `tab ${state.ui.tab === tab.id ? 'active' : ''} ${enabled ? '' : 'tab-locked'}`,
        title: enabled ? tab.label : `${tab.label} — switched off in Settings`,
        html: `${icon(tab.id === 'settings' ? 'sliders_h' : tab.icon, 15)}<span>${tab.short}</span>`,
        onclick: () => {
          if (!enabled) {
            toast(`${tab.label} is switched off. Turn it on in Settings › Features.`,
              { kind: 'warn', timeout: 4200 });
            return;
          }
          this.setTab(tab.id);
        },
      });
      this.nodes.tabs.appendChild(button);
    }
  }

  setTab(tab, { silent = false } = {}) {
    const meta = TABS.find((item) => item.id === tab) || TABS[0];
    // A tab whose feature is off lands on Settings. Before the first snapshot the
    // feature flags are only defaults, so the saved tab is kept as it is.
    const blocked = state.featuresKnown && GEN_TABS.has(meta.id) && meta.feature
      && !state.features[meta.feature];
    const target = blocked ? 'settings' : meta.id;
    // setUI publishes, so the history strip and the sidebar follow the tab.
    // The handler below calls back into setTab, which then returns early.
    if (state.ui.tab !== target) state.setUI({ tab: target });
    if (this._rendered === target) return;
    this._rendered = target;
    this.renderTabs();
    this.renderLayout();
    if (!silent) this.nodes.railScroll.scrollTop = 0;
  }

  renderLayout() {
    const tab = state.ui.tab;
    const gen = GEN_TABS.has(tab);
    const settings = tab === 'settings';
    document.body.classList.toggle('no-rail', tab === 'assets');
    document.body.classList.toggle('no-sidebar', settings || !state.ui.sidebarVisible);
    // The history strip belongs to the generation tabs only.
    document.body.classList.toggle('no-history', !gen);

    this.nodes.genView.classList.toggle('hidden', !gen);
    this.nodes.assetsView.classList.toggle('hidden', tab !== 'assets');
    this.nodes.settingsView.classList.toggle('hidden', !settings);

    this.nodes.stageHost.classList.toggle('hidden', tab === 'music');
    this.nodes.musicHost.classList.toggle('hidden', tab !== 'music');

    clear(this.nodes.composerHost);
    if (gen) this.nodes.composerHost.appendChild(this.composerFor(tab).host);

    if (this.nodes.railHead) {
      this.nodes.railHead.classList.toggle('hidden', !gen);
      if (this.nodes.railTitle) {
        this.nodes.railTitle.textContent = gen
          ? `${TABS.find((item) => item.id === tab).short} settings` : '';
      }
    }
    this.nodes.railScroll.classList.toggle('hidden', !gen);
    this.nodes.railNav.classList.toggle('hidden', !settings);

    if (gen && this.panel) {
      this.panel.setKind(tab);
      this.panel.render();
    }
    // The stage follows the tab: the video stage must not keep showing the
    // last image after a tab switch.
    if (gen && this.stage) this.stage.paint();
    if (settings && !this.settingsView) {
      this.settingsView = new SettingsView(this.nodes.settingsView, { nav: this.nodes.railNav });
    }
    if (tab === 'assets' && !this.assets) this.assets = new Assets(this.nodes.assetsView);

    this.renderActionbar();
  }

  // -------------------------------------------------------------- action bar
  renderActionbar() {
    const tab = state.ui.tab;
    if (!GEN_TABS.has(tab)) {
      clear(this.nodes.actionbar);
      this._actionKey = null;
      this._actionParts = null;
      return;
    }

    const active = state.activeJob();
    const running = !!active;
    const engineReady = tab === 'music' ? true : state.engineRunning;
    // The memo covers the tab, the job, the progress, and the two things that
    // change the button without changing the job: a submit in flight, and the
    // engine coming online after the first paint.
    const parts = [tab, running ? 'run' : 'idle', active ? active.id : '-',
                   active ? String(Math.round(active.progress || 0)) : '0',
                   this.busy ? 'busy' : 'ready', engineReady ? 'engine-on' : 'engine-off'];
    const key = parts.join('|');
    if (this._actionKey === key) return;
    const previous = this._actionParts;
    this._actionKey = key;
    this._actionParts = parts;
    const sameTab = !!previous && previous[0] === tab && previous[1] === parts[1]
      && previous[4] === parts[4] && previous[5] === parts[5];
    if (sameTab) { this.updateActionNote(tab, active); return; }

    const kindLabel = { image: 'Generate image', video: 'Generate video', music: 'Generate track' }[tab];
    const iconFor = { image: 'wand', video: 'film', music: 'music' }[tab];

    // Rebuilt in full: the button state (label, disabled, Stop) depends on more
    // than the memo key, so appending without clearing left stale buttons.
    const nodes = [
      el('button', {
        class: 'btn primary big',
        disabled: this.busy || !engineReady,
        title: engineReady ? `${kindLabel} (Ctrl+Enter)` : 'Start the engine first',
        html: `${icon(iconFor, 16)}<span>${this.busy ? 'Submitting…' : kindLabel}</span>`,
        onclick: () => this.generate(tab),
      }),
      running
        ? el('button', {
          class: 'btn danger big',
          html: `${icon('stop', 15)}<span>Stop</span>`,
          onclick: () => this.cancelJob(active),
        })
        : null,
      el('span', { class: 'grow' }),
      el('span', { class: 'queue-note', text: actionNote(tab, active) }),
    ].filter(Boolean);
    clear(this.nodes.actionbar);
    this.nodes.actionbar.append(...nodes);
  }

  updateActionNote(tab, active) {
    const note = this.nodes.actionbar.querySelector('.queue-note');
    if (note) note.textContent = actionNote(tab, active || state.activeJob());
  }

  // ------------------------------------------------------------ generation
  collectParams(kind) {
    const defaults = (state.settings.defaults && state.settings.defaults[kind]) || {};
    const form = structuredClone(state.form(kind));
    const params = { ...structuredClone(defaults), ...form };
    for (const [key, value] of Object.entries(params)) if (value === undefined) delete params[key];
    return params;
  }

  async generate(kind) {
    if (this.busy) return;
    // The prompt input is debounced; take the live text so a click or Enter
    // right after typing uses what the user actually wrote.
    const entry = this.composers.get(kind);
    if (entry && entry.instance && entry.instance.flush) entry.instance.flush();
    const params = this.collectParams(kind);
    const problem = missingPrimary(kind, params);
    if (problem) {
      toast(problem, { kind: 'warn', timeout: 3200 });
      const composer = this.composers.get(kind);
      if (composer && composer.instance.promptBox) composer.instance.promptBox.focus();
      return;
    }
    const modelKey = kind === 'music' ? 'model' : (kind === 'video' ? 'model' : 'checkpoint');
    const unetKey = kind === 'video' ? 'model' : 'checkpoint';
    if (kind !== 'music' && !params[unetKey] && !params[modelKey] && !params.unet) {
      toast('Pick a model first — the Model section has the list.', { kind: 'warn', timeout: 4200 });
      return;
    }
    this.busy = true;
    this.renderActionbar();
    try {
      const data = await api.generate(kind, params, state.session);
      if (data.job) state.upsertJob(data.job);
      state.setUI({ selectedJob: (data.job && data.job.id) || null });
    } catch (err) {
      toast(err.message, { kind: 'err', timeout: 9000 });
    } finally {
      this.busy = false;
      this.renderActionbar();
    }
  }

  async cancelJob(job) {
    const target = job || state.activeJob();
    if (!target) { toast('Nothing is running', { kind: 'info', timeout: 1500 }); return; }
    try {
      await api.cancelJob(target.id);
      toast('Cancelling', { kind: 'warn', timeout: 1800 });
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  // ---------------------------------------------------------------- chrome
  renderChrome() {
    this.renderTabs();
    this.renderEnginePill();
    this.renderQueuePill();
    this.renderMeter();
    const version = $('#version');
    if (version) version.textContent = state.demo ? 'demo' : (state.version || '');
    applyTheme();
  }

  renderEnginePill() {
    const node = this.nodes.enginePill;
    if (!node) return;
    clear(node);
    const engine = state.engine || {};
    if (state.demo) {
      node.className = 'pill run';
      node.innerHTML = `<span class="dot"></span>Demo engine`;
      return;
    }
    if (engine.installing || state.install && state.install.running) {
      node.className = 'pill run';
      node.innerHTML = `<span class="dot"></span>Installing`;
      return;
    }
    if (!engine.installed) {
      node.className = 'pill err';
      node.innerHTML = '<span class="dot"></span>ComfyUI not installed';
      return;
    }
    if (engine.connected || engine.online) {
      node.className = 'pill ok';
      node.innerHTML = `<span class="dot"></span>ComfyUI <b>${engine.version || 'ready'}</b>`;
      return;
    }
    if (engine.running) {
      node.className = 'pill run';
      node.innerHTML = '<span class="dot"></span>ComfyUI starting';
      return;
    }
    node.className = `pill ${engine.update_available ? 'warn' : ''}`.trim();
    node.innerHTML = engine.update_available
      ? `<span class="dot"></span>stopped · <b>${engine.latest_version}</b> available`
      : '<span class="dot"></span>ComfyUI stopped';
  }

  renderQueuePill() {
    const node = this.nodes.queuePill;
    if (!node) return;
    const active = state.activeJob();
    const queued = Object.keys((state.queue && state.queue.active) || {}).length;
    const running = !!active;
    node.className = `pill ${running ? 'run' : ''}`;
    clear(node);
    node.appendChild(el('span', { class: 'dot' }));
    node.appendChild(el('span', {
      text: running
        ? `${labelFor(active.kind)} · ${num(active.progress || 0, 0)}%`
        : (queued ? `${queued} queued` : 'Queue empty'),
    }));
  }

  renderMeter() {
    const node = this.nodes.meter;
    if (!node) return;
    const gpus = (state.system && state.system.gpus) || [];
    const memory = (state.system && state.system.memory) || {};
    const gpu = gpus[0];
    const percentValue = gpu ? Number(gpu.percent || 0) : Number(memory.percent || 0);
    const fill = node.querySelector('.fill');
    const num2 = node.querySelector('.num');
    if (fill) {
      fill.style.width = `${clamp(percentValue, 0, 100)}%`;
      fill.classList.toggle('hot', percentValue > 90);
    }
    if (num2) {
      num2.textContent = gpu
        ? gpu.name.replace(/^NVIDIA\s+/i, '').slice(0, 14)
        : `${num(percentValue, 0)}% RAM`;
    }
    node.title = gpu
      ? `VRAM ${bytes(gpu.memory_used)} / ${bytes(gpu.memory_total)} · driver ${gpu.driver || '?'}`
      : `RAM ${bytes(memory.used)} / ${bytes(memory.total)}`;
  }

  // ---------------------------------------------------------------- socket
  connect() {
    if (this.socket) return;
    const socket = new Socket({ onStatus: (status) => this.onSocketStatus(status) });
    this.socket = socket;

    socket.on('hello', (data) => {
      if (data.session_id) state.session = data.session_id;
      state.version = data.version || state.version;
      state.connected = true;
      this.refreshState();
    });

    socket.on('job', (data) => { if (data.job) state.upsertJob(data.job); });
    socket.on('job_done', (data) => this.onJobDone(data));
    socket.on('job_error', (data) => this.onJobError(data));

    socket.on('progress', (data) => state.upsertJob(patchOf({
      id: data.job_id,
      progress: data.percent !== undefined ? Number(data.percent) : (data.steps
        ? (100 * Number(data.step || 0)) / Number(data.steps) : undefined),
      step: data.step, steps: data.steps,
      stage: data.stage, eta: data.eta,
    })));
    socket.on('stage', (data) => {
      const label = data.node || data.phase || data.stage || '';
      state.upsertJob({ id: data.job_id, stage: String(label).slice(0, 40) });
      if (this.musicStage) this.musicStage.paintFoot();
    });
    socket.on('preview', (data) => { if (this.stage) this.stage.pushFrame(data); });
    socket.on('audio', (data) => state.setAudioPeaks(data.job_id, data));
    socket.on('expected', (data) => {
      state.setExpectedLength(data.job_id, data.seconds);
      if (this.musicStage) {
        this.musicStage.paintFoot();
        const shown = this.musicStage.current();
        if (shown && shown.id === data.job_id) this.musicStage.paintInfo(shown);
      }
    });
    socket.on('music_status', (data) => {
      state.music = { ...(state.music || {}), ...(data.music || {}) };
      if (this.musicStage) this.musicStage.pushStatus(data);
    });
    // SheetSage2 installed or removed: the Cover a song section changes shape.
    socket.on('transcriber', (data) => {
      state.transcriber = { ...(state.transcriber || {}), ...((data && data.transcriber) || {}) };
      state.emit(TOPICS.settings);
    });
    // The plan a running job just wrote: drawn as a piano roll before any sound.
    socket.on('score', (data) => {
      if (this.musicStage) this.musicStage.pushScore(data);
    });
    socket.on('transcription', (data) => this.onTranscription(data));

    socket.on('engine', (data) => {
      state.engine = { ...(state.engine || {}), ...(data.engine || {}) };
      this.renderEnginePill();
      if ((data.engine && (data.engine.connected || data.engine.online)) && !this.modelsLoaded) {
        this.loadCatalog();
      }
    });
    socket.on('engine_error', (data) => {
      toast(`Engine: ${data.error || 'error'}`, { kind: 'err', timeout: 8000 });
      this.refreshState();
    });
    socket.on('engine_log', (data) => state.pushLog(engineEntry(data.line, 'comfyui')));
    socket.on('music_log', (data) => state.pushLog(engineEntry(data.line, 'music')));
    socket.on('install', (data) => this.onInstall(data));
    socket.on('install_done', (data) => {
      if (this.setup) this.setup.installFinished(data);
      if (data.status) state.install = data.status;
      toast(data.ok === false ? `Install stopped: ${data.error || 'see the log'}`
        : 'Install finished', { kind: data.ok === false ? 'err' : 'ok', timeout: 6000 });
      this.buildViews();
      this.refreshState().then(() => this.loadCatalog());
    });

    socket.on('logs', (data) => { if (data.lines) state.pushLog(data.lines); });
    socket.on('settings', (data) => state.setSettings(data.settings, { problems: data.problems }));
    socket.on('nodes', (data) => { state.nodes = data.nodes || []; });
    socket.on('enhancements', (data) => { state.enhancements = data.enhancements || {}; });
    socket.on('history', () => this.reloadHistory());
    socket.on('history_delete', (data) => { if (data.job_id) state.removeJob(data.job_id); });

    socket.connect();
    setInterval(() => socket.ping(), 20000);
  }

  onSocketStatus(status) {
    if (status === 'open') {
      state.connected = true;
      this.reloadHistory();
    } else if (status === 'closed' || status === 'error') {
      state.connected = false;
    }
  }

  /**
   * SheetSage2 finished with a recording: the ABC of a song that was never written down.
   *
   * The result arrives with the choice attached, because transcribing is only half
   * of covering a song — the melody has to go back into a generation.
   */
  onTranscription(data) {
    if (!data) return;
    if (data.error) {
      toast(`Transcription failed: ${data.error}`, { kind: 'err', timeout: 9000 });
      return;
    }
    const summary = data.summary || {};
    const vocal = (summary.voices || {}).Vocal || {};
    const bars = Number(vocal.measures || 0);
    const notes = Number(vocal.notes || 0);
    const bits = [bars ? `${num(bars)} bars` : 'a score', notes ? `${num(notes)} notes` : ''];
    if (summary.bpm) bits.push(`${num(summary.bpm)} BPM`);
    if (summary.key) bits.push(String(summary.key));
    toast(`Transcribed ${bits.filter(Boolean).join(' · ')}`
      + (data.chords ? ` · ${num(data.chords)} chords` : ''), {
      kind: data.valid === false ? 'warn' : 'ok',
      timeout: 10000,
      action: { label: 'Use this melody', run: () => this.useTranscription(data) },
    });
    const warnings = data.warnings || [];
    if (warnings.length) {
      toast(`The transcriber flagged ${warnings.length} thing${warnings.length > 1 ? 's' : ''}`, {
        kind: 'warn',
        timeout: 8000,
        action: { label: 'Read them', run: () => showTranscriptionNotes(warnings) },
      });
    }
    state.transcription = data;
  }

  /**
   * Put a transcribed melody into the music form.
   *
   * Chain of thought moves to `melody`: the model is told to keep this tune and
   * invent the rest, which is what a cover needs. Anything else would let the
   * planner write a different melody over the top of the one just transcribed.
   */
  useTranscription(data) {
    const abc = String((data && data.abc) || '');
    if (!abc.trim()) { toast('That transcription holds no notes', { kind: 'warn' }); return; }
    this.setTab('music');
    state.setForm('music', { abc, cot: 'melody' });
    toast('Melody loaded into the form — the model will follow it', {
      kind: 'ok', timeout: 4200,
    });
  }

  onJobDone(data) {
    if (data.job) state.upsertJob(data.job);
    const job = data.job || { kind: 'image' };
    const name = (job.files && job.files[0] && fileName(job.files[0].name || '')) || job.title || '';
    toast(`${labelFor(job.kind)} finished${name ? ` · ${name}` : ''}${
      data.elapsed ? ` in ${seconds(data.elapsed)}` : ''}`, { kind: 'ok', timeout: 5200 });
    if (this.musicStage) this.musicStage.paint();
  }

  onJobError(data) {
    if (data.job) state.upsertJob(data.job);
    else if (data.job_id) state.upsertJob({ id: data.job_id, status: 'error', error: data.error });
    // Submit-time validation failures are toasted from the HTTP response by the
    // client that posted the job; toasting again here would show it twice.
    if (data.submit) return;
    // A music job that is missing a python package is one click from working.
    const missing = String(data.module || '');
    if (missing) {
      toast(`${missing} is missing from the music environment (.venv-music).`, {
        kind: 'err', title: 'Missing package', timeout: 0,
        action: { label: `Install ${missing}`, run: () => this.installMissingPackage(missing) },
      });
      return;
    }
    toast(`${data.error || 'Generation failed'}`, { kind: 'err', timeout: 9000 });
  }

  async installMissingPackage(name) {
    toast(`Installing ${name} into .venv-music…`, { kind: 'info', timeout: 5000 });
    try {
      await api.musicAction('install_deps', { modules: [name] });
      toast(`${name} installed. Generate the track again.`, { kind: 'ok', timeout: 7000 });
    } catch (err) {
      toast(err.message || 'Install failed', { kind: 'err', timeout: 9000 });
    }
  }

  onInstall(data) {
    if (!data) return;
    state.install = { ...(state.install || {}), stage: data.stage, message: data.message,
      percent: data.percent, at: data.at };
    if (this.setup) this.setup.pushInstall(data);
    this.renderEnginePill();
    if (data.stage === 'failed') toast(data.message || 'Install failed', { kind: 'err', timeout: 9000 });
  }

  async reloadHistory() {
    try {
      const data = await api.jobs({ limit: 200 });
      state.replaceJobs(data.jobs || []);
      state.historyStats = data.stats || state.historyStats;
      state.sessions = data.sessions || [];
    } catch (err) {
      /* the socket will retry */
    }
  }

  // --------------------------------------------------------------- polling
  startPolling() {
    const tick = async () => {
      if (document.hidden) return;
      try {
        const data = await api.system();
        state.system = data.system || state.system;
        state.queue = data.queue || state.queue;
        state.engine = { ...(state.engine || {}), ...(data.engine || {}) };
        this.renderQueuePill();
        this.renderMeter();
        this.renderEnginePill();
      } catch (err) {
        /* offline: the socket reconnect handles it */
      }
    };
    tick();
    setInterval(tick, 5000);
  }

  // ----------------------------------------------------------- global wiring
  bindGlobalEvents() {
    window.addEventListener('autumn:refresh', () => {
      this.refreshState().then(() => this.loadCatalog());
    });
    window.addEventListener('autumn:features', () => {
      this.refreshState().then(() => {
        this.buildViews();
        this.renderTabs();
        this.renderLayout();
      });
    });
    window.addEventListener('autumn:setup', () => {
      if (!this.setup) this.setup = new SetupWizard(this.nodes.setupHost);
      state.setupRequired = true;
      this.setup.done = false;
      this.setup.visible = true;
      this.setup.render();
    });

    document.addEventListener('keydown', (event) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (GEN_TABS.has(state.ui.tab)) this.generate(state.ui.tab);
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'l') {
        event.preventDefault();
        state.setUI({ logsOpen: !state.ui.logsOpen });
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'h') {
        event.preventDefault();
        state.setUI({ historyCollapsed: !state.ui.historyCollapsed });
        return;
      }
      if (event.altKey && event.key.toLowerCase() === 's') {
        event.preventDefault();
        state.setUI({ sidebarVisible: !state.ui.sidebarVisible });
      }
      if (event.key === 'Escape' && !typing) {
        if (state.ui.logsOpen) state.setUI({ logsOpen: false });
      }
    });

    state.on(TOPICS.ui, (patch) => {
      if (!patch) return;
      if (patch.tab) this.setTab(patch.tab, { silent: true });
      if ('historyCollapsed' in patch) this.applyHistoryCollapsed();
      if ('sidebarVisible' in patch) {
        document.body.classList.toggle('no-sidebar', !patch.sidebarVisible);
        const button = document.querySelector('#topbar-buttons .icon-btn[data-tool="side"]');
        if (button) button.classList.toggle('active', !!patch.sidebarVisible);
      }
      if ('theme' in patch || 'accent' in patch) applyTheme();
    });

    state.on(TOPICS.jobs, () => {
      this.renderQueuePill();
      this.renderActionbar();
    });
    // The defaults behind an empty form arrive with the settings, so a settings
    // refresh (first load, seeding, a reset) has to redraw the note too.
    state.on(TOPICS.settings, () => this.updateActionNote(state.ui.tab, state.activeJob()));
    state.on(TOPICS.state, () => {
      this.renderChrome();
      this.renderActionbar();
      // Music can be switched on after boot, and the stage is built once.
      const missingStage = state.features.image_video && !this.stage;
      const missingMusic = state.features.music && !this.musicStage;
      if (missingStage || missingMusic) this.buildViews();
    });
    state.on(TOPICS.models, () => {
      if (this.settingsView) this.settingsView.render();
      // Model defaults can arrive with the list, so the note has to be redrawn.
      this.updateActionNote(state.ui.tab, state.activeJob());
    });
    // The action bar summary reads the form, so it has to follow form edits:
    // without this the note kept the value it had when the tab was built.
    state.on(TOPICS.form, (kind) => {
      if (!kind || kind === state.ui.tab) this.updateActionNote(state.ui.tab, state.activeJob());
    });

    document.addEventListener('click', (event) => {
      const pill = event.target.closest('#engine-pill');
      if (pill) this.engineMenu(pill);
    });

    const chrome = $('#topbar-buttons');
    if (chrome) this.renderTopbarButtons(chrome);

    if (this.nodes.railSearch) {
      this.nodes.railSearch.addEventListener('input', debounce(() => {
        if (!this.panel) return;
        this.panel.setQuery(this.nodes.railSearch.value.trim());
      }, 160));
    }

    this.installResizers();
    this.applyHistoryCollapsed();
    applySizes();
  }

  renderTopbarButtons(host) {
    clear(host);
    const tool = (name, title, run, extraClass = '', data = null) => el('button', {
      class: `icon-btn ${extraClass}`,
      html: icon(name, 15), title, 'aria-label': title,
      dataset: data || {}, onclick: run,
    });
    host.append(
      mountVolumeControl(),
      tool('refresh', 'Re-read models, nodes and versions', () => {
        this.refreshState().then(() => this.loadCatalog());
        toast('Catalog refreshed', { kind: 'ok', timeout: 1600 });
      }, 'spin-on-hover', { tool: 'refresh' }),
      tool('terminal', 'Console (Ctrl+Shift+L)', () => state.setUI({ logsOpen: !state.ui.logsOpen }),
        state.ui.logsOpen ? 'active' : '', { tool: 'logs' }),
      tool('split', 'Session sidebar (Alt+S)', () => {
        state.setUI({ sidebarVisible: !state.ui.sidebarVisible });
      }, state.ui.sidebarVisible ? 'active' : '', { tool: 'side' }),
      tool(state.ui.theme === 'light' ? 'moon' : 'sun', 'Switch theme', () => {
        const theme = state.ui.theme === 'light' ? 'dark' : 'light';
        state.setUI({ theme });
        state.saveSettings({ app: { theme } }).catch(() => { /* local only */ });
      }, '', { tool: 'theme' }),
      tool('github', 'Autumn Diffusion on GitHub',
        () => window.open('https://github.com/frostyautumnleaf/autumn-diffusion', '_blank'),
        '', { tool: 'github' }),
    );
  }

  engineMenu(anchor) {
    const engine = state.engine || {};
    const items = [];
    if (engine.installed) {
      if (engine.running || engine.connected) {
        items.push({ label: 'Stop ComfyUI', icon: 'power', run: () => this.engineAction('stop') });
        items.push({ label: 'Restart ComfyUI', icon: 'refresh', run: () => this.engineAction('restart') });
      } else {
        items.push({ label: 'Start ComfyUI', icon: 'play', run: () => this.engineAction('start') });
      }
      items.push({ label: 'Free VRAM / unload models', icon: 'gpu', run: () => this.engineAction('free') });
      items.push({ sep: true });
    } else {
      items.push({ label: 'Install ComfyUI', icon: 'download', run: () => this.engineAction('install') });
    }
    if (engine.update_available) {
      items.push({ label: 'Update ComfyUI', icon: 'arrow_down',
        hint: engine.latest_version || '', run: () => this.engineAction('update') });
    }
    items.push({ label: 'Check for updates', icon: 'refresh', run: () => this.checkUpdate() });
    items.push({ label: 'Open the ComfyUI web UI', icon: 'external',
      run: () => window.open(engine.url || 'http://127.0.0.1:8188', '_blank') });
    items.push({ sep: true });
    items.push({ label: 'Engine settings', icon: 'sliders_h',
      run: () => { state.setUI({ tab: 'settings' }); } });
    menu(anchor, items, { align: 'left' });
  }

  async engineAction(action) {
    toast(action === 'stop' ? 'Stopping the engine…' : 'Working on the engine…',
      { kind: 'info', timeout: 3000 });
    try {
      await api.engineAction(action, {});
      await this.refreshState();
    } catch (err) {
      toast(err.message, { kind: 'err', timeout: 8000 });
    }
  }

  async checkUpdate() {
    try {
      const data = await api.engineAction('check_update', {});
      const latest = data.latest || '';
      const installed = data.installed || state.engine.version || '';
      if (latest && latest !== installed) {
        toast(`ComfyUI ${latest} is available (you have ${installed || 'unknown'})`,
          { kind: 'warn', timeout: 9000 });
      } else {
        toast(`ComfyUI ${installed || 'unknown'} is up to date`, { kind: 'ok', timeout: 4000 });
      }
      state.engine = { ...(state.engine || {}), latest_version: latest };
      this.renderEnginePill();
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  // --------------------------------------------------------------- resizers
  installResizers() {
    bindResizer($('#resizer-rail'), 'rail', (value) => {
      document.documentElement.style.setProperty('--rail-w', `${value}px`);
    }, (event, start) => start + event.deltaX);
    bindResizer($('#resizer-side'), 'side', (value) => {
      document.documentElement.style.setProperty('--side-w', `${value}px`);
    }, (event, start) => start - event.deltaX);
    bindResizer($('#resizer-history'), 'history', (value) => {
      document.documentElement.style.setProperty('--history-h', `${value}px`);
    }, (event, start) => start - event.deltaY);
  }

  applyHistoryCollapsed() {
    document.body.classList.toggle('history-collapsed', !!state.ui.historyCollapsed);
  }
}

// Resizer ------------------------------------------------------------------
function bindResizer(node, which, apply, deltaOf) {
  if (!node) return;
  const limits = LIMITS[which];
  node.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    const startValue = parseInt(getComputedStyle(document.documentElement)
      .getPropertyValue(limits.css), 10) || state.ui[limits.key] || limits.min;
    let value = startValue;
    node.classList.add('drag');
    node.setPointerCapture(event.pointerId);
    const move = (moveEvent) => {
      const delta = deltaOf ? deltaOf(moveEvent, startValue) : startValue;
      value = clamp(Number.isFinite(delta) ? delta : startValue, limits.min, limits.max);
      apply(value);
    };
    const up = () => {
      node.classList.remove('drag');
      try { node.releasePointerCapture(event.pointerId); } catch (err) { /* already up */ }
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      state.setUI({ [limits.key]: Math.round(value) });
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });
  node.addEventListener('dblclick', () => {
    const defaults = { rail: 372, side: 268, history: 232 };
    const value = defaults[which];
    apply(value);
    state.setUI({ [limits.key]: value });
  });
}

// Small helpers ------------------------------------------------------------
function labelFor(kind) {
  return { image: 'Image', video: 'Video', music: 'Track' }[kind] || 'Job';
}

/** Drop keys whose value is undefined so a sparse patch cannot blank a field. */
function patchOf(patch) {
  const out = {};
  for (const [key, value] of Object.entries(patch)) if (value !== undefined) out[key] = value;
  return out;
}

function actionNote(kind, active) {
  if (active) {
    const bits = [`${labelFor(active.kind)} ${active.id.slice(0, 6)}`];
    if (active.stage) bits.push(String(active.stage));
    if (active.elapsed) bits.push(seconds(active.elapsed));
    return bits.join(' · ');
  }
  const form = state.form(kind);
  const defaults = (state.settings.defaults && state.settings.defaults[kind]) || {};
  const read = (key, fallback = '') => {
    const value = form[key] !== undefined ? form[key] : defaults[key];
    return value === undefined || value === null || value === '' ? fallback : value;
  };
  if (kind === 'music') {
    // YuE decides the track length, so there is no duration to preview here.
    const bits = [String(read('model', 'YuE'))];
    const format = String(read('output_format', 'flac') || '').toUpperCase();
    if (format) bits.push(format);
    return bits.join(' · ');
  }
  const model = read(kind === 'video' ? 'model' : 'checkpoint', '');
  if (!model) {
    return kind === 'video' ? 'Pick a video model to start' : 'Pick a checkpoint to start';
  }
  const bits = [String(model), `${read('width')}x${read('height')}`, `${read('steps')} steps`];
  if (kind === 'video') bits.push(`${read('frames')}f @ ${read('fps')}fps`);
  if (Number(read('batch_size', 1)) > 1) bits.push(`x${read('batch_size')}`);
  return bits.join(' · ');
}

function applyTheme() {
  const root = document.documentElement;
  root.dataset.theme = state.ui.theme || 'dark';
  root.dataset.accent = state.ui.accent || 'amber';
  document.body.dataset.reducedMotion = state.settings && state.settings.app
    && state.settings.app.reduced_motion ? '1' : '0';
}

function applySizes() {
  document.documentElement.style.setProperty('--rail-w', `${state.ui.railWidth}px`);
  document.documentElement.style.setProperty('--side-w', `${state.ui.sideWidth}px`);
  document.documentElement.style.setProperty('--history-h', `${state.ui.historyHeight}px`);
}

/**
 * List what the transcriber was unsure about.
 *
 * SheetSage2 says when it heard a melody that could be one of two things, and a
 * person editing a cover wants to know which bars to listen to before trusting
 * the roll, so the warnings are shown rather than dropped.
 */
function showTranscriptionNotes(warnings) {
  modal({
    title: 'Transcription notes',
    icon: 'warning',
    width: 560,
    body: (root) => {
      clear(root);
      root.appendChild(el('div', { class: 'note-list' }, (warnings || []).map((line) => el(
        'div', { class: 'note-row' }, [
          el('span', { class: 'ic', html: icon('info', 13) }),
          el('span', { class: 'grow', text: String(line) }),
        ]))));
    },
    actions: [{ label: 'Close', kind: 'ghost', run: (close) => close() }],
  });
}

// Boot ---------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
  window.autumn = new App();
  // Debug handle: lets devtools and the screenshot tooling read app state.
  window.autumn.state = state;
});
