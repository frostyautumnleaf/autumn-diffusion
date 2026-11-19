// Central client state: server snapshot, model lists, per-tab forms, job index.
// Views subscribe to named topics and re-render on change.

import { api } from './api.js';
import { loadJSON, saveJSON } from './util.js';

export const TOPICS = {
  state: 'state',
  settings: 'settings',
  jobs: 'jobs',
  form: 'form',
  models: 'models',
  wildcards: 'wildcards',
  presets: 'presets',
  ui: 'ui',
  logs: 'logs',
  music: 'music',
};

const FORM_KEY = 'autumn.forms.v3';
const UI_KEY = 'autumn.ui.v3';

const EMPTY_FORM = {
  image: { regional: { enabled: false, mode: 'area', regions: [] }, loras: [], embeddings: [], controlnets: [], attachments: [] },
  video: { regional: { enabled: false, mode: 'area', regions: [] }, loras: [], controlnets: [], guides: [], references: [], attachments: [],
    interpolate: { enabled: false, model: 'rife20_', multiplier: 2 }, seedvr: { enabled: false } },
  music: { attachments: [] },
};

const listeners = new Map();
const pending = new Map();

/** Height the history block ships with, before the screen is measured. */
const DEFAULT_HISTORY_HEIGHT = 250;

/**
 * Fit the history block to the screen.
 *
 * 250 px is comfortable on a 1000 px window and swallows the stage on a 760 px
 * one, so the shipped default is capped at a quarter of the viewport. A height
 * the user dragged is theirs and is left alone.
 */
function fitHistoryHeight(value) {
  const wanted = Number(value) || DEFAULT_HISTORY_HEIGHT;
  const screen = (typeof window !== 'undefined' && window.innerHeight) || 1000;
  if (wanted !== DEFAULT_HISTORY_HEIGHT) return wanted;
  return Math.round(Math.max(170, Math.min(wanted, screen * 0.26)));
}
let frame = null;

function dispatch(topic, detail) {
  for (const handler of (listeners.get(topic) || []).slice()) {
    try { handler(detail, topic); } catch (err) { console.error('listener failed', topic, err); }
  }
  for (const handler of (listeners.get('*') || []).slice()) {
    try { handler(topic, detail); } catch (err) { console.error('wildcard listener failed', err); }
  }
}

// Notifications are coalesced into one animation frame so a burst of progress
// events does not re-render the panels more than once per frame.
function notify(topic, detail) {
  if (pending.get(topic) === 'suppress') return;
  const previous = pending.get(topic);
  // Two patches for the same topic in one frame are merged, otherwise the last
  // one hides the keys of the earlier one and subscribers miss a change.
  const mergeable = detail && typeof detail === 'object' && !Array.isArray(detail)
    && previous && typeof previous === 'object' && !Array.isArray(previous);
  pending.set(topic, mergeable ? { ...previous, ...detail } : (detail ?? null));
  if (frame !== null) return;
  frame = requestAnimationFrame(() => {
    frame = null;
    const batch = [...pending.entries()];
    pending.clear();
    for (const [name, payload] of batch) dispatch(name, payload);
  });
}

function merge(target, source) {
  const out = { ...target };
  for (const [key, value] of Object.entries(source || {})) {
    if (value && typeof value === 'object' && !Array.isArray(value) &&
        target[key] && typeof target[key] === 'object' && !Array.isArray(target[key])) {
      out[key] = merge(target[key], value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

export const state = {
  connected: false,
  version: '',
  configured: true,
  setupRequired: false,
  demo: false,
  features: { image_video: true, music: false, seedvr: false, configured: true },
  /** False until the first snapshot arrives, so a saved tab is not judged on defaults. */
  featuresKnown: false,
  settings: {},
  problems: [],
  engine: {},
  nodes: [],
  enhancements: { kjnodes: false, taeh3: false, seedvr2: false },
  capabilities: {},
  music: {},
  install: {},
  system: {},
  queue: {},
  session: '',
  paths: {},
  models: { models: {}, categories: [], samplers: [], schedulers: [], image_files: [], video_files: [], audio_files: [] },
  wildcards: [],
  wildcardFolders: [],
  presets: [],
  jobs: new Map(),
  historyStats: {},
  sessions: [],
  preview: null,
  audioPeaks: new Map(),
  logLines: [],
  ui: {
    tab: 'image',
    selectedJob: null,
    railFilter: '',
    sections: {},
    stageView: 'auto',
    historyCollapsed: false,
    sidebarVisible: true,
    railWidth: 372,
    sideWidth: 268,
    historyHeight: fitHistoryHeight(DEFAULT_HISTORY_HEIGHT),
    theme: 'dark',
    accent: 'amber',
    logsOpen: false,
    volume: 0.8,
    muted: false,
    assetsCategory: 'checkpoints',
    assetsSubfolder: '',
    assetsQuery: '',
    assetsSelect: null,
  },
  forms: loadJSON(FORM_KEY, null) || structuredClone(EMPTY_FORM),

  // Subscription ---------------------------------------------------------
  on(topic, handler) {
    if (!listeners.has(topic)) listeners.set(topic, []);
    listeners.get(topic).push(handler);
    return () => this.off(topic, handler);
  },
  off(topic, handler) {
    const list = listeners.get(topic);
    if (!list) return;
    const index = list.indexOf(handler);
    if (index >= 0) list.splice(index, 1);
  },
  emit(topic, detail) { notify(topic, detail); },

  // Server sync ----------------------------------------------------------
  applySnapshot(data) {
    const snapshot = data && data.settings ? data : (data && data.state ? data.state : data);
    if (!snapshot) return;
    this.version = snapshot.version || this.version;
    this.configured = snapshot.configured !== false;
    this.setupRequired = !!snapshot.setup_required;
    this.demo = !!snapshot.demo;
    if (snapshot.features) {
      this.features = snapshot.features;
      this.featuresKnown = true;
    }
    if (snapshot.settings) this.settings = snapshot.settings;
    this.problems = snapshot.problems || [];
    this.engine = snapshot.engine || this.engine;
    this.nodes = snapshot.nodes || this.nodes;
    this.enhancements = snapshot.enhancements || this.enhancements;
    this.capabilities = snapshot.capabilities || this.capabilities;
    this.music = snapshot.music || this.music;
    this.install = snapshot.install || this.install;
    this.system = snapshot.system || this.system;
    this.queue = snapshot.queue || this.queue;
    this.session = snapshot.session_id || this.session;
    this.paths = snapshot.paths || this.paths;
    if (snapshot.history) {
      this.replaceJobs(snapshot.history.jobs || []);
      this.historyStats = snapshot.history.stats || this.historyStats;
    }
    if (snapshot.preview) this.preview = snapshot.preview;
    this.applySettingsToUi();
    notify(TOPICS.state);
    notify(TOPICS.settings);
  },

  applySettingsToUi() {
    const app = (this.settings && this.settings.app) || {};
    if (app.theme) this.ui.theme = app.theme;
    if (app.accent) this.ui.accent = app.accent;
    if (typeof app.sidebar_visible === 'boolean') this.ui.sidebarVisible = app.sidebar_visible;
    if (typeof app.history_visible === 'boolean') this.ui.historyCollapsed = !app.history_visible;
    if (app.history_height) this.ui.historyHeight = fitHistoryHeight(app.history_height);
    this.persistUI();
  },

  setSettings(settings, meta = {}) {
    this.settings = merge(this.settings || {}, settings || {});
    if (meta.problems) this.problems = meta.problems;
    this.applySettingsToUi();
    notify(TOPICS.settings);
  },

  async saveSettings(patch) {
    const data = await api.saveSettings(patch);
    this.setSettings(data.settings);
    if (data.problems) this.problems = data.problems;
    notify(TOPICS.state, 'batched');
    return data;
  },

  setModels(data) {
    this.models = { ...this.models, ...data };
    notify(TOPICS.models);
  },

  setWildcards(entries, folders) {
    this.wildcards = entries || [];
    if (folders) this.wildcardFolders = folders;
    notify(TOPICS.wildcards);
  },

  setPresets(items) {
    this.presets = items || [];
    notify(TOPICS.presets);
  },

  // Jobs -----------------------------------------------------------------
  replaceJobs(jobs) {
    const seen = new Set();
    for (const job of jobs) {
      seen.add(job.id);
      const existing = this.jobs.get(job.id);
      this.jobs.set(job.id, existing ? { ...existing, ...job } : job);
    }
    for (const id of [...this.jobs.keys()]) if (!seen.has(id)) this.jobs.delete(id);
    notify(TOPICS.jobs);
  },

  upsertJob(job) {
    if (!job || !job.id) return;
    const existing = this.jobs.get(job.id);
    this.jobs.set(job.id, existing ? { ...existing, ...job } : job);
    notify(TOPICS.jobs, job);
  },

  removeJob(id) {
    this.jobs.delete(id);
    if (this.ui.selectedJob === id) this.ui.selectedJob = null;
    notify(TOPICS.jobs);
  },

  jobList({ kind, limit = 400, session, favorites = false, query = '' } = {}) {
    const items = [...this.jobs.values()]
      .filter((job) => !kind || job.kind === kind)
      .filter((job) => !session || job.session_id === session)
      .filter((job) => !favorites || job.favorite)
      .filter((job) => !query || `${job.title} ${job.prompt} ${job.id}`.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    return limit ? items.slice(0, limit) : items;
  },

  sessionJobs() {
    return [...this.jobs.values()]
      .filter((job) => job.session_id === this.session)
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  },

  activeJob() {
    const running = this.jobList({ limit: 400 }).filter(
      (job) => job.status === 'running' || job.status === 'queued');
    return running.length ? running[0] : null;
  },

  // Forms ----------------------------------------------------------------
  form(kind) {
    if (!this.forms[kind]) this.forms[kind] = structuredClone(EMPTY_FORM[kind] || {});
    return this.forms[kind];
  },

  formValue(kind, key) {
    const form = this.form(kind);
    if (form[key] !== undefined) return form[key];
    const defaults = (this.settings.defaults && this.settings.defaults[kind]) || {};
    if (defaults[key] !== undefined) return defaults[key];
    const section = (this.settings[kind === 'music' ? 'music' : kind]) || {};
    return section[key];
  },

  setForm(kind, patch) {
    this.forms[kind] = { ...this.forms[kind], ...patch };
    this.persistForms();
    notify(TOPICS.form, kind);
  },

  setFormValue(kind, key, value) {
    if (!String(key).includes('.')) {
      this.setForm(kind, { [key]: value });
      return;
    }
    // Dotted keys address a nested value (`regional.enabled`, `seedvr.enabled`).
    // Writing them flat left a stray `"regional.enabled"` key next to an object
    // that never changed, so the panel and the submit both saw the old value.
    const form = { ...this.form(kind) };
    const parts = String(key).split('.');
    let node = form;
    for (let index = 0; index < parts.length - 1; index += 1) {
      const current = node[parts[index]];
      node[parts[index]] = Array.isArray(current)
        ? current.slice()
        : (current && typeof current === 'object' ? { ...current } : {});
      node = node[parts[index]];
    }
    node[parts[parts.length - 1]] = value;
    this.forms[kind] = form;
    this.persistForms();
    notify(TOPICS.form, kind);
  },

  resetForm(kind) {
    const defaults = (this.settings.defaults && this.settings.defaults[kind]) || {};
    this.forms[kind] = { ...structuredClone(EMPTY_FORM[kind] || {}), ...structuredClone(defaults) };
    this.persistForms();
    notify(TOPICS.form, kind);
  },

  loadForm(kind, params) {
    this.forms[kind] = { ...structuredClone(EMPTY_FORM[kind] || {}), ...structuredClone(params || {}) };
    this.persistForms();
    notify(TOPICS.form, kind);
  },

  persistForms() { saveJSON(FORM_KEY, this.forms); },

  // Derived --------------------------------------------------------------
  get engineRunning() { return !!(this.engine.running || this.engine.online); },
  get engineInstalled() { return !!this.engine.installed; },
  get seedvrInstalled() { return !!this.enhancements.seedvr2; },
  get musicInstalled() { return !!(this.music && (this.music.installed || this.music.stub)); },
  get hqPreviewReady() { return !!(this.enhancements.kjnodes && this.enhancements.taeh3); },
  get hqPreviewOn() { return !!(this.settings.preview && this.settings.preview.hq_minimax_h3); },
  get tabList() {
    const tabs = [];
    if (this.features.image_video) { tabs.push('image', 'video'); }
    if (this.features.music) tabs.push('music');
    tabs.push('assets', 'settings');
    return tabs;
  },

  setUI(patch) {
    this.ui = { ...this.ui, ...patch };
    this.persistUI();
    notify(TOPICS.ui, patch);
  },

  persistUI() {
    const { tab, selectedJob, railFilter, sections, stageView, historyCollapsed,
      sidebarVisible, railWidth, sideWidth, historyHeight, theme, accent,
      assetsCategory, volume, muted } = this.ui;
    saveJSON(UI_KEY, { tab, selectedJob, railFilter, sections, stageView, historyCollapsed,
      sidebarVisible, railWidth, sideWidth, historyHeight, theme, accent, assetsCategory,
      volume, muted });
  },

  restoreUI() {
    const saved = loadJSON(UI_KEY, null);
    if (saved) this.ui = { ...this.ui, ...saved };
  },

  /** Length the audio of a music job is expected to reach, from the worker. */
  setExpectedLength(jobId, seconds) {
    const job = this.jobs.get(jobId);
    if (!job) return;
    const value = Number(seconds) || 0;
    const meta = { ...(job.meta || {}), expected_seconds: value };
    if ((job.meta && job.meta.expected_seconds) === value) return;
    this.jobs.set(jobId, { ...job, meta });
    notify(TOPICS.music, jobId);
    notify(TOPICS.jobs);
  },

  pushLog(entry) {
    this.logLines.push(entry);
    if (this.logLines.length > 1200) this.logLines.splice(0, this.logLines.length - 1200);
    notify(TOPICS.logs, entry);
  },

  setAudioPeaks(jobId, payload) {
    const current = this.audioPeaks.get(jobId)
      || { peaks: [], duration: 0, sampleRate: 48000, perSecond: 0 };
    const peaks = current.peaks.slice();
    // The first tile fixes the grid for the whole track. A tile bucketed at a
    // different rate is resampled onto it, otherwise the two rates drift apart and
    // the waveform shows silent holes where audio really plays.
    const rate = current.perSecond || payload.peaks_per_second || 25;
    const tileRate = payload.peaks_per_second
      || (payload.peaks && payload.peaks.length
        ? payload.peaks.length / Math.max(0.0001, payload.duration || 1) : rate);
    const scale = tileRate > 0 ? tileRate / rate : 1;
    const start = Math.floor((payload.start || 0) * rate);
    if (start > peaks.length) peaks.fill(0, peaks.length, start);
    for (let i = 0; i < payload.peaks.length; i += 1) {
      const source = Math.min(payload.peaks.length - 1, Math.floor(i / scale));
      peaks[start + i] = payload.peaks[source];
    }
    const played = Math.max(0, (payload.start || 0) + (payload.duration || 0));
    this.audioPeaks.set(jobId, {
      peaks,
      duration: Math.max(current.duration, played, payload.buffered || 0),
      sampleRate: payload.sample_rate || current.sampleRate,
      perSecond: rate,
      // The growing WAV the worker writes, so the decoded part can be listened to.
      url: payload.url || current.url || '',
      revision: Number(payload.revision) || current.revision || 0,
      buffered: Math.max(current.buffered || 0, Number(payload.buffered) || 0, played),
    });
    notify(TOPICS.music, jobId);
  },
};

state.restoreUI();
