// Music stage: the middle of the window on the music tab.
//
// Two views share the space: the interactive waveform of the selected track, with
// a live waveform that fills in as VAE tiles are decoded, and the piano roll of
// the ABC score YuE plans before it makes any sound. The waveform lives in
// waveform.js and the roll in score.js; this class keeps both fed and decides
// which one is on screen.

import { state, TOPICS } from './state.js';
import { icon } from './icons.js';
import { toast, lightbox, attachContextMenu } from './components.js';
import { Waveform } from './waveform.js';
import { ScoreRoll } from './score.js';
import { el, clear, baseName, mediaUrl, seconds as secs, dateTime, num, copyText } from './util.js';
import { describe, summarise, modelLabel } from './preview.js';
import { loadIntoForm } from './history.js';

/** YuE pipeline stages, in the order the worker reports them. */
const STAGES = [
  { id: 'load', label: 'Load model', hint: 'Weights are moved onto the GPU' },
  { id: 'plan', label: 'Token plan', hint: 'Lyrics and length become token slots' },
  { id: 'semantic', label: 'Semantic tokens', hint: 'Stage 1: melody and structure' },
  { id: 'nar', label: 'Acoustic tokens', hint: 'Stage 2: codebook frames' },
  { id: 'vae', label: 'Decode audio', hint: 'Waveform is decoded tile by tile' },
];

export class MusicStage {
  constructor(root) {
    this.root = root;
    this.jobId = null;
    this.status = null;      // last `music_status` payload
    this.scoreJobId = null;  // job the score on screen belongs to
    this.score = null;       // last score payload, kept across a repaint
    this.view = state.ui.musicView || 'wave';
    this.waveform = new Waveform(el('div'));
    this.roll = new ScoreRoll({
      // The player owns the clock, so the roll asks it where it is each frame and
      // clicks back into it. One play head, in both views.
      onTime: () => this.waveform.playhead,
      onSeek: (seconds) => this.waveform.seek(seconds),
    });
    this.render();

    state.on(TOPICS.jobs, () => this.syncJob());
    state.on(TOPICS.ui, (patch) => {
      if (!patch) return;
      if (patch.selectedJob !== undefined) this.syncJob();
      if (patch.musicView !== undefined && patch.musicView !== this.view) {
        this.view = patch.musicView;
        this.applyView();
      }
    });
  }

  /** A `music_status` event from the engine. */
  pushStatus(data) {
    this.status = data || null;
    if (data && data.job_id && data.job_id !== this.jobId) return;
    // The pill in the bar reads this status, so both pieces are refreshed together.
    this.paintFoot();
    this.paintBar();
  }

  /**
   * A `score` event: the ABC the planner wrote, already parsed server-side.
   *
   * The plan arrives before any audio, so the roll is the first thing a run puts
   * on screen. Selecting another job while one runs must not drop this score, so
   * it is keyed by job id and re-applied by `loadScore`.
   */
  pushScore(data) {
    const jobId = data && data.job_id ? String(data.job_id) : this.jobId;
    if (!jobId || jobId !== this.jobId) return;
    this.score = data;
    this.scoreJobId = jobId;
    this.roll.setScore(data);
    this.paintScoreFlag();
  }

  /** Fetch the score of a job, which is what brings back an old track's plan. */
  async loadScore(job) {
    const id = job ? job.id : null;
    if (!id) {
      this.score = null;
      this.scoreJobId = null;
      this.roll.setScore(null);
      return;
    }
    if (this.score && this.scoreJobId === id) {
      this.roll.setScore(this.score);
      this.paintScoreFlag();
      return;
    }
    this.scoreJobId = id;
    const meta = (job.meta || {}).score || {};
    // A compose job delivers a score and no audio, so ask even when nothing played.
    const hasScore = Boolean(meta.path || meta.abc || (job.files || []).length
      || (job.meta || {}).score);
    if (!hasScore && job.status !== 'running') {
      this.score = null;
      this.roll.setScore(null);
      this.paintScoreFlag();
      return;
    }
    try {
      const { api } = await import('./api.js');
      const data = await api.jobScore(id);
      if (this.scoreJobId !== id) return;
      this.score = data;
      this.roll.setScore(data);
    } catch (err) {
      if (this.scoreJobId !== id) return;
      this.score = null;
      this.roll.setScore(err.payload && err.payload.abc
        ? { abc: err.payload.abc, valid: false, error: err.message } : null);
    }
    this.paintScoreFlag();
  }

  render() {
    clear(this.root);
    this.root.classList.add('stage', 'music-stage');
    this.barBuilt = false;
    this.footBuilt = false;

    this.bar = el('div', { class: 'stage-bar' });
    this.body = el('div', { class: 'stage-body' });
    this.canvas = el('div', { class: 'stage-canvas music-canvas' });
    this.progress = el('div', { class: 'stage-progress' }, [el('div', { class: 'bar' })]);
    this.foot = el('div', { class: 'stage-foot' });

    this.body.appendChild(this.canvas);
    this.body.appendChild(this.progress);
    this.root.appendChild(this.bar);
    this.root.appendChild(this.body);
    this.root.appendChild(this.foot);
    this.paint();
    attachContextMenu(this.root, () => this.contextItems());
  }

  /**
   * Right-click items for the stage.
   *
   * The waveform canvas has its own menu for playback and zoom, so this one covers
   * the track as a job: its file, its settings and its place in the queue.
   */
  contextItems() {
    const job = this.current();
    const file = job ? (job.files || [])[0] : null;
    const running = !!job && (job.status === 'running' || job.status === 'queued');
    const playing = !!this.waveform.playing;
    return [
      { label: playing ? 'Pause the track' : 'Play the track', icon: playing ? 'pause' : 'play',
        disabled: !this.waveform.jobId, run: () => this.waveform.toggle() },
      { label: 'Back to the start', icon: 'skip', disabled: !this.waveform.jobId,
        run: () => this.waveform.seek(0) },
      { label: 'Stop every track', icon: 'stop', run: () => this.waveform.pauseAll() },
      { sep: true },
      { label: this.view === 'score' ? 'Show the waveform' : 'Show the planned score',
        icon: this.view === 'score' ? 'wave' : 'code',
        run: () => this.setView(this.view === 'score' ? 'wave' : 'score') },
      { label: 'Copy the ABC score', icon: 'copy', disabled: !(this.score && this.score.abc),
        run: () => copyText(this.score.abc)
          .then(() => toast('ABC copied — paste it into Composition to edit', { kind: 'ok', timeout: 2200 }))
          .catch(() => toast('Could not copy', { kind: 'err' })) },
      { sep: true },
      { label: 'Open in the viewer', icon: 'expand', disabled: !file, run: () => this.openResult() },
      { label: 'Download the track', icon: 'download', disabled: !file,
        run: () => window.open(mediaUrl(file), '_blank') },
      { sep: true },
      { label: 'Load these settings into the form', icon: 'sliders_h', disabled: !job,
        run: () => loadIntoForm(job) },
      { label: 'Copy the settings as JSON', icon: 'code', disabled: !job,
        run: () => copyTrackSettings(job) },
      { label: 'Copy the lyrics', icon: 'copy', disabled: !(job && job.prompt),
        run: () => copyText(job.prompt || '')
          .then(() => toast('Lyrics copied', { kind: 'ok', timeout: 1400 }))
          .catch(() => toast('Could not copy', { kind: 'err' })) },
      { label: 'Cancel this job', icon: 'stop', danger: true, disabled: !running,
        run: () => this.cancel() },
    ];
  }

  // ------------------------------------------------------------------ pieces
  /**
   * The bar above the stage.
   *
   * Built once and patched in place afterwards: a progress event arrives many times
   * a second, and a rebuild makes the whole strip flicker.
   */
  paintBar() {
    const job = this.current();
    const running = !!job && (job.status === 'running' || job.status === 'queued');
    if (!this.barBuilt) {
      clear(this.bar);
      this.titleNode = el('span', { class: 'stage-title' });
      this.statusNode = el('span', { class: 'pill' });
      this.bar.append(
        this.titleNode,
        this.statusNode,
        el('span', { class: 'grow' }),
        el('button', {
          class: 'icon-btn', html: icon('expand', 14), title: 'Open in the viewer',
          onclick: () => this.openResult(),
        }),
        el('button', {
          class: 'icon-btn danger', html: icon('stop', 14), title: 'Cancel the running job',
          onclick: () => this.cancel(),
        }),
      );
      this.barBuilt = true;
      this._barText = '';
      this._barState = '';
    }
    const title = job ? describe(job) : 'No track selected';
    if (title !== this._barText) {
      this.titleNode.textContent = title;
      this._barText = title;
    }
    const label = statusText(this.status);
    const flavour = running ? 'run' : (state.musicInstalled ? 'ok' : 'warn');
    const signature = `${flavour}|${label}`;
    if (signature !== this._barState) {
      this.statusNode.className = `pill ${flavour}`;
      this.statusNode.innerHTML = `<span class="dot"></span>${label}`;
      this._barState = signature;
    }
  }

  paint() {
    this.paintBar();
    const job = this.current();
    clear(this.canvas);

    const head = el('div', { class: 'music-head' }, [
      el('div', { class: 'grow' }, [
        el('div', { class: 'music-title', text: job ? (baseName(job.title || '') || 'Untitled track') : 'Nothing to play' }),
        el('div', { class: 'music-sub', text: job ? musicLine(job) : 'Write a prompt, then press Generate.' }),
      ]),
      this.viewSwitch(),
      job ? el('button', {
        class: 'btn sm ghost', html: `${icon('copy', 12)}<span>Copy settings</span>`,
        onclick: () => copyTrackSettings(job),
      }) : null,
    ]);

    const waveHost = el('div', { class: 'music-wave' });
    const scoreHost = el('div', { class: 'music-roll' });
    this.waveHost = waveHost;
    this.scoreHost = scoreHost;
    this.waveform.mount(waveHost);
    this.roll.mount(scoreHost);
    // The transport belongs to the track, not to one view: the piano roll follows
    // the play head, so it needs the same play / seek / volume row underneath it.
    this.canvas.appendChild(el('div', { class: 'music-shell anim-up' },
      [head, waveHost, scoreHost, this.waveform.player]));

    if (!job) {
      this.canvas.appendChild(el('div', { class: 'music-idle' }, [
        el('div', { class: 'e-ic', html: icon('wave', 24) }),
        el('div', { html: '<b>No track yet</b>' }),
        el('div', { class: 'tiny faint', text: 'Waveforms appear here as soon as audio starts decoding.' }),
      ]));
      this.waveform.setJob(null, { live: false });
      this.loadScore(null);
    } else {
      const running = job.status === 'running' || job.status === 'queued';
      this.waveform.setJob(job, { live: running && job.status === 'running' });
      this.paintInfo(job);
      this.loadScore(job);
    }

    this.applyView();
    this.paintProgress(job);
    this.paintFoot();
  }

  /**
   * The Waveform / Score switch.
   *
   * The score is only interesting for a job that has one, but the control stays
   * visible either way: a switch that appears and disappears moves the buttons
   * beside it, and a click meant for "copy settings" lands somewhere else.
   */
  viewSwitch() {
    const build = (id, label, iconName) => el('button', {
      class: this.view === id ? 'on' : '',
      html: `${icon(iconName, 12)}<span>${label}</span>`,
      title: id === 'wave' ? 'Show the waveform of the track'
        : 'Show the planned score as a piano roll',
      onclick: () => this.setView(id),
    });
    return el('div', { class: 'segmented music-views' }, [
      build('wave', 'Waveform', 'wave'),
      build('score', 'Score', 'code'),
    ]);
  }

  setView(view) {
    if (view === this.view) return;
    this.view = view;
    state.setUI({ musicView: view });
    // The head is rebuilt rather than patched: two buttons and no flicker risk.
    const job = this.current();
    const head = this.canvas.querySelector('.music-head');
    if (head && job) {
      const old = head.querySelector('.music-views');
      if (old) old.replaceWith(this.viewSwitch());
    }
    this.applyView();
  }

  /** Show one view, and let only the visible one animate. */
  applyView() {
    const score = this.view === 'score';
    if (this.waveHost) this.waveHost.style.display = score ? 'none' : '';
    if (this.scoreHost) this.scoreHost.style.display = score ? '' : 'none';
    this.root.classList.toggle('show-score', score);
    this.roll.setActive(score);
    if (score) {
      // The roll is laid out hidden, so its size is only known now.
      requestAnimationFrame(() => this.roll.resize());
    }
  }

  /** Mark the switch when a plan exists, so the score is not hidden by accident. */
  paintScoreFlag() {
    const node = this.canvas.querySelector('.music-views');
    if (!node) return;
    const has = Boolean(this.score && this.score.grid);
    node.classList.toggle('has-score', has);
    const score = node.lastElementChild;
    if (score && score.firstElementChild) {
      score.title = has ? 'Show the planned score as a piano roll'
        : 'Show the plan as a piano roll — this job has none';
    }
  }

  paintInfo(job) {
    const meta = job.meta || {};
    const running = job.status === 'running' || job.status === 'queued';
    // While a track decodes the worker reports the length it is aiming at, so the
    // number does not creep up as tiles arrive.
    const planned = Number(meta.expected_seconds || 0);
    const length = meta.audio_seconds
      || (running ? (planned || this.waveform.timeline()) : this.waveform.duration)
      || planned;
    const rows = [
      ['Model', modelLabel(job) || state.music.model || 'YuE2'],
      ['Length', secs(length)],
      ['Sample rate', meta.sample_rate ? `${Math.round(meta.sample_rate / 100) / 10} kHz` : '48 kHz'],
      ['Channels', meta.channels === 1 ? 'mono' : 'stereo'],
      ['Seed', String(job.params && job.params.seed !== undefined ? job.params.seed : meta.seed ?? '—')],
      ['Created', dateTime(job.created_at)],
    ];
    const shell = this.canvas.querySelector('.music-shell');
    if (!shell) return;
    // The block is repainted when the worker reports a length, so drop the old one.
    const old = shell.querySelector('.music-meta');
    if (old) old.remove();
    shell.appendChild(el('div', { class: 'music-meta' },
      rows.filter((row) => row[1] !== '' && row[1] !== null).map((row) => el('div', { class: 'stat' }, [
        el('div', { class: 'k', text: row[0] }),
        el('div', { class: 'v', text: String(row[1]) }),
      ]))));
  }

  paintProgress(job) {
    const running = !!job && (job.status === 'running' || job.status === 'queued');
    const raw = Number((job && job.progress) || 0);
    const value = Math.max(0, Math.min(100, raw <= 1 ? raw * 100 : raw));
    this.progress.classList.toggle('active', running);
    this.progress.querySelector('.bar').style.width = `${running ? (value || 4) : 0}%`;
  }

  /**
   * The pipeline strip under the stage.
   *
   * Patched rather than rebuilt: five stage chips and one note, with only the fields
   * that moved written. Rebuilding here made the foot blink on every decoded tile.
   */
  paintFoot() {
    const job = this.current();
    if (!this.footBuilt) {
      clear(this.foot);
      this.strip = el('div', { class: 'stage-strip' });
      this.stageNodes = STAGES.map((item) => el('span', { class: 'ss-item', title: item.hint }, [
        el('i', { class: 'dot' }),
        el('span', { text: item.label }),
        el('b'),
      ]));
      for (const node of this.stageNodes) this.strip.appendChild(node);
      this.queueNote = el('span', { class: 'queue-note' });
      this.foot.append(this.strip, this.queueNote);
      this.footBuilt = true;
      this._footState = '';
    }
    if (!job || (job.status !== 'running' && job.status !== 'queued')) {
      const last = state.music && state.music.last_error;
      const note = last ? `YuE error: ${String(last).slice(0, 90)}`
        : 'YuE worker idles; it loads on the first job.';
      this.strip.style.display = 'none';
      if (this.queueNote.textContent !== note) this.queueNote.textContent = note;
      this._footState = '';
      return;
    }
    this.strip.style.display = '';
    const stageName = String(job.stage || (job.meta && job.meta.stage)
      || (this.status && this.status.stage) || 'plan');
    const index = STAGES.findIndex((item) => item.id === stageName);
    const raw = Number(job.progress || 0);
    const eta = job.eta || (this.status && this.status.eta);
    const bits = [];
    if (eta) bits.push(`~${secs(eta)} left`);
    if (job.elapsed) bits.push(`${secs(job.elapsed)} elapsed`);
    const note = bits.join(' · ');
    const signature = `${index}|${Math.round(raw)}|${note}`;
    if (signature === this._footState) return;
    this._footState = signature;
    STAGES.forEach((item, position) => {
      const node = this.stageNodes[position];
      const done = index >= 0 && position < index;
      const active = item.id === stageName;
      const value = active ? (raw <= 1 ? raw * 100 : raw) : (done ? 100 : 0);
      const className = `ss-item ${done ? 'done' : ''} ${active ? 'on' : ''}`;
      if (node.className !== className) node.className = className;
      const shown = done ? '' : (active ? `${num(value, 0)}%` : '');
      const slot = node.lastElementChild;
      if (slot.textContent !== shown) slot.textContent = shown;
    });
    if (this.queueNote.textContent !== note) this.queueNote.textContent = note;
  }

  // ------------------------------------------------------------------ helpers
  current() {
    const selected = state.ui.selectedJob ? state.jobs.get(state.ui.selectedJob) : null;
    const running = state.jobList({ kind: 'music', limit: 20 }).find(
      (job) => job.status === 'running' || job.status === 'queued');
    if (selected && selected.kind === 'music') return selected;
    if (running) return running;
    const music = state.jobList({ kind: 'music', limit: 1 })[0];
    return music || null;
  }

  syncJob() {
    const job = this.current();
    const id = job ? job.id : null;
    const changed = id !== this.jobId;
    this.jobId = id;
    if (changed) { this.paint(); return; }
    this.paintProgress(job);
    this.paintFoot();
    this.paintBar();
    this.paintSub(job);
  }

  /** Keep the line under the title current, including its percentage. */
  paintSub(job) {
    const node = this.canvas.querySelector('.music-sub');
    if (node && job) node.textContent = musicLine(job);
  }

  async cancel() {
    const job = state.activeJob();
    if (!job) { toast('Nothing to cancel', { kind: 'info', timeout: 1400 }); return; }
    try {
      await (await import('./api.js')).api.cancelJob(job.id);
      toast('Cancelling the track', { kind: 'warn', timeout: 1600 });
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  openResult() {
    const job = this.current();
    const file = job && job.files && job.files[0];
    if (!file) { toast('This track has no output', { kind: 'warn' }); return; }
    lightbox({
      title: baseName(file.name || job.title || job.id),
      subtitle: job.prompt || '',
      url: mediaUrl(file),
      kind: 'audio',
      meta: [
        ['Job', job.id],
        ['Created', dateTime(job.created_at)],
        ['Model', modelLabel(job)],
        ['Settings', summarise(job)],
      ],
      actions: [{ label: 'Download', icon: 'download', run: () => window.open(mediaUrl(file), '_blank') }],
    });
  }
}

function statusText(data) {
  if (!data) return state.musicInstalled ? 'YuE' : 'YuE not installed';
  const stage = data.stage || data.state || '';
  if (stage) return `YuE · ${stage}`;
  if (data.state === 'ready') return 'YuE ready';
  return state.musicInstalled ? 'YuE' : 'YuE not installed';
}

function musicLine(job) {
  const meta = job.meta || {};
  const bits = [job.prompt ? job.prompt.slice(0, 90) : ''];
  if (meta.audio_seconds) bits.push(`${secs(meta.audio_seconds)}`);
  if (job.status === 'running') {
    const raw = Number(job.progress || 0);
    bits.push(`${num(raw <= 1 ? raw * 100 : raw, 0)}%`);
  }
  return bits.filter(Boolean).join(' · ') || summarise(job);
}

async function copyTrackSettings(job) {
  const { copyText } = await import('./util.js');
  copyText(JSON.stringify({ ...(job.params || {}), summary: summarise(job) }, null, 2))
    .then(() => toast('Track settings copied', { kind: 'ok', timeout: 1600 }))
    .catch((err) => toast(err.message, { kind: 'err' }));
}

export default MusicStage;
