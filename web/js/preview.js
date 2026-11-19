// Stage: the middle of the window. Live preview while a job runs, the finished
// result when it does not, and a step timeline while anything is in flight.
//
// Frames arrive as `preview` websocket events (data URL JPEG, decoded either by
// latent2rgb or, for MiniMax H3 with KJNodes + taeh3 installed, by the H3
// temporal autoencoder through KJNodes' ModelPreviewOverride).

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, lightbox } from './components.js';
import { el, clear, clamp, num, percent as pct, seconds as secs, mediaUrl, baseName, dateTime } from './util.js';
import { installHqPreview } from './sections.js';

const SOURCE_LABEL = {
  latent2rgb: 'latent2rgb',
  taehv: 'TAEHV',
  taeh3: 'taeh3 (high quality)',
  vae: 'VAE',
  none: 'no preview',
};

export class Stage {
  constructor(root) {
    this.root = root;
    this.view = state.ui.stageView || 'auto';
    this.frame = null;
    this.frameAt = 0;
    this.fps = 0;
    this.jobId = null;
    this.render();
    state.on(TOPICS.jobs, () => this.syncJob());
    state.on(TOPICS.ui, (patch) => {
      if (patch && patch.stageView) { this.view = patch.stageView; this.render(); }
      // Clicking a result in the sidebar or the history strip shows it here,
      // unless a generation owns the stage.
      if (patch && patch.selectedJob !== undefined) this.syncJob();
    });
  }

  /** Feed one `preview` websocket frame into the stage. */
  pushFrame(data) {
    if (!data || !data.image) return;
    const now = performance.now();
    if (this.lastFrameAt && now - this.lastFrameAt < 40) return;
    const gap = this.lastFrameAt ? (now - this.lastFrameAt) / 1000 : 0;
    this.lastFrameAt = now;
    if (gap > 0) this.fps = this.fps ? this.fps * 0.7 + (1 / gap) * 0.3 : 1 / gap;
    this.frame = data;
    this.jobId = data.job_id || this.jobId;
    this.frameAt = now;
    if (this.view === 'live' || (this.view === 'auto' && this.isRunning())) {
      this.paintFrame();
    }
  }

  isRunning() {
    const job = this.current();
    return !!job && (job.status === 'running' || job.status === 'queued');
  }

  /** The generation kind the stage should follow: the image stage shows images. */
  kindFilter() {
    return (state.ui.tab === 'image' || state.ui.tab === 'video') ? state.ui.tab : null;
  }

  current() {
    if (this.jobId && state.jobs.has(this.jobId)) {
      const job = state.jobs.get(this.jobId);
      if (job.status === 'running' || job.status === 'queued') return job;
    }
    const kind = this.kindFilter();
    const match = (job) => !kind || job.kind === kind;
    const running = state.jobList({ limit: 400 }).filter(
      (job) => (job.status === 'running' || job.status === 'queued') && match(job));
    if (running.length) return running[0];
    const selected = state.ui.selectedJob ? state.jobs.get(state.ui.selectedJob) : null;
    if (selected && match(selected)) return selected;
    return state.jobList({ limit: 400 }).find(match) || null;
  }

  setView(view) {
    this.view = view;
    state.setUI({ stageView: view });
    this.render();
  }

  render() {
    clear(this.root);
    this.root.classList.add('stage');
    this.root.appendChild(this.renderBar());

    this.body = el('div', { class: 'stage-body' });
    this.canvas = el('div', { class: 'stage-canvas' });
    this.body.appendChild(this.canvas);

    this.progress = el('div', { class: 'stage-progress' }, [el('div', { class: 'bar' })]);
    this.body.appendChild(this.progress);
    this.foot = el('div', { class: 'stage-foot' });
    this.body.appendChild(this.foot);
    this.root.appendChild(this.body);

    this.paint();
  }

  renderBar() {
    const tabs = el('div', { class: 'stage-tabs' });
    for (const [value, label, iconName] of [
      ['live', 'Live', 'bolt'], ['result', 'Result', 'frame'], ['timeline', 'Steps', 'list']]) {
      tabs.appendChild(el('button', {
        class: this.view === value || (this.view === 'auto' && value === 'live' && this.isRunning()) ? 'on' : '',
        html: `${icon(iconName, 13)}<span>${label}</span>`,
        onclick: () => this.setView(value),
      }));
    }

    this.titleEl = el('span', { class: 'stage-title' });
    const bar = el('div', { class: 'stage-bar' }, [
      this.titleEl,
      el('span', { class: 'grow' }),
      tabs,
      el('button', {
        class: 'icon-btn', html: icon('expand', 14), title: 'Open in the viewer',
        onclick: () => this.openResult(),
      }),
      el('button', {
        class: 'icon-btn danger', html: icon('stop', 14), title: 'Cancel the running job',
        onclick: () => this.cancel(),
      }),
    ]);
    this.paintTitle();
    return bar;
  }

  /** The bar title follows the job the stage is showing, so it repaints on tab switch. */
  paintTitle() {
    if (!this.titleEl) return;
    const job = this.current();
    const text = job ? describe(job) : 'Nothing running';
    if (this.titleEl.textContent !== text) this.titleEl.textContent = text;
    this.titleEl.classList.toggle('running', !!(job && (job.status === 'running' || job.status === 'queued')));
  }

  // Painting ---------------------------------------------------------------
  paint() {
    const job = this.current();
    this.paintTitle();
    const running = !!job && (job.status === 'running' || job.status === 'queued');
    const showLive = this.view === 'live' || (this.view === 'auto' && running);
    if (showLive) {
      if (running) { this.paintFrame(); this.paintProgress(job); this.paintFoot(job); }
      else this.paintIdle(job);
      return;
    }
    if (this.view === 'timeline') { this.paintTimeline(job); this.paintProgress(job); this.paintFoot(job); return; }
    this.paintResult(job);
    this.paintProgress(null);
    this.paintFoot(job);
  }

  paintFrame() {
    const data = this.frame;
    clear(this.canvas);
    if (!data || !data.image) {
      this.canvas.appendChild(el('div', { class: 'preview-skeleton' }, [
        el('div', { class: 'skel' }),
      ]));
      return;
    }
    const image = el('img', { src: data.image, alt: 'Live preview', class: 'live' });
    this.canvas.appendChild(el('div', { class: 'preview-frame' }, [
      image,
      el('div', { class: 'preview-hud' }, [
        el('span', { text: `${data.step ?? 0}/${data.total ?? '?'}` }),
        el('span', { text: pct((data.percent ?? 0) / 100) }),
        el('span', { text: SOURCE_LABEL[data.source] || data.source || '' }),
        el('span', { text: this.fps ? `${num(this.fps, 1)} fps` : '' }),
        el('span', { text: `${data.width}×${data.height}` }),
      ]),
    ]));
  }

  paintIdle(job) {
    clear(this.canvas);
    if (job && job.status === 'succeeded') { this.paintResult(job); return; }
    this.canvas.appendChild(el('div', { class: 'empty' }, [
      el('div', { class: 'e-ic', html: icon('wand', 26) }),
      el('div', {}, [
        el('b', { text: 'Nothing is generating' }),
        el('span', { text: 'Frames appear here as soon as a run starts. Live preview can be tuned in Settings.' }),
      ]),
    ]));
  }

  paintResult(job) {
    clear(this.canvas);
    if (!job) { this.paintIdle(null); return; }
    const outputs = (job.files || []).filter((item) => item && (item.url || item.path));
    if (!outputs.length) {
      this.canvas.appendChild(el('div', { class: 'empty' }, [
        el('div', { class: 'e-ic', html: icon(job.kind === 'music' ? 'music' : 'image', 26) }),
        el('div', {}, [
          el('b', { text: job.status === 'error' ? 'The run failed' : 'No output yet' }),
          el('span', { text: job.error || describe(job) }),
        ]),
      ]));
      return;
    }
    const primary = outputs[0];
    const kind = primary.kind || (job.kind === 'music' ? 'audio' : mediaKind(primary.name || primary.path || ''));
    const url = mediaUrl(primary);
    let media;
    if (kind === 'video') media = el('video', { src: url, controls: true, loop: true, muted: true, preload: 'metadata' });
    else if (kind === 'audio') media = el('audio', { src: url, controls: true });
    else media = el('img', { src: url, alt: baseName(primary.name || '') });
    const frame = el('div', { class: 'preview-frame' }, [media]);
    frame.addEventListener('dblclick', () => this.openResult());
    this.canvas.appendChild(frame);
    if (outputs.length > 1) {
      const strip = el('div', { class: 'row', style: { gap: '6px', padding: '8px 12px', flexWrap: 'wrap' } });
      for (const item of outputs.slice(1)) {
        strip.appendChild(el('button', {
          class: 'btn sm ghost', text: baseName(item.name || item.path),
          onclick: () => { primary.url = item.url; primary.path = item.path; this.paintResult(job); },
        }));
      }
      this.canvas.appendChild(strip);
    }
  }

  paintProgress(job) {
    const total = job ? (job.steps || job.total || 0) : 0;
    const value = job && total ? clamp((job.step || 0) / total, 0, 1)
      : (job && job.progress ? job.progress : 0);
    const running = job && (job.status === 'running' || job.status === 'queued');
    this.progress.classList.toggle('active', !!running);
    const bar = this.progress.querySelector('.bar');
    bar.style.width = `${(value * 100).toFixed(1)}%`;
    if (running) bar.style.animationDuration = '900ms';
  }

  paintFoot(job) {
    clear(this.foot);
    if (!job) {
      this.foot.appendChild(el('span', { class: 'queue-note', text: 'Queue is empty' }));
      return;
    }
    const outputs = (job.files || []).length;
    const bits = [
      `#${job.id.slice(0, 7)}`,
      job.status,
      (job.steps || job.total) ? `${job.step}/${job.steps || job.total} steps` : '',
      job.elapsed ? secs(job.elapsed) : '',
      outputs ? `${outputs} output${outputs > 1 ? 's' : ''}` : '',
      job.engine === 'yue' ? 'YuE' : 'ComfyUI',
    ].filter(Boolean);
    this.foot.appendChild(el('span', { class: 'queue-note', text: bits.join(' · ') }));
    this.foot.appendChild(el('span', { class: 'grow' }));
    if (job.kind !== 'music' && job.params && job.params.family === 'minimax_h3'
      && !state.hqPreviewReady && (job.status === 'running' || job.status === 'queued')) {
      this.foot.appendChild(el('button', {
        class: 'btn sm ghost',
        html: `${icon('sparkles', 12)}<span>Upgrade the H3 preview</span>`,
        title: 'Decode previews with taeh3 instead of latent2rgb',
        onclick: () => installHqPreview({ refresh: () => this.render() }),
      }));
    }
    this.foot.appendChild(el('button', {
      class: 'btn sm ghost', html: `${icon('copy', 12)}<span>Copy settings</span>`,
      onclick: () => copySettings(job),
    }));
  }

  paintTimeline(job) {
    clear(this.canvas);
    const wrap = el('div', { class: 'timeline' });
    const steps = job && job.steps_log && job.steps_log.length ? job.steps_log : guessSteps(job);
    for (const step of steps) {
      wrap.appendChild(el('div', { class: `timeline-row ${step.active ? 'active' : ''}` }, [
        el('span', { class: 't-label', text: step.label }),
        el('span', { class: 't-bar' }, [el('i', { style: { width: `${Math.round(step.value * 100)}%` } })]),
        el('span', { class: 't-val', text: step.value >= 1 ? 'done' : pct(step.value) }),
      ]));
    }
    if (job && job.error) {
      wrap.appendChild(el('div', { class: 'tiny', style: { color: 'var(--err)' }, text: job.error }));
    }
    this.canvas.appendChild(wrap);
  }

  // Actions ----------------------------------------------------------------
  async cancel() {
    const job = this.current();
    if (!job) { toast('Nothing to cancel', { kind: 'info', timeout: 1400 }); return; }
    try {
      await api.cancelJob(job.id);
      toast('Cancellation requested', { kind: 'warn' });
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  openResult() {
    const job = this.current();
    if (!job) return;
    const outputs = job.files || [];
    if (!outputs.length) { toast('This job has no output', { kind: 'warn' }); return; }
    const primary = outputs[0];
    const kind = primary.kind || (job.kind === 'music' ? 'audio' : mediaKind(primary.name || primary.path || ''));
    lightbox({
      title: baseName(primary.name || job.title || job.id),
      subtitle: job.prompt || '',
      url: mediaUrl(primary),
      kind,
      meta: [
        ['Job', job.id],
        ['Created', dateTime(job.created_at)],
        ['Model', modelLabel(job)],
        ['Settings', summarise(job)],
        ['Engine', job.engine || ''],
      ],
      actions: [
        { label: 'Download', icon: 'download', keepOpen: true, run: () => { window.open(mediaUrl(primary), '_blank'); } },
        { label: 'Rerun', icon: 'refresh', run: () => rerun(job) },
      ],
    });
  }

  syncJob() {
    const job = this.current();
    const id = job ? job.id : null;
    if (id !== this.jobId && job && (job.status === 'succeeded' || job.status === 'error')) {
      this.frame = null;
    }
    if (!job || job.id !== this.jobId) {
      if (job && (job.status === 'running' || job.status === 'queued')) {
        this.jobId = job.id;
        this.frame = null;
        this.fps = 0;
      }
    }
    this.paint();
  }
}

// Helpers ------------------------------------------------------------------
export function describe(job) {
  if (!job) return 'Nothing running';
  const label = { image: 'Image', video: 'Video', music: 'Track' }[job.kind] || job.kind;
  if (job.status === 'queued') return `${label} queued`;
  if (job.status === 'running') return `${label} · ${job.stage || 'sampling'}`;
  if (job.status === 'error') return `${label} failed`;
  if (job.status === 'cancelled') return `${label} cancelled`;
  const first = job.preview || (job.files && job.files[0]);
  return `${label} · ${baseName((first && first.name) || job.title || '')}`;
}

function modelLabel(job) {
  const params = job.params || {};
  return params.checkpoint || params.model || params.unet || params.family || params.model_repo || '';
}

function summarise(job) {
  const params = job.params || {};
  if (job.kind === 'music') {
    return [params.style, params.instrumental ? 'instrumental' : 'vocals'].filter(Boolean).join(' · ');
  }
  const bits = [];
  if (params.width) bits.push(`${params.width}×${params.height}`);
  if (params.frames) bits.push(`${params.frames}f @ ${params.fps}`);
  if (params.steps) bits.push(`${params.steps} steps`);
  if (params.sampler) bits.push(`${params.sampler}/${params.scheduler}`);
  if (params.cfg !== undefined) bits.push(`cfg ${params.cfg}`);
  if (params.seed !== undefined) bits.push(`seed ${params.seed}`);
  return bits.join(' · ');
}

function guessSteps(job) {
  if (!job) return [];
  const total = job.steps || job.total || job.params?.steps || 0;
  const done = job.step || 0;
  const steps = [
    { label: 'Submit', value: 1 },
    { label: 'Load model', value: job.status === 'queued' ? 0 : 1 },
  ];
  if (total) steps.push({ label: 'Sample', value: total ? clamp(done / total, 0, 1) : 0, active: true });
  else steps.push({ label: 'Sample', value: job.status === 'running' ? 0.5 : 0, active: job.status === 'running' });
  const finished = job.status === 'succeeded';
  if (job.kind === 'video') {
    steps.push({ label: 'Decode', value: finished ? 1 : (done >= total ? 0.5 : 0) });
    if ((job.params || {}).interpolate && (job.params.interpolate).enabled) {
      steps.push({ label: 'Interpolate', value: finished ? 1 : 0 });
    }
    if ((job.params || {}).seedvr && (job.params.seedvr).enabled) {
      steps.push({ label: 'SeedVR2', value: finished ? 1 : 0 });
    }
    steps.push({ label: 'Mux', value: finished ? 1 : 0 });
  } else if (job.kind === 'music') {
    steps.push({ label: 'Semantic tokens', value: finished ? 1 : clamp(done / Math.max(1, total), 0, 1) });
    steps.push({ label: 'Acoustic tokens', value: finished ? 1 : 0 });
    steps.push({ label: 'VAE decode', value: finished ? 1 : 0 });
  } else {
    steps.push({ label: 'Decode', value: finished ? 1 : 0 });
    steps.push({ label: 'Save', value: finished ? 1 : 0 });
  }
  return steps;
}

async function copySettings(job) {
  const { copyText } = await import('./util.js');
  copyText(JSON.stringify(job.params || {}, null, 2))
    .then(() => toast('Settings copied as JSON', { kind: 'ok' }))
    .catch((err) => toast(err.message, { kind: 'err' }));
}

async function rerun(job) {
  try {
    await api.rerunJob(job.id);
    toast('Requeued from history', { kind: 'ok' });
  } catch (err) {
    toast(err.message, { kind: 'err' });
  }
}

export { guessSteps, summarise, modelLabel };
