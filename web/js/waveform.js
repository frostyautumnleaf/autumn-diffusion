// Waveform: interactive canvas for generated audio.
//
// Peaks arrive two ways. While a track is generating the backend streams decoded
// VAE tiles as `audio` events, which state.audioPeaks accumulates, so the
// waveform grows from left to right. When a job finishes the full peak list is
// attached to the job and the same drawing path is reused.

import { state, TOPICS } from './state.js';
import { icon } from './icons.js';
import { el, clear, clamp, seconds as secs, baseName } from './util.js';

const BAR = 3;      // px per bar at zoom 1
const GAP = 1.5;
const TOKENS_PER_SECOND = 25;   // YuE2 semantic tokens, i.e. latent frames

export class Waveform {
  constructor(root) {
    this.root = root;
    this.jobId = null;
    this.peaks = [];
    this.duration = 0;        // seconds of audio that exist so far
    this.expected = 0;        // seconds the finished track is expected to last
    this.perSecond = 25;      // peak buckets per second of audio
    this.sampleRate = 48000;
    this.zoom = 1;          // > 1 shows a shorter window
    this.offset = 0;        // seconds from the start that is visible
    this.hover = null;      // seconds under the cursor
    this.live = false;
    this.playhead = 0;
    this.playing = false;
    this._frame = null;

    this.wrap = el('div', { class: 'wave-wrap' });
    this.shell = el('div', { class: 'wave-shell' });
    this.canvas = el('canvas', { class: 'wave' });
    this.shell.appendChild(this.canvas);
    this.liveBar = el('div', { class: 'wave-live', style: { display: 'none' } }, [
      el('span', { class: 'dot' }),
      el('span', { text: 'Decoding audio — the waveform is live' }),
    ]);
    this.player = el('div', { class: 'player' }, [
      el('button', {
        class: 'icon-btn play', html: icon('play', 16), title: 'Play / pause',
        onclick: () => this.toggle(),
      }),
      el('span', { class: 'p-time', text: '0:00.0' }),
      el('input', { class: 'seek', type: 'range', min: 0, max: 1000, value: 0, title: 'Seek' }),
      el('span', { class: 'p-time right', text: '0:00.0' }),
      el('button', {
        class: 'icon-btn', html: icon('zoom_out', 14), title: 'Fit the whole track',
        onclick: () => this.fit(),
      }),
      el('a', {
        class: 'icon-btn', html: icon('download', 14), title: 'Download the track',
        download: '', href: '#',
      }),
    ]);
    this.audio = el('audio', { preload: 'metadata' });
    this.wrap.appendChild(this.shell);
    this.wrap.appendChild(this.liveBar);
    this.wrap.appendChild(this.player);

    this.ctx = this.canvas.getContext('2d');
    this.wire();
    state.on(TOPICS.music, (jobId) => {
      if (jobId && jobId === this.jobId) { this.pullPeaks(); this.animate(); }
    });
  }

  /** Insert the player into `target`, which the caller owns. */
  mount(target) {
    this.host = target || this.root;
    clear(this.host);
    this.host.appendChild(this.wrap);
    this.resize();
    requestAnimationFrame(() => this.resize());
  }

  /** Show a job: full peaks when it finished, streaming peaks while it runs.
   *
   * `expected` is the length the finished track should reach. While a job runs the
   * timeline uses it, so the waveform grows from the left instead of being
   * rescaled to fit whatever has decoded so far.
   */
  setJob(job, { live = false, expected = 0 } = {}) {
    this.jobId = job ? job.id : null;
    this.live = live;
    this.zoom = 1;
    this.offset = 0;
    this.playhead = 0;
    this.playing = false;
    this.expected = Math.max(0, Number(expected) || 0);
    this.liveBar.style.display = live ? 'flex' : 'none';
    this.pullPeaks();
    if (!this.expected) this.expected = this.guessLength(job);
    const file = job && job.files && job.files[0];
    const url = file && (file.url || '');
    const link = this.player.querySelector('a.icon-btn');
    if (url) {
      this.audio.src = url;
      link.href = url;
      link.setAttribute('download', baseName(file.name || 'audio'));
      this.audio.onended = () => { this.playing = false; this.setPlayButton(); this.animate(); };
      this.audio.ontimeupdate = () => { this.playhead = this.audio.currentTime; this.animate(); };
      this.audio.onloadedmetadata = () => { this.animate(); this.updateTime(); };
    } else {
      this.audio.removeAttribute('src');
      link.href = '#';
    }
    this.animate();
    this.updateTime();
  }

  pullPeaks() {
    const job = this.jobId ? state.jobs.get(this.jobId) : null;
    const meta = (job && job.meta) || {};
    const expected = Number(meta.expected_seconds || 0);
    if (expected > 0) this.expected = expected;

    const entry = this.jobId ? state.audioPeaks.get(this.jobId) : null;
    if (entry) {
      this.peaks = entry.peaks || [];
      this.duration = entry.duration || 0;
      this.sampleRate = entry.sampleRate || 48000;
      this.perSecond = entry.perSecond || this.perSecond;
      return;
    }
    if (meta.peaks && meta.peaks.length) {
      this.peaks = meta.peaks;
      this.duration = meta.audio_seconds || meta.duration || 0;
      this.sampleRate = meta.sample_rate || this.sampleRate;
      this.perSecond = meta.peaks_per_second || this.perSecond;
    } else if (!this.live) {
      this.peaks = [];
    }
  }

  /**
   * Length the finished track is expected to reach, in seconds.
   *
   * YuE emits one semantic token per latent frame, and a frame is 1920 samples of
   * 48 kHz audio, so 25 tokens is one second of sound. The token ceiling of the
   * request is therefore the length the decode runs to.
   */
  guessLength(job) {
    const meta = (job && job.meta) || {};
    const params = (job && job.params) || {};
    const planned = Number(meta.expected_seconds || 0);
    if (planned > 0) return planned;
    const seconds = Number(meta.audio_seconds || 0);
    if (seconds > 0) return seconds;
    const sampling = params.semantic_sampling || {};
    const tokens = Number(sampling.max_tokens || 0);
    if (tokens > 0) return tokens / TOKENS_PER_SECOND;
    return 0;
  }

  /** Seconds the timeline spans: the expected length while a job is still running. */
  timeline() {
    return this.live ? Math.max(this.duration, this.expected) : this.duration;
  }

  // Interaction ------------------------------------------------------------
  wire() {
    const move = (event) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const visible = this.visibleDuration();
      this.hover = this.offset + (x / rect.width) * visible;
      if (!this.playing && !this.live) this.animate();
    };
    this.canvas.addEventListener('mousemove', move);
    this.canvas.addEventListener('mouseleave', () => { this.hover = null; this.animate(); });
    this.canvas.addEventListener('click', (event) => {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = (event.clientX - rect.left) / rect.width;
      const at = this.offset + ratio * this.visibleDuration();
      this.seek(at);
    });
    this.canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const ratio = (event.clientX - rect.left) / rect.width;
      const anchor = this.offset + ratio * this.visibleDuration();
      const factor = event.deltaY > 0 ? 1 / 1.2 : 1.2;
      this.zoom = clamp(this.zoom * factor, 1, 40);
      const visible = this.visibleDuration();
      this.offset = clamp(anchor - ratio * visible, 0, Math.max(0, this.duration - visible));
      this.animate();
    }, { passive: false });
    this.player.querySelector('.seek').addEventListener('input', (event) => {
      const ratio = Number(event.target.value) / 1000;
      this.seek(ratio * this.duration);
    });
    new ResizeObserver(() => this.resize()).observe(this.shell);
  }

  visibleDuration() {
    const total = this.timeline();
    if (!total) return 1;
    return Math.max(0.25, total / this.zoom);
  }

  seek(at) {
    // A live track can only be seeked as far as audio that has already decoded.
    const limit = this.live ? this.duration : this.timeline();
    this.playhead = clamp(at, 0, limit || 0);
    if (this.audio.src) {
      try { this.audio.currentTime = this.playhead; } catch (err) { /* not loaded */ }
    }
    this.updateTime();
    this.animate();
  }

  toggle() {
    if (!this.audio.src) {
      if (this.live) this.seek(this.duration);
      return;
    }
    if (this.playing) { this.audio.pause(); this.playing = false; }
    else {
      this.audio.currentTime = this.playhead;
      this.audio.play().catch(() => { this.playing = false; });
      this.playing = true;
    }
    this.setPlayButton();
    this.animate();
  }

  setPlayButton() {
    const button = this.player.querySelector('.icon-btn.play');
    button.innerHTML = icon(this.playing ? 'pause' : 'play', 16);
  }

  /** Reset zoom and scroll so the whole track fits. */
  fit() {
    this.zoom = 1;
    this.offset = 0;
    this.animate();
  }

  updateTime() {
    const total = this.timeline();
    const ratio = total ? this.playhead / total : 0;
    const times = this.player.querySelectorAll('.p-time');
    if (times[0]) times[0].textContent = fmt(this.playhead);
    if (times[1]) times[1].textContent = fmt(total);
    const seek = this.player.querySelector('.seek');
    seek.value = String(Math.round(ratio * 1000));
    // The slider paints its own filled track up to the thumb (see app.css).
    seek.style.setProperty('--pct', `${Math.round(ratio * 100)}%`);
  }

  // Drawing ----------------------------------------------------------------
  resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = this.shell.clientWidth || 600;
    const height = this.shell.clientHeight || 168;
    this.height = height;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.draw();
  }

  animate() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = null;
      this.draw();
      if (this.playing || this.live) this.animate();
    });
  }

  draw() {
    const width = this.shell.clientWidth || 600;
    const height = this.height || this.shell.clientHeight || 168;
    const ctx = this.ctx;
    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue('--accent').trim() || '#f59e0b';
    const dim = styles.getPropertyValue('--bg-4').trim() || '#1b2027';
    const line = styles.getPropertyValue('--line').trim() || '#232a33';
    const text = styles.getPropertyValue('--text-faint').trim() || '#6b7684';

    ctx.clearRect(0, 0, width, height);
    const mid = height / 2;
    const visible = this.visibleDuration();

    // Time grid. The labels sit along the bottom so the live badge in the top left
    // of the shell never covers them.
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    const gridStep = niceStep(visible / 8);
    const firstTick = Math.ceil(this.offset / gridStep) * gridStep;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = text;
    for (let t = firstTick; t <= this.offset + visible; t += gridStep) {
      const x = ((t - this.offset) / visible) * width;
      ctx.beginPath();
      ctx.moveTo(x, 14);
      ctx.lineTo(x, height - 18);
      ctx.stroke();
      ctx.fillText(fmt(t), x + 4, height - 5);
    }

    const peaks = this.peaks;
    const perSecond = this.perSecond || 25;
    const step = (perSecond * visible) / (width / (BAR + GAP));
    const start = this.offset * perSecond;
    if (peaks.length) {
      for (let x = 0; x < width; x += BAR + GAP) {
        const index = Math.floor(start + ((x / width) * visible) * perSecond);
        if (index < 0 || index >= peaks.length) continue;
        let peak = 0;
        const span = Math.max(1, Math.floor(step));
        for (let k = 0; k < span && index + k < peaks.length; k += 1) {
          peak = Math.max(peak, peaks[index + k] || 0);
        }
        const amplitude = clamp(peak, 0, 1) * (mid - 12);
        const at = this.offset + (x / width) * visible;
        const played = at <= this.playhead;
        const pending = this.live && at > this.duration;
        ctx.fillStyle = pending ? line : (played ? accent : dim);
        ctx.globalAlpha = pending ? 0.55 : (played ? 0.95 : 0.75);
        ctx.beginPath();
        const radius = Math.min(BAR / 2, 2);
        roundRect(ctx, x, mid - amplitude, BAR, amplitude * 2, radius);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = text;
      ctx.fillText(this.live ? 'Waiting for the first decoded tile…' : 'No waveform data for this track.',
        12, mid);
    }

    // Decode edge: the point the live waveform has reached so far.
    const total = this.timeline();
    if (this.live && total && this.duration < total) {
      const edge = ((this.duration - this.offset) / visible) * width;
      if (edge >= 0 && edge <= width) {
        const fade = ctx.createLinearGradient(edge - 26, 0, edge + 4, 0);
        fade.addColorStop(0, 'rgba(245, 158, 11, 0)');
        fade.addColorStop(1, 'rgba(245, 158, 11, 0.20)');
        ctx.fillStyle = fade;
        ctx.fillRect(edge - 26, 8, 30, height - 16);
        ctx.strokeStyle = accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(edge, 8);
        ctx.lineTo(edge, height - 8);
        ctx.stroke();
      }
    }

    // Playhead
    if (total) {
      const x = ((this.playhead - this.offset) / visible) * width;
      if (x >= 0 && x <= width) {
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, 8);
        ctx.lineTo(x, height - 8);
        ctx.stroke();
      }
    }
    if (this.hover !== null) {
      const x = ((this.hover - this.offset) / visible) * width;
      ctx.strokeStyle = 'rgba(255,255,255,0.22)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 8);
      ctx.lineTo(x, height - 8);
      ctx.stroke();
      ctx.setLineDash([]);
      // No label while a track decodes: the live badge owns that corner.
      if (!this.live) {
        ctx.fillStyle = text;
        ctx.fillText(fmt(this.hover), Math.min(width - 46, x + 5), 12);
      }
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
}

function niceStep(value) {
  const steps = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  for (const step of steps) if (value <= step) return step;
  return 600;
}

function fmt(value) {
  if (!Number.isFinite(value)) return '0:00.0';
  const minutes = Math.floor(value / 60);
  const rest = value - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`;
}

export { fmt as formatTime };
