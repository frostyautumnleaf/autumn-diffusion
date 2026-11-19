// Waveform: interactive canvas and player for generated audio.
//
// Peaks arrive two ways. While a track is generating the backend streams decoded
// VAE tiles as `audio` events, which state.audioPeaks accumulates, so the
// waveform grows from left to right. When a job finishes the full peak list is
// attached to the job and the same drawing path is reused.
//
// Playback follows the same two phases. A running job is played from the growing
// WAV the worker keeps in the cache, re-requested only when playback catches up
// with it; when the finished file lands the player hands over to it without moving
// the play head. Position, zoom and scroll are remembered per job, so clicking
// away to an older track and back does not lose your place.

import { state, TOPICS } from './state.js';
import { icon } from './icons.js';
import { adoptVolume } from './volume.js';
import { el, clear, clamp, baseName, mediaUrl } from './util.js';

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
    this.wantPlay = false;  // the user asked for sound: survive a live reload
    this._frame = null;
    this._catchTimer = null;
    this._pending = null;
    this.memory = new Map();   // jobId -> { playhead, zoom, offset }
    this.srcKind = 'none';
    this.srcUrl = '';
    this.srcRev = 0;
    this._resumeAt = 0;
    this._resume = false;

    this.wrap = el('div', { class: 'wave-wrap' });
    this.shell = el('div', { class: 'wave-shell' });
    this.canvas = el('canvas', { class: 'wave' });
    this.shell.appendChild(this.canvas);
    this.liveBar = el('div', { class: 'wave-live', style: { display: 'none' } }, [
      el('span', { class: 'dot' }),
      el('span', { text: 'Decoding audio — the waveform is live' }),
    ]);
    this.playBtn = el('button', {
      class: 'icon-btn play', html: icon('play', 16), title: 'Play / pause',
      onclick: () => this.toggle(),
    });
    this.dlBtn = el('a', {
      class: 'icon-btn', html: icon('download', 14), title: 'Download the track',
      download: '', href: '#',
    });
    this.player = el('div', { class: 'player' }, [
      this.playBtn,
      el('span', { class: 'p-time', text: '0:00.0' }),
      el('input', { class: 'seek', type: 'range', min: 0, max: 1000, value: 0, title: 'Seek' }),
      el('span', { class: 'p-time right', text: '0:00.0' }),
      el('button', {
        class: 'icon-btn', html: icon('zoom_out', 14), title: 'Fit the whole track',
        onclick: () => this.fit(),
      }),
      this.dlBtn,
    ]);
    this.audio = el('audio', { preload: 'auto' });
    this.wrap.appendChild(this.shell);
    this.wrap.appendChild(this.liveBar);
    this.wrap.appendChild(this.player);

    this.ctx = this.canvas.getContext('2d');
    this.wire();
    state.on(TOPICS.music, (jobId) => {
      if (!jobId || jobId !== this.jobId) return;
      this.pullPeaks();
      this.syncSource();
      this.updateTime();
      this.animate();
    });
    // A job that just finished gains its final file and its full peak list.
    state.on(TOPICS.jobs, () => {
      if (!this.jobId || !state.jobs.has(this.jobId)) return;
      const wanted = this.desiredSource();
      if (wanted.kind !== this.srcKind) this.syncSource();
      else this.pullPeaks();
      this.updateTime();
      this.animate();
    });
  }

  /** Insert the player into `target`, which the caller owns. */
  mount(target) {
    const host = target || this.root;
    if (this.wrap.parentNode === host) { this.resize(); return; }
    clear(host);
    host.appendChild(this.wrap);
    this.resize();
    requestAnimationFrame(() => this.resize());
  }

  /** Show a job: full peaks when it finished, streaming peaks while it runs.
   *
   * `expected` is the length the finished track should reach. While a job runs the
   * timeline uses it, so the waveform grows from the left instead of being
   * rescaled to fit whatever has decoded so far.
   *
   * Calling this again for the job already on screen only refreshes the data: play
   * head, zoom and offset are kept, so a progress event can never interrupt
   * playback or jump the view back to the start.
   */
  setJob(job, { live = false, expected = 0 } = {}) {
    const id = job ? job.id : null;
    if (id !== this.jobId) {
      if (this.jobId) {
        this.memory.set(this.jobId, {
          playhead: this.playhead, zoom: this.zoom, offset: this.offset,
        });
      }
      if (this.audio && !this.audio.paused) this.audio.pause();
      if (this._catchTimer) { clearTimeout(this._catchTimer); this._catchTimer = null; }
      this.jobId = id;
      const saved = id ? this.memory.get(id) : null;
      this.playhead = saved ? saved.playhead : 0;
      this.zoom = saved ? saved.zoom : 1;
      this.offset = saved ? saved.offset : 0;
      this.wantPlay = false;
      this.peaks = [];
      this.duration = 0;
      this.playing = false;
      this.srcKind = 'none';
      this.srcUrl = '';
      this.setPlayButton();
    }
    this.live = live;
    this.expected = Math.max(0, Number(expected) || 0);
    this.liveBar.style.display = live ? 'flex' : 'none';
    this.pullPeaks();
    if (!this.expected) this.expected = this.guessLength(job);
    this.syncSource();

    const file = job && job.files && job.files[0];
    const url = file && job.status === 'succeeded' ? mediaUrl(file) : '';
    if (url) {
      this.dlBtn.href = url;
      this.dlBtn.setAttribute('download', baseName(file.name || 'audio'));
      this.dlBtn.classList.remove('off');
    } else {
      this.dlBtn.href = '#';
      this.dlBtn.classList.add('off');
    }
    this.updateTime();
    this.animate();
  }

  /**
   * Where the peaks come from.
   *
   * A finished job uses the peak list stored with it, which is the accurate one:
   * it covers the whole file at the full bucket rate. While a job runs the streamed
   * tiles are used instead.
   */
  pullPeaks() {
    const job = this.jobId ? state.jobs.get(this.jobId) : null;
    const meta = (job && job.meta) || {};
    const expected = Number(meta.expected_seconds || 0);
    if (expected > 0) this.expected = expected;
    if (meta.peaks && meta.peaks.length) {
      this.peaks = meta.peaks;
      this.duration = meta.audio_seconds || meta.duration || this.duration;
      this.sampleRate = meta.sample_rate || this.sampleRate;
      this.perSecond = meta.peaks_per_second || this.perSecond;
      return;
    }
    const entry = this.jobId ? state.audioPeaks.get(this.jobId) : null;
    if (entry) {
      this.peaks = entry.peaks || [];
      this.duration = entry.duration || 0;
      this.sampleRate = entry.sampleRate || 48000;
      this.perSecond = entry.perSecond || this.perSecond;
      return;
    }
    if (!this.live) this.peaks = [];
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

  // Source selection -------------------------------------------------------
  /**
   * The URL the player should hold.
   *
   * `final` is the finished file. `live` is the growing WAV in the cache, with the
   * revision that was current when it was last reported - the query string is what
   * defeats the browser cache. `none` means there is nothing to play yet.
   */
  desiredSource() {
    const job = this.jobId ? state.jobs.get(this.jobId) : null;
    if (!job) return { kind: 'none', url: '', rev: 0 };
    const file = (job.files || [])[0];
    const finalUrl = file ? mediaUrl(file) : '';
    if (finalUrl && job.status === 'succeeded') return { kind: 'final', url: finalUrl, rev: 0 };
    const entry = state.audioPeaks.get(this.jobId);
    if (entry && entry.url && (entry.buffered || entry.duration) > 0) {
      const rev = Number(entry.revision) || 0;
      const join = entry.url.includes('?') ? '&' : '?';
      return { kind: 'live', url: `${entry.url}${join}v=${rev}`, rev };
    }
    return { kind: 'none', url: '', rev: 0 };
  }

  /**
   * Point the `<audio>` element at the source the job should play.
   *
   * A live source is only re-requested when playback has run out of what was
   * downloaded, otherwise every tile would restart the download and the sound would
   * stutter. `force` reloads regardless, which is what a seek past the current end
   * needs.
   */
  syncSource(force = false) {
    const want = this.desiredSource();
    if (want.kind === 'live' && this.srcKind === 'live' && want.url !== this.srcUrl) {
      if (!force && !this.outOfAudio()) {
        this._pending = want;
        return;
      }
    } else if (want.kind === this.srcKind && want.url === this.srcUrl) {
      return;
    }
    this._resumeAt = Math.max(0, this.playhead || (this.audio.currentTime || 0));
    this._resume = this.wantPlay;
    this.srcKind = want.kind;
    this.srcUrl = want.url;
    this.srcRev = want.rev;
    this._pending = want.kind === 'live' ? want : null;
    if (!want.url) {
      if (!this.audio.paused) this.audio.pause();
      this.audio.removeAttribute('src');
      this.playing = false;
      this.setPlayButton();
      return;
    }
    this.audio.src = want.url;
    adoptVolume(this.audio);
    this.audio.load();
  }

  /** True when the element has played everything it has downloaded. */
  outOfAudio() {
    const media = this.audio;
    if (media.ended) return true;
    const end = Number(media.duration) || 0;
    if (!end) return false;
    return (media.currentTime || 0) >= end - 0.15;
  }

  /** After a live stream ran out of audio, catch up with the newest revision. */
  catchUp() {
    if (this.srcKind !== 'live') return;
    const want = (this._pending && this._pending.url) ? this._pending : this.desiredSource();
    if (!want.url || want.url === this.srcUrl) return;
    this.syncSource(true);
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
      this.seek(ratio * this.timeline());
    });

    const media = this.audio;
    media.addEventListener('loadedmetadata', () => {
      // A live reload resumes where the previous download stopped, never at its end.
      const end = Number(media.duration) || 0;
      const at = Math.min(this._resumeAt || 0, end > 0.05 ? end - 0.05 : 0);
      if (at > 0.05) {
        try { media.currentTime = at; } catch (err) { /* range not ready */ }
        this.playhead = at;
      }
      if (this._resume && this.srcKind === 'live') this.playMedia();
      this._resume = false;
      this.updateTime();
      this.animate();
    });
    media.addEventListener('play', () => { this.playing = true; this.setPlayButton(); this.animate(); });
    media.addEventListener('pause', () => { this.playing = false; this.setPlayButton(); this.animate(); });
    media.addEventListener('timeupdate', () => {
      this.playhead = media.currentTime || 0;
      this.keepPlayheadVisible();
      this.updateTime();
      this.animate();
    });
    media.addEventListener('ended', () => {
      this.playing = false;
      this.setPlayButton();
      // A live track is not over: the worker still has audio on the way.
      if (this.srcKind === 'live' && this.wantPlay) {
        this.catchUp();
        if (this._catchTimer) clearTimeout(this._catchTimer);
        this._catchTimer = setTimeout(() => {
          this._catchTimer = null;
          if (this.wantPlay && this.srcKind === 'live') this.catchUp();
        }, 1200);
      }
      this.animate();
    });
    media.addEventListener('error', () => {
      this.playing = false;
      this.setPlayButton();
    });
    new ResizeObserver(() => this.resize()).observe(this.shell);
  }

  visibleDuration() {
    const total = this.timeline();
    if (!total) return 1;
    return Math.max(0.25, total / this.zoom);
  }

  /** Keep the play head inside the window while a live track plays into new space. */
  keepPlayheadVisible() {
    if (this.zoom <= 1) return;
    const visible = this.visibleDuration();
    if (this.playhead > this.offset + visible * 0.9
      || this.playhead < this.offset + visible * 0.05) {
      this.offset = clamp(this.playhead - visible * 0.2, 0,
        Math.max(0, this.timeline() - visible));
    }
  }

  seek(at) {
    const total = this.timeline();
    // A live track can only be seeked as far as audio that has already decoded.
    const limit = this.live ? this.duration : total;
    this.playhead = clamp(at, 0, limit || 0);
    if (this.srcKind === 'live' && this.outOfAudio()) {
      // The new part of the file is not downloaded yet: reload, then seek.
      this._resumeAt = this.playhead;
      this._resume = this.wantPlay;
      this.syncSource(true);
      return;
    }
    if (this.audio.src) {
      try { this.audio.currentTime = this.playhead; } catch (err) { /* not loaded */ }
    }
    this.updateTime();
    this.animate();
  }

  toggle() {
    if (this.playing) {
      this.wantPlay = false;
      this.audio.pause();
      this.playing = false;
      this.setPlayButton();
      this.animate();
      return;
    }
    if (!this.audio.src) {
      // Nothing to hear yet: park the head at the decode edge so playback starts
      // there as soon as the first tile lands.
      if (this.live) this.seek(this.duration);
      return;
    }
    this.wantPlay = true;
    this.playMedia();
  }

  playMedia() {
    const media = this.audio;
    if (!media.src) return;
    if (this.playhead > 0 && Math.abs((media.currentTime || 0) - this.playhead) > 0.25) {
      try { media.currentTime = this.playhead; } catch (err) { /* range not ready */ }
    }
    adoptVolume(media);
    const started = media.play();
    if (started && started.catch) {
      started.catch(() => {
        // Autoplay without a user gesture is blocked, and a live source can fail
        // while it waits for the next chunk; the `ended` handler retries.
        this.playing = false;
        this.setPlayButton();
      });
    }
    this.playing = !media.paused;
    this.setPlayButton();
    this.animate();
  }

  setPlayButton() {
    this.playBtn.innerHTML = icon(this.playing ? 'pause' : 'play', 16);
    this.playBtn.title = this.playing ? 'Pause' : 'Play';
  }

  /** Reset zoom and scroll so the whole track fits. */
  fit() {
    this.zoom = 1;
    this.offset = 0;
    this.animate();
  }

  updateTime() {
    const total = this.timeline();
    const ratio = clamp(total ? this.playhead / total : 0, 0, 1);
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
    const accent = styles.getPropertyValue('--accent').trim() || '#f0a24b';
    const dim = styles.getPropertyValue('--bg-4').trim() || '#232323';
    const line = styles.getPropertyValue('--line').trim() || '#262626';
    const text = styles.getPropertyValue('--text-faint').trim() || '#585858';

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
  ctx.quadraticCurveTo(x + w, y + h - radius, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.lineTo(x, y, x + radius, y);
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

/**
 * Static waveform picture used as a thumbnail for audio.
 *
 * Peaks come from the job metadata, so a track has a recognisable preview before
 * any audio is decoded. The bars are centred on the middle line and the drawing
 * stretches to whatever box it is placed in.
 */
export function waveThumb(peaks, bars = 34) {
  const SVG = 'http://www.w3.org/2000/svg';
  const values = reducePeaks(peaks, bars);
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('class', 'wave-thumb');
  svg.setAttribute('viewBox', `0 0 ${values.length} 100`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  values.forEach((peak, index) => {
    const height = Math.max(4, peak * 88);
    const rect = document.createElementNS(SVG, 'rect');
    rect.setAttribute('x', String(index + 0.18));
    rect.setAttribute('y', String((100 - height) / 2));
    rect.setAttribute('width', '0.64');
    rect.setAttribute('height', String(height));
    rect.setAttribute('rx', '0.3');
    svg.appendChild(rect);
  });
  return svg;
}

/** Collapse a peak list to ``count`` values, keeping the loudest of each bucket. */
function reducePeaks(peaks, count) {
  const source = Array.isArray(peaks) ? peaks.filter(Number.isFinite) : [];
  const out = new Array(Math.max(1, count)).fill(0);
  if (!source.length) return out;
  for (let index = 0; index < out.length; index += 1) {
    const from = Math.floor((index * source.length) / out.length);
    const to = Math.max(from + 1, Math.floor(((index + 1) * source.length) / out.length));
    let peak = 0;
    for (let i = from; i < to; i += 1) peak = Math.max(peak, Math.abs(source[i]));
    out[index] = Math.min(1, peak);
  }
  return out;
}

