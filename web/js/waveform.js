// Waveform: interactive canvas and player for generated audio.
//
// Peaks arrive two ways. While a track is generating the backend streams decoded
// VAE tiles as `audio` events, which state.audioPeaks accumulates, so the
// waveform grows from left to right. When a job finishes the full peak list is
// attached to the job and the same drawing path is reused.
//
// Playback is one `<audio>` element per job, held in a pool, not one element that
// is re-pointed at whatever is on screen. Switching to another track therefore
// does not stop the track you were listening to: it keeps playing, turned down,
// and its play head is exactly where you left it when you come back. The icon on
// the play button is read from the element itself, so it can never disagree with
// what is audible.
//
// A live track plays from the growing WAV the worker keeps in the cache. Seeking
// inside the part that has already decoded must land on the requested second, so
// a seek carries an explicit target that survives a source reload instead of
// being clamped to the end of whatever had downloaded. Position, zoom and scroll
// are remembered per job.

import { state, TOPICS } from './state.js';
import { icon } from './icons.js';
import { menu } from './components.js';
import { adoptVolume, setDuck } from './volume.js';
import { el, clear, clamp, baseName, mediaUrl } from './util.js';

const BAR = 3;      // px per bar at zoom 1
const GAP = 1.5;
const TOKENS_PER_SECOND = 25;   // YuE2 semantic tokens, i.e. latent frames
const DUCK = 0.25;              // volume of a track playing in the background
const MAX_CHANNELS = 6;         // tracks that keep their player alive
const RECOMPUTE_RATE = 25;      // buckets per second used by the browser recompute

/** Envelopes recomputed in the browser, keyed by job id. */
const fixedPeaks = new Map();
/** Jobs whose envelope was already checked for a recompute in this session. */
const checked = new Set();
/** One AudioContext, created on first need, for decoding audio to measure it. */
let measureCtx = null;

/**
 * The URL one job should play.
 *
 * `final` is the finished file. `live` is the growing WAV in the cache, with the
 * revision that was current when it was last reported - the query string is what
 * defeats the browser cache. `none` means there is nothing to play yet.
 */
function desiredSource(jobId) {
  const job = jobId ? state.jobs.get(jobId) : null;
  if (!job) return { kind: 'none', url: '', rev: 0 };
  const file = (job.files || [])[0];
  const finalUrl = file ? mediaUrl(file) : '';
  if (finalUrl && job.status === 'succeeded') return { kind: 'final', url: finalUrl, rev: 0 };
  const entry = state.audioPeaks.get(jobId);
  if (entry && entry.url && (entry.buffered || entry.duration) > 0) {
    const rev = Number(entry.revision) || 0;
    const join = entry.url.includes('?') ? '&' : '?';
    return { kind: 'live', url: `${entry.url}${join}v=${rev}`, rev };
  }
  return { kind: 'none', url: '', rev: 0 };
}

/**
 * One track's player.
 *
 * A channel outlives the moment its job is on screen, which is what lets several
 * tracks play at once and lets a play head survive browsing through history.
 */
class Channel {
  constructor(owner, jobId) {
    this.owner = owner;
    this.jobId = jobId;
    this.audio = el('audio', { preload: 'auto' });
    this.srcKind = 'none';
    this.srcUrl = '';
    this.srcRev = 0;
    this.playhead = 0;
    this.playing = false;
    this.wantPlay = false;
    this.resume = false;      // start playback again once the new source is ready
    this.resumeAt = 0;        // the second a reload is expected to come back at
    this.seekTarget = null;   // second to land on, honoured across a reload
    this.seekTries = 0;
    this.pending = null;      // live source not fetched until playback needs it
    this.catchTimer = null;
    this.lastUsed = Date.now();
    this.wire();
  }

  wire() {
    const media = this.audio;
    media.addEventListener('loadedmetadata', () => {
      this.applySeek();
      // Handing over from the live file to the finished one, or catching up with a
      // live stream, must keep the sound running.
      if (this.resume) {
        this.resume = false;
        this.wantPlay = true;
        this.playMedia();
      }
      this.refresh();
    });
    media.addEventListener('canplay', () => this.applySeek());
    media.addEventListener('timeupdate', () => {
      if (this.seekTarget === null) this.playhead = media.currentTime || 0;
      this.refresh(true);
    });
    media.addEventListener('play', () => { this.playing = true; this.owner.mix(); this.refresh(); });
    media.addEventListener('pause', () => { this.playing = false; this.owner.mix(); this.refresh(); });
    media.addEventListener('ended', () => {
      this.playing = false;
      this.owner.mix();
      // A live track is not over: the worker still has audio on the way.
      if (this.srcKind === 'live' && this.wantPlay) this.catchUpSoon();
      this.refresh();
    });
    media.addEventListener('error', () => {
      this.playing = false;
      this.owner.mix();
      this.refresh();
    });
  }

  /** Tell the view about a change, and let it ignore a channel that is off screen. */
  refresh(follow = false) {
    this.owner.onChannelChange(this, follow);
  }

  /**
   * Point the element at the source the job should play.
   *
   * A live source is only re-requested when playback has run out of what was
   * downloaded, otherwise every tile would restart the download and the sound would
   * stutter. `force` reloads regardless, which is what a seek past the current end
   * needs. An explicit seek target always beats "resume where the last download
   * stopped".
   */
  syncSource(force = false) {
    const want = desiredSource(this.jobId);
    if (want.kind === 'live' && this.srcKind === 'live' && want.url !== this.srcUrl) {
      if (!force && !this.outOfAudio()) {
        this.pending = want;
        return;
      }
    } else if (want.kind === this.srcKind && want.url === this.srcUrl) {
      return;
    }
    this.resume = this.wantPlay;
    this.resumeAt = this.seekTarget !== null ? this.seekTarget : Math.max(0, this.playhead);
    this.srcKind = want.kind;
    this.srcUrl = want.url;
    this.srcRev = want.rev;
    this.pending = want.kind === 'live' ? want : null;
    this.lastUsed = Date.now();
    if (!want.url) {
      this.stop();
      return;
    }
    // The play head is about to be re-applied, so keep the channel's own record.
    this.playhead = this.resumeAt;
    this.seekTarget = this.resumeAt > 0.02 ? this.resumeAt : null;
    this.seekTries = 0;
    this.audio.src = want.url;
    adoptVolume(this.audio);
    this.owner.mix();
    this.audio.load();
  }

  /** Stop this track and drop its source. */
  stop() {
    this.wantPlay = false;
    this.resume = false;
    this.seekTarget = null;
    if (this.catchTimer) {
      clearTimeout(this.catchTimer);
      this.catchTimer = null;
    }
    if (!this.audio.paused) this.audio.pause();
    this.audio.removeAttribute('src');
    this.playing = false;
    this.srcKind = 'none';
    this.srcUrl = '';
    this.srcRev = 0;
    this.pending = null;
    this.owner.mix();
    this.refresh();
  }

  /**
   * Land on `seekTarget`, retrying until the element agrees.
   *
   * Setting `currentTime` before the bytes for that offset are available is
   * silently ignored, so the target is re-applied until it sticks or the element
   * is clearly never going to serve it. A live source that has not grown far
   * enough yet stops just before its end and keeps playing, because that is the
   * newest audio that exists.
   */
  applySeek() {
    const media = this.audio;
    if (this.seekTarget === null) return;
    const end = Number(media.duration) || 0;
    let target = this.seekTarget;
    if (end > 0.05 && target > end - 0.05) {
      target = Math.max(0, end - 0.05);
      this.playhead = target;
      this.seekTarget = null;
    }
    try { media.currentTime = target; } catch (err) { /* bytes not ready */ }
    const got = Number(media.currentTime) || 0;
    if (Math.abs(got - target) < 0.12) {
      this.seekTarget = null;
      this.seekTries = 0;
      this.playhead = got;
      this.refresh();
      return;
    }
    this.seekTries += 1;
    if (this.seekTries > 12) {
      this.seekTries = 0;
      this.seekTarget = null;
      return;
    }
    setTimeout(() => this.applySeek(), 60);
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
    const want = (this.pending && this.pending.url) ? this.pending : desiredSource(this.jobId);
    if (!want.url || want.url === this.srcUrl) return;
    this.resume = this.wantPlay;
    this.resumeAt = Math.max(0, this.playhead);
    this.syncSource(true);
  }

  /** Catch up now, and again shortly in case the next tile has not landed yet. */
  catchUpSoon() {
    this.catchUp();
    if (this.catchTimer) clearTimeout(this.catchTimer);
    this.catchTimer = setTimeout(() => {
      this.catchTimer = null;
      if (this.wantPlay && this.srcKind === 'live' && !this.playing) this.catchUpSoon();
    }, 900);
  }

  playMedia() {
    const media = this.audio;
    if (!media.src) return;
    if (this.playhead > 0 && Math.abs((media.currentTime || 0) - this.playhead) > 0.25) {
      this.seekTarget = this.playhead;
      this.seekTries = 0;
      this.applySeek();
    }
    adoptVolume(media);
    const started = media.play();
    if (started && started.catch) {
      started.catch(() => {
        // Autoplay without a user gesture is blocked, and a live source can fail
        // while it waits for the next chunk; the `ended` handler retries.
        this.playing = false;
        this.owner.mix();
        this.refresh();
      });
    }
    this.playing = !media.paused;
    this.owner.mix();
    this.refresh();
  }
}

export class Waveform {
  constructor(root) {
    this.root = root;
    this.jobId = null;
    this.peaks = [];
    this.duration = 0;        // seconds of audio that exist so far
    this.buffered = 0;        // seconds of that the player can actually seek to
    this.expected = 0;        // seconds the finished track is expected to last
    this.perSecond = 25;      // peak buckets per second of audio
    this.sampleRate = 48000;
    this.zoom = 1;          // > 1 shows a shorter window
    this.offset = 0;        // seconds from the start that is visible
    this.hover = null;      // seconds under the cursor
    this.live = false;
    this._frame = null;
    this.channels = new Map();   // jobId -> Channel
    this.memory = new Map();     // jobId -> { playhead, zoom, offset }

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
      el('span', { class: 'p-time', text: '0:00' }),
      el('input', { class: 'seek', type: 'range', min: 0, max: 1000, value: 0, title: 'Seek' }),
      el('span', { class: 'p-time right', text: '0:00' }),
      el('button', {
        class: 'icon-btn', html: icon('zoom_out', 14), title: 'Fit the whole track',
        onclick: () => this.fit(),
      }),
      el('button', {
        class: 'icon-btn', html: icon('sliders', 14), title: 'More track options',
        onclick: (event) => this.menuFor(this.player, event),
      }),
      this.dlBtn,
    ]);
    this.wrap.appendChild(this.shell);
    this.wrap.appendChild(this.liveBar);
    this.wrap.appendChild(this.player);

    this.ctx = this.canvas.getContext('2d');
    this.wire();
    state.on(TOPICS.music, (jobId) => {
      if (!jobId || jobId !== this.jobId) return;
      this.pullPeaks();
      const channel = this.active();
      if (channel) channel.syncSource();
      this.updateTime();
      this.animate();
    });
    // A job that just finished gains its final file and its full peak list.
    state.on(TOPICS.jobs, () => {
      if (!this.jobId || !state.jobs.has(this.jobId)) return;
      const channel = this.active();
      if (!channel) return;
      if (desiredSource(this.jobId).kind !== channel.srcKind) channel.syncSource();
      else this.pullPeaks();
      this.updateTime();
      this.animate();
    });
  }

  /** The channel of the job on screen, created on demand. */
  active(create = true) {
    if (!this.jobId) return null;
    let channel = this.channels.get(this.jobId);
    if (!channel && create) {
      channel = new Channel(this, this.jobId);
      const saved = this.memory.get(this.jobId);
      if (saved) channel.playhead = saved.playhead || 0;
      this.channels.set(this.jobId, channel);
    }
    return channel || null;
  }

  /** The play head and the playing flag belong to the channel of the visible job. */
  get playhead() {
    const channel = this.active();
    return channel ? channel.playhead : 0;
  }

  set playhead(value) {
    const channel = this.active();
    if (channel) channel.playhead = Number(value) || 0;
  }

  get playing() {
    const channel = this.active(false);
    return channel ? !channel.audio.paused : false;
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

  /**
   * Show a job: full peaks when it finished, streaming peaks while it runs.
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
      const previous = this.active(false);
      if (this.jobId) {
        this.memory.set(this.jobId, {
          playhead: previous ? previous.playhead : this.playhead,
          zoom: this.zoom, offset: this.offset,
        });
      }
      this.jobId = id;
      const saved = id ? this.memory.get(id) : null;
      this.zoom = saved ? saved.zoom : 1;
      this.offset = saved ? saved.offset : 0;
      this.peaks = [];
      this.duration = 0;
      this.buffered = 0;
      if (id) {
        const channel = this.active();
        if (saved && !channel.seekTarget) channel.playhead = saved.playhead || 0;
      }
      this.pruneChannels();
      this.mix();
      this.setPlayButton();
    }
    this.live = live;
    this.expected = Math.max(0, Number(expected) || 0);
    this.liveBar.style.display = live ? 'flex' : 'none';
    this.pullPeaks();
    if (!this.expected) this.expected = this.guessLength(job);
    const channel = this.active();
    if (channel) {
      channel.lastUsed = Date.now();
      channel.syncSource();
    }

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
   * An envelope measured in the browser from the audio itself wins, because it is
   * always on the grid the drawing uses. Next comes the list stored with a finished
   * job, then the tiles streamed while a job runs.
   */
  pullPeaks() {
    const job = this.jobId ? state.jobs.get(this.jobId) : null;
    const meta = (job && job.meta) || {};
    const expected = Number(meta.expected_seconds || 0);
    if (expected > 0) this.expected = expected;
    const fixed = this.jobId ? fixedPeaks.get(this.jobId) : null;
    if (fixed && fixed.peaks.length) {
      this.peaks = fixed.peaks;
      this.perSecond = fixed.perSecond;
      this.duration = Math.max(this.duration, fixed.duration || 0);
      if (job && job.status === 'succeeded') this.buffered = this.duration;
      return;
    }
    if (meta.peaks && meta.peaks.length) {
      this.peaks = meta.peaks;
      this.duration = meta.audio_seconds || meta.duration || this.duration;
      this.sampleRate = meta.sample_rate || this.sampleRate;
      this.perSecond = meta.peaks_per_second || this.perSecond;
      if (job.status === 'succeeded') this.buffered = this.duration;
      this.ensurePeaks(job);
      return;
    }
    const entry = this.jobId ? state.audioPeaks.get(this.jobId) : null;
    if (entry) {
      this.peaks = entry.peaks || [];
      this.duration = entry.duration || 0;
      this.buffered = entry.buffered || entry.duration || 0;
      this.sampleRate = entry.sampleRate || 48000;
      this.perSecond = entry.perSecond || this.perSecond;
      return;
    }
    if (!this.live) this.peaks = [];
  }

  /**
   * Measure the envelope in the browser when the stored one cannot be trusted.
   *
   * A long slice is bucketed coarser than asked for, and a list whose rate does not
   * match its length puts the bars on the wrong grid, which is what leaves silent
   * holes in the drawing. The result is sent back to the server so the next session
   * does not have to measure it again.
   */
  ensurePeaks(job) {
    if (!job || job.status !== 'succeeded' || checked.has(job.id)) return;
    const file = (job.files || [])[0];
    const url = file ? mediaUrl(file) : '';
    if (!url) return;
    const meta = job.meta || {};
    const seconds = Number(meta.audio_seconds || meta.duration || file.duration || 0);
    const rate = Number(meta.peaks_per_second || RECOMPUTE_RATE);
    const peaks = Array.isArray(meta.peaks) ? meta.peaks : [];
    const wanted = seconds * rate;
    if (peaks.length && (!wanted || Math.abs(peaks.length - wanted) / wanted < 0.08)) return;
    checked.add(job.id);
    recomputePeaks(url).then((result) => {
      if (!result || !result.peaks.length) return;
      fixedPeaks.set(job.id, result);
      if (job.id === this.jobId) {
        this.peaks = result.peaks;
        this.perSecond = result.perSecond;
        this.duration = Math.max(this.duration, result.duration);
        this.buffered = this.duration;
        this.updateTime();
      }
      this.animate();
      savePeaks(job.id, result);
    }).catch(() => { /* the stored envelope stays as it is */ });
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

  /** The last second the player can actually be placed at. */
  playableEnd() {
    if (!this.live) return this.duration;
    const known = this.buffered || this.duration;
    return Math.max(0, Math.min(this.duration || known, known));
  }

  /** Forget a job entirely: its channel, its remembered view and its peaks. */
  forget(jobId) {
    const channel = this.channels.get(jobId);
    if (channel) {
      channel.stop();
      this.channels.delete(jobId);
    }
    this.memory.delete(jobId);
    fixedPeaks.delete(jobId);
    checked.delete(jobId);
    if (jobId === this.jobId) this.peaks = [];
  }

  /** Pause a track without throwing away its play head. */
  pause(jobId) {
    const channel = this.channels.get(jobId || this.jobId);
    if (channel) {
      channel.wantPlay = false;
      channel.resume = false;
      channel.audio.pause();
    }
  }

  /** Pause every track this view is holding a player for. */
  pauseAll() {
    for (const channel of this.channels.values()) {
      channel.wantPlay = false;
      channel.resume = false;
      channel.audio.pause();
    }
    this.mix();
    this.setPlayButton();
    this.animate();
  }

  /**
   * Keep only the players that are useful, so a long session cannot pile up
   * audio elements. Playing tracks are never dropped.
   */
  pruneChannels() {
    if (this.channels.size <= MAX_CHANNELS) return;
    const keep = [...this.channels.entries()]
      .filter(([jobId]) => jobId !== this.jobId)
      .sort((a, b) => b[1].lastUsed - a[1].lastUsed);
    for (const [jobId, channel] of keep.slice(Math.max(0, MAX_CHANNELS - 1))) {
      if (channel.playing) continue;
      this.memory.set(jobId, {
        playhead: channel.playhead,
        zoom: this.zoom, offset: this.offset,
      });
      channel.stop();
      this.channels.delete(jobId);
    }
  }

  /**
   * Set the level of every track.
   *
   * Only the track on screen is heard at full volume while it plays; the others are
   * turned down rather than stopped, so a long generation can keep running while
   * you look at something else.
   */
  mix() {
    const active = this.active(false);
    const duckOthers = !!(active && active.playing);
    for (const channel of this.channels.values()) {
      setDuck(channel.audio, channel === active || !duckOthers ? 1 : DUCK);
    }
  }

  /** Called by a channel when its time or play state moved. */
  onChannelChange(channel, follow = false) {
    if (channel.jobId !== this.jobId) return;
    this.setPlayButton();
    if (follow) this.keepPlayheadVisible();
    this.updateTime();
    this.animate();
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
    this.canvas.addEventListener('dblclick', () => this.fit());
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
    this.canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      this.menuFor(this.canvas, event);
    });
    this.player.querySelector('.seek').addEventListener('input', (event) => {
      const ratio = Number(event.target.value) / 1000;
      this.seek(ratio * this.timeline());
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

  /**
   * Move the play head to `at`, and get the element to agree.
   *
   * A live track can only be seeked as far as audio that has already been written
   * to its file. Inside that region the target is applied directly when the bytes
   * are already held, and through a reload when they are not - the target travels
   * with the reload so playback lands on the second that was asked for, never on
   * the end of the previous download.
   */
  seek(at) {
    const channel = this.active();
    if (!channel) return;
    const limit = this.live ? this.playableEnd() : this.duration;
    const target = clamp(Number(at) || 0, 0, Math.max(0, limit));
    channel.playhead = target;
    channel.seekTarget = target;
    channel.seekTries = 0;
    channel.lastUsed = Date.now();
    const served = Number(channel.audio.duration) || 0;
    if (channel.srcKind === 'none') {
      this.updateTime();
      this.animate();
      return;
    }
    if (!served || target <= served - 0.05) channel.applySeek();
    else if (channel.srcKind === 'live') {
      channel.resume = channel.wantPlay;
      channel.syncSource(true);
    } else {
      channel.applySeek();
    }
    this.keepPlayheadVisible();
    this.updateTime();
    this.animate();
  }

  toggle() {
    const channel = this.active();
    if (!channel) return;
    channel.lastUsed = Date.now();
    if (channel.playing) {
      channel.wantPlay = false;
      channel.resume = false;
      channel.audio.pause();
      this.setPlayButton();
      this.animate();
      return;
    }
    if (!channel.audio.src) channel.syncSource();
    if (!channel.audio.src) {
      // Nothing to hear yet: park the head at the decode edge so playback starts
      // there as soon as the first tile lands.
      if (this.live) this.seek(this.playableEnd());
      return;
    }
    channel.wantPlay = true;
    channel.playMedia();
    this.setPlayButton();
    this.animate();
  }

  /** The icon follows the element, so it can never disagree with what is audible. */
  setPlayButton() {
    const channel = this.active(false);
    const playing = !!(channel && !channel.audio.paused);
    this.playBtn.innerHTML = icon(playing ? 'pause' : 'play', 16);
    this.playBtn.title = playing ? 'Pause' : 'Play';
  }

  /** Reset zoom and scroll so the whole track fits. */
  fit() {
    this.zoom = 1;
    this.offset = 0;
    this.animate();
  }

  zoomBy(factor) {
    const anchor = this.offset + this.visibleDuration() / 2;
    this.zoom = clamp(this.zoom * factor, 1, 40);
    const visible = this.visibleDuration();
    this.offset = clamp(anchor - visible / 2, 0, Math.max(0, this.duration - visible));
    this.animate();
  }

  /** Right-click menu for the waveform and the transport row. */
  menuFor(anchor, event) {
    const channel = this.active(false);
    const playing = !!(channel && !channel.audio.paused);
    const background = [...this.channels.values()].filter((entry) => entry.playing
      && entry.jobId !== this.jobId);
    const at = event ? { x: event.clientX, y: event.clientY } : null;
    menu(anchor, [
      { label: playing ? 'Pause' : 'Play', icon: playing ? 'pause' : 'play', run: () => this.toggle() },
      {
        label: 'Back to the start', icon: 'skip', disabled: !channel,
        run: () => this.seek(0),
      },
      { sep: true },
      { label: 'Fit the whole track', icon: 'zoom_out', run: () => this.fit() },
      { label: 'Zoom in', icon: 'zoom_in', hint: 'wheel', run: () => this.zoomBy(1.4) },
      { label: 'Zoom out', icon: 'zoom_out', hint: 'wheel', run: () => this.zoomBy(1 / 1.4) },
      { sep: true },
      {
        label: 'Measure the waveform again', icon: 'wave',
        hint: 'fixes holes',
        run: () => {
          const job = state.jobs.get(this.jobId);
          if (!job) return;
          checked.delete(job.id);
          fixedPeaks.delete(job.id);
          this.ensurePeaks(job);
        },
      },
      {
        label: 'Stop the tracks playing behind', icon: 'stop',
        disabled: !background.length,
        hint: background.length ? `${background.length}` : '',
        run: () => {
          for (const entry of background) {
            entry.wantPlay = false;
            entry.resume = false;
            entry.audio.pause();
          }
          this.mix();
          this.setPlayButton();
        },
      },
    ], at ? { at } : {});
  }

  updateTime() {
    const total = this.timeline();
    const head = this.playhead;
    const ratio = clamp(total ? head / total : 0, 0, 1);
    const times = this.player.querySelectorAll('.p-time');
    const short = total > 0 && total < 60;
    const left = fmt(head, short);
    const right = fmt(total, short);
    if (times[0] && times[0].textContent !== left) times[0].textContent = left;
    if (times[1] && times[1].textContent !== right) times[1].textContent = right;
    const seek = this.player.querySelector('.seek');
    const value = String(Math.round(ratio * 1000));
    if (seek.value !== value) seek.value = value;
    // The slider paints its own filled track up to the thumb (see app.css).
    const pct = `${Math.round(ratio * 100)}%`;
    if (seek.style.getPropertyValue('--pct') !== pct) seek.style.setProperty('--pct', pct);
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
      ctx.fillText(fmt(t, gridStep < 1), x + 4, height - 5);
    }

    const peaks = this.peaks;
    const perSecond = this.perSecond || 25;
    const step = (perSecond * visible) / (width / (BAR + GAP));
    const start = this.offset * perSecond;
    const playhead = this.playhead;
    if (peaks.length) {
      for (let x = 0; x < width; x += BAR + GAP) {
        const index = Math.floor(start + ((x / width) * visible) * perSecond);
        if (index < 0 || index >= peaks.length) continue;
        let peak = 0;
        const span = Math.max(1, Math.floor(step));
        for (let k = 0; k < span && index + k < peaks.length; k += 1) {
          peak = Math.max(peak, peaks[index + k] || 0);
        }
        // A hairline floor, so a quiet passage still reads as continuous audio
        // instead of looking like a gap in the recording.
        const amplitude = Math.max(0.8, clamp(peak, 0, 1) * (mid - 12));
        const at = this.offset + (x / width) * visible;
        const played = at <= playhead;
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
      const x = ((playhead - this.offset) / visible) * width;
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
        ctx.fillText(fmt(this.hover, true), Math.min(width - 46, x + 5), 12);
      }
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.lineTo(x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.lineTo(x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + h - radius);
  ctx.lineTo(x, y, x + radius, y);
}

function niceStep(value) {
  const steps = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  for (const step of steps) if (value <= step) return step;
  return 600;
}

/**
 * Time label. Tenths are shown only where they say something: on a sub-second
 * grid, on the scrub readout, and on the transport of a short clip.
 */
function fmt(value, tenths = false) {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0;
  let minutes = Math.floor(safe / 60);
  let rest = safe - minutes * 60;
  if (!tenths) {
    rest = Math.round(rest);
    if (rest >= 60) { minutes += 1; rest = 0; }
    return `${minutes}:${String(rest).padStart(2, '0')}`;
  }
  return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`;
}

/**
 * Decode a track in the browser and measure its peak envelope.
 *
 * The AudioContext is only used to read samples; nothing is routed to the output,
 * so measuring is silent and cannot disturb what is playing.
 */
async function recomputePeaks(url, perSecond = RECOMPUTE_RATE) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) return null;
  const bytes = await response.arrayBuffer();
  if (!bytes.byteLength) return null;
  const context = await getMeasureContext();
  if (!context) return null;
  const decoded = await context.decodeAudioData(bytes);
  return {
    peaks: peaksFromBuffer(decoded, perSecond),
    perSecond,
    duration: round2(decoded.duration),
  };
}

async function getMeasureContext() {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) return null;
  if (!measureCtx) {
    try {
      measureCtx = new Ctor({ sampleRate: 44100 });
    } catch (err) {
      measureCtx = new Ctor();
    }
  }
  return measureCtx;
}

/** Peak envelope of a decoded track, normalised to the loudest bucket. */
function peaksFromBuffer(buffer, perSecond) {
  const rate = Math.max(1, Math.round(perSecond));
  const length = Math.max(1, Math.floor(buffer.duration * rate));
  const out = new Array(length).fill(0);
  const channels = Math.min(2, Math.max(1, buffer.numberOfChannels));
  const data = [];
  for (let channel = 0; channel < channels; channel += 1) data.push(buffer.getChannelData(channel));
  const total = buffer.length;
  const window = Math.max(1, Math.floor(total / length));
  for (let index = 0; index < length; index += 1) {
    const from = index * window;
    const to = Math.min(total, from + window);
    let peak = 0;
    // Sampling every 3rd frame keeps a 5 minute measurement under a few ms while
    // still catching every transient that matters to the drawing.
    const stride = Math.max(1, Math.floor((to - from) / 512));
    for (let i = from; i < to; i += stride) {
      for (let c = 0; c < channels; c += 1) {
        const value = Math.abs(data[c][i] || 0);
        if (value > peak) peak = value;
      }
    }
    out[index] = peak;
  }
  let top = 0;
  for (const value of out) if (value > top) top = value;
  if (top > 0) for (let index = 0; index < out.length; index += 1) out[index] /= top;
  return out.map((value) => Math.round(value * 10000) / 10000);
}

/**
 * Hand a browser-measured envelope back to the server.
 *
 * The stored one is wrong for some long tracks, and measuring takes a second of
 * decoding, so the fix is worth keeping for the next session.
 */
function savePeaks(jobId, result) {
  fetch('/api/jobs/' + encodeURIComponent(jobId) + '/peaks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      peaks: result.peaks,
      peaks_per_second: result.perSecond,
      audio_seconds: result.duration,
    }),
  }).catch(() => { /* the server keeps the envelope it has */ });
}

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

function round2(value) {
  return Math.round((Number(value) || 0) * 100) / 100;
}

export { fmt as formatTime };
