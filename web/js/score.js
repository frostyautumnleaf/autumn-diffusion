// Piano roll for the ABC scores YuE plans.
//
// YuE2 writes an ABC transcription before it makes any sound, and that score is
// the editable face of a composition: cover a song by transcribing it, fix a
// melody by hand, reharmonise without touching the tune. Reading ABC in the
// browser would mean a second parser that can disagree with the one the server
// validated, so the server sends note data (`abc.grid`) and this file only draws.
//
// The roll is canvas-only: no DOM per note, because a two-minute song is a few
// hundred notes across two voices and a div each would stutter. Times travel in
// seconds so the play head can follow the audio player without a tempo maths
// step on every frame.

import { el, clamp, num } from './util.js';
import { icon } from './icons.js';

const KEY_W = 30;          // width of the pitch strip on the left
const CHORD_H = 20;        // chord-symbol lane
const RULER_H = 20;        // bar numbers under the roll
const MIN_ROW = 2;         // px per semitone when zoomed out
const MAX_ROW = 26;        // px per semitone when zoomed in
const MIN_PPS = 6;         // px per second
const MAX_PPS = 640;
const PAD_Q = 0.5;         // quarters of slack at each end when fitting
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/** MIDI note number to scientific pitch name, A4 = 69. */
export function noteName(midi) {
  const n = Math.round(Number(midi) || 0);
  return `${NAMES[((n % 12) + 12) % 12]}${Math.floor(n / 12) - 1}`;
}

/** True for the black keys of the piano, used only to shade the strip. */
const isBlack = (midi) => [1, 3, 6, 8, 10].includes(((Math.round(midi) % 12) + 12) % 12);

/**
 * A scrollable, zoomable piano roll of one score.
 *
 * `onSeek` is called with seconds when the roll is clicked; `onTime` is asked for
 * the current play head each frame so the line can follow an audio element that
 * owns the clock. Neither is required.
 */
export class ScoreRoll {
  constructor({ onSeek = null, onTime = null } = {}) {
    this.grid = null;
    this.abc = '';
    this.valid = true;
    this.error = '';
    this.voices = { Vocal: true, Ins: true };
    this.pps = 60;              // px per second, i.e. horizontal zoom
    this.offset = 0;            // seconds at the left edge
    this.rowH = 11;             // px per semitone, i.e. vertical zoom
    this.low = 48;              // lowest MIDI number drawn
    this.high = 84;
    this.userAdjusted = false;    // true once the person has zoomed or scrolled
    this.hover = null;          // { x, y, note } while the cursor is over a note
    this.playhead = 0;
    this.active = false;        // animate only while the roll is on screen
    this.onSeek = onSeek;
    this.onTime = onTime;
    this._frame = null;
    this._drag = null;
    this._signature = '';

    this.shell = el('div', { class: 'roll-shell' });
    this.bar = el('div', { class: 'roll-bar' });
    this.canvas = el('canvas', { class: 'roll' });
    this.tip = el('div', { class: 'roll-tip', style: { display: 'none' } });
    this.empty = el('div', { class: 'roll-empty' });
    this.body = el('div', { class: 'roll-body' }, [this.canvas, this.tip, this.empty]);
    this.rulerLabel = el('span', { class: 'roll-read tiny faint' });
    this.bar.append(this.rulerLabel, el('span', { class: 'grow' }));
    this.shell.append(this.bar, this.body);
    this.ctx = this.canvas.getContext('2d');

    this.paintBar();
    this.wire();
  }

  // ------------------------------------------------------------------ chrome
  /** Buttons above the roll. Rebuilt only when the score appears or disappears. */
  paintBar() {
    const button = (name, title, run, extra = '') => el('button', {
      class: `icon-btn ${extra}`, html: icon(name, 14), title, onclick: () => run(),
    });
    this.bar.querySelectorAll('.icon-btn').forEach((node) => node.remove());
    const has = !!this.grid;
    this.bar.append(...[
      button('zoom_out', 'Fit the whole score', () => this.fit()),
      button('zoom_in', 'Zoom in on time', () => this.zoomBy(1.35)),
      el('button', {
        class: 'icon-btn', html: icon('shrink', 14), title: 'Fit the pitch range',
        onclick: () => this.fitPitch(),
      }),
      el('button', {
        class: 'icon-btn', html: icon('layers', 14), title: 'Voices to show',
        onclick: (event) => this.voiceMenu(event),
      }),
      has && el('button', {
        class: 'icon-btn', html: icon('download', 14), title: 'Save the score as .abc',
        onclick: () => this.saveAbc(),
      }),
    ].filter(Boolean));
  }

  /** Choose which of the two voices the roll draws. */
  voiceMenu(event) {
    import('./components.js').then(({ menu }) => menu(event.currentTarget, [
      { label: 'Vocal melody', checked: this.voices.Vocal, run: () => this.toggleVoice('Vocal') },
      { label: 'Instrumental', checked: this.voices.Ins, run: () => this.toggleVoice('Ins') },
      { sep: true },
      { label: 'Fit the pitch range', icon: 'shrink', run: () => this.fitPitch() },
      { label: 'Fit the whole song', icon: 'zoom_out', run: () => this.fit() },
    ]));
  }

  toggleVoice(name) {
    const other = name === 'Vocal' ? 'Ins' : 'Vocal';
    // Showing nothing would leave an empty box, so the last voice on stays on.
    this.voices[name] = !this.voices[name];
    if (!this.voices.Vocal && !this.voices.Ins) this.voices[other] = true;
    this.fitPitch();
    this.draw();
  }

  saveAbc() {
    if (!this.abc) return;
    const blob = new Blob([this.abc], { type: 'text/plain' });
    const link = el('a', { href: URL.createObjectURL(blob), download: 'score.abc' });
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  }

  /** Insert the roll into `target`, which the caller owns. */
  mount(target) {
    if (this.shell.parentNode === target) { this.resize(); return; }
    target.appendChild(this.shell);
    this.resize();
    requestAnimationFrame(() => this.resize());
  }

  /**
   * Show a parsed score. `payload` is the shape `/api/jobs/{id}/score` returns.
   *
   * A score that has not changed since the last call is ignored, so a progress
   * event cannot throw away the zoom. The first score of a job is fitted; after
   * that the view is left where the person put it.
   */
  setScore(payload) {
    const data = payload || {};
    const signature = `${data.abc || ''}|${data.valid ? 1 : 0}|${data.error || ''}`;
    if (signature === this._signature) return;
    this._signature = signature;
    this.abc = String(data.abc || '');
    this.valid = data.valid !== false;
    this.error = String(data.error || '');
    this.grid = this.valid ? (data.grid || null) : null;
    if (!this.grid) {
      this.paintEmpty();
      return;
    }
    this.empty.style.display = 'none';
    if (!this.userAdjusted) {
      this.fit();
      this.fitPitch();
    }
    this.paintHead();
    this.draw();
  }

  /** The line under the buttons: tempo, key, meter and length. */
  paintHead() {
    const grid = this.grid;
    if (!grid) { this.rulerLabel.textContent = ''; return; }
    const bars = (grid.bars || []).length;
    const beats = grid.meter ? grid.meter.split('/')[0] : 4;
    this.rulerLabel.textContent = `${grid.bpm} BPM · ${grid.meter} · ${grid.key || '—'}`
      + ` · ${bars} bars · ${num(grid.duration_seconds, 1)} s · ${beats} beats/bar`;
  }

  paintEmpty() {
    this.empty.style.display = '';
    this.empty.innerHTML = '';
    if (!this.valid && this.error) {
      this.empty.append(
        el('div', { class: 'e-ic', html: icon('warning', 22) }),
        el('div', { html: '<b>The score does not parse</b>' }),
        el('div', { class: 'tiny faint', text: this.error.slice(0, 220) }),
      );
      return;
    }
    this.empty.append(
      el('div', { class: 'e-ic', html: icon('code', 22) }),
      el('div', { html: '<b>No score yet</b>' }),
      el('div', {
        class: 'tiny faint',
        text: 'YuE writes its plan as ABC notation. It appears here while the job plans, '
          + 'and can be edited afterwards under Composition.',
      }),
    );
    this.rulerLabel.textContent = '';
  }

  // ------------------------------------------------------------------- input
  wire() {
    new ResizeObserver(() => this.resize()).observe(this.body);
    this.canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      if (event.shiftKey) {
        // Shift + wheel walks the pitch range, the axis a singer cares about.
        this.low -= event.deltaY * 0.02;
        this.userAdjusted = true;
        this.clampView();
        this.draw();
        return;
      }
      const at = this.secondAt(event.offsetX);
      this.pps = clamp(this.pps * Math.exp(-event.deltaY * 0.0016), MIN_PPS, MAX_PPS);
      // Hold the second under the cursor still, so zooming reads as magnifying.
      this.offset = at - (event.offsetX - KEY_W) / this.pps;
      this.userAdjusted = true;
      this.clampView();
      this.draw();
    }, { passive: false });

    this.canvas.addEventListener('pointerdown', (event) => {
      if (event.button === 2) return;
      this._drag = {
        x: event.clientX, y: event.clientY, moved: 0,
        offset: this.offset, low: this.low, rowH: this.rowH,
      };
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener('pointermove', (event) => {
      if (this._drag) {
        const dx = event.clientX - this._drag.x;
        const dy = event.clientY - this._drag.y;
        this._drag.moved = Math.max(this._drag.moved, Math.abs(dx) + Math.abs(dy));
        if (this._drag.moved > 3) {
          this.offset = this._drag.offset - dx / this.pps;
          this.low = this._drag.low + Math.round(dy / this._drag.rowH);
          this.userAdjusted = true;
          this.clampView();
          this.draw();
          return;
        }
      }
      this.trackHover(event);
    });
    this.canvas.addEventListener('pointerup', (event) => {
      const drag = this._drag;
      this._drag = null;
      if (drag && drag.moved > 3) return;
      if (event.target === this.canvas && this.onSeek) {
        this.onSeek(clamp(this.secondAt(event.offsetX), 0, this.duration()));
      }
    });
    this.canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      this.tip.style.display = 'none';
      this.draw();
    });
    this.canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      import('./components.js').then(({ menu }) => menu(event.currentTarget, [
        { label: 'Fit the whole score', icon: 'zoom_out', run: () => this.fit() },
        { label: 'Fit the pitch range', icon: 'shrink', run: () => this.fitPitch() },
        { label: 'Jump to the play head', icon: 'target', run: () => this.reveal(this.playhead) },
        { sep: true },
        { label: 'Copy the ABC text', icon: 'copy', run: () => this.copyAbc() },
      ]));
    });
  }

  copyAbc() {
    if (!this.abc) return;
    import('./util.js').then(({ copyText }) => copyText(this.abc));
  }

  /** Show what the cursor is over: pitch, onset, length. */
  trackHover(event) {
    const found = this.noteAt(event.offsetX, event.offsetY);
    const same = found && this.hover && found.voice === this.hover.voice
      && found.index === this.hover.index;
    this.hover = found;
    if (!found) {
      this.tip.style.display = 'none';
      this.draw();
      return;
    }
    const [onset, midi, length] = found.note;
    const beat = 60 / (this.grid.bpm || 120);
    this.tip.style.display = '';
    this.tip.innerHTML = '';
    this.tip.append(
      el('b', { text: noteName(midi) }),
      el('span', { class: 'tiny faint', text: found.voice === 'Vocal' ? 'vocal melody' : 'instrumental' }),
    );
    this.tip.append(el('div', {
      class: 'tiny faint',
      text: `bar ${this.barIndexAt(found.start) + 1}, beat ${this.beatAt(found.start)}`
        + ` · ${num(found.start, 2)} s in · ${num(length * beat, 2)} s long`
        + ` · ${num(onset, 2)} quarters`,
    }));
    this.tip.style.left = `${clamp(event.offsetX + 12, 6, Math.max(6, this.body.clientWidth - 180))}px`;
    this.tip.style.top = `${clamp(event.offsetY - 46, 4, Math.max(4, this.body.clientHeight - 60))}px`;
    this.draw();
  }

  // ---------------------------------------------------------------- geometry
  resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = this.body.clientWidth;
    const height = this.body.clientHeight;
    if (!width || !height) return;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = width;
    this.height = height;
    // A roll that was laid out while hidden fitted at width 0, which leaves the
    // music squeezed into a third of the canvas. Fit again once it is measured.
    if (this.grid && !this.userAdjusted) {
      this.fit();
      this.fitPitch();
      return;
    }
    this.clampView();
    this.draw();
  }

  rollWidth() { return Math.max(0, (this.width || 0) - KEY_W); }
  rollTop() { return CHORD_H; }
  rollHeight() { return Math.max(0, (this.height || 0) - CHORD_H - RULER_H); }
  duration() { return this.grid ? Number(this.grid.duration_seconds) || 0 : 0; }
  rowOf(midi) { return this.rollTop() + (this.high - midi) * this.rowH; }
  xOf(seconds) { return KEY_W + (seconds - this.offset) * this.pps; }
  secondAt(x) { return this.offset + (x - KEY_W) / this.pps; }
  midiAt(y) { return Math.round(this.high - (y - this.rollTop()) / this.rowH); }

  /** Keep the view inside the score, so a drag cannot lose the music entirely. */
  clampView() {
    const length = this.duration();
    const span = this.rollWidth() / this.pps;
    const min = -Math.min(1, span * 0.2);
    this.offset = clamp(this.offset, min, Math.max(min, length - Math.min(span, length) * 0.25));
    const rows = this.rollHeight() / Math.max(this.rowH, MIN_ROW);
    this.low = clamp(this.low, 12, Math.max(12, 127 - rows));
    this.high = this.low + rows;
  }

  /** Show the whole song, keeping the current pitch zoom. */
  fit() {
    const width = this.rollWidth();
    const length = Math.max(this.duration(), 1) + PAD_Q * 2 * (60 / (this.grid?.bpm || 120));
    if (width > 0 && length > 0) this.pps = clamp(width / length, MIN_PPS, MAX_PPS);
    this.offset = -0.25;
    this.draw();
  }

  zoomBy(factor, anchor = null) {
    const at = anchor === null ? this.secondAt(KEY_W + this.rollWidth() / 2) : anchor;
    this.pps = clamp(this.pps * factor, MIN_PPS, MAX_PPS);
    this.offset = at - (this.rollWidth() / 2) / this.pps;
    this.clampView();
    this.draw();
  }

  panBy(deltaSeconds) {
    this.offset += deltaSeconds;
    this.userAdjusted = true;
    this.clampView();
    this.draw();
  }

  /** Set the vertical range to the notes on show, plus a little air. */
  fitPitch() {
    const notes = this.notesToShow();
    let low = 60;
    let high = 72;
    if (notes.length) {
      low = Math.min(...notes.map((item) => item[1]));
      high = Math.max(...notes.map((item) => item[1]));
    }
    const rows = Math.max(8, high - low + 4);
    const height = this.rollHeight();
    this.rowH = clamp(height > 0 ? height / rows : 11, MIN_ROW, MAX_ROW);
    const span = height > 0 ? height / this.rowH : rows;
    this.low = Math.max(12, low - (span - (high - low)) / 2);
    this.high = this.low + span;
    this.draw();
  }

  /** Scroll so the play head is on screen, keeping the zoom. */
  reveal(seconds) {
    const span = this.rollWidth() / this.pps;
    if (seconds < this.offset || seconds > this.offset + span * 0.9) {
      this.offset = Math.max(-0.25, seconds - span * 0.35);
    }
    this.draw();
  }

  /** Notes of every visible voice, as the grid holds them. */
  notesToShow() {
    if (!this.grid) return [];
    const out = [];
    for (const [name, voice] of Object.entries(this.grid.voices || {})) {
      if (!this.voices[name]) continue;
      (voice.notes || []).forEach((note) => out.push([note, name]));
    }
    return out.map(([note]) => note);
  }

  /** The note under a point, if any, in the voice it belongs to. */
  noteAt(x, y) {
    if (!this.grid || x < KEY_W) return null;
    const seconds = this.secondAt(x);
    const midi = this.midiAt(y);
    for (const [name, voice] of Object.entries(this.grid.voices || {})) {
      if (!this.voices[name]) continue;
      const notes = voice.notes || [];
      for (let index = 0; index < notes.length; index += 1) {
        const note = notes[index];
        const top = this.rowOf(note[1]);
        if (Math.abs(y - (top + this.rowH / 2)) > Math.max(this.rowH, 3) * 0.75) continue;
        if (seconds >= note[3] - 0.02 && seconds <= note[3] + note[4] + 0.02) {
          return { note, voice: name, index, start: note[3] };
        }
        if (midi === note[1] && seconds >= note[3] - 0.05
            && seconds <= note[3] + note[4] + 0.05) {
          return { note, voice: name, index, start: note[3] };
        }
      }
    }
    return null;
  }

  barIndexAt(seconds) {
    const bars = (this.grid && this.grid.bars) || [];
    for (let index = 0; index < bars.length; index += 1) {
      if (seconds < bars[index].seconds + 1e-6) return index;
    }
    return bars.length;
  }

  beatAt(seconds) {
    const grid = this.grid;
    if (!grid) return 1;
    const beats = Number(String(grid.meter).split('/')[0]) || 4;
    const quarter = 60 / (grid.bpm || 120);
    return (Math.floor(seconds / quarter) % beats) + 1;
  }

  // ------------------------------------------------------------------ drawing
  /** Start or stop the animation loop; it costs nothing when the roll is hidden. */
  setActive(on) {
    this.active = !!on;
    if (this.active) this.animate();
  }

  animate() {
    if (!this.active || this._frame) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = null;
      const now = this.onTime ? Number(this.onTime()) : this.playhead;
      const moved = Math.abs(now - this.playhead) > 0.0005;
      this.playhead = now;
      if (moved && this.grid) {
        this.reveal(now);
        this.draw();
      }
      this.animate();
    });
  }

  /**
   * Colours for one frame, read from the theme.
   *
   * The roll is painted, not styled, so it has to re-read the palette each draw:
   * light mode and the accent picker change the same variables the CSS uses, and a
   * hardcoded grey would leave a dark slab in a light window.
   */
  palette() {
    const styles = getComputedStyle(document.documentElement);
    const read = (name, fallback) => (styles.getPropertyValue(name) || fallback).trim();
    return {
      bg: read('--bg-2', '#161616'),
      lane: read('--bg-3', '#1c1c1c'),
      line: read('--line', '#262626'),
      strong: read('--line-strong', '#343434'),
      text: read('--text-dim', '#adadad'),
      faint: read('--text-faint', '#585858'),
      accent: read('--accent', '#f0a24b'),
      second: read('--info', '#60a5fa'),
      warn: read('--warn', '#fbbf24'),
      err: read('--err', '#f87171'),
    };
  }

  draw() {
    const ctx = this.ctx;
    const width = this.width || 0;
    const height = this.height || 0;
    if (!ctx || !width || !height) return;
    const pal = this.pal = this.palette();

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = pal.bg;
    ctx.fillRect(0, 0, width, height);
    if (!this.grid) return;

    const top = this.rollTop();
    const bottom = height - RULER_H;
    const left = KEY_W;
    const right = width;

    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, Math.max(0, right - left), Math.max(0, bottom - top));
    ctx.clip();
    this.drawSections(ctx, top, bottom);
    this.drawGrid(ctx, top, bottom);
    this.drawNotes(ctx, top);
    this.drawPlayhead(ctx, top, bottom);
    ctx.restore();

    this.drawChords(ctx, right);
    this.drawKeys(ctx, top, bottom);
    this.drawRuler(ctx, left, right, height);
    this.drawFrame(ctx, width, height, top, bottom);
  }

  /** Alternating bands and labels for [Intro], [Chorus] and friends. */
  drawSections(ctx, top, bottom) {
    const grid = this.grid;
    const sections = grid.sections || [];
    const pal = this.pal;
    if (!sections.length) return;
    ctx.save();
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'top';
    sections.forEach((section, index) => {
      const start = this.xOf(section.seconds);
      const end = index + 1 < sections.length
        ? this.xOf(sections[index + 1].seconds) : this.xOf(this.duration());
      if (end < KEY_W || start > this.width) return;
      if (index % 2 === 1) {
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = pal.lane;
        ctx.fillRect(start, top, Math.max(1, end - start), bottom - top);
      }
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = pal.accent;
      ctx.fillRect(start, top, 1, bottom - top);
      if (section.name && end - start > 34) {
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = pal.text;
        ctx.fillText(section.name.slice(0, 18).toUpperCase(), start + 5, top + 3);
      }
      ctx.globalAlpha = 1;
    });
    ctx.restore();
  }

  /** Bar and beat lines, with the bar line stronger than the beats inside it. */
  drawGrid(ctx, top, bottom) {
    const grid = this.grid;
    const pal = this.pal;
    const quarter = 60 / (grid.bpm || 120);
    const beats = Number(String(grid.meter).split('/')[0]) || 4;
    const bars = grid.bars || [];
    ctx.save();
    ctx.lineWidth = 1;
    const pxPerBeat = quarter * this.pps;
    if (pxPerBeat > 26) {
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = pal.line;
      ctx.beginPath();
      for (const bar of bars) {
        for (let beat = 1; beat < beats; beat += 1) {
          const x = Math.round(this.xOf(bar.seconds + beat * quarter)) + 0.5;
          ctx.moveTo(x, top);
          ctx.lineTo(x, bottom);
        }
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = pal.strong;
    ctx.beginPath();
    for (const bar of bars) {
      const x = Math.round(this.xOf(bar.seconds)) + 0.5;
      if (x < KEY_W - 1 || x > this.width + 1) continue;
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  drawNotes(ctx, top) {
    if (!this.grid) return;
    const pal = this.pal;
    const height = Math.max(2, this.rowH - 1.5);
    const radius = Math.min(2.5, height / 2);
    const hover = this.hover;
    Object.entries(this.grid.voices || {}).forEach(([name, voice], order) => {
      if (!this.voices[name]) return;
      // The vocal line is the one a person edits, so it keeps the accent colour.
      ctx.fillStyle = order === 0 ? pal.accent : pal.second;
      ctx.globalAlpha = order === 0 ? 0.95 : 0.55;
      const notes = voice.notes || [];
      for (let index = 0; index < notes.length; index += 1) {
        const note = notes[index];
        const x = this.xOf(note[3]);
        const w = Math.max(2, note[4] * this.pps - 1);
        if (x + w < KEY_W || x > this.width) continue;
        const y = this.rowOf(note[1]);
        if (y < top - height || y > this.height) continue;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, y, w, height, radius);
        else ctx.rect(x, y, w, height);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    });
    if (hover) {
      ctx.save();
      ctx.strokeStyle = pal.text;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 1;
      const note = hover.note;
      const x = this.xOf(note[3]);
      const w = Math.max(2, note[4] * this.pps - 1);
      ctx.strokeRect(x - 0.5, this.rowOf(note[1]) - 0.5, w + 1, height + 1);
      ctx.restore();
    }
  }

  drawPlayhead(ctx, top, bottom) {
    const length = this.duration();
    const at = clamp(this.playhead, 0, Math.max(length, 0.001));
    const x = Math.round(this.xOf(at)) + 0.5;
    if (x < KEY_W || x > this.width) return;
    const pal = this.pal;
    ctx.save();
    ctx.strokeStyle = pal.err;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = pal.err;
    ctx.fillRect(x - 3, top, 6, bottom - top);
    ctx.restore();
  }

  /** Chord symbols along the top, each holding the ground until the next one. */
  drawChords(ctx, right) {
    const grid = this.grid;
    if (!grid) return;
    const pal = this.pal;
    ctx.save();
    ctx.fillStyle = pal.lane;
    ctx.globalAlpha = 0.6;
    ctx.fillRect(0, 0, right, CHORD_H);
    ctx.globalAlpha = 1;
    ctx.font = '600 10px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const chords = ((grid.voices || {}).Vocal || {}).chords || [];
    // A symbol holds the ground until the next one, which is the space it may use.
    const endOf = (index) => (index + 1 < chords.length
      ? Number(chords[index + 1][1]) : this.duration());
    chords.forEach((chord, index) => {
      const [, seconds, symbol] = chord;
      if (!symbol) return;
      const x = this.xOf(Number(seconds));
      const room = Math.max(26, this.xOf(endOf(index)) - x - 6);
      if (x + room < KEY_W || x > this.width) return;
      // Parentheses mark harmony the plan inferred rather than heard.
      ctx.fillStyle = symbol.startsWith('(') ? pal.warn : pal.text;
      ctx.globalAlpha = 0.9;
      ctx.save();
      ctx.beginPath();
      ctx.rect(Math.max(KEY_W, x), 0, room, CHORD_H);
      ctx.clip();
      ctx.fillText(symbol, x + 4, CHORD_H / 2);
      ctx.restore();
      ctx.globalAlpha = 1;
    });
    ctx.restore();
  }

  /** The pitch strip: C and A rows are labelled, black keys are shaded. */
  drawKeys(ctx, top, bottom) {
    const pal = this.pal;
    ctx.save();
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = pal.lane;
    ctx.fillRect(0, top, KEY_W, bottom - top);
    ctx.globalAlpha = 1;
    for (let midi = Math.ceil(this.low); midi <= Math.floor(this.high); midi += 1) {
      const y = this.rowOf(midi);
      if (y < top - this.rowH || y > bottom) continue;
      ctx.globalAlpha = isBlack(midi) ? 0.5 : 0.28;
      ctx.fillStyle = isBlack(midi) ? pal.bg : pal.strong;
      ctx.fillRect(0, y, KEY_W, Math.max(1, this.rowH));
      const natural = ((midi % 12) + 12) % 12;
      // C rows name the octave first, A rows join in once there is room; below
      // 4.6 px per row text would overlap, so the strip falls back to shading.
      const label = (natural === 0 && this.rowH >= 4.6)
        || (natural === 9 && this.rowH >= 8);
      if (label) {
        ctx.globalAlpha = 0.9;
        ctx.font = `${this.rowH >= 8 ? 8 : 7}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = pal.text;
        ctx.textBaseline = 'middle';
        ctx.fillText(noteName(midi), 4, y + this.rowH / 2);
      } else if (natural === 0) {
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = pal.strong;
        ctx.fillRect(0, y, KEY_W, 1);
      }
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.line;
    ctx.beginPath();
    ctx.moveTo(KEY_W + 0.5, top);
    ctx.lineTo(KEY_W + 0.5, bottom);
    ctx.stroke();
    ctx.restore();
  }

  /** Bar numbers, and seconds when the zoom is high enough to need them. */
  drawRuler(ctx, left, right, height) {
    const grid = this.grid;
    const pal = this.pal;
    const bars = grid.bars || [];
    const y = height - RULER_H;
    ctx.save();
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = pal.lane;
    ctx.fillRect(0, y, right, RULER_H);
    ctx.globalAlpha = 1;
    ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const quarter = 60 / (grid.bpm || 120);
    const barPx = bars.length ? (bars[0].end - bars[0].start) * quarter * this.pps : 100;
    const every = Math.max(1, Math.round(46 / Math.max(barPx, 1)));
    ctx.fillStyle = pal.text;
    ctx.globalAlpha = 0.85;
    bars.forEach((bar, index) => {
      const x = this.xOf(bar.seconds);
      if (x < left - 40 || x > right) return;
      if (index % every === 0) ctx.fillText(String(index + 1), x + 4, y + RULER_H / 2);
    });
    if (this.pps > 90) {
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = pal.faint;
      const step = this.pps > 260 ? 1 : this.pps > 140 ? 5 : 10;
      for (let s = 0; s <= this.duration(); s += step) {
        const x = this.xOf(s);
        if (x < left || x > right - 26) continue;
        ctx.fillText(`${s}s`, x + 2, y + RULER_H / 2);
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  drawFrame(ctx, width, height, top, bottom) {
    ctx.save();
    ctx.strokeStyle = this.pal.line;
    ctx.beginPath();
    ctx.moveTo(0, CHORD_H + 0.5);
    ctx.lineTo(width, CHORD_H + 0.5);
    ctx.moveTo(0, bottom + 0.5);
    ctx.lineTo(width, bottom + 0.5);
    ctx.stroke();
    ctx.restore();
  }
}

export default ScoreRoll;
