// Bottom history block: every past generation, this session and earlier ones.
// The strip is filtered by the tab the user is on, so the image tab shows
// images and the music tab shows tracks.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, menu, confirmDialog, lightbox } from './components.js';
import { el, clear, fileUrl, mediaUrl, baseName, timeAgo, seconds as secs, debounce } from './util.js';
import { waveThumb } from './waveform.js';

const KIND_LABEL = { image: 'Images', video: 'Videos', music: 'Tracks' };

export function thumbOf(job) {
  const file = job.preview || (job.files || [])[0];
  if (!file) return '';
  if (file.thumb) return fileUrl(file.thumb);
  if (file.kind === 'audio' || job.kind === 'music') return '';
  return mediaUrl(file);
}

/**
 * Thumbnail node for a job.
 *
 * Audio has no picture, so a track is drawn as a static waveform from the peaks
 * the worker reports. A picture that cannot be read is swapped for the kind icon,
 * so a cleared thumbnail cache never leaves a broken image on screen.
 */
export function thumbNode(job, size = 20) {
  const fallback = () => el('span', {
    class: 'thumb-fallback',
    html: icon(job.kind === 'music' ? 'music' : (job.kind || 'image'), size),
  });
  const src = thumbOf(job);
  if (!src) {
    const peaks = (job.meta || {}).peaks;
    if (job.kind === 'music' && Array.isArray(peaks) && peaks.length) return waveThumb(peaks);
    return fallback();
  }
  const img = el('img', { src, alt: '', loading: 'lazy' });
  img.addEventListener('error', () => img.replaceWith(fallback()), { once: true });
  return img;
}

export class History {
  constructor(root) {
    this.root = root;
    this.query = '';
    this.favoritesOnly = false;
    this.render();
    // Progress events arrive many times a second. The last job that changed is
    // carried into the debounced repaint so that only its card is written.
    this._pendingJob = null;
    this._refresh = debounce(() => {
      const job = this._pendingJob;
      this._pendingJob = null;
      this.renderBody(job);
    }, 60);
    state.on(TOPICS.jobs, (job) => {
      if (job && job.id) this._pendingJob = job;
      this._refresh();
    });
    state.on(TOPICS.ui, (patch) => {
      if (!patch) return;
      if (patch.tab || patch.selectedJob !== undefined) this.renderBody();
    });
  }

  get kind() {
    const tab = state.ui.tab;
    return tab === 'music' ? 'music' : (tab === 'video' ? 'video' : 'image');
  }

  render() {
    clear(this.root);
    this.root.className = 'history';
    this.root.appendChild(this.renderHead());
    this.body = el('div', { class: 'history-body' });
    this.strip = el('div', { class: 'history-strip' });
    this.body.appendChild(this.strip);
    this.root.appendChild(this.body);
    this._signature = null;   // the new strip is empty, so it must be filled
    this.renderBody();
  }

  renderHead() {
    const input = el('input', {
      class: 'input sm', placeholder: 'Filter history…', style: { width: '170px' },
      oninput: debounce((event) => {
        this.query = event.target.value.trim().toLowerCase();
        this.renderBody();
      }, 180),
    });

    return el('div', { class: 'history-head' }, [
      el('span', { class: 'section-icon', html: icon('history', 15) }),
      this.titleEl = el('span', { class: 'rail-title', text: KIND_LABEL[this.kind] || 'History' }),
      input,
      el('button', {
        class: `icon-btn ${this.favoritesOnly ? 'active' : ''}`,
        html: icon('star', 14), title: 'Show favourites only',
        onclick: () => { this.favoritesOnly = !this.favoritesOnly; this.render(); },
      }),
      el('span', { class: 'grow' }),
      this.countEl = el('span', { class: 'queue-note', text: '' }),
      el('button', {
        class: 'icon-btn', html: icon('grid', 14), title: 'History options',
        onclick: (event) => this.openMenu(event.currentTarget),
      }),
      el('button', {
        class: 'icon-btn',
        html: icon(state.ui.historyCollapsed ? 'chevron_up' : 'chevron_down', 14),
        title: 'Collapse / expand',
        onclick: () => {
          state.setUI({ historyCollapsed: !state.ui.historyCollapsed });
          document.body.classList.toggle('history-collapsed', state.ui.historyCollapsed);
          this.render();
        },
      }),
    ]);
  }

  jobs() {
    return state.jobList({
      kind: this.kind,
      limit: 300,
      favorites: this.favoritesOnly,
      query: this.query,
    });
  }

  renderBody(changed) {
    if (!this.strip) return;
    const jobs = this.jobs();
    if (this.titleEl) this.titleEl.textContent = KIND_LABEL[this.kind] || 'History';
    if (this.countEl) this.countEl.textContent = `${jobs.length} shown`;
    const signature = jobs.map((job) => job.id).join(',');
    if (signature !== this._signature) {
      // The set or the order changed, so the strip is rebuilt. Progress updates do
      // not change it, which is what keeps thumbnails from flickering.
      this._signature = signature;
      this.cards = new Map();
      clear(this.strip);
      if (!jobs.length) {
        this.strip.appendChild(el('div', { class: 'empty', style: { width: '100%' } }, [
          el('div', { class: 'e-ic', html: icon('history', 22) }),
          el('div', {}, [
            el('b', { text: `No ${KIND_LABEL[this.kind] ? KIND_LABEL[this.kind].toLowerCase() : 'history'} yet` }),
            el('span', { text: 'Finished generations land here and stay between sessions.' }),
          ]),
        ]));
        return;
      }
      for (const job of jobs) {
        const card = this.card(job);
        this.cards.set(job.id, card);
        this.strip.appendChild(card);
      }
      return;
    }
    // The set and the order are unchanged. When the event named one job and that
    // card is still part of what the filter shows, patching that card is enough.
    if (changed && !this.query && !this.favoritesOnly && changed.kind === this.kind
      && this.cards && this.cards.has(changed.id)) {
      this.updateCard(this.cards.get(changed.id), changed);
      return;
    }
    for (const job of jobs) {
      const card = this.cards && this.cards.get(job.id);
      if (card) this.updateCard(card, job);
    }
  }

  card(job) {
    const card = el('div', {
      class: 'hist-card',
      dataset: { job: job.id },
      onclick: () => selectJob(job),
      ondblclick: () => openJob(job),
      oncontextmenu: (event) => {
        event.preventDefault();
        this.jobMenu(null, job, { x: event.clientX, y: event.clientY });
      },
    }, [
      el('div', { class: 'hist-thumb' }),
      el('div', { class: 'hist-bar' }, [el('i')]),
      el('div', { class: 'hist-info' }, [
        el('div', { class: 'hist-title' }),
        el('div', { class: 'hist-meta' }, [
          el('span', { class: 'm-when' }),
          el('span', { class: 'm-el' }),
          el('span', { class: 'm-state' }),
        ]),
      ]),
      el('button', {
        class: 'hist-star', html: icon('star', 11), title: 'Favourite',
        onclick: async (event) => {
          event.stopPropagation();
          // The closure holds a snapshot, whose favourite flag goes stale after the
          // first toggle, so the current flag is read from the store.
          const next = !(state.jobs.get(job.id) || job).favorite;
          state.upsertJob({ id: job.id, favorite: next });
          try { await api.favoriteJob(job.id, next); } catch (err) { toast(err.message, { kind: 'err' }); }
        },
      }),
      el('button', {
        class: 'hist-run', html: icon('refresh', 11), title: 'Load these settings into the form',
        onclick: (event) => { event.stopPropagation(); loadIntoForm(job); },
      }),
      el('button', {
        class: 'hist-run', style: { right: '28px' }, html: icon('grip', 11), title: 'More',
        onclick: (event) => { event.stopPropagation(); this.jobMenu(event.currentTarget, job); },
      }),
    ]);
    card._refs = {
      thumb: card.querySelector('.hist-thumb'),
      bar: card.querySelector('.hist-bar i'),
      title: card.querySelector('.hist-title'),
      when: card.querySelector('.m-when'),
      elapsed: card.querySelector('.m-el'),
      state: card.querySelector('.m-state'),
    };
    this.updateCard(card, job);
    return card;
  }

  /**
   * Refresh one card in place.
   *
   * Progress events arrive many times a second. Rebuilding the card would drop the
   * <img> and the browser would reload the thumbnail, which reads as a flash, so
   * only the parts that changed are written.
   */
  updateCard(card, job) {
    const refs = card._refs;
    if (!refs) return;
    const running = job.status === 'running' || job.status === 'queued';
    const progress = job.steps ? (job.step || 0) / job.steps : (job.progress || 0);
    const title = `${job.title || job.prompt || job.id} \u2014 ${job.status}`;
    const heading = job.title || job.prompt || job.id;
    const when = timeAgo(job.created_at);
    const elapsed = job.elapsed ? secs(job.elapsed) : '';
    const stateText = job.status === 'error' ? 'failed'
      : (running ? `${Math.round(progress * 100)}%` : '');
    // Each write is guarded. Assigning a value that has not changed still replaces
    // the text node, and with a strip full of cards that is what flickered.
    card.classList.toggle('active', state.ui.selectedJob === job.id);
    card.classList.toggle('running', running);
    card.classList.toggle('failed', job.status === 'error');
    if (card.title !== title) card.title = title;
    if (running) refs.bar.style.width = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
    if (refs.title.textContent !== heading) refs.title.textContent = heading;
    if (refs.when.textContent !== when) refs.when.textContent = when;
    if (refs.elapsed.textContent !== elapsed) refs.elapsed.textContent = elapsed;
    if (refs.state.textContent !== stateText) refs.state.textContent = stateText;
    refs.state.classList.toggle('err', job.status === 'error');
    const star = card.querySelector('.hist-star');
    star.classList.toggle('on', !!job.favorite);
    // The spinner lives inside the thumb, so it is patched rather than rebuilt.
    let live = refs.thumb.querySelector(':scope > .hist-live');
    if (running && !live) {
      live = el('div', { class: 'hist-live' }, [el('div', { class: 'spinner' })]);
      refs.thumb.appendChild(live);
    } else if (!running && live) {
      live.remove();
      // The node is gone, so it must not be used as the insertion reference below.
      live = null;
    }
    const src = thumbOf(job);
    // A finished job must show something even when the thumbnail path never
    // changed, which is the case for audio and for a job that failed.
    const empty = !running && !refs.thumb.querySelector(':scope > :not(.hist-live)');
    if (src !== refs.thumbSrc || empty) {
      refs.thumbSrc = src;
      for (const child of Array.from(refs.thumb.children)) {
        if (child !== live) child.remove();
      }
      if (src || !running) refs.thumb.appendChild(thumbNode(job));
    }
  }

  jobMenu(anchor, job, at = null) {
    menu(anchor, [
      { label: 'Open in the viewer', icon: 'expand', run: () => openJob(job) },
      { label: 'Show in the stage', icon: 'arrow_right', run: () => selectJob(job) },
      { label: 'Load settings into the form', icon: 'sliders_h', run: () => loadIntoForm(job) },
      { label: 'Run again', icon: 'refresh', run: () => rerun(job) },
      { label: job.favorite ? 'Remove favourite' : 'Favourite', icon: 'star', run: () => favourite(job, !job.favorite) },
      { label: 'Copy the prompt', icon: 'copy', run: () => copyPrompt(job) },
      { label: 'Copy the settings', icon: 'code', run: () => copySettings(job) },
      { sep: true },
      { label: 'Delete from history', icon: 'trash', danger: true, run: () => remove(this, job) },
    ], at ? { at } : {});
  }

  openMenu(anchor) {
    menu(anchor, [
      { label: 'Clear this tab\'s history', icon: 'trash', danger: true, run: () => clearHistory(this) },
      { label: 'Clear all history', icon: 'trash', danger: true, run: () => clearHistory(this, true) },
      { label: 'Reload history', icon: 'refresh', run: () => window.dispatchEvent(new CustomEvent('autumn:refresh')) },
    ]);
  }
}

// Actions ------------------------------------------------------------------
export function openJob(job) {
  const file = (job.files || [])[0];
  if (!file) { toast('This job has no files', { kind: 'warn' }); return; }
  lightbox({
    title: baseName(file.name || job.title || job.id),
    subtitle: job.prompt || '',
    url: mediaUrl(file),
    kind: file.kind === 'audio' || job.kind === 'music' ? 'audio' : (file.kind || 'image'),
    meta: [
      ['Job', job.id],
      ['Status', job.status],
      ['Engine', job.engine || ''],
      ['Elapsed', job.elapsed ? `${job.elapsed}s` : ''],
    ],
    actions: [{ label: 'Run again', icon: 'refresh', run: () => rerun(job) }],
  });
}

/**
 * Make `job` the selected result, moving to its tab if the open tab shows a
 * different kind. Selection and view have to agree, otherwise the outline moves
 * while the stage keeps showing something else.
 */
export function selectJob(job) {
  const patch = { selectedJob: job.id };
  const kind = job.kind;
  if (kind && kind !== state.ui.tab && ['image', 'video', 'music'].includes(kind)) {
    patch.tab = kind;
  }
  state.setUI(patch);
}

export function loadIntoForm(job) {
  if (!job.params) { toast('Nothing to load', { kind: 'warn' }); return; }
  const kind = job.kind === 'music' ? 'music' : job.kind;
  state.loadForm(kind, job.params);
  state.setUI({ tab: kind });
  toast('Settings loaded into the form', { kind: 'ok' });
}

async function rerun(job) {
  try {
    await api.rerunJob(job.id);
    toast('Requeued', { kind: 'ok' });
  } catch (err) { toast(err.message, { kind: 'err' }); }
}

async function favourite(job, value) {
  state.upsertJob({ id: job.id, favorite: value });
  try { await api.favoriteJob(job.id, value); } catch (err) { toast(err.message, { kind: 'err' }); }
}

async function copyPrompt(job) {
  const { copyText } = await import('./util.js');
  try { await copyText(job.prompt || ''); toast('Prompt copied', { kind: 'ok' }); }
  catch (err) { toast(err.message, { kind: 'err' }); }
}

async function copySettings(job) {
  const { copyText } = await import('./util.js');
  try {
    await copyText(JSON.stringify(job.params || {}, null, 2));
    toast('Settings copied as JSON', { kind: 'ok' });
  } catch (err) { toast(err.message, { kind: 'err' }); }
}

async function remove(history, job) {
  const sure = await confirmDialog({
    title: 'Delete from history',
    message: `Remove ${job.title || job.id} from the history list? The files on disk are kept.`,
    confirmLabel: 'Delete entry',
  });
  if (!sure) return;
  try {
    await api.deleteJob(job.id);
    state.removeJob(job.id);
    toast('History entry removed', { kind: 'ok' });
  } catch (err) { toast(err.message, { kind: 'err' }); }
}

async function clearHistory(history, all = false) {
  const sure = await confirmDialog({
    title: 'Clear history',
    message: all
      ? 'Remove every history entry? Generated files stay on disk.'
      : `Remove the ${KIND_LABEL[history.kind] || ''} history? Generated files stay on disk.`,
    confirmLabel: 'Clear',
  });
  if (!sure) return;
  try {
    await api.clearHistory(all ? '' : history.kind, '');
    window.dispatchEvent(new CustomEvent('autumn:refresh'));
    toast('History cleared', { kind: 'ok' });
  } catch (err) { toast(err.message, { kind: 'err' }); }
}
