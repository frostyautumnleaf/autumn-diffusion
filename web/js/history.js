// Bottom history block: every past generation, this session and earlier ones.
// The strip is filtered by the tab the user is on, so the image tab shows
// images and the music tab shows tracks.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, menu, confirmDialog, lightbox } from './components.js';
import { el, clear, fileUrl, mediaUrl, baseName, timeAgo, seconds as secs, debounce } from './util.js';

const KIND_LABEL = { image: 'Images', video: 'Videos', music: 'Tracks' };

export function thumbOf(job) {
  const file = job.preview || (job.files || [])[0];
  if (!file) return '';
  if (file.thumb) return fileUrl(file.thumb);
  if (file.kind === 'audio' || job.kind === 'music') return '';
  return mediaUrl(file);
}

/**
 * Thumbnail node for a job. A picture that cannot be read is swapped for the
 * kind icon, so a cleared thumbnail cache never leaves a broken image on screen.
 */
export function thumbNode(job, size = 20) {
  const fallback = () => el('span', {
    class: 'thumb-fallback',
    html: icon(job.kind === 'music' ? 'music' : (job.kind || 'image'), size),
  });
  const src = thumbOf(job);
  if (!src) return fallback();
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
    this._refresh = debounce(() => this.renderBody(), 60);
    state.on(TOPICS.jobs, this._refresh);
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

  renderBody() {
    if (!this.strip) return;
    const jobs = this.jobs();
    if (this.titleEl) this.titleEl.textContent = KIND_LABEL[this.kind] || 'History';
    if (this.countEl) this.countEl.textContent = `${jobs.length} shown`;
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
    for (const job of jobs) this.strip.appendChild(this.card(job));
  }

  card(job) {
    const active = state.ui.selectedJob === job.id;
    const running = job.status === 'running' || job.status === 'queued';
    const progress = job.steps ? (job.step || 0) / job.steps : (job.progress || 0);

    const card = el('div', {
      class: `hist-card ${active ? 'active' : ''}`,
      dataset: { job: job.id },
      title: `${job.title || job.prompt || job.id} — ${job.status}`,
      onclick: () => state.setUI({ selectedJob: job.id }),
      ondblclick: () => openJob(job),
    }, [
      el('div', { class: 'hist-thumb' }, [
        thumbNode(job),
        running ? el('div', { class: 'hist-live' }, [el('div', { class: 'spinner' })]) : null,
      ]),
      el('div', { class: 'hist-bar' }, [el('i', { style: { width: `${Math.round(progress * 100)}%` } })]),
      el('div', { class: 'hist-info' }, [
        el('div', { class: 'hist-title', text: job.title || job.prompt || job.id }),
        el('div', { class: 'hist-meta' }, [
          el('span', { text: timeAgo(job.created_at) }),
          el('span', { text: job.elapsed ? secs(job.elapsed) : '' }),
          el('span', { text: job.status === 'error' ? 'failed' : '' }),
        ]),
      ]),
      el('button', {
        class: `hist-star ${job.favorite ? 'on' : ''}`,
        html: icon('star', 11), title: 'Favourite',
        onclick: async (event) => {
          event.stopPropagation();
          const next = !job.favorite;
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
    return card;
  }

  jobMenu(anchor, job) {
    menu(anchor, [
      { label: 'Open in the viewer', icon: 'expand', run: () => openJob(job) },
      { label: 'Load settings into the form', icon: 'sliders_h', run: () => loadIntoForm(job) },
      { label: 'Run again', icon: 'refresh', run: () => rerun(job) },
      { label: job.favorite ? 'Remove favourite' : 'Favourite', icon: 'star', run: () => favourite(job, !job.favorite) },
      { label: 'Copy the prompt', icon: 'copy', run: () => copyPrompt(job) },
      { sep: true },
      { label: 'Delete from history', icon: 'trash', danger: true, run: () => remove(this, job) },
    ]);
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
