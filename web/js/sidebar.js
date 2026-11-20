// Right sidebar: what this session produced, plus the live queue.
// Everything here is scoped to the current session id; the bottom history
// block is where older sessions live.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, menu } from './components.js';
import { el, clear, mediaUrl, baseName, clockTime, seconds as secs } from './util.js';
import { thumbNode, thumbOf, loadIntoForm, openJob, selectJob } from './history.js';

export class Sidebar {
  constructor(root) {
    this.root = root;
    this.render();
    // The event carries the job that changed when it can, so a progress tick only
    // has to write one row instead of the whole list.
    state.on(TOPICS.jobs, (job) => this.renderList(job));
    // A full snapshot only matters here when it brought a new session id, because
    // the list is scoped to the session.
    this._session = state.session;
    state.on(TOPICS.state, () => {
      if (state.session === this._session) return;
      this._session = state.session;
      this.render();
    });
    // The selection outline has to follow clicks, and selecting is a UI change
    // rather than a job change, so this listens on its own topic.
    state.on(TOPICS.ui, (patch) => {
      if (patch && patch.selectedJob !== undefined) this.syncActive();
    });
  }

  /** Move the selection outline to the item the stage is showing. */
  syncActive() {
    if (!this.list) return;
    for (const node of this.list.querySelectorAll('.session-item')) {
      node.classList.toggle('active', node.dataset.job === state.ui.selectedJob);
    }
  }

  render() {
    clear(this.root);
    this._signature = null;
    this.nodes = null;
    this.root.appendChild(el('div', { class: 'side-head' }, [
      el('span', { class: 'rail-title', text: 'This session' }),
      el('span', { class: 'grow' }),
      el('span', { class: 'queue-note', text: `#${(state.session || '').slice(0, 6)}` }),
      el('button', {
        class: 'icon-btn', html: icon('sliders_h', 14), title: 'Sidebar options',
        onclick: (event) => this.openMenu(event.currentTarget),
      }),
    ]));
    this.list = el('div', { class: 'side-scroll' });
    this.root.appendChild(this.list);
    this.renderList();
  }

  renderList(changed) {
    if (!this.list) return;
    const active = state.jobList({ limit: 500 }).filter(
      (job) => job.status === 'running' || job.status === 'queued');
    const done = state.sessionJobs().filter(
      (job) => job.status !== 'running' && job.status !== 'queued');
    const signature = `${active.map((job) => job.id).join(',')}`
      + `|${done.map((job) => job.id).join(',')}`;
    if (signature === this._signature && this.nodes) {
      // Only job data changed: patch rows instead of rebuilding them, so the
      // thumbnails do not reload and flash on every step.
      if (changed && this.patchOne(changed)) return;
      for (const job of [...active, ...done]) {
        const node = this.nodes.get(job.id);
        if (node) this.updateItem(node, job);
      }
      if (this.doneLabel) this.doneLabel.textContent = `Completed (${done.length})`;
      return;
    }
    this._signature = signature;
    this.nodes = new Map();
    clear(this.list);
    this.runningLabel = sectionLabel('Running');
    this.doneLabel = sectionLabel(`Completed (${done.length})`);
    if (active.length) {
      this.list.appendChild(this.runningLabel);
      for (const job of active) this.add(job, true);
    }
    this.list.appendChild(this.doneLabel);
    if (!done.length && !active.length) {
      this.list.appendChild(el('div', { class: 'empty' }, [
        el('div', { class: 'e-ic', html: icon('history', 20) }),
        el('div', {}, [
          el('b', { text: 'Nothing yet' }),
          el('span', { text: 'Generations from this session appear here.' }),
        ]),
      ]));
      return;
    }
    for (const job of done) this.add(job, false);
  }

  /**
   * Write one job onto its row, when that row is in the right group.
   *
   * A job that started, finished or left the session changes group, and the list
   * has to be rebuilt for it; anything else is a progress tick that one row can
   * carry. Returning false hands the work back to the full pass.
   */
  patchOne(job) {
    const node = this.nodes.get(job.id);
    if (!node) return false;
    const running = job.status === 'running' || job.status === 'queued';
    const inSession = job.session_id === state.session;
    if (!!node._live !== running) return false;
    if (!running && !inSession) return false;
    this.updateItem(node, job);
    return true;
  }

  add(job, live) {
    const node = this.item(job, live);
    this.nodes.set(job.id, node);
    this.list.appendChild(node);
  }

  item(job, live) {
    const node = el('div', {
      class: 'session-item',
      dataset: { job: job.id },
      onclick: () => selectJob(job),
      ondblclick: () => openJob(job),
      oncontextmenu: (event) => {
        event.preventDefault();
        this.itemMenu(null, job, { x: event.clientX, y: event.clientY });
      },
    }, [
      el('div', { class: 'session-thumb' }),
      el('div', { class: 'session-main' }, [
        el('div', { class: 'session-title' }),
        el('div', { class: 'session-meta' }, [
          el('span', { class: `badge ${job.kind}`, text: job.kind }),
          el('span', { class: 'm-when', text: clockTime(job.created_at) }),
          el('span', { class: 'm-el' }),
          el('span', { class: 'm-state' }),
        ]),
        el('div', { class: 'bar', style: { marginTop: '5px' } }, [
          el('i', { style: { display: 'block', height: '2px', background: 'var(--accent)' } }),
        ]),
      ]),
      el('button', {
        class: 'icon-btn', style: { width: '22px', height: '22px' },
        html: icon('grip', 12), title: 'Actions',
        onclick: (event) => { event.stopPropagation(); this.itemMenu(event.currentTarget, job); },
      }),
    ]);
    node._refs = {
      thumb: node.querySelector('.session-thumb'),
      title: node.querySelector('.session-title'),
      elapsed: node.querySelector('.m-el'),
      state: node.querySelector('.m-state'),
      bar: node.querySelector('.bar'),
      barFill: node.querySelector('.bar i'),
    };
    node._live = !!live;
    this.updateItem(node, job);
    return node;
  }

  /** Write the current progress, title and thumbnail of a job onto its row. */
  updateItem(node, job) {
    const refs = node._refs;
    if (!refs) return;
    const running = job.status === 'running' || job.status === 'queued';
    const progress = job.steps ? (job.step || 0) / job.steps : (job.progress || 0);
    const pct = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
    const title = `${job.title || job.prompt || job.id} \u2014 ${job.status}`;
    const heading = job.title || job.prompt || job.id;
    const elapsed = job.elapsed ? secs(job.elapsed) : '';
    const stateText = job.status === 'error' ? 'error' : (running ? `${Math.round(progress * 100)}%` : '');
    // Every write is guarded: assigning the same value still replaces the text node,
    // and dozens of rows per progress event is what made the list flicker.
    if (node.title !== title) node.title = title;
    if (refs.title.textContent !== heading) refs.title.textContent = heading;
    if (refs.elapsed.textContent !== elapsed) refs.elapsed.textContent = elapsed;
    if (refs.state.textContent !== stateText) refs.state.textContent = stateText;
    const stateClass = job.status === 'error' ? 'm-state err' : 'm-state';
    if (refs.state.className !== stateClass) refs.state.className = stateClass;
    const barShown = running ? 'block' : 'none';
    if (refs.bar.style.display !== barShown) refs.bar.style.display = barShown;
    if (running && refs.barFill.style.width !== pct) refs.barFill.style.width = pct;
    const activeOn = state.ui.selectedJob === job.id;
    if (node.classList.contains('active') !== activeOn) node.classList.toggle('active', activeOn);
    const src = thumbOf(job);
    if (src !== refs.thumbSrc) {
      refs.thumbSrc = src;
      clear(refs.thumb);
      refs.thumb.appendChild(thumbNode(job, 18));
    }
  }

  itemMenu(anchor, job, at = null) {
    const items = [
      { label: 'Open in the viewer', icon: 'expand', run: () => openJob(job) },
      { label: 'Show in the stage', icon: 'arrow_right', run: () => selectJob(job) },
      { label: 'Load settings', icon: 'sliders_h', run: () => loadIntoForm(job) },
      { label: 'Run again', icon: 'refresh', run: async () => {
        try { await api.rerunJob(job.id); toast('Requeued', { kind: 'ok' }); }
        catch (err) { toast(err.message, { kind: 'err' }); }
      } },
    ];
    if (job.status === 'running' || job.status === 'queued') {
      items.push({ label: 'Cancel', icon: 'stop', danger: true, run: async () => {
        try { await api.cancelJob(job.id); toast('Cancelling', { kind: 'warn' }); }
        catch (err) { toast(err.message, { kind: 'err' }); }
      } });
    }
    items.push({ sep: true });
    items.push({ label: 'Hide from this session', icon: 'eye_off', run: () => state.removeJob(job.id) });
    menu(anchor, items, at ? { at } : {});
  }

  openMenu(anchor) {
    const counts = state.historyStats || {};
    menu(anchor, [
      { label: 'Hide the sidebar', icon: 'eye_off', run: () => state.setUI({ sidebarVisible: false }) },
      { label: 'Cancel everything queued', icon: 'stop', danger: true, run: async () => {
        for (const job of state.jobList({ limit: 100 })) {
          if (job.status === 'running' || job.status === 'queued') {
            try { await api.cancelJob(job.id); } catch (err) { /* already gone */ }
          }
        }
      } },
      { label: `History: ${counts.total || state.jobs.size} entries`, disabled: true },
    ]);
  }
}

function sectionLabel(text) {
  return el('div', { class: 'sub-label', text, style: { margin: '8px 4px 2px' } });
}
