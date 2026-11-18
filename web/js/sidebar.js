// Right sidebar: what this session produced, plus the live queue.
// Everything here is scoped to the current session id; the bottom history
// block is where older sessions live.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, menu } from './components.js';
import { el, clear, mediaUrl, baseName, clockTime, seconds as secs } from './util.js';
import { thumbNode, loadIntoForm, openJob } from './history.js';

export class Sidebar {
  constructor(root) {
    this.root = root;
    this.render();
    state.on(TOPICS.jobs, () => this.renderList());
    state.on(TOPICS.state, () => this.render());
  }

  render() {
    clear(this.root);
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

  renderList() {
    if (!this.list) return;
    clear(this.list);
    const active = state.jobList({ limit: 500 }).filter(
      (job) => job.status === 'running' || job.status === 'queued');
    const done = state.sessionJobs().filter(
      (job) => job.status !== 'running' && job.status !== 'queued');

    if (active.length) {
      this.list.appendChild(sectionLabel('Running'));
      for (const job of active) this.list.appendChild(this.item(job, true));
    }
    this.list.appendChild(sectionLabel(`Completed (${done.length})`));
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
    for (const job of done) this.list.appendChild(this.item(job, false));
  }

  item(job, live) {
    const progress = job.steps ? (job.step || 0) / job.steps : (job.progress || 0);
    const node = el('div', {
      class: `session-item ${state.ui.selectedJob === job.id ? 'active' : ''}`,
      dataset: { job: job.id },
      onclick: () => state.setUI({ selectedJob: job.id }),
      ondblclick: () => openJob(job),
    }, [
      el('div', { class: 'session-thumb' }, [thumbNode(job, 18)]),
      el('div', { class: 'session-main' }, [
        el('div', { class: 'session-title', text: job.title || job.prompt || job.id }),
        el('div', { class: 'session-meta' }, [
          el('span', { class: `badge ${job.kind}` , text: job.kind }),
          el('span', { text: clockTime(job.created_at) }),
          job.elapsed ? el('span', { text: secs(job.elapsed) }) : null,
        ]),
        live ? el('div', {
          class: 'bar',
          style: { marginTop: '5px' },
        }, [el('i', { style: { width: `${Math.round(progress * 100)}%`, display: 'block', height: '2px', background: 'var(--accent)' } })]) : null,
      ]),
    ]);
    if (job.status === 'error') {
      node.querySelector('.session-meta').appendChild(
        el('span', { class: 'badge error', text: 'error' }),
      );
    }
    node.appendChild(el('button', {
      class: 'icon-btn', style: { width: '22px', height: '22px' },
      html: icon('grip', 12), title: 'Actions',
      onclick: (event) => { event.stopPropagation(); this.itemMenu(event.currentTarget, job); },
    }));
    return node;
  }

  itemMenu(anchor, job) {
    const items = [
      { label: 'Open in the viewer', icon: 'expand', run: () => openJob(job) },
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
    menu(anchor, items);
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
