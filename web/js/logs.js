// Console drawer: the program log plus the ComfyUI and music engine output.
//
// Lines arrive over the WebSocket (`logs`, `engine_log`, `music_log`) and are
// kept in `state.logLines`, so reopening the drawer or switching filters is a
// local operation. New lines are appended to the DOM one at a time instead of
// re-rendering, which keeps a chatty engine from costing frames.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast } from './components.js';
import { el, clear, clockTime, copyText } from './util.js';

const SOURCES = [
  { id: 'all', label: 'All', match: () => true },
  { id: 'app', label: 'Program', match: (e) => !e.logger || (!e.logger.startsWith('comfyui') && !e.logger.startsWith('music')) },
  { id: 'comfyui', label: 'ComfyUI', match: (e) => String(e.logger || '').startsWith('comfyui') },
  { id: 'music', label: 'Music', match: (e) => String(e.logger || '').startsWith('music') },
];

const LEVELS = [
  { id: 'all', label: 'All', floor: 0 },
  { id: 'info', label: 'Info+', floor: 20 },
  { id: 'warning', label: 'Warn+', floor: 30 },
  { id: 'error', label: 'Error+', floor: 40 },
];

const LEVEL_RANK = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };

/** Turn an engine stderr line into a log entry, guessing its severity. */
export function engineEntry(line, source = 'comfyui') {
  const text = String(line || '');
  let level = 'INFO';
  if (/\b(error|exception|traceback|failed)\b/i.test(text)) level = 'ERROR';
  else if (/\b(warning|warn|deprecated)\b/i.test(text)) level = 'WARNING';
  else if (/^\s*(debug|\[D\])/.test(text)) level = 'DEBUG';
  return { t: Date.now() / 1000, level, logger: source, msg: text.replace(/\s+$/, '') };
}

export class LogDrawer {
  constructor(root) {
    this.root = root;
    this.source = 'all';
    this.level = 'all';
    this.query = '';
    this.autoScroll = true;
    this.nodes = new Map();
    this.render();

    state.on(TOPICS.logs, (entry) => this.onEntry(entry));
    state.on(TOPICS.ui, (patch) => {
      if (patch && 'logsOpen' in patch) this.syncOpen();
    });
  }

  // --------------------------------------------------------------- chrome
  render() {
    clear(this.root);
    this.root.className = 'log-drawer';

    this.root.appendChild(el('div', { class: 'log-head' }, [
      el('span', { class: 'log-title', html: icon('terminal', 14) }),
      el('span', { class: 'log-name', text: 'Console' }),
      this.sourceTabs(),
      this.levelNode = this.levelSelect(),
      el('span', { class: 'grow' }),
      el('input', {
        class: 'input sm log-search',
        placeholder: 'Filter lines…',
        spellcheck: 'false',
        oninput: (event) => {
          this.query = event.target.value.trim().toLowerCase();
          this.renderLines();
        },
      }),
      el('label', { class: 'check tiny' }, [
        el('input', {
          type: 'checkbox', checked: this.autoScroll,
          onchange: (event) => { this.autoScroll = event.target.checked; if (this.autoScroll) this.scrollToEnd(); },
        }),
        el('span', { class: 'box', html: icon('check', 11) }),
        el('span', { class: 'txt', text: 'Follow' }),
      ]),
      el('button', {
        class: 'icon-btn', title: 'Copy visible lines',
        html: icon('copy', 14), onclick: () => this.copy(),
      }),
      el('button', {
        class: 'icon-btn', title: 'Clear the view',
        html: icon('trash', 14), onclick: () => this.clearView(),
      }),
      el('button', {
        class: 'icon-btn', title: 'Close the console',
        html: icon('x', 15), onclick: () => state.setUI({ logsOpen: false }),
      }),
    ]));

    this.body = el('div', { class: 'logs' });
    this.lines = el('div', { class: 'log-lines' });
    this.body.appendChild(this.lines);
    this.root.appendChild(this.body);

    this.footer = el('div', { class: 'log-foot' }, [
      el('span', { class: 'queue-note', text: '' }),
      el('span', { class: 'grow' }),
      el('span', { class: 'queue-note', text: 'logs/autumn.log on disk' }),
    ]);
    this.root.appendChild(this.footer);

    this.syncOpen();
  }

  sourceTabs() {
    const node = el('div', { class: 'segmented log-tabs' });
    for (const source of SOURCES) {
      node.appendChild(el('button', {
        class: this.source === source.id ? 'on' : '',
        text: source.label,
        onclick: () => {
          this.source = source.id;
          [...node.children].forEach((child, index) => child.classList.toggle('on', SOURCES[index].id === source.id));
          this.renderLines();
        },
      }));
    }
    return node;
  }

  levelSelect() {
    return el('div', { class: 'segmented log-tabs' }, LEVELS.map((level) => el('button', {
      class: this.level === level.id ? 'on' : '',
      text: level.label,
      onclick: () => {
        this.level = level.id;
        [...(this.levelNode ? this.levelNode.children : [])].forEach((child, index) => {
          child.classList.toggle('on', LEVELS[index].id === level.id);
        });
        this.renderLines();
      },
    })));
  }

  // ------------------------------------------------------------- visibility
  syncOpen() {
    const open = !!state.ui.logsOpen;
    this.root.classList.toggle('open', open);
    this.root.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open) {
      this.loadHistory();
      if (this.autoScroll) requestAnimationFrame(() => this.scrollToEnd());
    }
  }

  // ---------------------------------------------------------------- loading
  loaded = false;

  async loadHistory() {
    if (this.loaded) return;
    this.loaded = true;
    try {
      const data = await api.logs(600);
      const lines = data.lines || [];
      if (!state.logLines.length && lines.length) {
        for (const line of lines) state.logLines.push(line);
        this.renderLines();
      }
    } catch (err) {
      this.loaded = false;
    }
  }

  // ----------------------------------------------------------------- render
  matches(entry) {
    const source = SOURCES.find((item) => item.id === this.source) || SOURCES[0];
    if (!source.match(entry)) return false;
    const floor = (LEVELS.find((item) => item.id === this.level) || LEVELS[0]).floor;
    if (floor && (LEVEL_RANK[String(entry.level || 'INFO').toUpperCase()] || 20) < floor) return false;
    if (this.query) {
      const hay = `${entry.logger || ''} ${entry.msg || ''}`.toLowerCase();
      if (!hay.includes(this.query)) return false;
    }
    return true;
  }

  lineNode(entry) {
    const level = String(entry.level || 'INFO').toUpperCase();
    return el('div', { class: `log-line ${level.toLowerCase()}` }, [
      el('span', { class: 't', text: clockTime(entry.t) }),
      el('span', { class: 'n', text: shortLogger(entry.logger) }),
      el('span', { class: 'm', text: String(entry.msg || '') }),
    ]);
  }

  onEntry(entry) {
    if (Array.isArray(entry)) {
      for (const item of entry) this.appendOne(item);
      return;
    }
    this.appendOne(entry);
  }

  appendOne(entry) {
    if (!entry || !this.lines) return;
    if (!this.matches(entry)) return;
    this.lines.appendChild(this.lineNode(entry));
    while (this.lines.childElementCount > 1200) this.lines.removeChild(this.lines.firstChild);
    if (this.autoScroll) this.scrollToEnd();
    this.updateCount();
  }

  renderLines() {
    if (!this.lines) return;
    clear(this.lines);
    const fragment = document.createDocumentFragment();
    for (const entry of state.logLines) {
      if (this.matches(entry)) fragment.appendChild(this.lineNode(entry));
    }
    this.lines.appendChild(fragment);
    this.updateCount();
    if (this.autoScroll) requestAnimationFrame(() => this.scrollToEnd());
  }

  scrollToEnd() {
    if (!this.lines) return;
    this.lines.scrollTop = this.lines.scrollHeight;
  }

  updateCount() {
    const foot = this.root.querySelector('.log-foot .queue-note');
    if (foot) foot.textContent = `${this.lines.childElementCount} shown / ${state.logLines.length} buffered`;
  }

  visibleText() {
    return [...this.lines.children].map((node) => node.textContent.trim()).join('\n');
  }

  async copy() {
    const text = this.visibleText();
    if (!text) {
      toast('Nothing to copy', { kind: 'warn' });
      return;
    }
    await copyText(text);
    toast(`${this.lines.childElementCount} lines copied`, { kind: 'ok' });
  }

  clearView() {
    state.logLines.length = 0;
    clear(this.lines);
    this.updateCount();
  }
}

function shortLogger(logger) {
  const name = String(logger || '').replace(/^autumn\./, '');
  return name.length > 14 ? `${name.slice(0, 13)}…` : name;
}

/** A throttled sink that turns raw engine output into log entries. */
export function makeEngineSink(source) {
  return (line) => state.pushLog(engineEntry(line, source));
}

export default LogDrawer;
