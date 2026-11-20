// Toasts, modals, context menus, tooltips and confirm dialogs.

import { el, clear, escapeHtml } from './util.js';
import { icon } from './icons.js';

// Toasts -------------------------------------------------------------------
const toastRoot = () => {
  let root = document.querySelector('.toasts');
  if (!root) {
    root = el('div', { class: 'toasts' });
    document.body.appendChild(root);
  }
  return root;
};

const TOAST_ICON = { ok: 'check', err: 'alert', warn: 'warning', info: 'info', run: 'bolt' };

export function toast(message, { kind = 'info', title = '', timeout = 4200, action = null } = {}) {
  const node = el('div', { class: `toast ${kind}` }, [
    el('span', { class: 'ic', html: icon(TOAST_ICON[kind] || 'info', 15) }),
    el('div', { class: 'grow' }, [
      title ? el('b', { text: title }) : null,
      el('span', { class: 'msg', text: String(message) }),
    ]),
    action ? el('button', {
      class: 'btn sm ghost', text: action.label,
      onclick: () => { action.run(); dismiss(); },
    }) : null,
    el('button', { class: 'icon-btn', style: { width: '22px', height: '22px' }, html: icon('x', 13), onclick: () => dismiss() }),
  ]);
  toastRoot().appendChild(node);
  let done = false;
  function dismiss() {
    if (done) return;
    done = true;
    node.classList.add('out');
    setTimeout(() => node.remove(), 200);
  }
  if (timeout > 0) setTimeout(dismiss, timeout);
  return { dismiss, node };
}

// Modal --------------------------------------------------------------------
/**
 * Open a modal. `render(body, close)` fills the body; buttons come from
 * `actions: [{label, kind, run(close)}]`. Returns {close, body, root}.
 */
export function modal({ title = '', icon: iconName = '', width = 620, body = null,
  actions = [], onClose = null, dismissable = true, className = '' } = {}) {
  let root = document.querySelector('.modal-root');
  if (!root) {
    root = el('div', { class: 'modal-root' });
    document.body.appendChild(root);
  }
  clear(root);
  const close = (result) => {
    root.classList.remove('open');
    clear(root);
    document.removeEventListener('keydown', onKey);
    if (onClose) onClose(result);
  };
  const onKey = (event) => {
    if (event.key === 'Escape' && dismissable) { event.preventDefault(); close(null); }
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      const primary = actions.find((action) => action.primary);
      if (primary) { event.preventDefault(); primary.run(close); }
    }
  };

  const bodyNode = el('div', { class: 'modal-body' });
  const dialog = el('div', {
    class: `modal ${className}`,
    style: { '--modal-w': `${width}px` },
    role: 'dialog', 'aria-modal': 'true',
  }, [
    title === null ? null : el('div', { class: 'modal-head' }, [
      iconName ? el('span', { class: 'acc-t', html: icon(iconName, 16) }) : null,
      el('div', { class: 'modal-title', text: title }),
      el('button', { class: 'icon-btn', html: icon('x', 15), onclick: () => close(null), 'aria-label': 'Close' }),
    ]),
    bodyNode,
    actions.length ? el('div', { class: 'modal-foot' }, actions.map((action) => el('button', {
      class: `btn ${action.kind || 'ghost'}`,
      text: action.label,
      onclick: () => action.run(close),
    }))) : null,
  ]);

  root.appendChild(el('div', {
    class: 'modal-scrim',
    onclick: () => { if (dismissable) close(null); },
  }));
  root.appendChild(dialog);
  root.classList.add('open');
  document.addEventListener('keydown', onKey);

  if (typeof body === 'function') body(bodyNode, close);
  else if (body) bodyNode.appendChild(body);

  const first = dialog.querySelector('input, textarea, select, button.primary');
  if (first) setTimeout(() => first.focus(), 40);
  return { close, body: bodyNode, root: dialog };
}

export function confirmDialog({ title = 'Are you sure?', message = '', confirmLabel = 'Confirm',
  kind = 'danger' } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => { if (!settled) { settled = true; resolve(value); } };
    modal({
      title,
      icon: 'warning',
      width: 460,
      body: (node) => node.appendChild(el('div', { class: 'dim', text: message })),
      onClose: () => finish(false),
      actions: [
        { label: 'Cancel', kind: 'ghost', run: (close) => { finish(false); close(false); } },
        { label: confirmLabel, kind, primary: true, run: (close) => { finish(true); close(true); } },
      ],
    });
  });
}

export function promptDialog({ title = 'Input', label = '', value = '', placeholder = '',
  confirmLabel = 'Save', multiline = false } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => { if (!settled) { settled = true; resolve(result); } };
    let field = null;
    modal({
      title,
      width: 460,
      body: (node) => {
        field = multiline
          ? el('textarea', { class: 'textarea', placeholder, rows: 6 })
          : el('input', { class: 'input', placeholder, value });
        if (multiline) field.value = value;
        node.appendChild(el('div', { class: 'field' }, [
          label ? el('div', { class: 'field-top' }, [el('span', { class: 'field-label', text: label })]) : null,
          field,
        ]));
      },
      onClose: () => finish(null),
      actions: [
        { label: 'Cancel', kind: 'ghost', run: (close) => { finish(null); close(null); } },
        { label: confirmLabel, kind: 'primary', primary: true,
          run: (close) => { const text = field.value.trim(); finish(text); close(text); } },
      ],
    });
  });
}

// Context menu -------------------------------------------------------------
/**
 * Open a menu, either anchored to an element or at a point on the screen.
 *
 * items: [{label, icon, run, hint, danger, disabled} | {sep:true} | {label, group:true}]
 * Options: `align` ('left' | 'right'), `offset`, and `at: {x, y}` to place the
 * menu under the pointer, which is what a right-click menu wants.
 */
export function menu(anchor, items, { align = 'right', offset = 4, at = null } = {}) {
  closeMenu();
  const node = el('div', { class: 'popover menu' });
  for (const item of items) {
    if (!item) continue;
    if (item.sep) { node.appendChild(el('div', { class: 'menu-sep' })); continue; }
    if (item.group) { node.appendChild(el('div', { class: 'menu-label', text: item.label })); continue; }
    node.appendChild(el('div', {
      class: `menu-item ${item.danger ? 'danger' : ''}`,
      onclick: (event) => { event.stopPropagation(); closeMenu(); if (!item.disabled) item.run && item.run(); },
    }, [
      item.icon ? el('span', { html: icon(item.icon, 14) }) : el('span', { style: { width: '14px' } }),
      el('span', { class: 'grow', text: item.label }),
      item.hint ? el('span', { class: 'tiny faint mono', text: item.hint }) : null,
    ]));
  }
  document.body.appendChild(node);
  const menuRect = node.getBoundingClientRect();
  let left;
  let top;
  if (at) {
    left = Math.max(8, Math.min(at.x, window.innerWidth - menuRect.width - 8));
    top = Math.max(8, Math.min(at.y, window.innerHeight - menuRect.height - 8));
  } else {
    const rect = anchor.getBoundingClientRect();
    left = align === 'right' ? rect.right - menuRect.width : rect.left;
    left = Math.max(8, Math.min(left, window.innerWidth - menuRect.width - 8));
    top = rect.bottom + offset;
    if (top + menuRect.height > window.innerHeight - 8) {
      top = Math.max(8, rect.top - menuRect.height - offset);
    }
  }
  node.style.left = `${left}px`;
  node.style.top = `${top}px`;
  node.dataset.menu = '1';
  const dismiss = (event) => {
    if (!node.contains(event.target)) closeMenu();
  };
  setTimeout(() => {
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', onKey);
  }, 0);
  node._cleanup = () => {
    document.removeEventListener('pointerdown', dismiss);
    document.removeEventListener('keydown', onKey);
  };
  function onKey(event) { if (event.key === 'Escape') closeMenu(); }
  return node;
}

/**
 * Wire a right-click menu to a node.
 *
 * `build` returns the items for the menu. Inside a text box the browser menu
 * (cut, copy, paste, spellcheck) is more useful than an app menu, so the app menu
 * is skipped there unless Shift is held, which is the one gesture the user has to
 * remember to get at it.
 */
export function attachContextMenu(node, build) {
  if (!node || typeof build !== 'function') return node;
  node.addEventListener('contextmenu', (event) => {
    if (event.defaultPrevented) return;   // a nested handler already owns this right-click
    const editing = event.target && event.target.closest
      && event.target.closest('input, textarea, select');
    if (editing && !event.shiftKey) return;
    const items = build(event);
    if (!items || !items.length) return;
    event.preventDefault();
    menu(null, items, { at: { x: event.clientX, y: event.clientY } });
  });
  return node;
}

export function closeMenu() {
  document.querySelectorAll('.popover[data-menu]').forEach((node) => {
    if (node._cleanup) node._cleanup();
    node.remove();
  });
}

// Tooltip ------------------------------------------------------------------
//
// A tooltip carries three parts: a title (usually the setting name), the
// explanation, and an optional example. All three are optional, and a plain
// string still works, so callers can be as terse as they like.
let tipNode = null;

/**
 * Attach a hover tooltip to a node.
 *
 * `spec` is a string or `{ title, text, example }`. The native `title` attribute
 * is cleared so the browser does not draw a second box.
 */
export function attachTooltip(node, spec) {
  if (!spec) return;
  const detail = normaliseTip(spec);
  node.setAttribute('title', '');
  node.addEventListener('mouseenter', (event) => showTip(event.currentTarget, detail));
  node.addEventListener('mouseleave', hideTip);
  node.addEventListener('click', hideTip);
}

function normaliseTip(spec) {
  if (typeof spec === 'string') return { text: spec };
  return {
    title: spec.title || '',
    text: spec.text || spec.body || '',
    example: spec.example || '',
  };
}

export function showTip(anchor, spec) {
  hideTip();
  const detail = normaliseTip(spec);
  if (!detail.title && !detail.text && !detail.example) return;
  tipNode = el('div', { class: 'popover tip rich' }, [
    detail.title ? el('div', { class: 'tip-title', text: detail.title }) : null,
    detail.text ? el('div', { class: 'tip-text', text: detail.text }) : null,
    detail.example ? el('div', { class: 'tip-example' }, [
      el('span', { class: 'tip-eg-label', text: 'Example' }),
      el('code', { text: detail.example }),
    ]) : null,
  ]);
  tipNode.style.pointerEvents = 'none';
  document.body.appendChild(tipNode);
  const rect = anchor.getBoundingClientRect();
  const tipRect = tipNode.getBoundingClientRect();
  let left = rect.left + rect.width / 2 - tipRect.width / 2;
  left = Math.max(6, Math.min(left, window.innerWidth - tipRect.width - 6));
  let top = rect.top - tipRect.height - 6;
  if (top < 6) top = rect.bottom + 6;
  if (top + tipRect.height > window.innerHeight - 6) {
    top = Math.max(6, window.innerHeight - tipRect.height - 6);
  }
  tipNode.style.left = `${left}px`;
  tipNode.style.top = `${top}px`;
}

export function hideTip() {
  if (tipNode) { tipNode.remove(); tipNode = null; }
}

// Lightbox -----------------------------------------------------------------
export function lightbox({ title = '', subtitle = '', url = '', poster = '', kind = 'image',
  meta = [], actions = [] }) {
  const media = kind === 'video'
    ? el('video', { src: url, poster, controls: true, autoplay: true, loop: true, muted: true })
    : kind === 'audio'
      ? el('audio', { src: url, controls: true, style: { width: 'min(520px, 80vw)' } })
      : el('img', { src: url, alt: title });
  modal({
    title: title || 'Result',
    width: 900,
    className: 'lightbox-modal',
    body: (node) => {
      node.appendChild(el('div', { class: 'center', style: { background: 'var(--bg-0)', padding: '8px', borderRadius: '10px' } }, [media]));
      if (subtitle) node.appendChild(el('div', { class: 'tiny dim', style: { marginTop: '10px', whiteSpace: 'pre-wrap' }, text: subtitle }));
      if (meta.length) {
        const list = el('dl', { class: 'kv', style: { marginTop: '10px' } });
        for (const [key, value] of meta) {
          list.appendChild(el('dt', { text: key }));
          list.appendChild(el('dd', { text: String(value) }));
        }
        node.appendChild(list);
      }
    },
    actions: actions.map((action) => ({
      label: action.label, kind: action.kind || 'ghost', icon: action.icon,
      run: (close) => { action.run(); if (!action.keepOpen) close(null); },
    })),
  });
}

export { escapeHtml };
