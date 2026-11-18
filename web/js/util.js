// Small DOM and formatting helpers shared by every module.

export function el(tag, attrs = {}, children = []) {
  const parts = tag.split(/(?=[.#])/);
  const node = document.createElement(parts[0] || 'div');
  for (const token of parts.slice(1)) {
    if (token[0] === '.') node.classList.add(token.slice(1));
    else if (token[0] === '#') node.id = token.slice(1);
  }
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = `${node.className} ${value}`.trim();
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  addChildren(node, children);
  return node;
}

export function addChildren(node, children) {
  const list = Array.isArray(children) ? children : [children];
  for (const child of list.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function debounce(fn, wait = 220) {
  let timer = null;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
  wrapped.cancel = () => clearTimeout(timer);
  wrapped.flush = (...args) => { clearTimeout(timer); fn(...args); };
  return wrapped;
}

export function throttle(fn, wait = 120) {
  let last = 0;
  let timer = null;
  let queued = null;
  return (...args) => {
    const now = performance.now();
    const remain = wait - (now - last);
    queued = args;
    if (remain <= 0) {
      last = now;
      fn(...queued);
      queued = null;
    } else if (!timer) {
      timer = setTimeout(() => {
        last = performance.now();
        timer = null;
        if (queued) fn(...queued);
        queued = null;
      }, remain);
    }
  };
}

/** Coalesce high-frequency updates into one animation frame. */
export function raf(fn) {
  let pending = false;
  const run = () => { pending = false; fn(); };
  return () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(run);
  };
}

export function uid(prefix = '') {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return prefix ? `${prefix}-${hex}` : hex;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Formatting ---------------------------------------------------------------
export function bytes(value, digits = 1) {
  const n = Number(value) || 0;
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const exp = Math.min(units.length - 1, Math.floor(Math.log(Math.abs(n)) / Math.log(1024)));
  const scaled = n / 1024 ** exp;
  return `${scaled.toFixed(exp === 0 ? 0 : digits)} ${units[exp]}`;
}

export function num(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '0';
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function seconds(value) {
  const n = Math.max(0, Number(value) || 0);
  if (n < 1) return `${Math.round(n * 1000)} ms`;
  if (n < 60) return `${n.toFixed(1)} s`;
  const mins = Math.floor(n / 60);
  const rest = Math.round(n % 60);
  if (mins < 60) return `${mins}m ${String(rest).padStart(2, '0')}s`;
  const hours = Math.floor(mins / 60);
  return `${hours}h ${String(mins % 60).padStart(2, '0')}m`;
}

export function clockTime(value) {
  const date = new Date((Number(value) || 0) * 1000);
  return date.toLocaleTimeString('en-GB', { hour12: false });
}

export function timeAgo(value) {
  const stamp = Number(value) || 0;
  if (!stamp) return 'never';
  const diff = Math.max(0, Date.now() / 1000 - stamp);
  if (diff < 45) return 'just now';
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  if (diff < 86400 * 30) return `${Math.round(diff / 86400)} d ago`;
  return new Date(stamp * 1000).toLocaleDateString();
}

export function dateTime(value) {
  const date = new Date((Number(value) || 0) * 1000);
  return date.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
}

export function durationOf(job) {
  if (!job || !job.started_at) return '';
  const end = job.finished_at || (Date.now() / 1000);
  return seconds(end - job.started_at);
}

export function plural(count, word, suffix = 's') {
  return `${count} ${word}${count === 1 ? '' : suffix}`;
}

export function percent(value) {
  return `${Math.round((Number(value) || 0) * 100) / 100}%`;
}

// Storage ------------------------------------------------------------------
export function loadJSON(key, fallback = null) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (err) {
    return fallback;
  }
}

export function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (err) {
    return false;
  }
}

// Misc ---------------------------------------------------------------------
export function fileName(path) {
  const parts = String(path || '').split(/[\\/]/);
  return parts[parts.length - 1] || String(path || '');
}

export function baseName(path) {
  const name = fileName(path);
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(0, dot) : name;
}

export function extension(path) {
  const name = fileName(path);
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
}

export const IMAGE_EXT = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tif', 'tiff', 'avif']);
export const VIDEO_EXT = new Set(['mp4', 'webm', 'mkv', 'mov', 'avi', 'gif']);
export const AUDIO_EXT = new Set(['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'opus']);

export function mediaKind(path) {
  const ext = extension(path);
  if (IMAGE_EXT.has(ext)) return 'image';
  if (VIDEO_EXT.has(ext)) return 'video';
  if (AUDIO_EXT.has(ext)) return 'audio';
  return 'file';
}

export function fileUrl(path) {
  if (!path) return '';
  if (/^https?:/.test(path) || path.startsWith('data:')) return path;
  return `/file?path=${encodeURIComponent(path)}`;
}

export function mediaUrl(item) {
  if (!item) return '';
  if (item.url) return item.url;
  const params = new URLSearchParams({
    filename: item.name || '',
    subfolder: item.subfolder || '',
    type: item.type || 'output',
  });
  return `/media?${params}`;
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(String(text));
    return true;
  } catch (err) {
    const area = el('textarea', { value: String(text), style: { position: 'fixed', opacity: '0' } });
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  }
}

export function download(url, filename) {
  const anchor = el('a', { href: url, download: filename || '' });
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function matchScore(needle, haystack) {
  const a = String(needle || '').toLowerCase().trim();
  const b = String(haystack || '').toLowerCase();
  if (!a) return 0;
  if (b.includes(a)) return a.length / Math.max(1, b.length) + (b.startsWith(a) ? 0.5 : 0);
  let index = 0;
  for (const ch of a) {
    const found = b.indexOf(ch, index);
    if (found < 0) return -1;
    index = found + 1;
  }
  return 0.2;
}
