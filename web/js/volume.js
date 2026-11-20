// App-wide playback volume.
//
// One volume serves every media element the app plays. The slider and the mute
// button write `state.ui.volume` and `state.ui.muted`, and those values are pushed
// onto audio and video elements as they appear, so a track started in the history
// block obeys the same level as one started from the music stage.
//
// Elements the app mutes on purpose (thumbnails, hover loops, silent video
// previews) are remembered as "own mute" and never touched again, otherwise
// unmuting the app would turn a silent thumbnail into sound.

import { state, TOPICS } from './state.js';
import { icon } from './icons.js';
import { attachContextMenu } from './components.js';
import { el, clamp } from './util.js';

/** Media the app muted itself: leave it alone forever. */
const ownMute = new WeakSet();
/** Media this module muted because the app is muted, so it can be unmuted again. */
const appMute = new WeakSet();
/** Tracks playing in the background are turned down; this is their factor. */
const duckFactor = new WeakMap();

let installed = false;
let control = null;

/**
 * Push the app volume onto one media element.
 *
 * @param {HTMLMediaElement} media
 * @param {{force?: boolean}} [options] `force` applies the level even to media the
 *   app muted itself, which is what a caller that adopts an element wants.
 */
export function applyVolume(media, options = {}) {
  if (!media || !(media instanceof HTMLMediaElement)) return media;
  if (!options.force && !media.dataset.autumnAdopted) {
    if (ownMute.has(media)) return media;
    if (media.muted && !appMute.has(media)) {
      ownMute.add(media);
      return media;
    }
  }
  const muted = !!state.ui.muted;
  const level = clamp(Number(state.ui.volume), 0, 1)
    * clamp(duckFactor.has(media) ? duckFactor.get(media) : 1, 0, 1);
  if (media.volume !== level) media.volume = level;
  if (muted) {
    if (!media.muted) media.muted = true;
    appMute.add(media);
  } else if (appMute.has(media)) {
    media.muted = false;
    appMute.delete(media);
  }
  return media;
}

/** Mark an element as owned by the app so the global level always applies. */
export function adoptVolume(media) {
  if (media && media instanceof HTMLMediaElement) {
    media.dataset.autumnAdopted = '1';
    ownMute.delete(media);
  }
  return applyVolume(media, { force: true });
}

/**
 * Turn one media element down or back up, relative to the app volume.
 *
 * The music player keeps several tracks open at once, so a track that is playing
 * while another one is on screen is ducked instead of stopped.
 */
export function setDuck(media, factor) {
  if (!media || !(media instanceof HTMLMediaElement)) return media;
  const value = clamp(Number(factor) || 1, 0, 1);
  if (value >= 1) duckFactor.delete(media);
  else duckFactor.set(media, value);
  if (media.dataset.autumnAdopted) return applyVolume(media, { force: true });
  return media;
}

/** Apply the level to every media element under `root`. */
export function applyAllVolume(root = document) {
  for (const media of root.querySelectorAll('audio, video')) applyVolume(media);
}

/**
 * The top-bar control: a mute button and a slider.
 *
 * The node keeps itself in step with `state.ui`, so callers build it once.
 */
export function mountVolumeControl() {
  if (control) return control;
  const slider = el('input', {
    class: 'vol-range', type: 'range', min: 0, max: 100, value: String(Math.round(pct() * 100)),
    title: 'Playback volume', 'aria-label': 'Playback volume',
    oninput: (event) => {
      const value = clamp(Number(event.target.value) / 100, 0, 1);
      state.setUI({ volume: value, muted: value === 0 });
    },
  });
  const button = el('button', {
    class: 'icon-btn vol-btn', 'aria-label': 'Mute / unmute',
    onclick: () => state.setUI({ muted: !state.ui.muted }),
  });
  control = el('div', { class: 'volume' }, [button, slider]);
  control.slider = slider;
  control.button = button;
  attachContextMenu(control, () => volumeMenu());
  paintControl();
  state.on(TOPICS.ui, (patch) => {
    if (patch && ('volume' in patch || 'muted' in patch)) paintControl();
  });
  return control;
}

function pct() {
  return state.ui.muted ? 0 : clamp(Number(state.ui.volume) || 0, 0, 1);
}

/** The volume the app ships with, which is what the reset item writes back. */
const DEFAULT_VOLUME = 0.8;

/** Right-click items for the volume control. */
function volumeMenu() {
  const level = clamp(Number(state.ui.volume) || 0, 0, 1);
  const items = [
    { label: state.ui.muted ? 'Unmute' : 'Mute', icon: state.ui.muted ? 'volume_up' : 'volume_off',
      run: () => state.setUI({ muted: !state.ui.muted }) },
    { sep: true },
    { group: true, label: 'Level' },
  ];
  for (const value of [1, 0.75, 0.5, 0.25]) {
    items.push({
      label: `${Math.round(value * 100)}%`,
      icon: value >= 0.75 ? 'volume_up' : 'volume_down',
      hint: Math.abs(level - value) < 0.005 && !state.ui.muted ? 'current' : '',
      run: () => state.setUI({ volume: value, muted: false }),
    });
  }
  items.push({ sep: true });
  items.push({ label: `Reset to ${Math.round(DEFAULT_VOLUME * 100)}%`, icon: 'refresh',
    disabled: Math.abs(level - DEFAULT_VOLUME) < 0.005 && !state.ui.muted,
    run: () => state.setUI({ volume: DEFAULT_VOLUME, muted: false }) });
  return items;
}

function paintControl() {
  if (!control) return;
  const value = pct();
  const name = state.ui.muted || value <= 0.001
    ? 'volume_off' : (value < 0.5 ? 'volume_down' : 'volume_up');
  control.button.innerHTML = icon(name, 15);
  control.button.title = state.ui.muted ? 'Unmute' : `Volume ${Math.round(value * 100)}%`;
  control.slider.value = String(Math.round(value * 100));
  control.slider.style.setProperty('--pct', `${Math.round(value * 100)}%`);
  control.classList.toggle('muted', !!state.ui.muted);
}

/**
 * Start watching the document for media elements and follow the stored level.
 *
 * `play` and `loadstart` are caught in the capture phase, so an element is turned
 * to the app volume before it makes a sound.
 */
export function initVolume() {
  if (installed) return;
  installed = true;
  const sink = (event) => applyVolume(event.target);
  for (const type of ['loadstart', 'play', 'loadedmetadata']) {
    document.addEventListener(type, sink, true);
  }
  // Elements added without ever playing (a poster-only <video>) still get the level.
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        if (node.matches && node.matches('audio, video')) applyVolume(node);
        else applyAllVolume(node);
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
  applyAllVolume();
  state.on(TOPICS.ui, (patch) => {
    if (patch && ('volume' in patch || 'muted' in patch)) applyAllVolume();
  });
}

export default { applyVolume, adoptVolume, setDuck, applyAllVolume, mountVolumeControl, initVolume };
