// Prompt composer: the text box every generation starts from, plus the
// attachments dropped onto it and the presets saved against a tab.
//
// Attachments are uploaded into the engine input folder on drop
// (POST /api/upload) and stored on the form as {name, kind, url}, so a
// generation can pick them again from any of the media dropdowns.

import { state, TOPICS } from './state.js';
import { api } from './api.js';
import { icon } from './icons.js';
import { toast, menu, modal, lightbox, promptDialog, attachContextMenu } from './components.js';
import { el, clear, debounce, fileName, baseName, mediaKind, mediaUrl, copyText } from './util.js';

const mounted = new Map();

const PLACEHOLDERS = {
  image: 'Describe the image. Use __wildcards__ for random lists, (word:1.2) to weight a token, '
    + 'and drop a picture here to condition on it.',
  video: 'Describe the shot: subject, motion, camera, lighting. Drop an image or a video here for '
    + 'image-to-video, video-to-video or reference work.',
  music: '[Verse]\nWrite the words here. Blank lines separate lines.\n[Chorus]\n…',
};

/**
 * Which form field the main text box writes to.
 *
 * Image and video generate from a prompt. YuE takes a style description plus
 * lyrics, and the lyrics are the long text a person types, so on the music tab
 * the composer owns `lyrics` and the sidebar owns the style description.
 */
const PRIMARY_FIELD = { image: 'prompt', video: 'prompt', music: 'lyrics' };

const PRIMARY_LABEL = { image: 'Prompt', video: 'Prompt', music: 'Lyrics' };

/** Form field the main text box of `kind` writes to. */
export function fieldFor(kind) {
  return PRIMARY_FIELD[kind] || 'prompt';
}

/** Display name of the main text box of `kind`. */
export function labelFor(kind) {
  return PRIMARY_LABEL[kind] || 'Prompt';
}

/**
 * Why a generation cannot start, or null when it can.
 *
 * YuE can make an instrumental out of a style description alone, so the music
 * tab does not insist on lyrics.
 */
export function missingPrimary(kind, params) {
  const text = String((params || {})[fieldFor(kind)] || '').trim();
  if (text) return null;
  if (kind === 'music') {
    return String((params || {}).style || '').trim()
      ? null
      : 'Write a style description, or lyrics for a song with vocals';
  }
  return 'Write a prompt first';
}

/** Rough CLIP-ish token count: commas and words, ignoring weighting syntax. */
export function tokenCount(text) {
  if (!text) return 0;
  const cleaned = text.replace(/[()]/g, ' ').replace(/:\s*-?\d*\.?\d+/g, ' ');
  return cleaned.split(/[,\n]+/).reduce((total, chunk) => total
    + chunk.split(/\s+/).filter(Boolean).length, 0);
}

export function insertIntoPrompt(kind, text, { target = 'prompt' } = {}) {
  const composer = mounted.get(kind);
  if (!composer) return;
  composer.insert(text, target);
}

export class Composer {
  constructor(root, kind) {
    this.root = root;
    this.kind = kind;
    this.field = fieldFor(kind);
    this.label = labelFor(kind);
    // YuE has no negative prompt, so the music tab never shows one.
    this.hasNegative = kind !== 'music';
    this.negativeOpen = false;
    mounted.set(kind, this);
    this.render();
    this._onForm = (changed) => {
      if (changed && changed !== kind) return;
      this.syncValues();
    };
    state.on(TOPICS.form, this._onForm);
  }

  get form() { return state.form(this.kind); }

  /**
   * Push the live textarea text into the form state.
   *
   * The input handler is debounced, so a click or Enter right after typing
   * would otherwise generate with the previous prompt.
   */
  flush() {
    if (this.promptBox) state.setFormValue(this.kind, this.field, this.promptBox.value);
    if (this.negativeBox) state.setFormValue(this.kind, 'negative', this.negativeBox.value);
  }

  insert(text, target = 'prompt') {
    const box = this[target === 'negative' ? 'negativeBox' : 'promptBox'];
    if (!box) return;
    const start = box.selectionStart ?? box.value.length;
    const end = box.selectionEnd ?? start;
    const before = box.value.slice(0, start);
    const after = box.value.slice(end);
    const glue = before && !/[\s(,]$/.test(before) ? ' ' : '';
    box.value = `${before}${glue}${text}${after}`;
    box.focus();
    const caret = start + glue.length + text.length;
    box.setSelectionRange(caret, caret);
    state.setFormValue(this.kind, target === 'negative' ? 'negative' : this.field, box.value);
    this.updateMeta();
  }

  render() {
    clear(this.root);
    this.root.className = 'composer';

    this.box = el('div', { class: 'composer-box' });
    this.promptBox = el('textarea', {
      class: 'composer-input',
      placeholder: PLACEHOLDERS[this.kind] || PLACEHOLDERS.image,
      spellcheck: this.kind === 'music' ? 'true' : 'false',
      rows: this.kind === 'music' ? 8 : 5,
      dataset: { field: this.field },
      'aria-label': this.label,
      oninput: debounce((event) => {
        state.setFormValue(this.kind, this.field, event.target.value);
        this.updateMeta();
      }, 220),
    });
    this.promptBox.value = this.form[this.field]
      || state.formValue(this.kind, this.field) || '';

    this.negativeBox = null;
    if (this.hasNegative) {
      this.negativeBox = el('textarea', {
        class: 'composer-input composer-neg',
        placeholder: 'Negative prompt: what to keep out of the frame.',
        spellcheck: 'false',
        rows: 2,
        dataset: { field: 'negative' },
        oninput: debounce((event) => {
          state.setFormValue(this.kind, 'negative', event.target.value);
          this.updateMeta();
        }, 220),
      });
      this.negativeBox.value = state.formValue(this.kind, 'negative') || '';
    }

    this.attachStrip = el('div', { class: 'attach-strip' });
    this.renderAttachments();

    this.box.appendChild(this.promptBox);
    this.box.appendChild(this.attachStrip);
    this.box.appendChild(el('div', {
      class: 'drop-hint',
      text: this.kind === 'music'
        ? 'Drop an audio file to attach it'
        : 'Drop an image, video or audio file to attach it',
    }));

    this.root.appendChild(this.box);
    this.root.appendChild(this.renderFoot());

    this.wireDrop(this.box);
    attachContextMenu(this.box, () => this.contextItems());
    this.updateMeta();
  }

  /** Right-click items for the composer box. */
  contextItems() {
    const text = this.promptBox ? String(this.promptBox.value || '') : '';
    const items = [
      { label: 'Attach a file from disk', icon: 'upload', run: () => this.pickFiles() },
    ];
    if (this.kind === 'music') {
      items.push({ label: 'Insert a section marker', icon: 'wave',
        run: () => this.insert('[Chorus]') });
    } else {
      items.push({ label: 'Insert a wildcard', icon: 'shuffle', run: () => this.pickWildcard() });
    }
    items.push({ label: 'Randomise the seed', icon: 'dice', run: () => this.randomSeed() });
    if (this.hasNegative) {
      items.push({ label: 'Swap prompt and negative', icon: 'branch',
        disabled: !text.trim() && !String((this.negativeBox && this.negativeBox.value) || '').trim(),
        run: () => this.swapNegative() });
    }
    items.push({ sep: true });
    items.push({ label: `Copy the ${this.label.toLowerCase()}`, icon: 'copy', disabled: !text.trim(),
      run: () => copyText(text)
        .then(() => toast(`${this.label} copied`, { kind: 'ok', timeout: 1200 }))
        .catch(() => toast('Could not copy', { kind: 'err' })) });
    items.push({ label: 'Clear the text', icon: 'x', disabled: !text.trim(), run: () => {
      this.promptBox.value = '';
      state.setFormValue(this.kind, this.field, '');
      this.updateMeta();
      this.promptBox.focus();
    } });
    items.push({ label: `Open the large ${this.label.toLowerCase()} editor`, icon: 'expand',
      run: () => this.openBigEditor() });
    items.push({ sep: true });
    items.push({ label: 'Load a preset', icon: 'save',
      run: () => this.openPresets(this.presetTool) });
    items.push({ label: 'Reset the form to defaults', icon: 'refresh', run: () => {
      state.resetForm(this.kind);
      toast('Form reset to the default settings', { kind: 'ok', timeout: 1800 });
    } });
    return items;
  }

  /** Write a fresh seed into the form, as the dice button does. */
  randomSeed() {
    const seed = Math.floor(Math.random() * 4294967295);
    state.setForm(this.kind, { seed, random_seed: true });
    toast(`Seed ${seed}`, { kind: 'info', timeout: 1400 });
  }

  /** Exchange the prompt and the negative prompt. */
  swapNegative() {
    if (!this.negativeBox) return;
    const prompt = this.promptBox.value;
    const negative = this.negativeBox.value;
    this.promptBox.value = negative;
    this.negativeBox.value = prompt;
    state.setForm(this.kind, { [this.field]: negative, negative: prompt });
    this.updateMeta();
    toast('Prompt and negative prompt swapped', { kind: 'ok', timeout: 1600 });
  }

  /** Open the browser file picker and attach whatever comes back. */
  pickFiles() {
    const input = el('input', {
      type: 'file', multiple: true, style: { display: 'none' },
      accept: this.kind === 'music' ? 'audio/*' : 'image/*,video/*,audio/*',
    });
    input.addEventListener('change', async () => {
      await this.attachFiles(Array.from(input.files || []));
      input.remove();
    });
    document.body.appendChild(input);
    input.click();
  }

  renderFoot() {
    const tool = (name, title, run, extraClass = '') => el('button', {
      class: `icon-btn ${extraClass}`,
      html: icon(name, 14), title, 'aria-label': title, onclick: run,
    });

    const meta = el('div', { class: 'composer-meta' });
    this.metaNode = meta;

    const tools = [
      tool(this.kind === 'music' ? 'wave' : 'shuffle', 'Insert a wildcard', () => this.pickWildcard()),
      tool('dice', 'Randomise the seed', () => {
        const seed = Math.floor(Math.random() * 4294967295);
        state.setForm(this.kind, { seed, random_seed: true });
        toast(`Seed ${seed}`, { kind: 'info', timeout: 1400 });
      }),
    ];
    if (this.hasNegative) {
      tools.push(tool(this.negativeOpen ? 'chevron_up' : 'chevron_down', 'Negative prompt', () => {
        this.negativeOpen = !this.negativeOpen;
        this.render();
      }));
    }
    tools.push(tool('expand', `Open the large ${this.label.toLowerCase()} editor`,
      () => this.openBigEditor()));
    this.presetTool = tool('branch', 'Presets', (event) => this.openPresets(event.currentTarget));
    tools.push(this.presetTool);
    tools.push(tool('copy', `Copy the ${this.label.toLowerCase()}`, () => {
      copyText(this.promptBox.value).then(() => toast(`${this.label} copied`,
        { kind: 'ok', timeout: 1200 }));
    }));

    const row = el('div', { class: 'composer-foot' }, [
      meta,
      el('div', { class: 'grow' }),
      ...tools,
    ]);
    if (this.negativeOpen && this.negativeBox) this.promptBox.after(this.negativeBox);
    return row;
  }

  updateMeta() {
    if (!this.metaNode) return;
    const text = this.promptBox.value;
    const tokens = tokenCount(text);
    const negative = this.negativeOpen && this.negativeBox
      ? tokenCount(this.negativeBox.value) : 0;
    const attachments = (this.form.attachments || []).length;
    clear(this.metaNode);
    if (this.kind === 'music') {
      // YuE reads the lyric lines as the arrangement, so lines are the useful count.
      const lines = text.split('\n').filter((line) => line.trim()).length;
      this.metaNode.appendChild(el('span', { text: `${lines} line${lines === 1 ? '' : 's'}` }));
      this.metaNode.appendChild(el('span', { text: `· ${tokens} tok` }));
    } else {
      this.metaNode.appendChild(el('span', { text: `${tokens} tok` }));
    }
    if (negative) this.metaNode.appendChild(el('span', { text: `· ${negative} neg` }));
    if (attachments) this.metaNode.appendChild(el('span', { text: `· ${attachments} attached` }));
    const wildcards = (state.wildcards || []).length;
    if (wildcards) this.metaNode.appendChild(el('span', { text: `· ${wildcards} wildcards` }));
  }

  syncValues() {
    const value = state.formValue(this.kind, this.field) || '';
    const negative = this.negativeBox ? state.formValue(this.kind, 'negative') || '' : '';
    if (this.promptBox && document.activeElement !== this.promptBox && this.promptBox.value !== value) {
      this.promptBox.value = value;
    }
    if (this.negativeBox && document.activeElement !== this.negativeBox
      && this.negativeBox.value !== negative) {
      this.negativeBox.value = negative;
    }
    this.renderAttachments();
    this.updateMeta();
  }

  // Attachments ------------------------------------------------------------
  renderAttachments() {
    if (!this.attachStrip) return;
    clear(this.attachStrip);
    const items = this.form.attachments || [];
    for (const [index, item] of items.entries()) {
      const url = item.url || '';
      const kind = item.kind || mediaKind(item.name || '');
      const thumb = kind === 'video'
        ? el('video', { src: url, muted: true, playsinline: true })
        : (kind === 'image' ? el('img', { src: url, alt: '' })
          : el('span', { class: 'kind', html: icon(kind === 'audio' ? 'mic' : 'file', 12) }));
      this.attachStrip.appendChild(el('div', {
        class: 'attach',
        title: `${item.name} — click to use as the source ${kind}`,
        onclick: (event) => this.attachmentMenu(event, item, index),
      }, [thumb, el('span', {
        class: 'x', html: icon('x', 10),
        onclick: (event) => {
          event.stopPropagation();
          this.setAttachments(items.filter((_, i) => i !== index));
        },
      })]));
    }
    this.updateMeta();
  }

  setAttachments(items) {
    state.setFormValue(this.kind, 'attachments', items);
    this.renderAttachments();
  }

  attachmentMenu(anchor, item, index) {
    const kind = item.kind || mediaKind(item.name || '');
    const items = [];
    if (kind === 'image') {
      items.push({ label: 'Use as initial image', icon: 'image', run: () => state.setForm(this.kind, { init_image: item.name }) });
      items.push({ label: 'Use as first frame', icon: 'frame', run: () => state.setForm(this.kind, { source_image: item.name }) });
      items.push({ label: 'Use as last frame', icon: 'frame', run: () => state.setForm(this.kind, { last_frame: item.name }) });
      items.push({ label: 'Use as mask', icon: 'mask', run: () => state.setForm(this.kind, { mask_image: item.name }) });
    }
    if (kind === 'video') {
      items.push({ label: 'Use as source video', icon: 'video', run: () => state.setForm(this.kind, { source_video: item.name }) });
      items.push({ label: 'Use as control video', icon: 'branch', run: () => state.setForm(this.kind, { control_video: item.name }) });
    }
    if (kind === 'audio') {
      items.push({ label: 'Use as driving audio', icon: 'mic', run: () => state.setForm(this.kind, { driving_audio: item.name }) });
    }
    items.push({ sep: true });
    items.push({ label: 'Open preview', icon: 'external', run: () => this.preview(item) });
    items.push({ label: 'Remove', icon: 'trash', danger: true, run: () => {
      this.setAttachments((this.form.attachments || []).filter((_, i) => i !== index));
    } });
    menu(anchor, items);
  }

  preview(item) {
    lightbox({
      title: baseName(item.name), url: item.url,
      kind: item.kind || mediaKind(item.name || ''),
    });
  }

  wireDrop(node) {
    const hasFiles = (event) => Array.from(event.dataTransfer ? event.dataTransfer.types : []).includes('Files');
    node.addEventListener('dragenter', (event) => { if (hasFiles(event)) { event.preventDefault(); this.box.classList.add('drag'); } });
    node.addEventListener('dragover', (event) => { if (hasFiles(event)) { event.preventDefault(); this.box.classList.add('drag'); } });
    node.addEventListener('dragleave', (event) => {
      if (!node.contains(event.relatedTarget)) this.box.classList.remove('drag');
    });
    node.addEventListener('drop', async (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      this.box.classList.remove('drag');
      await this.attachFiles(Array.from(event.dataTransfer.files));
    });
  }

  /** Upload dropped or picked files and attach each one to this tab. */
  async attachFiles(files) {
    for (const file of files) {
      try {
        const data = await api.upload(file, { category: 'input' });
        const attachment = {
          name: data.file.name,
          kind: data.file.kind || mediaKind(file.name),
          url: data.file.url || '',
          size: file.size,
        };
        const items = (this.form.attachments || []).slice();
        items.push(attachment);
        this.setAttachments(items);
        await this.refreshMedia();
        this.autoWire(attachment);
        toast(`Attached ${file.name}`, { kind: 'ok', timeout: 2200 });
      } catch (err) {
        toast(`${file.name}: ${err.message}`, { kind: 'err', timeout: 6000 });
      }
    }
  }

  /** Put a freshly dropped file into the field the current tab cares about. */
  autoWire(attachment) {
    const kind = attachment.kind;
    const patch = {};
    if (this.kind === 'image') {
      if (kind === 'image' && !state.formValue('image', 'init_image')) patch.init_image = attachment.name;
      if (kind === 'image' && /mask/i.test(attachment.name)) patch.mask_image = attachment.name;
    } else if (this.kind === 'video') {
      const mode = state.formValue('video', 'mode');
      if (kind === 'image') {
        if (mode === 'flf2v' && !state.formValue('video', 'last_frame')) patch.last_frame = attachment.name;
        else if (!state.formValue('video', 'source_image')) patch.source_image = attachment.name;
      }
      if (kind === 'video' && !state.formValue('video', 'source_video')) patch.source_video = attachment.name;
      if (kind === 'audio' && !state.formValue('video', 'driving_audio')) patch.driving_audio = attachment.name;
    }
    if (Object.keys(patch).length) state.setForm(this.kind, patch);
  }

  async refreshMedia() {
    try {
      const data = await api.models();
      state.setModels(data);
    } catch (err) { /* the next snapshot picks it up */ }
  }

  // Wildcards --------------------------------------------------------------
  async pickWildcard() {
    let entries = state.wildcards || [];
    if (!entries.length) {
      try {
        const data = await api.wildcards();
        state.setWildcards(data.entries || [], data.folders || []);
        entries = data.entries || [];
      } catch (err) { /* ignore */ }
    }
    if (!entries.length) {
      toast('No wildcard lists yet. Add them in the Assets tab.', { kind: 'warn' });
      return;
    }
    const anchor = this.root.querySelector('.icon-btn');
    menu(anchor, entries.slice(0, 60).map((entry) => ({
      label: entry.name || entry,
      hint: entry.count ? `${entry.count} values` : '',
      run: () => this.insert(`__${entry.name || entry}__`),
    })), { align: 'left' });
  }

  // Big editor -------------------------------------------------------------
  openBigEditor() {
    const body = el('div', { class: 'col', style: { gap: '10px' } });
    const prompt = el('textarea', { class: 'textarea', rows: 12, spellcheck: 'false' });
    prompt.value = this.promptBox.value;
    body.appendChild(el('div', { class: 'sub-label', text: this.label }));
    body.appendChild(prompt);
    let negative = null;
    if (this.hasNegative) {
      negative = el('textarea', { class: 'textarea', rows: 5, spellcheck: 'false' });
      negative.value = this.negativeBox ? this.negativeBox.value : '';
      body.appendChild(el('div', { class: 'sub-label', text: 'Negative prompt' }));
      body.appendChild(negative);
    }
    modal({
      title: `${this.label} editor`,
      icon: this.kind === 'music' ? 'music' : 'wand',
      width: 760,
      body,
      actions: [{
        label: 'Apply', kind: 'primary', run: (close) => {
          const patch = { [this.field]: prompt.value };
          if (negative) patch.negative = negative.value;
          state.setForm(this.kind, patch);
          this.render();
          close();
        },
      }],
    });
  }

  // Presets ----------------------------------------------------------------
  async openPresets(anchor, at = null) {
    let presets = state.presets || [];
    try {
      const data = await api.presets(this.kind);
      presets = data.presets || [];
      state.presets = presets;
      state.emit(TOPICS.presets);
    } catch (err) { /* keep whatever we have */ }
    const items = presets.filter((preset) => !preset.kind || preset.kind === this.kind)
      .map((preset) => ({
        label: preset.name,
        hint: preset.label || '',
        run: () => this.applyPreset(preset),
      }));
    items.push({ sep: true });
    items.push({
      label: 'Save the current form as a preset', icon: 'save',
      run: () => this.savePreset(),
    });
    items.push({
      label: 'Reset the form to defaults', icon: 'refresh',
      run: () => {
        state.resetForm(this.kind);
        toast('Form reset to the default settings', { kind: 'ok', timeout: 1800 });
      },
    });
    if (!items.length) items.push({ label: 'No presets for this tab yet', disabled: true });
    menu(anchor, items, at ? { at } : {});
  }

  async savePreset() {
    const name = await promptDialog({
      title: 'Save preset', label: 'Preset name', placeholder: 'Cinematic look',
    });
    if (!name) return;
    try {
      await api.savePreset(name, this.kind, state.form(this.kind), '');
      toast(`Preset "${name}" saved`, { kind: 'ok' });
      const data = await api.presets();
      state.presets = data.presets || [];
      state.emit(TOPICS.presets);
    } catch (err) {
      toast(err.message, { kind: 'err' });
    }
  }

  applyPreset(preset) {
    state.loadForm(this.kind, preset.params || {});
    this.render();
    toast(`Preset "${preset.name}" loaded`, { kind: 'ok', timeout: 1800 });
  }
}
