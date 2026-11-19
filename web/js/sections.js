// Declarative option panels for the three generation tabs.
//
// A section is { id, label, icon, hint?, when?, fields: [...] }. `fields` holds
// the specs understood by fields.js. Nothing here touches the DOM: panel.js
// renders the specs and writes values straight into the per-tab form, and the
// form is what the backend receives as `params`.
//
// The key names mirror the backend builders one-for-one
// (autumn/workflow/image.py, autumn/workflow/video.py, autumn/workflow/seedvr.py
// and autumn/config.py). Renaming a key on one side means renaming it on the
// other.

import { state } from './state.js';
import { toast, confirmDialog } from './components.js';
import { baseName, copyText } from './util.js';

// Model family table mirrors autumn/workflow/video.py.
export const FAMILIES = [
  { value: 'wan22', label: 'Wan 2.2', hint: 'T2V / I2V / VACE, 16 px canvas, 4n+1 frames' },
  { value: 'hunyuan', label: 'Hunyuan Video', hint: 'Text and image to video, 4n+1 frames' },
  { value: 'hunyuan15', label: 'Hunyuan 1.5', hint: 'Compact Hunyuan, 4n+1 frames' },
  { value: 'ltx2', label: 'LTX-2', hint: 'Fast, 32 px canvas, 8n+1 frames, optional audio' },
  { value: 'minimax_h3', label: 'MiniMax H3', hint: 'Reference and keyframe guided, 17n+5 frames' },
  { value: 'svd', label: 'Stable Video Diffusion', hint: 'Image to video only' },
  { value: 'generic', label: 'Generic checkpoint', hint: 'Any video checkpoint that loads as a model' },
];

export const MODE_LABELS = {
  t2v: 'Text to video', i2v: 'Image to video', flf2v: 'First / last frame',
  v2v: 'Video to video (denoise)', vace: 'Video to video (VACE)', inpaint: 'Video inpaint',
  control: 'Control video', animate: 'Animate (reference)', ref2v: 'Reference to video',
  guide: 'Keyframe guides', t2va: 'Text to video + audio', i2va: 'Image to video + audio',
};

const MODES = {
  wan22: ['t2v', 'i2v', 'flf2v', 'v2v', 'vace', 'inpaint', 'control', 'animate'],
  hunyuan: ['t2v', 'i2v', 'v2v'],
  hunyuan15: ['t2v', 'i2v', 'v2v'],
  ltx2: ['t2v', 'i2v', 't2va', 'i2va', 'v2v'],
  minimax_h3: ['t2v', 'i2v', 'flf2v', 'ref2v', 'guide'],
  svd: ['i2v'],
  generic: ['t2v', 'v2v'],
};

// Frame counts must land on the temporal grid of the family.
const GRIDS = {
  wan22: { step: 4, base: 1 }, hunyuan: { step: 4, base: 1 }, hunyuan15: { step: 4, base: 1 },
  ltx2: { step: 8, base: 1 }, minimax_h3: { step: 17, base: 5 }, svd: { step: 1, base: 0 },
  generic: { step: 4, base: 1 },
};

const FAMILY_FPS = {
  wan22: 24, hunyuan: 24, hunyuan15: 24, ltx2: 25, minimax_h3: 24, svd: 6, generic: 24,
};

const CANVAS_MULTIPLE = {
  wan22: 16, hunyuan: 16, hunyuan15: 16, ltx2: 32, minimax_h3: 32, svd: 8, generic: 16,
};

export function modesFor(family) {
  return (MODES[family] || MODES.generic).map((mode) => ({ value: mode, label: MODE_LABELS[mode] || mode }));
}

export function snapFrames(family, frames) {
  const grid = GRIDS[family] || GRIDS.generic;
  const value = Math.max(1, Math.round(Number(frames) || 1));
  if (grid.step <= 1) return value;
  if ((value - grid.base) % grid.step === 0) return value;
  const k = Math.max(0, Math.round((value - grid.base) / grid.step));
  return grid.base + k * grid.step;
}

export function frameHint(family) {
  const grid = GRIDS[family] || GRIDS.generic;
  if (grid.step <= 1) return 'any frame count';
  const first = [grid.base, grid.base + grid.step, grid.base + grid.step * 2].join(', ');
  return `${grid.step}n+${grid.base} (e.g. ${first})`;
}

function multipleFor(family) {
  return CANVAS_MULTIPLE[family] || 16;
}

// Option resolvers ---------------------------------------------------------
function merged(...categories) {
  const out = [];
  const seen = new Set();
  for (const category of categories) {
    for (const name of (state.models.models[category] || [])) {
      if (seen.has(name)) continue;
      seen.add(name);
      out.push(name);
    }
  }
  return out;
}

const checkpoints = () => merged('checkpoints');
const diffusionModels = () => merged('diffusion_models', 'unet');
const textEncoders = () => merged('text_encoders', 'clip');
const vaes = () => ['auto'].concat(merged('vae'));
const upscaleModels = () => merged('upscale_models');
const wildcards = () => state.wildcards.map((entry) => entry.name || entry);

// SeedVR2 parameters only make sense once the node pack is in and switched on.
const svOn = (ctx) => state.seedvrInstalled && !!(ctx.get('seedvr') || {}).enabled;

// Ask app.js to pull a fresh snapshot from the server.
export function requestRefresh() {
  window.dispatchEvent(new CustomEvent('autumn:refresh'));
}

// Actions ------------------------------------------------------------------
async function installEnhancement(name, label) {
  const { api } = await import('./api.js');
  toast(`Installing ${label}…`, { kind: 'info', timeout: 60000 });
  try {
    await api.setEnhancement(name, true);
    requestRefresh();
    toast(`${label} installed`, { kind: 'ok' });
  } catch (err) {
    toast(`${label}: ${err.message}`, { kind: 'err', timeout: 9000 });
  }
}

async function installSeedvr(ctx) {
  const okToGo = await confirmDialog({
    title: 'Install SeedVR2',
    message: 'SeedVR2 clones the ComfyUI custom node into the engine and downloads two large '
      + 'models (a DiT and a VAE, several GB). The ComfyUI engine reloads its node list when '
      + 'this finishes. Continue?',
    confirmLabel: 'Install SeedVR2',
  });
  if (!okToGo) return;
  await installEnhancement('seedvr2', 'SeedVR2');
  if (ctx) ctx.refresh();
}

async function installHqPreview(ctx) {
  const okToGo = await confirmDialog({
    title: 'Enable high quality H3 preview',
    message: 'This installs the KJNodes custom node pack and downloads taeh3.safetensors '
      + '(about 300 MB) so MiniMax H3 previews decode through the H3 temporal autoencoder '
      + 'instead of latent2rgb.',
    confirmLabel: 'Install',
  });
  if (!okToGo) return;
  await installEnhancement('kjnodes', 'KJNodes');
  await installEnhancement('taeh3', 'taeh3 decoder');
  if (ctx) ctx.refresh();
}

function insertWildcard(ctx, name) {
  return async () => {
    const { insertIntoPrompt } = await import('./composer.js');
    insertIntoPrompt(ctx.kind, `__${name}__`);
  };
}

// Shared field groups ------------------------------------------------------
function seedFields(kind) {
  return [
    {
      key: 'seed', type: 'number', label: 'Seed', min: 0, max: 4294967295, step: 1,
      hint: 'fixed output',
    },
    {
      key: 'random_seed', type: 'toggle', label: 'Randomise seed',
      help: 'A new seed per generation. The seed used is written to the history entry.',
    },
    { key: 'seed_increment', type: 'number', label: 'Seed step', min: 0, max: 1000, step: 1 },
  ];
}

function outputSection(kind, extra) {
  const fields = [
    {
      key: 'filename_prefix', type: 'text', label: 'Output folder / prefix',
      placeholder: 'autumn/image/Autumn',
      help: 'Relative to the program output directory. ComfyUI appends the index and extension.',
    },
    ...extra,
    {
      key: '_output_path', type: 'buttons', label: null, items: [
        {
          label: 'Copy output path', icon: 'copy', kind: 'ghost', run: () => {
            const dir = (state.paths && (kind === 'music' ? state.paths.music : state.paths.output)) || 'output';
            copyText(dir).then(() => toast('Output path copied', { kind: 'ok' }));
          },
        },
      ],
    },
  ];
  return { id: 'output', label: 'Output', icon: 'save', fields };
}

function regionalSection(kind, { video = false } = {}) {
  return {
    id: 'regional',
    label: 'Regional prompting',
    icon: 'crop',
    hint: 'per-area prompts',
    fields: [
      {
        key: 'regional.enabled', type: 'toggle', label: 'Enable regional prompting',
        help: 'Each region is encoded separately and blended into the canvas, so one prompt can '
          + 'describe different subjects per area.',
      },
      {
        key: 'regional.mode', type: 'select', label: 'Blend mode',
        when: (ctx) => !!ctx.get('regional.enabled'),
        options: [
          { value: 'area', label: 'Additive area (ComfyUI regional style)' },
          { value: 'mask', label: 'Masked rectangle (cross attention window)' },
          { value: 'and', label: 'AND conditioning (attention mask)' },
          { value: 'concat', label: 'Concat chain (per-region prompt chain)' },
        ],
        help: 'AND gives the sharpest separation, additive area the softest bleed. Regions keep '
          + 'their own override when set.',
      },
      {
        key: 'regional.regions', type: 'regions', label: null,
        when: (ctx) => !!ctx.get('regional.enabled'),
        video,
      },
      {
        key: '_regional_syntax', type: 'note',
        when: (ctx) => !!ctx.get('regional.enabled'),
        title: 'Typing a region instead',
        text: 'Regions also parse out of the prompt text: ' +
          '@region[label=sky w=1 h=0.35 s=0.9]storm clouds@end — attributes may also sit on the '
          + 'directive line with the body under it, ended by @end. Text regions are listed here '
          + 'before the ones you drag.',
      },
      {
        key: '_regional_help', type: 'note',
        when: (ctx) => !ctx.get('regional.enabled'),
        title: 'How it works',
        text: 'Drag the rectangles on the canvas to place a region, then give it a prompt and a '
          + 'weight. Regions are stored as fractions of the canvas, so they survive a resolution '
          + 'change.',
      },
    ],
  };
}

function adapterSection(kind, { embeddings = true, controlnets = true } = {}) {
  const fields = [
    { key: 'loras', type: 'loras', label: null, help: null },
  ];
  if (embeddings) fields.push({ key: 'embeddings', type: 'embeddings', label: null });
  if (controlnets) fields.push({ key: 'controlnets', type: 'controlnets', label: null });
  return {
    id: 'adapters',
    label: 'LoRAs, embeddings & ControlNet',
    icon: 'layers',
    hint: (ctx) => {
      const loras = (ctx.get('loras') || []).filter((l) => l && l.enabled !== false).length;
      const cn = (ctx.get('controlnets') || []).filter((c) => c && c.enabled !== false).length;
      const emb = (ctx.get('embeddings') || []).filter((e) => e && e.enabled !== false).length;
      const parts = [];
      if (loras) parts.push(`${loras} LoRA`);
      if (emb) parts.push(`${emb} emb`);
      if (cn) parts.push(`${cn} CN`);
      return parts.join(' · ');
    },
    fields,
  };
}

function wildcardSection() {
  return {
    id: 'wildcards',
    label: 'Wildcards',
    icon: 'shuffle',
    hint: (ctx) => `${wildcards().length} lists`,
    fields: [
      {
        key: '_wildcard_note', type: 'note', title: 'Syntax',
        text: 'Write __name__ in the prompt. The list is picked from at generation time; the '
          + 'expanded prompt is stored with the history entry.',
      },
      { key: 'wildcard_seed', type: 'select', label: 'Pick mode', options: [
        { value: 'batch', label: 'Fixed per seed' },
        { value: 'random', label: 'Random every run' },
        { value: 'index', label: 'Cycle by index' },
      ] },
      {
        key: '_wildcard_list', type: 'buttons', label: 'Insert into prompt',
        items: () => wildcards().slice(0, 40).map((name) => ({
          label: name, kind: 'ghost', run: (ctx) => insertWildcard(ctx, name)(),
        })),
      },
    ],
  };
}

// Image -------------------------------------------------------------------
const IMAGE_SECTIONS = [
  {
    id: 'model', label: 'Model', icon: 'box',
    hint: (ctx) => (ctx.get('checkpoint') ? baseName(ctx.get('checkpoint'))
      : (ctx.get('unet') ? baseName(ctx.get('unet')) : 'Pick a checkpoint')),
    fields: [
      {
        key: 'checkpoint', type: 'combo', label: 'Checkpoint', optionsCategory: 'checkpoints',
        emptyLabel: 'No checkpoints found', placeholder: 'Pick a checkpoint…',
        help: 'Single-file models carry UNet, CLIP and VAE. For separate components leave this '
          + 'empty and fill the diffusion model below.',
      },
      {
        key: 'unet', type: 'combo', label: 'Diffusion model / UNet',
        options: diffusionModels, emptyLabel: 'None (use checkpoint)',
        placeholder: 'Pick a diffusion model…',
      },
      {
        key: 'weight_dtype', type: 'select', label: 'Weight dtype', options: [
          { value: 'default', label: 'Default (file dtype)' },
          { value: 'fp32', label: 'fp32' },
          { value: 'fp16_fast', label: 'fp16 fast' },
          { value: 'bf16_fast', label: 'bf16 fast' },
          { value: 'fp8_e4m3fn_fast', label: 'fp8 e4m3fn fast' },
          { value: 'fp8_e4m3fn_scaled', label: 'fp8 scaled' },
        ], help: 'Fast dtypes trade precision for speed and VRAM.' },
      {
        key: 'model_dtype', type: 'select', label: 'Model dtype', options: [
          { value: 'auto', label: 'Auto' }, { value: 'default', label: 'Default' },
          { value: 'fp16', label: 'fp16' }, { value: 'bf16', label: 'bf16' },
          { value: 'fp32', label: 'fp32' }, { value: 'fp8_e4m3fn', label: 'fp8 e4m3fn' },
          { value: 'fp8_e5m2', label: 'fp8 e5m2' },
        ] },
      {
        key: 'clip', type: 'combo', label: 'CLIP / text encoder', options: textEncoders,
        emptyLabel: 'From checkpoint', placeholder: 'Optional override…',
      },
      {
        key: 'clip2', type: 'combo', label: 'Second text encoder', options: textEncoders,
        emptyLabel: 'None', placeholder: 'Optional (dual-encoder models)…',
      },
      {
        key: 'clip3', type: 'combo', label: 'Third text encoder', options: textEncoders,
        emptyLabel: 'None', placeholder: 'Optional (tri-encoder models)…',
      },
      {
        key: 'clip_type', type: 'select', label: 'CLIP type', options: [
          { value: 'auto', label: 'Auto' }, { value: 'clip', label: 'CLIP-L' },
          { value: 'clip_vit_h', label: 'ViT-H' }, { value: 'clip_vit_big_g', label: 'ViT-bigG' },
          { value: 't5', label: 'T5-XXL' }, { value: 'qwen', label: 'Qwen' },
        ] },
      {
        key: 'vae', type: 'combo', label: 'VAE', options: vaes, emptyLabel: 'From checkpoint',
        placeholder: 'auto', help: '"auto" uses the VAE baked into the checkpoint.',
      },
      {
        key: 'clip_skip', type: 'number', label: 'CLIP skip', min: 0, max: 12, step: 1,
        hint: '0 = off', help: 'Use an earlier CLIP layer. SD1.5 users often set 2.',
      },
      {
        key: 'latent', type: 'select', label: 'Latent space', options: [
          { value: 'auto', label: 'Auto' }, { value: 'sd', label: 'SD (4x, 4 channel)' },
          { value: 'sd3', label: 'SD3 / Flux (2x)' }, { value: 'flux', label: 'Flux' },
          { value: 'flux2', label: 'Flux 2' },
        ] },
    ],
  },
  {
    id: 'canvas', label: 'Canvas', icon: 'frame',
    hint: (ctx) => `${ctx.get('width') || 1024}×${ctx.get('height') || 1024}`,
    fields: [
      { key: 'width', type: 'number', label: 'Width', min: 64, max: 8192, step: 8 },
      { key: 'height', type: 'number', label: 'Height', min: 64, max: 8192, step: 8 },
      {
        key: '_aspect', type: 'buttons', label: 'Aspect ratio', items: [
          { label: '1:1', kind: 'ghost', run: (ctx) => ctx.setSize(1024, 1024) },
          { label: '3:2', kind: 'ghost', run: (ctx) => ctx.setSize(1216, 832) },
          { label: '2:3', kind: 'ghost', run: (ctx) => ctx.setSize(832, 1216) },
          { label: '4:3', kind: 'ghost', run: (ctx) => ctx.setSize(1152, 896) },
          { label: '16:9', kind: 'ghost', run: (ctx) => ctx.setSize(1344, 768) },
          { label: '9:16', kind: 'ghost', run: (ctx) => ctx.setSize(768, 1344) },
          { label: '21:9', kind: 'ghost', run: (ctx) => ctx.setSize(1536, 640) },
        ],
      },
      {
        key: 'batch_size', type: 'number', label: 'Batch size', min: 1, max: 64, step: 1,
        help: 'Images per run. Each image gets its own seed when randomising.',
      },
    ],
  },
  {
    id: 'sampling', label: 'Sampling', icon: 'noise',
    hint: (ctx) => `${ctx.get('steps') || 0} · ${ctx.get('sampler') || 'euler'}`,
    fields: [
      { key: 'steps', type: 'number', label: 'Steps', min: 1, max: 200, step: 1 },
      { key: 'sampler', type: 'select', label: 'Sampler', options: 'samplers' },
      { key: 'scheduler', type: 'select', label: 'Scheduler', options: 'schedulers' },
      { key: 'cfg', type: 'number', label: 'CFG', min: 0, max: 30, step: 0.05 },
      {
        key: 'denoise', type: 'percent', label: 'Denoise', unit: '%',
        help: 'Below 100 % keeps the composition of the input image.',
      },
      {
        key: 'shift', type: 'number', label: 'Shift', min: 0, max: 20, step: 0.1,
        help: 'Noise schedule shift. Higher shifts weight to the noisier end.',
      },
      ...seedFields('image'),
    ],
  },
  {
    id: 'input', label: 'Image input', icon: 'image',
    hint: (ctx) => {
      const parts = [];
      if (ctx.get('init_image')) parts.push('init');
      if (ctx.get('mask_image')) parts.push('mask');
      if (ctx.get('outpaint') && ctx.get('outpaint.enabled')) parts.push('outpaint');
      return parts.length ? parts.join(' + ') : 'text to image';
    },
    fields: [
      {
        key: 'init_image', type: 'combo', label: 'Initial image', options: 'image_files',
        emptyLabel: 'None (txt2img)', placeholder: 'Pick an uploaded image…',
        help: 'Drop an image on the prompt box to upload it, then pick it here.',
      },
      {
        key: 'init_fit', type: 'select', label: 'Fit to canvas', when: (ctx) => !!ctx.get('init_image'),
        options: [
          { value: 'stretch', label: 'Stretch' }, { value: 'fit', label: 'Fit (pad)' },
          { value: 'crop', label: 'Cover (crop)' },
        ] },
      {
        key: 'mask_image', type: 'combo', label: 'Mask image', options: 'image_files',
        emptyLabel: 'None', placeholder: 'Pick a mask…',
        help: 'White repaints, black keeps. The mask channel picks which channel is read.',
      },
      {
        key: 'mask_channel', type: 'select', label: 'Mask channel', when: (ctx) => !!ctx.get('mask_image'),
        options: [
          { value: 'red', label: 'Red' }, { value: 'green', label: 'Green' },
          { value: 'blue', label: 'Blue' }, { value: 'alpha', label: 'Alpha' },
        ] },
      {
        key: 'inpaint_grow', type: 'number', label: 'Grow mask (px)', min: 0, max: 256, step: 1,
        when: (ctx) => !!ctx.get('mask_image'),
      },
      {
        key: 'outpaint.enabled', type: 'toggle', label: 'Outpaint (expand canvas)',
        when: (ctx) => !!ctx.get('init_image'),
      },
      {
        key: 'outpaint.top', type: 'number', label: 'Top', min: 0, max: 4096, step: 8,
        when: (ctx) => !!(ctx.get('outpaint') || {}).enabled },
      {
        key: 'outpaint.bottom', type: 'number', label: 'Bottom', min: 0, max: 4096, step: 8,
        when: (ctx) => !!(ctx.get('outpaint') || {}).enabled },
      {
        key: 'outpaint.left', type: 'number', label: 'Left', min: 0, max: 4096, step: 8,
        when: (ctx) => !!(ctx.get('outpaint') || {}).enabled },
      {
        key: 'outpaint.right', type: 'number', label: 'Right', min: 0, max: 4096, step: 8,
        when: (ctx) => !!(ctx.get('outpaint') || {}).enabled },
      {
        key: 'outpaint.feather', type: 'number', label: 'Feather (px)', min: 0, max: 256, step: 1,
        when: (ctx) => !!(ctx.get('outpaint') || {}).enabled },
    ],
  },
  {
    id: 'hires', label: 'High-res fix', icon: 'zoom_in',
    hint: (ctx) => (ctx.get('hires_fix') ? `×${ctx.get('hires_scale') || 1.5}` : 'off'),
    fields: [
      {
        key: 'hires_fix', type: 'toggle', label: 'Enable hires fix',
        help: 'Upscales the latent and samples a second pass, which recovers detail that a single '
          + 'pass at high resolution loses.',
      },
      { key: 'hires_scale', type: 'number', label: 'Scale', min: 1, max: 4, step: 0.05,
        when: (ctx) => !!ctx.get('hires_fix') },
      { key: 'hires_steps', type: 'number', label: 'Steps', min: 1, max: 200, step: 1,
        when: (ctx) => !!ctx.get('hires_fix') },
      { key: 'hires_denoise', type: 'percent', label: 'Denoise', unit: '%',
        when: (ctx) => !!ctx.get('hires_fix') },
      { key: 'hires_cfg', type: 'number', label: 'CFG', min: 0, max: 30, step: 0.05,
        when: (ctx) => !!ctx.get('hires_fix') },
      {
        key: 'hires_method', type: 'select', label: 'Upscale method',
        when: (ctx) => !!ctx.get('hires_fix'),
        options: [
          { value: 'bislerp', label: 'Bislerp (latent)' },
          { value: 'bilinear', label: 'Bilinear (latent)' },
          { value: 'nearest-exact', label: 'Nearest (latent)' },
          { value: 'area', label: 'Area (latent)' },
          { value: 'model', label: 'Model upscaler (pixel space)' },
        ] },
      {
        key: 'hires_upscale_model', type: 'combo', label: 'Upscaler model',
        when: (ctx) => ctx.get('hires_method') === 'model',
        options: upscaleModels, emptyLabel: 'None installed',
      },
    ],
  },
  {
    id: 'refiner', label: 'Refiner', icon: 'sparkles',
    hint: (ctx) => (ctx.get('refiner') ? `${baseName(ctx.get('refiner'))} @ ${Math.round((ctx.get('refiner_switch') || 0) * 100)}%` : 'off'),
    fields: [
      {
        key: 'refiner', type: 'combo', label: 'Refiner checkpoint', optionsCategory: 'checkpoints',
        emptyLabel: 'None', placeholder: 'Pick a refiner…',
        help: 'Sampling hands over to the refiner at the switch point.',
      },
      { key: 'refiner_switch', type: 'percent', label: 'Switch at', unit: '%',
        when: (ctx) => !!ctx.get('refiner') },
      { key: 'refiner_prompt', type: 'text', label: 'Refiner prompt', placeholder: 'Optional extra detail',
        when: (ctx) => !!ctx.get('refiner') },
      { key: 'refiner_sampler', type: 'select', label: 'Refiner sampler', options: 'samplers',
        when: (ctx) => !!ctx.get('refiner') },
      { key: 'refiner_scheduler', type: 'select', label: 'Refiner scheduler', options: 'schedulers',
        when: (ctx) => !!ctx.get('refiner') },
      { key: 'refiner_cfg', type: 'number', label: 'Refiner CFG', min: 0, max: 30, step: 0.05,
        when: (ctx) => !!ctx.get('refiner') },
    ],
  },
  {
    id: 'upscale', label: 'Final upscale', icon: 'expand',
    hint: (ctx) => (ctx.get('final_upscale') ? `${ctx.get('final_megapixels') || 2} MP` : 'off'),
    fields: [
      {
        key: 'final_upscale', type: 'toggle', label: 'Upscale after sampling',
        help: 'Pixel-space upscale of the finished image, after every sampling pass.',
      },
      { key: 'upscale_model', type: 'combo', label: 'Upscaler model',
        when: (ctx) => !!ctx.get('final_upscale'), options: upscaleModels,
        emptyLabel: 'None installed' },
      { key: 'final_megapixels', type: 'number', label: 'Target megapixels', min: 0.25, max: 32,
        step: 0.25, when: (ctx) => !!ctx.get('final_upscale'),
        help: 'Width and height are scaled to hit this area.' },
    ],
  },
  regionalSection('image'),
  adapterSection('image'),
  wildcardSection(),
  outputSection('image', [
    { key: 'output_format', type: 'segmented', label: 'Format', options: [
      { value: 'png', label: 'PNG' }, { value: 'webp', label: 'WebP' }, { value: 'jpg', label: 'JPG' }] },
    { key: 'image_quality', type: 'number', label: 'Quality', min: 1, max: 100, step: 1,
      when: (ctx) => ctx.get('output_format') !== 'png' },
  ]),
];

// Video -------------------------------------------------------------------
const VIDEO_SECTIONS = [
  {
    id: 'model', label: 'Model & pipeline', icon: 'film',
    hint: (ctx) => {
      const family = ctx.get('family') || 'wan22';
      const label = (FAMILIES.find((f) => f.value === family) || {}).label || family;
      return `${label} · ${MODE_LABELS[ctx.get('mode')] || ctx.get('mode')}`;
    },
    fields: [
      {
        key: 'family', type: 'select', label: 'Architecture',
        options: FAMILIES.map((f) => ({ value: f.value, label: f.label, hint: f.hint })),
        help: 'Sets the node graph, the canvas multiple and the frame grid.',
      },
      {
        key: 'mode', type: 'select', label: 'Task',
        options: (ctx) => modesFor(ctx.get('family') || 'wan22'),
      },
      {
        key: 'model', type: 'combo', label: 'Diffusion model', options: diffusionModels,
        emptyLabel: 'None found', placeholder: 'Pick the video model…',
      },
      {
        key: 'checkpoint', type: 'combo', label: 'Checkpoint (all-in-one)',
        options: checkpoints, emptyLabel: 'None (use separate components)',
        placeholder: 'Optional…',
      },
      {
        key: 'weight_dtype', type: 'select', label: 'Weight dtype', options: [
          { value: 'default', label: 'Default (file dtype)' },
          { value: 'fp32', label: 'fp32' },
          { value: 'fp16_fast', label: 'fp16 fast' },
          { value: 'bf16_fast', label: 'bf16 fast' },
          { value: 'fp8_e4m3fn_fast', label: 'fp8 e4m3fn fast' },
        ] },
      {
        key: 'model_dtype', type: 'select', label: 'Model dtype', options: [
          { value: 'auto', label: 'Auto' }, { value: 'fp16', label: 'fp16' },
          { value: 'bf16', label: 'bf16' }, { value: 'fp32', label: 'fp32' },
          { value: 'fp8_e4m3fn', label: 'fp8 e4m3fn' },
        ] },
      {
        key: 'clip', type: 'combo', label: 'Text encoder', options: textEncoders,
        emptyLabel: 'None', placeholder: 'Pick a text encoder…',
      },
      {
        key: 'clip2', type: 'combo', label: 'Second text encoder', options: textEncoders,
        emptyLabel: 'None', placeholder: 'Optional…',
      },
      {
        key: 'clip_type', type: 'select', label: 'CLIP type', options: [
          { value: 'auto', label: 'Auto (from family)' },
          { value: 'wan', label: 'Wan' },
          { value: 'ltxv', label: 'LTX-Video' },
          { value: 'cogvideox', label: 'CogVideoX' },
          { value: 'minimax', label: 'MiniMax H3' },
          { value: 'lumina2', label: 'Lumina 2' },
          { value: 'qwen_image', label: 'Qwen' },
          { value: 'stable_diffusion', label: 'Stable Diffusion' },
        ],
        help: 'Which encoder recipe the loader uses. Auto picks the one for the model family.',
      },
      { key: 'clip_skip', type: 'number', label: 'CLIP skip', min: 0, max: 12, step: 1, hint: '0 = off' },
      {
        key: 'vae', type: 'combo', label: 'Video VAE', options: vaes, emptyLabel: 'None found',
        placeholder: 'Pick a VAE…',
      },
      {
        key: 'audio_vae', type: 'combo', label: 'Audio VAE',
        options: () => merged('audio_encoders', 'vae'),
        emptyLabel: 'None', placeholder: 'Only for audio-capable models…',
        when: (ctx) => ['t2va', 'i2va'].includes(ctx.get('mode')),
      },
    ],
  },
  {
    id: 'canvas', label: 'Resolution & duration', icon: 'frame',
    hint: (ctx) => {
      const fps = ctx.get('fps') || 24;
      const frames = ctx.get('frames') || 1;
      return `${ctx.get('width')}×${ctx.get('height')} · ${(frames / fps).toFixed(1)} s`;
    },
    fields: [
      { key: 'width', type: 'number', label: 'Width', min: 64, max: 4096, step: 8,
        hint: (ctx) => `×${multipleFor(ctx.get('family'))}` },
      { key: 'height', type: 'number', label: 'Height', min: 64, max: 4096, step: 8,
        hint: (ctx) => `×${multipleFor(ctx.get('family'))}` },
      {
        key: '_aspect', type: 'buttons', label: 'Aspect ratio', items: [
          { label: '16:9', kind: 'ghost', run: (ctx) => ctx.setSize(1280, 720) },
          { label: '9:16', kind: 'ghost', run: (ctx) => ctx.setSize(720, 1280) },
          { label: '1:1', kind: 'ghost', run: (ctx) => ctx.setSize(768, 768) },
          { label: '4:3', kind: 'ghost', run: (ctx) => ctx.setSize(1024, 768) },
          { label: '21:9', kind: 'ghost', run: (ctx) => ctx.setSize(1536, 640) },
          { label: 'HD', kind: 'ghost', run: (ctx) => ctx.setSize(1920, 1080) },
        ] },
      {
        key: 'frames', type: 'number', label: 'Frames', min: 1, max: 10000, step: 1,
        hint: (ctx) => frameHint(ctx.get('family') || 'wan22'),
        help: 'Snapped to the temporal grid of the architecture on save.',
      },
      {
        key: '_duration', type: 'buttons', label: 'Quick duration', items: [
          { label: '2 s', kind: 'ghost', run: (ctx) => ctx.setDuration(2) },
          { label: '4 s', kind: 'ghost', run: (ctx) => ctx.setDuration(4) },
          { label: '5 s', kind: 'ghost', run: (ctx) => ctx.setDuration(5) },
          { label: '8 s', kind: 'ghost', run: (ctx) => ctx.setDuration(8) },
          { label: '10 s', kind: 'ghost', run: (ctx) => ctx.setDuration(10) },
        ] },
      { key: 'fps', type: 'number', label: 'Frame rate', min: 1, max: 120, step: 1,
        hint: (ctx) => `default ${FAMILY_FPS[ctx.get('family')] || 24}` },
      { key: 'batch_size', type: 'number', label: 'Batch size', min: 1, max: 8, step: 1 },
    ],
  },
  {
    id: 'sampling', label: 'Sampling', icon: 'noise',
    hint: (ctx) => `${ctx.get('steps') || 0} · ${ctx.get('sampler') || 'euler'}`,
    fields: [
      { key: 'steps', type: 'number', label: 'Steps', min: 1, max: 200, step: 1 },
      { key: 'sampler', type: 'select', label: 'Sampler', options: 'samplers' },
      { key: 'scheduler', type: 'select', label: 'Scheduler', options: 'schedulers' },
      { key: 'cfg', type: 'number', label: 'CFG', min: 0, max: 30, step: 0.05 },
      {
        key: 'denoise', type: 'percent', label: 'Denoise', unit: '%',
        when: (ctx) => ['v2v', 'inpaint', 'animate'].includes(ctx.get('mode')),
      },
      {
        key: 'strength', type: 'percent', label: 'Strength', unit: '%',
        when: (ctx) => ['v2v', 'vace', 'inpaint', 'control', 'animate'].includes(ctx.get('mode')),
        help: 'How far the source video moves away from the original.',
      },
      { key: 'shift', type: 'number', label: 'Shift', min: 0, max: 20, step: 0.1,
        help: 'Noise schedule shift. Wan uses 5, Flux-style models about 1.' },
      { key: 'base_shift', type: 'number', label: 'Base shift', min: 0, max: 5, step: 0.05,
        when: (ctx) => ['wan22', 'minimax_h3'].includes(ctx.get('family')) },
      { key: 'max_shift', type: 'number', label: 'Max shift', min: 0, max: 10, step: 0.05,
        when: (ctx) => ['wan22', 'minimax_h3'].includes(ctx.get('family')) },
      {
        key: 'guidance_type', type: 'select', label: 'Guidance', options: [
          { value: 'auto', label: 'Auto' }, { value: 'classifier_free', label: 'Classifier free' },
          { value: 'cfg++', label: 'CFG++' }, { value: 'distilled', label: 'Distilled (fixed)' },
        ] },
      { key: 'svd_min_cfg', type: 'number', label: 'Min CFG (SVD)', min: 0, max: 10, step: 0.05,
        when: (ctx) => ctx.get('family') === 'svd' },
      { key: 'motion_bucket_id', type: 'number', label: 'Motion bucket', min: 0, max: 511, step: 1,
        when: (ctx) => ctx.get('family') === 'svd', help: 'Higher moves more.' },
      { key: 'augmentation_level', type: 'number', label: 'Augmentation', min: 0, max: 10, step: 0.05,
        when: (ctx) => ctx.get('family') === 'svd' },
      { key: 'shift_audio', type: 'number', label: 'Audio shift', min: 0, max: 20, step: 0.1,
        when: (ctx) => ['t2va', 'i2va'].includes(ctx.get('mode')) },
      ...seedFields('video'),
    ],
  },
  {
    id: 'inputs', label: 'Source media', icon: 'upload',
    hint: (ctx) => {
      const parts = [];
      if (ctx.get('source_image')) parts.push('image');
      if (ctx.get('last_frame')) parts.push('last');
      if (ctx.get('source_video')) parts.push('video');
      if (ctx.get('control_video')) parts.push('control');
      const refs = (ctx.get('reference_images') || []).length;
      if (refs) parts.push(`${refs} ref`);
      if (ctx.get('driving_audio')) parts.push('audio');
      return parts.length ? parts.join(' · ') : 'none attached';
    },
    fields: [
      {
        key: 'source_image', type: 'combo', label: 'First frame / source image',
        options: 'image_files', emptyLabel: 'None', placeholder: 'Pick an image…',
        help: 'Drop an image on the prompt box to upload it.',
      },
      {
        key: 'last_frame', type: 'combo', label: 'Last frame', options: 'image_files',
        emptyLabel: 'None', placeholder: 'Pick an image…',
        when: (ctx) => ctx.get('mode') === 'flf2v',
      },
      {
        key: 'source_video', type: 'combo', label: 'Source video', options: 'video_files',
        emptyLabel: 'None', placeholder: 'Pick a video…',
        when: (ctx) => ['v2v', 'vace', 'inpaint', 'animate'].includes(ctx.get('mode')),
      },
      {
        key: 'control_video', type: 'combo', label: 'Control video', options: 'video_files',
        emptyLabel: 'None', placeholder: 'Pick a video…',
        when: (ctx) => ctx.get('mode') === 'control',
      },
      {
        key: 'reference_images', type: 'references', label: null,
        when: (ctx) => ['ref2v', 'animate', 'guide', 'i2v'].includes(ctx.get('mode')),
      },
      {
        key: 'guides', type: 'guides', label: null,
        when: (ctx) => ctx.get('mode') === 'guide',
      },
      {
        key: 'driving_audio', type: 'combo', label: 'Driving audio', options: 'audio_files',
        emptyLabel: 'None', placeholder: 'Pick an audio file…',
        when: (ctx) => ['t2va', 'i2va', 'animate'].includes(ctx.get('mode')),
      },
      {
        key: 'keep_source_audio', type: 'toggle', label: 'Keep audio from the source video',
        when: (ctx) => ['v2v', 'vace', 'inpaint', 'control', 'animate'].includes(ctx.get('mode')),
      },
    ],
  },
  regionalSection('video', { video: true }),
  adapterSection('video', { embeddings: false }),
  {
    id: 'decode', label: 'Decoding', icon: 'cpu_fan',
    hint: (ctx) => (ctx.get('tiled_decode') ? 'tiled' : 'full frame'),
    fields: [
      {
        key: 'tiled_decode', type: 'toggle', label: 'Tiled decode',
        help: 'Decodes in tiles so long or large videos fit in VRAM. Slightly slower.',
      },
      { key: 'tile_size', type: 'number', label: 'Tile size', min: 64, max: 1024, step: 16,
        when: (ctx) => !!ctx.get('tiled_decode') },
      { key: 'tile_overlap', type: 'number', label: 'Tile overlap', min: 0, max: 512, step: 8,
        when: (ctx) => !!ctx.get('tiled_decode') },
      { key: 'temporal_size', type: 'number', label: 'Temporal tile (frames)', min: 1, max: 256, step: 1,
        when: (ctx) => !!ctx.get('tiled_decode') },
      { key: 'temporal_overlap', type: 'number', label: 'Temporal overlap', min: 0, max: 64, step: 1,
        when: (ctx) => !!ctx.get('tiled_decode') },
    ],
  },
  {
    id: 'interpolate', label: 'Frame interpolation', icon: 'split',
    hint: (ctx) => ((ctx.get('interpolate') || {}).enabled
      ? `×${(ctx.get('interpolate') || {}).multiplier || 2}` : 'off'),
    fields: [
      {
        key: 'interpolate.enabled', type: 'toggle', label: 'Interpolate frames (RIFE)',
        help: 'Adds in-between frames after sampling. Output frame rate is multiplied to match.',
      },
      {
        key: 'interpolate.model', type: 'combo', label: 'Interpolation model',
        options: 'interpolation_models', emptyLabel: 'None installed',
        when: (ctx) => !!(ctx.get('interpolate') || {}).enabled,
      },
      {
        key: 'interpolate.multiplier', type: 'select', label: 'Multiplier', options: [
          { value: 2, label: '2×' }, { value: 3, label: '3×' }, { value: 4, label: '4×' }],
        when: (ctx) => !!(ctx.get('interpolate') || {}).enabled,
      },
    ],
  },
  {
    id: 'seedvr', label: 'SeedVR2 upscaler', icon: 'rocket',
    hint: () => (state.seedvrInstalled ? 'installed' : 'optional'),
    fields: [
      {
        type: 'note',
        text: 'SeedVR2 is an optional diffusion video upscaler. Installing it clones '
          + 'numz/ComfyUI-SeedVR2_VideoUpscaler into the engine; the DiT and VAE weights are '
          + 'several GB and are fetched on first use.',
        when: () => !state.seedvrInstalled,
      },
      {
        type: 'buttons', label: null,
        items: () => [{ label: state.seedvrInstalled ? 'Installed' : 'Install SeedVR2',
                        icon: state.seedvrInstalled ? 'check' : 'download',
                        kind: state.seedvrInstalled ? 'ghost' : 'primary',
                        run: installSeedvr }],
        when: () => true,
      },
      {
        key: 'seedvr.enabled', type: 'toggle', label: 'Upscale with SeedVR2',
        help: 'Diffusion-based video restoration between decode and encode. Needs the SeedVR2 '
          + 'DiT and VAE models.',
        when: () => state.seedvrInstalled,
      },
      { key: 'seedvr.dit_model', type: 'combo', label: 'DiT model', options: 'seedvr_models',
        emptyLabel: 'None found', when: svOn },
      { key: 'seedvr.vae_model', type: 'combo', label: 'SeedVR VAE', options: 'seedvr_vae_models',
        emptyLabel: 'ema_vae_fp16.safetensors', when: svOn },
      { key: 'seedvr.resolution', type: 'number', label: 'Output resolution', min: 256, max: 4096,
        step: 64, when: (ctx) => !!(ctx.get('seedvr') || {}).enabled,
        help: 'Long edge of the upscaled video.' },
      { key: 'seedvr.max_resolution', type: 'number', label: 'Max resolution (0 = off)', min: 0,
        max: 4096, step: 64, when: svOn },
      {
        key: 'seedvr.color_correction', type: 'select', label: 'Colour correction', options: [
          { value: 'wavelet', label: 'Wavelet' }, { value: 'wavelet_adaptive', label: 'Wavelet adaptive' },
          { value: 'lab', label: 'Lab' }, { value: 'hsv', label: 'HSV' },
          { value: 'adain', label: 'AdaIN' }, { value: 'none', label: 'None' }],
        when: svOn },
      { key: 'seedvr.batch_size', type: 'number', label: 'Batch (frames)', min: 1, max: 64, step: 1,
        when: svOn },
      { key: 'seedvr.uniform_batch_size', type: 'check', label: null, text: 'Uniform batch size',
        when: svOn },
      { key: 'seedvr.temporal_overlap', type: 'number', label: 'Temporal overlap', min: 0, max: 64,
        step: 1, when: svOn },
      { key: 'seedvr.prepend_frames', type: 'number', label: 'Prepend frames', min: 0, max: 64,
        step: 1, when: svOn },
      { key: 'seedvr.input_noise_scale', type: 'number', label: 'Input noise', min: 0, max: 1,
        step: 0.01, when: svOn },
      { key: 'seedvr.latent_noise_scale', type: 'number', label: 'Latent noise', min: 0, max: 1,
        step: 0.01, when: svOn },
      { key: 'seedvr.device', type: 'select', label: 'Device', options: [
        { value: 'cuda:0', label: 'cuda:0' }, { value: 'cpu', label: 'cpu' }],
        when: svOn },
      { key: 'seedvr.offload_device', type: 'select', label: 'Offload device', options: [
        { value: 'cpu', label: 'cpu' }, { value: 'cuda:0', label: 'cuda:0' }],
        when: svOn },
      { key: 'seedvr.blocks_to_swap', type: 'number', label: 'Blocks to swap', min: 0, max: 64,
        step: 1, when: (ctx) => !!(ctx.get('seedvr') || {}).enabled,
        help: 'Trade speed for VRAM on small cards.' },
      { key: 'seedvr.swap_io_components', type: 'check', label: null, text: 'Swap I/O components',
        when: svOn },
      { key: 'seedvr.cache_model', type: 'check', label: null, text: 'Keep model in memory',
        when: svOn },
      { key: 'seedvr.attention_mode', type: 'select', label: 'Attention', options: [
        { value: 'auto', label: 'Auto' }, { value: 'sdpa', label: 'SDPA' },
        { value: 'flash_attn', label: 'Flash attention' }, { value: 'xformers', label: 'xFormers' }],
        when: svOn },
      { key: 'seedvr.encode_tiled', type: 'check', label: null, text: 'Tiled encode',
        when: svOn },
      { key: 'seedvr.encode_tile_size', type: 'number', label: 'Encode tile', min: 256, max: 2048,
        step: 64, when: svOn },
      { key: 'seedvr.encode_tile_overlap', type: 'number', label: 'Encode overlap', min: 0, max: 1024,
        step: 16, when: svOn },
      { key: 'seedvr.decode_tiled', type: 'check', label: null, text: 'Tiled decode',
        when: svOn },
      { key: 'seedvr.decode_tile_size', type: 'number', label: 'Decode tile', min: 256, max: 2048,
        step: 64, when: svOn },
      { key: 'seedvr.decode_tile_overlap', type: 'number', label: 'Decode overlap', min: 0, max: 1024,
        step: 16, when: svOn },
    ],
  },
  outputSection('video', [
    { key: 'container', type: 'select', label: 'Container', options: [
      { value: 'auto', label: 'Auto (MP4)' }, { value: 'mp4', label: 'MP4 (H.264)' },
      { value: 'webm', label: 'WebM (VP9)' }, { value: 'mkv', label: 'MKV (ProRes)' },
      { value: 'gif', label: 'GIF' }] },
    { key: 'bit_depth', type: 'select', label: 'Bit depth', options: [
      { value: 'auto', label: 'Auto (8-bit)' }, { value: '8', label: '8-bit' },
      { value: '10', label: '10-bit (10-bit H.264 / HEVC)' }] },
    { key: 'color_space', type: 'select', label: 'Colour space', options: [
      { value: 'sRGB', label: 'sRGB' }, { value: 'Rec.709', label: 'Rec.709' },
      { value: 'Display P3', label: 'Display P3' }] },
  ]),
];

// Music -------------------------------------------------------------------
/** Show a YuE token count as the seconds of audio it is worth (25 tokens per second). */
function tokenSeconds(value) {
  const tokens = Number(value || 0);
  if (tokens <= 0) return '';
  const seconds = tokens / 25;
  return seconds >= 60 ? `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, '0')}`
    : `${Math.round(seconds * 10) / 10} s`;
}

const MUSIC_SECTIONS = [
  {
    id: 'song', label: 'Song', icon: 'music',
    hint: (ctx) => (ctx.get('instrumental') ? 'instrumental'
      : (ctx.get('lyrics') ? `${(ctx.get('lyrics') || '').split('\n').length} lines` : 'no lyrics')),
    fields: [
      {
        key: 'style', type: 'area', label: 'Style description', rows: 4,
        placeholder: '90s british rock, male vocal, driving drums, melodic bass…',
        help: 'Genre, era, instruments, voice. YuE reads this as the generation condition. '
          + 'The words of the song go in the box under the waveform.',
      },
      {
        key: 'instrumental', type: 'toggle', label: 'Instrumental',
        help: 'Skips lyric generation and the vocal track.',
      },
      {
        key: 'cot', type: 'select', label: 'Reasoning depth', options: [
          { value: 'full', label: 'Full (genre, mood, capability, lyrics)' },
          { value: 'genre', label: 'Genre only' },
          { value: 'mood', label: 'Mood only' },
          { value: 'capability', label: 'Capability only' },
          { value: 'lyrics', label: 'Lyrics only' },
          { value: 'none', label: 'Off' }],
        help: 'How much chain-of-thought the model writes before it commits to music.' },
      {
        key: 'abc', type: 'code', label: 'ABC notation (optional)', rows: 4,
        placeholder: 'optional melody the model follows',
      },
    ],
  },
  {
    id: 'engine', label: 'Engine', icon: 'download',
    hint: () => (state.musicInstalled ? 'ready' : 'not installed'),
    fields: [
      {
        key: '_engine_status', type: 'note', title: 'YuE',
        text: 'Music runs through YuE in its own virtual environment inside the program folder. '
          + 'Weights are downloaded from Hugging Face into models/huggingface.',
      },
      {
        key: 'model', type: 'text', label: 'Model repo', placeholder: 'm-a-p/YuE2-3B',
        help: 'Hugging Face repository id.',
      },
      { key: 'vae', type: 'text', label: 'VAE repo', placeholder: 'm-a-p/YuE2-Vae' },
      {
        key: 'device', type: 'select', label: 'Device', options: [
          { value: 'auto', label: 'Auto' }, { value: 'cuda', label: 'CUDA' },
          { value: 'mps', label: 'Metal' }, { value: 'cpu', label: 'CPU' }] },
      { key: 'low_vram', type: 'toggle', label: 'Low VRAM mode',
        help: 'Runs the stages one after another and frees memory between them.' },
      {
        key: '_engine_actions', type: 'buttons', label: null, items: [
          { label: 'Install / repair YuE', icon: 'download', kind: 'ghost', run: async () => {
            const { api } = await import('./api.js');
            const okToGo = await confirmDialog({
              title: 'Install YuE',
              message: 'YuE is cloned into the program folder and gets its own virtual '
                + 'environment. Weights download from Hugging Face on first run. Continue?',
              confirmLabel: 'Install YuE',
            });
            if (!okToGo) return;
            try {
              await api.musicAction('install');
              requestRefresh();
              toast('YuE install started. Watch the log panel for progress.', { kind: 'ok' });
            } catch (err) {
              toast(err.message, { kind: 'err' });
            }
          } },
          { label: 'Unload model', icon: 'stop', kind: 'ghost', run: async () => {
            const { api } = await import('./api.js');
            try { await api.musicAction('unload'); toast('YuE unloaded', { kind: 'ok' }); }
            catch (err) { toast(err.message, { kind: 'err' }); }
          } },
        ],
      },
    ],
  },
  {
    id: 'semantic', label: 'Semantic sampling', icon: 'brain',
    hint: (ctx) => `T ${(ctx.get('semantic_sampling') || {}).temperature ?? 1.0}`,
    fields: [
      { key: 'semantic_sampling.temperature', type: 'number', label: 'Temperature', min: 0.1, max: 3, step: 0.01 },
      { key: 'semantic_sampling.top_p', type: 'number', label: 'Top p', min: 0, max: 1, step: 0.01 },
      { key: 'semantic_sampling.top_k', type: 'number', label: 'Top k', min: 1, max: 1000, step: 1 },
      { key: 'semantic_sampling.repetition_penalty', type: 'number', label: 'Repetition penalty',
        min: 1, max: 2, step: 0.005 },
      { key: 'semantic_sampling.penalty_window', type: 'number', label: 'Penalty window', min: 1,
        max: 2000, step: 1 },
      { key: 'semantic_sampling.min_tokens', type: 'number', label: 'Min tokens', min: 1, max: 20000, step: 1,
        hint: (ctx) => tokenSeconds(ctx.get('semantic_sampling.min_tokens')) },
      { key: 'semantic_sampling.max_tokens', type: 'number', label: 'Max tokens', min: 1, max: 40000, step: 1,
        hint: (ctx) => tokenSeconds(ctx.get('semantic_sampling.max_tokens')),
        help: 'Hard ceiling on the length. YuE2 emits 25 semantic tokens per second of audio, '
          + 'so 3000 is about two minutes. A track can stop earlier at its end token.' },
    ],
  },
  {
    id: 'abc', label: 'Acoustic (ABC) sampling', icon: 'wave',
    hint: (ctx) => `T ${(ctx.get('abc_sampling') || {}).temperature ?? 0.7}`,
    fields: [
      { key: 'abc_sampling.temperature', type: 'number', label: 'Temperature', min: 0.1, max: 3, step: 0.01 },
      { key: 'abc_sampling.top_p', type: 'number', label: 'Top p', min: 0, max: 1, step: 0.01 },
      { key: 'abc_sampling.top_k', type: 'number', label: 'Top k', min: 1, max: 1000, step: 1 },
      { key: 'abc_sampling.repetition_penalty', type: 'number', label: 'Repetition penalty', min: 1,
        max: 2, step: 0.005 },
      { key: 'abc_sampling.penalty_window', type: 'number', label: 'Penalty window', min: 1, max: 2000, step: 1 },
      { key: 'abc_sampling.min_tokens', type: 'number', label: 'Min tokens', min: 1, max: 20000, step: 1 },
      { key: 'abc_sampling.max_tokens', type: 'number', label: 'Max tokens', min: 1, max: 40000, step: 1 },
      { key: 'ode_steps', type: 'number', label: 'Decoder ODE steps', min: 1, max: 128, step: 1,
        help: 'Steps of the diffusion head. More is smoother and slower.' },
      { key: 'cfg_scale', type: 'number', label: 'CFG scale', min: 0, max: 10, step: 0.05,
        hint: 'blank = model default' },
      ...seedFields('music'),
    ],
  },
  outputSection('music', [
    { key: 'output_format', type: 'segmented', label: 'Format', options: [
      { value: 'flac', label: 'FLAC' }, { value: 'wav', label: 'WAV' }, { value: 'mp3', label: 'MP3' }] },
    { key: 'live_preview', type: 'toggle', label: 'Live waveform while generating',
      help: 'Streams decoded peaks as the acoustic stage runs.' },
    { key: 'trim_preview_seconds', type: 'number', label: 'Trim to first N seconds (0 = full)',
      min: 0, max: 600, step: 1 },
  ]),
];

// Registry -----------------------------------------------------------------
const REGISTRY = { image: IMAGE_SECTIONS, video: VIDEO_SECTIONS, music: MUSIC_SECTIONS };

export function sectionsFor(kind) {
  return REGISTRY[kind] || [];
}

export function seedvrInstallAction() {
  return installSeedvr;
}

export { installSeedvr, installHqPreview, insertWildcard };

export const SEEDVR_URL = 'https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler';
export const WILDCARD_NAMES = wildcards;
