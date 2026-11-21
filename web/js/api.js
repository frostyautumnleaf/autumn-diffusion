// Thin fetch wrapper for the dashboard API. Every call returns parsed JSON and
// throws Error(message) when the server answers {ok:false}.

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload || {};
  }
}

async function call(path, { method = 'GET', body, signal, raw } = {}) {
  const options = { method, signal, headers: {} };
  if (body !== undefined && !(body instanceof FormData)) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  } else if (body instanceof FormData) {
    options.body = body;
  }
  const response = await fetch(path, options);
  if (raw) return response;
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = { ok: false, error: `${response.status}: ${text.slice(0, 200)}` };
    }
  }
  if (!response.ok || data.ok === false) {
    throw new ApiError(data.error || `HTTP ${response.status}`, response.status, data);
  }
  return data;
}

const post = (path, body) => call(path, { method: 'POST', body: body || {} });
const patch = (path, body) => call(path, { method: 'PATCH', body: body || {} });

export const api = {
  ApiError,

  state: () => call('/api/state'),
  health: () => call('/api/health'),

  settings: () => call('/api/settings'),
  saveSettings: (settings) => post('/api/settings', { settings }),
  resetSection: (section) => post('/api/settings/reset', { section }),

  setup: (features, app, install) => post('/api/setup', { features, app, install }),

  models: (limit = 600) => call(`/api/models?limit=${limit}`),
  wildcards: () => call('/api/wildcards'),
  saveWildcard: (name, text) => post('/api/wildcards', { name, text }),
  deleteWildcard: (name) => call('/api/wildcards', { method: 'DELETE', body: { name } }),

  generate: (kind, params, sessionId) => post('/api/generate', { kind, params, session_id: sessionId }),
  jobs: (query = {}) => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value !== null && value !== undefined && value !== '') params.set(key, String(value));
    }
    const tail = params.toString();
    return call(`/api/jobs${tail ? `?${tail}` : ''}`);
  },
  job: (id) => call(`/api/jobs/${encodeURIComponent(id)}`),
  cancelJob: (id) => post(`/api/jobs/${encodeURIComponent(id)}/cancel`, {}),
  rerunJob: (id) => post(`/api/jobs/${encodeURIComponent(id)}/rerun`, {}),
  reuseJob: (id, index = 0) => post(`/api/jobs/${encodeURIComponent(id)}/reuse`, { index }),
  favoriteJob: (id, favorite) => post(`/api/jobs/${encodeURIComponent(id)}/favorite`, { favorite }),
  deleteJob: (id) => call(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  clearQueue: () => post('/api/queue/clear', {}),
  clearHistory: (kind, engine) => post('/api/history/clear', { kind, engine }),

  preview: (jobId) => call(`/api/preview${jobId ? `?job=${encodeURIComponent(jobId)}` : ''}`),
  system: () => call('/api/system'),
  logs: (count = 400, level = '') =>
    call(`/api/logs?count=${count}${level ? `&level=${encodeURIComponent(level)}` : ''}`),

  engine: () => call('/api/engine'),
  engineAction: (action, body) => post(`/api/engine/${action}`, body || {}),

  nodes: () => call('/api/nodes'),
  installNode: (url, name, requirements = true) =>
    post('/api/nodes/install', { url, name, requirements }),
  toggleNode: (name, enabled) => post('/api/nodes/toggle', { name, enabled }),
  removeNode: (name) => post('/api/nodes/remove', { name }),

  enhancements: () => call('/api/enhancements'),
  setEnhancement: (name, enabled, extra) => post('/api/enhancements', { name, enabled, ...(extra || {}) }),

  installStatus: () => call('/api/install'),
  startInstall: (features) => post('/api/install', { features }),

  music: () => call('/api/music'),
  musicAction: (action, extra) => post('/api/music', { action, ...(extra || {}) }),

  // Scores. A composition is edited as ABC, so these are cheap round trips: the
  // server parses with the same parser the generator uses.
  jobScore: (id) => call(`/api/jobs/${encodeURIComponent(id)}/score`),
  abcValidate: (abc) => post('/api/abc/validate', { abc }),
  abcCompare: (before, after, allowTempoChange = false) =>
    post('/api/abc/compare', { before, after, allow_tempo_change: allowTempoChange }),
  abcStrip: (abc, keepVoice = 'both') => post('/api/abc/strip', { abc, keep_voice: keepVoice }),
  transcribe: (audio, options = {}) => post('/api/music/transcribe', { audio, ...options }),
  installTranscriber: (enabled = true) => post('/api/enhancements', { name: 'transcriber', enabled }),

  categories: () => call('/api/assets/categories'),
  assets: (category, subfolder, root = 'models') => {
    const params = new URLSearchParams({ category, subfolder: subfolder || '', root });
    return call(`/api/assets?${params}`);
  },
  assetTree: (category, root = 'models') =>
    call(`/api/assets?category=${encodeURIComponent(category)}&root=${root}&tree=1`),
  assetAction: (action, body) => post(`/api/assets/${action}`, body),
  writeAsset: (category, path, text, root = 'models') =>
    post('/api/assets', { category, path, text, root }),
  deleteAsset: (category, path, root = 'models', purge = false) =>
    call('/api/assets', { method: 'DELETE', body: { category, path, root, purge } }),

  presets: (kind) => call(`/api/presets${kind ? `?kind=${kind}` : ''}`),
  savePreset: (name, kind, params, label) => post('/api/presets', { name, kind, params, label }),
  deletePreset: (name) => call('/api/presets', { method: 'DELETE', body: { name } }),

  upload: (file, { category = 'input', subfolder = '', name = '' } = {}) => {
    const params = new URLSearchParams({ category, subfolder });
    if (name) params.set('name', name);
    const form = new FormData();
    form.append('file', file, file.name);
    return call(`/api/upload?${params}`, { method: 'POST', body: form });
  },
};

export { ApiError as error };
