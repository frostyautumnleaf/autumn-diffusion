// WebSocket client: auto-reconnect, fan-out to subscribers, latency ping.

import { uid } from './util.js';

const BACKOFF = [400, 800, 1500, 2500, 4000, 6000, 10000];

export class Socket {
  constructor({ url, onStatus } = {}) {
    this.clientId = uid('ws');
    this.url = url || `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws?client=${this.clientId}`;
    this.onStatus = onStatus || (() => {});
    this.handlers = new Map();
    this.globals = new Set();
    this.ws = null;
    this.attempt = 0;
    this.closedByUser = false;
    this.lastEvent = 0;
    this.latency = 0;
    this.connected = false;
  }

  connect() {
    this.closedByUser = false;
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.attempt = 0;
      this.connected = true;
      this.onStatus('open', this);
    };
    this.ws.onmessage = (event) => this._receive(event);
    this.ws.onerror = () => this.onStatus('error', this);
    this.ws.onclose = () => {
      this.connected = false;
      this.onStatus('closed', this);
      if (!this.closedByUser) this.scheduleReconnect();
    };
  }

  scheduleReconnect() {
    const wait = BACKOFF[Math.min(this.attempt, BACKOFF.length - 1)];
    this.attempt += 1;
    this.onStatus('retry', this);
    setTimeout(() => this.connect(), wait);
  }

  close() {
    this.closedByUser = true;
    if (this.ws) {
      try { this.ws.close(); } catch (err) { /* already gone */ }
    }
  }

  _receive(event) {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (err) {
      return;
    }
    const name = payload.event || 'message';
    const data = payload.data || {};
    this.lastEvent = Date.now();
    if (name === 'pong') this.latency = Math.max(0, Date.now() - (data.at ? data.at * 1000 : Date.now()));
    const list = this.handlers.get(name);
    if (list) for (const handler of list.slice()) {
      try { handler(data, name); } catch (err) { console.error(`handler ${name} failed`, err); }
    }
    for (const handler of this.globals) {
      try { handler(name, data); } catch (err) { console.error('global handler failed', err); }
    }
  }

  on(name, handler) {
    if (!this.handlers.has(name)) this.handlers.set(name, []);
    this.handlers.get(name).push(handler);
    return () => this.off(name, handler);
  }

  off(name, handler) {
    const list = this.handlers.get(name);
    if (!list) return;
    const index = list.indexOf(handler);
    if (index >= 0) list.splice(index, 1);
  }

  onAny(handler) {
    this.globals.add(handler);
    return () => this.globals.delete(handler);
  }

  send(command, data = {}) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify({ command, data }));
    return true;
  }

  ping() {
    const at = Date.now() / 1000;
    this.latencyStart = at;
    return this.send('ping', { at });
  }
}
