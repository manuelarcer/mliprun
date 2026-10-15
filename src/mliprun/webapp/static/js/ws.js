// WebSocket client (PLAN section 6.5): subscriptions, reconnect with backoff
// (1, 2, 5, 10 s) and replay with `since = last seen seq + 1`.

// `transport.WebSocket` is replaceable by a test double.
export const transport = {
  WebSocket: window.WebSocket,
  url: () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
};

const BACKOFF_S = [1, 2, 5, 10];

class WsClient {
  constructor() {
    this.sock = null;
    this.attempt = 0;
    this.subs = new Map(); // job_id -> {lastSeq: number|null}  ("*" -> {})
    this.listeners = new Set(); // fn(event)
    this.stateListeners = new Set(); // fn("open"|"closed"|"connecting", info)
    this.state = "closed";
    this.reconnectTimer = null;
    this.everOpened = false;
  }

  connect() {
    if (this.sock) return;
    this._setState("connecting");
    let sock;
    try {
      sock = new transport.WebSocket(transport.url());
    } catch (err) {
      this._scheduleReconnect();
      return;
    }
    this.sock = sock;
    sock.onopen = () => {
      const reconnect = this.everOpened;
      this.everOpened = true;
      this.attempt = 0;
      this._setState("open", { reconnect });
      for (const [jobId, sub] of this.subs) this._sendSubscribe(jobId, sub);
    };
    sock.onmessage = (msg) => {
      let ev;
      try {
        ev = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (ev.type === "ws_ping" || ev.type === "ws_ack") return;
      if (ev.type === "ws_error") {
        console.warn("mliprun ws:", ev.code, ev.message);
        return;
      }
      if (typeof ev.seq === "number" && ev.job_id) {
        const sub = this.subs.get(ev.job_id);
        if (sub && (sub.lastSeq === null || ev.seq > sub.lastSeq)) sub.lastSeq = ev.seq;
      }
      for (const fn of this.listeners) {
        try {
          fn(ev);
        } catch (err) {
          console.error(err);
        }
      }
    };
    sock.onclose = () => {
      this.sock = null;
      this._setState("closed");
      this._scheduleReconnect();
    };
    sock.onerror = () => {
      /* onclose follows */
    };
  }

  _scheduleReconnect() {
    if (this.reconnectTimer) return;
    const delay = BACKOFF_S[Math.min(this.attempt, BACKOFF_S.length - 1)];
    this.attempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay * 1000);
  }

  _setState(state, info = {}) {
    this.state = state;
    for (const fn of this.stateListeners) fn(state, info);
  }

  _send(obj) {
    if (this.sock && this.sock.readyState === 1) this.sock.send(JSON.stringify(obj));
  }

  _sendSubscribe(jobId, sub) {
    if (jobId === "*") this._send({ action: "subscribe", job_id: "*" });
    else
      this._send({
        action: "subscribe",
        job_id: jobId,
        since: sub.lastSeq === null ? sub.initialSince : sub.lastSeq + 1,
      });
  }

  /** Subscribe to one job (replay from `since`) or to "*" (live status of all jobs). */
  subscribe(jobId, since = null) {
    const sub = { lastSeq: null, initialSince: since };
    if (since !== null && since > 0) sub.lastSeq = since - 1;
    this.subs.set(jobId, sub);
    this._sendSubscribe(jobId, sub);
  }

  unsubscribe(jobId) {
    if (!this.subs.has(jobId)) return;
    this.subs.delete(jobId);
    this._send({ action: "unsubscribe", job_id: jobId });
  }

  onEvent(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  onState(fn) {
    this.stateListeners.add(fn);
    return () => this.stateListeners.delete(fn);
  }
}

export const ws = new WsClient();
