// Job list (queue) and job detail views (PLAN 4.5, 6, 11.1).
import { api, ApiError, fileUrl, zipUrl } from "./api.js";
import { ws } from "./ws.js";
import {
  el,
  clear,
  fmtNum,
  fmtTime,
  fmtDuration,
  fmtBytes,
  unitForKey,
  labelForKey,
  toXyz,
  splitXyzFrames,
  frameThrottle,
  debounce,
  statusBadge,
  TERMINAL,
} from "./util.js";
import { viewerWithSlider } from "./viewer.js";
import { createPlots, imageLabel } from "./plots.js";

const CANCELLABLE = new Set(["pending", "running"]);
const PROGRESS_TYPES = new Set(["opt_step", "md_step", "neb_iter", "freq_progress"]);
const TEXT_EXT = /\.(txt|log|csv|json|jsonl|sh|xyz|extxyz|vasp|cif|out|py|md|dat|in)$|(^|\/)(POSCAR|CONTCAR|OUTCAR)$/i;

function briefText(job) {
  const r = job.result_brief;
  if (!r) return null;
  const parts = [];
  if ("converged" in r) parts.push(r.converged ? "converged" : "not converged");
  if (r.energy_eV !== undefined) parts.push(`E = ${fmtNum(r.energy_eV, 8)} eV`);
  if (r.final_energy_eV !== undefined) parts.push(`E = ${fmtNum(r.final_energy_eV, 8)} eV`);
  if (r.final_fmax_eV_per_A !== undefined && r.final_fmax_eV_per_A !== null) parts.push(`fmax ${fmtNum(r.final_fmax_eV_per_A, 3)} eV/Å`);
  if (r.n_imaginary !== undefined) parts.push(`${r.n_imaginary} imaginary`);
  if (r.zpe_eV !== undefined) parts.push(`ZPE ${fmtNum(r.zpe_eV, 5)} eV`);
  if (r.mean_temperature_K !== undefined) parts.push(`⟨T⟩ ${fmtNum(r.mean_temperature_K, 5)} K`);
  if (r.forward_barrier_eV !== undefined) parts.push(`Ea ${fmtNum(r.forward_barrier_eV, 4)} eV`);
  return parts.join(", ");
}

async function cancelJob(jobId, button, onDone) {
  button.disabled = true;
  button.textContent = "Cancelling…";
  try {
    const { job } = await api.cancelJob(jobId);
    if (onDone) onDone(job);
  } catch (err) {
    button.disabled = false;
    button.textContent = "Cancel";
    const msg = err instanceof ApiError && err.status === 409 ? "The job already finished." : err.message;
    alert(`Could not cancel: ${msg}`);
  }
}

// ================================================================= job list

function isRemote(job) {
  return !!(job && job.target && job.target.kind === "remote");
}

export class JobList {
  constructor(root) {
    this.root = root;
    this.jobs = new Map();
    this.order = [];
    this.selectedId = null;
    this.table = el("div", { class: "job-list", testid: "job-list" });
    this.summary = el("span", { class: "muted small" });
    this.root.append(el("div", { class: "panel-head" }, el("h2", {}, "Queue"), this.summary), this.table);
    this.refreshSoon = debounce(() => this.refresh(), 250);
    ws.onEvent((ev) => {
      if (ev.type === "job_status" || ev.type === "result" || ev.type === "error") {
        const j = this.jobs.get(ev.job_id);
        if (j && ev.type === "job_status") {
          j.status = ev.status;
          this._renderRow(j);
        }
        this.refreshSoon();
      }
    });
    this.pollTimer = setInterval(() => {
      if ([...this.jobs.values()].some((j) => CANCELLABLE.has(j.status))) this.refresh();
    }, 2500);
  }

  async refresh() {
    try {
      const { jobs } = await api.listJobs({ limit: 500 });
      this.jobs = new Map(jobs.map((j) => [j.job_id, j]));
      this.order = jobs.map((j) => j.job_id);
      this.render();
    } catch (err) {
      clear(this.table).append(el("div", { class: "error-text" }, `Could not load jobs: ${err.message}`));
    }
  }

  upsert(job) {
    if (!this.jobs.has(job.job_id)) this.order.unshift(job.job_id);
    this.jobs.set(job.job_id, { ...(this.jobs.get(job.job_id) || {}), ...job });
    this.render();
  }

  select(jobId) {
    this.selectedId = jobId;
    for (const row of this.table.querySelectorAll('[data-testid="job-row"]')) {
      row.classList.toggle("active", row.dataset.jobId === jobId);
    }
  }

  render() {
    clear(this.table);
    const counts = { pending: 0, running: 0 };
    for (const j of this.jobs.values()) if (j.status in counts) counts[j.status]++;
    this.summary.textContent = `${this.order.length} job${this.order.length === 1 ? "" : "s"} · ${counts.running} running · ${counts.pending} queued`;
    if (!this.order.length) {
      this.table.append(el("div", { class: "muted empty" }, "No jobs yet. Submit one from the New calculation panel."));
      return;
    }
    for (const id of this.order) {
      const row = el("div");
      this.table.append(row);
      this._renderRow(this.jobs.get(id), row);
    }
  }

  _renderRow(job, existing) {
    const row = existing || this.table.querySelector(`[data-testid="job-row"][data-job-id="${CSS.escape(job.job_id)}"]`);
    if (!row) return;
    const fresh = el("div", {
      class: "job-row" + (this.selectedId === job.job_id ? " active" : ""),
      testid: "job-row",
      dataset: { jobId: job.job_id, status: job.status },
      tabindex: "0",
      role: "link",
    });
    const progress = job.progress && job.status === "running" ? job.progress.text || "" : "";
    const brief = job.status === "done" ? briefText(job) : null;
    const extra =
      job.status === "pending" && job.queue_position
        ? `queued #${job.queue_position}`
        : job.status === "failed" && job.error
          ? job.error
          : brief || progress;
    let cancel = null;
    if (CANCELLABLE.has(job.status)) {
      cancel = el("button", { type: "button", class: "btn btn-small btn-danger", testid: "cancel-button", title: "Cancel this job" }, "Cancel");
      if (job.cancel_requested) {
        cancel.disabled = true;
        cancel.textContent = "Cancelling…";
      }
      cancel.addEventListener("click", (e) => {
        e.stopPropagation();
        cancelJob(job.job_id, cancel, (j) => this.upsert(j));
      });
    }
    fresh.append(
      el("div", { class: "jr-status" }, statusBadge(job.status)),
      el(
        "div",
        { class: "jr-main" },
        el("div", {}, el("span", { class: "jr-type" }, job.calc_type), " ", el("span", { class: "jr-label" }, job.label || ""), " ", el("span", { class: "muted" }, job.formula ? `${job.formula}` : ""),
          isRemote(job) ? [" ", el("span", { class: "tag target-tag", testid: "job-target", title: `Runs on ${job.target.env_name} (${job.target.machine_name})` }, job.target.env_name || job.target.machine_name)] : null,
        ),
        el("div", { class: "jr-extra muted small" + (job.status === "failed" ? " error-text" : "") }, extra || fmtTime(job.created_at)),
        job.folder || job.run_dir
          ? el("div", { class: "jr-folder muted small", title: job.run_dir || "" }, "folder ", job.folder || job.run_dir)
          : null,
      ),
      el("div", { class: "jr-calc muted small" }, job.calculator ? job.calculator.name : ""),
      el("div", { class: "jr-actions" }, cancel),
    );
    const go = () => (location.hash = `#/jobs/${job.job_id}`);
    fresh.addEventListener("click", go);
    fresh.addEventListener("keydown", (e) => {
      if (e.key === "Enter") go();
    });
    row.replaceWith(fresh);
  }
}

// ================================================================= job detail

export class JobDetail {
  constructor(root, { calcTypes, onJobChange }) {
    this.root = root;
    this.calcTypes = calcTypes;
    this.onJobChange = onJobChange;
    this.jobId = null;
    this.unsub = null;
    this.showPlaceholder();
  }

  showPlaceholder() {
    this._teardown();
    clear(this.root).append(
      el(
        "div",
        { class: "placeholder" },
        el("h2", {}, "Job detail"),
        el("p", { class: "muted" }, "Select a job in the queue to see its live plots, structure and results."),
      ),
    );
  }

  _showRemoteError(msg) {
    if (!this.remoteWarn) return;
    clear(this.remoteWarn);
    this.remoteWarn.classList.toggle("hidden", !msg);
    if (msg) this.remoteWarn.append(`Connection problem: ${msg} — the job keeps running on the remote; retrying`);
  }

  async _pollRemoteError() {
    const id = this.jobId;
    if (!id || TERMINAL.has(this.status)) {
      this._showRemoteError(null);
      return;
    }
    try {
      const fresh = await api.getJob(id);
      if (this.jobId !== id) return;
      this.detail.remote_error = fresh.remote_error;
      this._showRemoteError(fresh.remote_error);
    } catch {
      /* the next tick will catch up */
    }
  }

  _teardown() {
    if (this.remoteTimer) clearInterval(this.remoteTimer);
    this.remoteTimer = null;
    if (this.jobId) ws.unsubscribe(this.jobId);
    if (this.unsub) this.unsub();
    this.unsub = null;
    if (this.plots) this.plots.destroy();
    this.plots = null;
    if (this.view) this.view.viewer.destroy();
    this.view = null;
    if (this.uiTick) this.uiTick.cancel();
    this.jobId = null;
  }

  async open(jobId) {
    this._teardown();
    this.jobId = jobId;
    clear(this.root).append(el("div", { class: "muted" }, `Loading ${jobId}…`));
    let detail;
    try {
      detail = await api.getJob(jobId);
    } catch (err) {
      if (this.jobId !== jobId) return;
      clear(this.root).append(
        el("div", { class: "error-text" }, err.status === 404 ? `Job ${jobId} not found.` : `Could not load job: ${err.message}`),
      );
      return;
    }
    if (this.jobId !== jobId) return;
    this.detail = detail;
    this.status = detail.status;
    this.maxSeq = -1;
    this.loading = true;
    this.buffer = [];
    this.pendingFrame = null;
    this.pendingBand = null;
    this.latestProgress = null;
    this.resultShown = false;
    this._render();

    // Live events: buffer while history loads, then apply in seq order.
    this.unsub = ws.onEvent((ev) => {
      if (ev.job_id !== jobId || typeof ev.seq !== "number") return;
      if (this.loading) this.buffer.push(ev);
      else this._apply(ev);
    });

    // History via REST, then subscribe with since = next_seq (PLAN 6.5).
    let since = 0;
    try {
      for (let guard = 0; guard < 10000; guard++) {
        const page = await api.events(jobId, since, 2000);
        if (this.jobId !== jobId) return;
        for (const ev of page.events) this._apply(ev);
        if (!page.events.length || page.next_seq <= since) break;
        since = page.next_seq;
        if (page.terminal) break;
      }
    } catch (err) {
      this._log({ level: "error", message: `Could not load event history: ${err.message}` });
    }
    if (this.jobId !== jobId) return;
    this.loading = false;
    for (const ev of this.buffer.sort((a, b) => a.seq - b.seq)) this._apply(ev);
    this.buffer = [];
    ws.subscribe(jobId, this.maxSeq + 1);
    this.uiTick.flush();
    if (TERMINAL.has(this.status) && (!this.detail.reproduce || (this.status === "done" && !this.resultShown))) this._onTerminal();
    else this._loadFiles();
  }

  // ------------------------------------------------------------- rendering

  _fieldSpec(name) {
    const ct = this.calcTypes.calc_types.find((c) => c.name === this.detail.calc_type);
    return ct ? ct.fields.find((f) => f.name === name) : null;
  }

  _inputRole() {
    const ins = this.detail.inputs || {};
    return ins.structure ? "structure" : ins.band ? "band" : ins.initial ? "initial" : Object.keys(ins)[0];
  }

  _render() {
    const d = this.detail;
    const isNeb = d.calc_type === "neb" || d.calc_type === "autoneb";
    this.statusBadgeHost = el("span");
    this.cancelHost = el("span");
    this.phaseEl = el("span", { class: "muted small" });
    this.errorBox = el("div", { class: "error-box hidden" });
    this.progressBar = el("div", { class: "progress-fill" });
    this.progressText = el("span", { class: "small" });
    this.progressBox = el("div", { class: "progress hidden" }, el("div", { class: "progress-track" }, this.progressBar), this.progressText);
    this.timesEl = el("div", { class: "muted small" });
    this.remoteWarn = el("div", { class: "banner banner-warn small hidden", testid: "job-remote-error", role: "status" });
    this._showRemoteError(d.remote_error);
    if (isRemote(d)) {
      // The connection state changes without any job event: poll the detail while the job is live.
      this.remoteTimer = setInterval(() => this._pollRemoteError(), 3000);
    }

    const c = d.calculator || {};
    const task = c.uma_task || c.mace_head || c.sevennet_task;
    const header = el(
      "div",
      { class: "detail-head" },
      el(
        "div",
        { class: "dh-title" },
        this.statusBadgeHost,
        el("h2", {}, `${d.calc_type} — ${d.label || ""}`),
        this.cancelHost,
      ),
      el(
        "div",
        { class: "muted small" },
        `${d.job_id} · ${d.formula || ""} (${d.natoms ?? "?"} atoms) · ${c.name}${task ? ` / ${task}` : ""} · device ${c.device || "auto"}`,
        " ",
        this.phaseEl,
      ),
      d.run_dir || d.folder
        ? el("div", { class: "muted small dh-folder", title: "Run folder" }, "Folder: ", el("code", {}, d.run_dir || d.folder))
        : null,
      isRemote(d)
        ? el(
            "div",
            { class: "muted small dh-remote", testid: "job-remote-info" },
            el("div", { testid: "job-runs-on" }, "Runs on: ", el("strong", {}, `${d.target.env_name} (${d.target.machine_name})`)),
            d.remote_run_dir ? el("div", { testid: "job-remote-folder" }, "Remote folder: ", el("code", {}, d.remote_run_dir)) : null,
          )
        : null,
      this.remoteWarn,
      this.timesEl,
      this.progressBox,
      this.errorBox,
    );

    // Viewer
    const vhost = el("div");
    this.viewerNote = el("div", { class: "muted small viewer-note" });
    this.viewerTitle = isNeb ? el("div", { class: "viewer-title hidden", testid: "viewer-title" }) : null;
    const viewerCard = el("section", { class: "card viewer-card" }, el("h3", {}, isNeb ? "Band" : "Structure"), this.viewerTitle, vhost, this.viewerNote);
    this.view = viewerWithSlider(vhost, { height: 340, testid: "viewer", label: isNeb ? "Image" : "Frame" });
    this._initialStructure();

    // Plots
    this.plotMain = el("div", { class: "plot", testid: "plot-main" });
    this.plotSecondary = el("div", { class: "plot", testid: "plot-secondary" });
    this.plotExtra = el("div", { class: "plot hidden", testid: "plot-extra" });
    this.plots = createPlots(d.calc_type, { main: this.plotMain, secondary: this.plotSecondary, extra: this.plotExtra }, { params: d.params || {} });
    if (!this.plots.usesSecondary()) this.plotSecondary.classList.add("hidden");
    this.plotWait = el("div", { class: "muted small plot-wait" }, d.calc_type === "freq" || d.calc_type === "singlepoint" ? "Plots appear when the result is available." : "Waiting for the first step…");
    const plotsCard = el("section", { class: "card plots-card" }, el("h3", {}, "Live plots"), this.plotWait, this.plotMain, this.plotSecondary, this.plotExtra);

    // Results, tables, log, files, reproduce
    this.resultPanel = el("section", { class: "card result-panel", testid: "result-panel" }, el("h3", {}, "Result"), el("div", { class: "muted small" }, "No result yet."));
    this.tablesHost = el("div");
    this.logBody = el("div", { class: "log-body" });
    this.logPanel = el("section", { class: "card log-panel", testid: "log-panel" }, el("h3", {}, "Log"), this.logBody);
    this.filesList = el("ul", { class: "files-list", testid: "files-list" });
    const refresh = el("button", { type: "button", class: "btn btn-small" }, "Refresh");
    refresh.addEventListener("click", () => this._loadFiles());
    const filesCard = el(
      "section",
      { class: "card files-card" },
      el(
        "div",
        { class: "panel-head" },
        el("h3", {}, "Files"),
        el("span", {}, el("a", { href: zipUrl(d.job_id), class: "btn btn-small", testid: "zip-link", download: `${d.folder || d.job_id}.zip` }, "Download all (.zip)"), " ", refresh),
      ),
      el("div", { class: "muted small" }, `Run directory: ${d.run_dir || "–"}`),
      this.filesList,
    );
    this.reproducePanel = el("section", { class: "card reproduce-panel", testid: "reproduce-panel" });
    this.paramsCard = this._paramsCard();

    this.container = el(
      "div",
      { class: "job-detail", testid: "job-detail", dataset: { jobId: d.job_id, status: d.status } },
      header,
      el("div", { class: "detail-grid" }, viewerCard, plotsCard),
      this.resultPanel,
      this.tablesHost,
      el("div", { class: "detail-grid" }, this.logPanel, el("div", {}, filesCard, this.reproducePanel, this.paramsCard)),
    );
    clear(this.root).append(this.container);

    this.uiTick = frameThrottle(() => this._flushUi(), 150);
    this._setStatus(d.status, true);
    if (d.result) this._showResult(d.result);
    this._renderReproduce();
  }

  _paramsCard() {
    const d = this.detail;
    const rows = Object.entries(d.params || {}).map(([k, v]) => {
      const f = this._fieldSpec(k);
      const src = (d.param_sources || {})[k];
      return el(
        "tr",
        {},
        el("th", { title: f ? f.help : "" }, f ? f.label : k),
        el("td", {}, `${typeof v === "number" ? fmtNum(v) : v === null ? "auto" : String(v)}${f && f.unit ? ` ${f.unit}` : ""}`),
        el("td", { class: src === "user" ? "src-user" : "muted small" }, src || ""),
      );
    });
    rows.push(el("tr", {}, el("th", {}, "Live structure every"), el("td", {}, `${d.structure_every} events`), el("td")));
    return el(
      "details",
      { class: "card params-card" },
      el("summary", {}, el("h3", { class: "inline" }, "Parameters")),
      el("table", { class: "kv" }, rows),
    );
  }

  _initialStructure() {
    const d = this.detail;
    const ins = d.inputs || {};
    const role = this._inputRole();
    const info = ins[role] || {};
    this.symbols = info.symbols || null;
    this.pbc = info.pbc || [true, true, true];
    if (role === "band" && info.frames_xyz) {
      this.view.setFrames(splitXyzFrames(info.frames_xyz), (i, n) => `Image ${i} of ${n - 1}`);
      this.viewerNote.textContent = "Uploaded starting band.";
    } else if (ins.initial && ins.final && ins.initial.xyz && ins.final.xyz) {
      this.view.setFrames([ins.initial.xyz, ins.final.xyz], (i) => (i === 0 ? "Initial" : "Final"));
      this.viewerNote.textContent = "Endpoints (before interpolation).";
    } else if (info.xyz) {
      this.view.viewer.showXyz(info.xyz);
      this.viewerNote.textContent = "Input structure.";
    }
  }

  _setStatus(status, initial = false) {
    this.status = status;
    this.detail.status = status;
    if (this.container) this.container.dataset.status = status;
    clear(this.statusBadgeHost).append(statusBadge(status));
    clear(this.cancelHost);
    if (CANCELLABLE.has(status)) {
      const b = el("button", { type: "button", class: "btn btn-small btn-danger", testid: "cancel-button" }, "Cancel job");
      if (this.detail.cancel_requested) {
        b.disabled = true;
        b.textContent = "Cancelling…";
      }
      b.addEventListener("click", () =>
        cancelJob(this.detail.job_id, b, (job) => {
          this.detail.cancel_requested = true;
          if (job.status !== this.status) this._setStatus(job.status);
        }),
      );
      this.cancelHost.append(b);
    }
    const d = this.detail;
    const parts = [`created ${fmtTime(d.created_at)}`];
    if (d.started_at) parts.push(`started ${fmtTime(d.started_at)}`);
    if (d.finished_at) parts.push(`finished ${fmtTime(d.finished_at)}`);
    if (d.started_at && d.finished_at) parts.push(`wall time ${fmtDuration((new Date(d.finished_at) - new Date(d.started_at)) / 1000)}`);
    if (d.queue_position) parts.push(`queue position ${d.queue_position}`);
    this.timesEl.textContent = parts.join(" · ");
    if (TERMINAL.has(status)) {
      this.phaseEl.textContent = ""; // PLAN 15.2: no "phase" once finished
      // A finished job has no live progress (replayed events would otherwise
      // leave e.g. "iteration 33 / 600" with a part-filled bar); freq keeps
      // its force-call count.
      if (this.detail.calc_type !== "freq") this.progressBox.classList.add("hidden");
    }
    if (status === "failed" && d.error) this._showError(d.error);
    if (!initial && TERMINAL.has(status)) this._onTerminal();
    if (this.onJobChange) this.onJobChange(d);
  }

  async _onTerminal() {
    // Pick up result, reproduce, returncode and final timestamps.
    try {
      const fresh = await api.getJob(this.detail.job_id);
      if (fresh.job_id !== this.jobId) return;
      Object.assign(this.detail, fresh);
      this._setStatus(fresh.status, true);
      if (fresh.result && !this.resultShown) this._showResult(fresh.result);
      this._renderReproduce();
    } catch {
      /* the next refresh will catch up */
    }
    this.progressBox.classList.toggle("hidden", this.detail.calc_type !== "freq" || !this.latestProgress);
    this._loadFiles();
  }

  _showError(message, traceback) {
    clear(this.errorBox).append(el("strong", {}, "Error: "), message);
    if (traceback) this.errorBox.append(el("details", {}, el("summary", {}, "Traceback"), el("pre", {}, traceback)));
    this.errorBox.classList.remove("hidden");
  }

  // ------------------------------------------------------------- events

  _apply(ev) {
    if (ev.seq <= this.maxSeq) return; // replay overlap / duplicate
    this.maxSeq = ev.seq;
    switch (ev.type) {
      case "job_status":
        if (ev.reason) this.detail.error = ev.reason;
        this._log({ level: ev.status === "failed" ? "error" : "status", message: `status → ${ev.status}${ev.reason ? ` (${ev.reason})` : ""}`, t: ev.t });
        if (ev.status === "running" && !this.detail.started_at) this.detail.started_at = new Date(ev.t * 1000).toISOString();
        if (TERMINAL.has(ev.status) && !this.detail.finished_at) this.detail.finished_at = new Date(ev.t * 1000).toISOString();
        if (ev.status !== this.status || this.loading) this._setStatus(ev.status, this.loading);
        break;
      case "phase":
        this.phaseEl.textContent = TERMINAL.has(this.status) ? "" : `phase: ${ev.phase}`;
        this._log({ level: "phase", message: `[${ev.phase}] ${ev.message || ""}`, t: ev.t });
        break;
      case "log":
        this._log({ level: ev.level, message: ev.message, t: ev.t, logger: ev.logger });
        break;
      case "error":
        this._log({ level: "error", message: `${ev.exception_type ? ev.exception_type + ": " : ""}${ev.message}`, t: ev.t });
        this._showError(ev.message, ev.traceback);
        break;
      case "result":
        this._showResult(ev);
        break;
      default:
        if (PROGRESS_TYPES.has(ev.type)) {
          this.plots.ingest(ev);
          this.latestProgress = ev;
          if (ev.frame) this.pendingFrame = ev.frame;
          if (ev.band) {
            this.pendingBand = ev.band;
            this.pendingHighest = Number.isInteger(ev.highest_image) ? ev.highest_image : null;
          }
          this.uiTick();
        }
    }
  }

  _flushUi() {
    const ev = this.latestProgress;
    if (ev) {
      this.plotWait.classList.add("hidden");
      const p = this.detail.params || {};
      let frac = null;
      let text = "";
      if (ev.type === "opt_step") {
        frac = p.max_steps ? ev.step / p.max_steps : null;
        text = `step ${ev.step}${p.max_steps ? ` / ${p.max_steps}` : ""} · E = ${fmtNum(ev.energy, 9)} eV · fmax = ${fmtNum(ev.fmax, 4)} eV/Å`;
      } else if (ev.type === "md_step") {
        frac = p.steps ? ev.step / p.steps : null;
        text = `step ${ev.step}${p.steps ? ` / ${p.steps}` : ""} · t = ${fmtNum(ev.time_fs, 6)} fs · T = ${fmtNum(ev.temperature, 5)} K · Etot = ${fmtNum(ev.etot, 8)} eV`;
      } else if (ev.type === "neb_iter") {
        const total = this.detail.calc_type === "neb" ? p.neb_max_steps : null;
        frac = total ? ev.iteration / total : null;
        text = `iteration ${ev.iteration}${total ? ` / ${total}` : ""} · barrier = ${fmtNum(ev.barrier, 5)} eV · fmax = ${fmtNum(ev.fmax, 4)} eV/Å · highest image ${ev.highest_image}`;
        if (ev.autoneb) text += ` · sub-step ${ev.autoneb.sub_step}, ${ev.n_images} images`;
      } else if (ev.type === "freq_progress") {
        frac = ev.total ? ev.done / ev.total : null;
        text = `${ev.done} / ${ev.total} force calls`;
      }
      if (!TERMINAL.has(this.status) || ev.type === "freq_progress") {
        this.progressBox.classList.remove("hidden");
        this.progressBar.style.width = frac === null ? "100%" : `${Math.min(100, frac * 100).toFixed(1)}%`;
        this.progressBar.classList.toggle("indeterminate", frac === null);
        this.progressText.textContent = text;
      }
    }
    if (this.pendingBand && this.symbols) {
      const band = this.pendingBand;
      this.pendingBand = null;
      const frames = band.map((fr) => toXyz(this.symbols, fr.positions, fr.cell, this.pbc));
      const h = this.pendingHighest;
      this.view.setFrames(frames, (i, n) => `Image ${imageLabel(i)} of ${imageLabel(n - 1)}${i === h ? " (highest)" : ""}`);
      if (h !== null && h < frames.length) {
        this.view.viewer.goto(h);
        this._setViewerTitle(h);
      }
      this.view.refreshCaption();
      this.viewerNote.textContent = `Live band (iteration ${this.latestProgress ? this.latestProgress.iteration : "?"}).`;
    } else if (this.pendingFrame && this.symbols) {
      const fr = this.pendingFrame;
      this.pendingFrame = null;
      this.view.setFrames(null);
      this.view.viewer.showFrame(this.symbols, fr, this.pbc);
      const e = this.latestProgress;
      this.viewerNote.textContent = `Live structure (step ${e && e.step !== undefined ? e.step : "?"}).`;
    }
  }

  _setViewerTitle(index) {
    if (!this.viewerTitle) return;
    this.viewerTitle.textContent = `Image with highest energy - ${imageLabel(index)}`;
    this.viewerTitle.classList.remove("hidden");
  }

  _log({ level, message, t }) {
    const atBottom = this.logBody.scrollTop + this.logBody.clientHeight >= this.logBody.scrollHeight - 8;
    const time = t ? new Date(t * 1000).toLocaleTimeString(undefined, { hour12: false }) : "";
    this.logBody.append(el("div", { class: `log-line log-${level}` }, el("span", { class: "log-time" }, time), " ", message));
    while (this.logBody.childElementCount > 3000) this.logBody.firstChild.remove();
    if (atBottom) this.logBody.scrollTop = this.logBody.scrollHeight;
  }

  // ------------------------------------------------------------- result

  _showResult(r) {
    this.resultShown = true;
    this.detail.result = r;
    this.plotWait.classList.add("hidden");
    this.plots.setResult(r);
    this.uiTick.flush();

    // Viewer: final band or frame.
    if (Array.isArray(r.band) && this.symbols) {
      const frames = r.band.map((fr) => toXyz(this.symbols, fr.positions, fr.cell, this.pbc));
      const ts = Number.isInteger(r.ts_image_index) ? r.ts_image_index : 0;
      this.view.setFrames(frames, (i, n) => `Image ${imageLabel(i)} of ${imageLabel(n - 1)}${i === r.ts_image_index ? " (highest)" : ""}`);
      this.view.viewer.goto(ts);
      this.pendingBand = null; // the final band wins over any queued live band
      if (Number.isInteger(r.ts_image_index)) this._setViewerTitle(r.ts_image_index);
      this.view.refreshCaption();
      this.viewerNote.textContent = "Final band.";
    } else if (r.final_frame && this.symbols) {
      this.view.setFrames(null);
      this.view.viewer.showFrame(this.symbols, r.final_frame, this.pbc);
      this.viewerNote.textContent = r.calc_type === "freq" ? "Input geometry (the one the Hessian was computed at)." : "Final structure.";
    }
    if (r.calc_type === "singlepoint" && Array.isArray(r.forces)) {
      const fn = r.forces.map((f) => f.f_norm);
      this.view.viewer.colorBy(fn);
      const max = Math.max(...fn.filter(Number.isFinite), 0);
      this.viewerNote.textContent = `Atoms coloured by |F|: blue = 0, red = ${fmtNum(max, 4)} eV/Å.`;
    }

    // Scalars
    const skip = new Set(["job_id", "seq", "t", "type", "calc_type", "record_results", "forces", "modes", "band", "final_frame", "energies_eV", "relative_energies_eV", "mode_files", "stress_GPa", "stress_unavailable_reason"]);
    const rows = [];
    if ("converged" in r) {
      rows.push(el("tr", {}, el("th", {}, "Converged"), el("td", {}, el("span", { class: `badge ${r.converged ? "status-done" : "status-failed"}` }, r.converged ? "yes" : "no"))));
    }
    for (const [k, v] of Object.entries(r)) {
      if (skip.has(k) || k === "converged") continue;
      if (v !== null && typeof v === "object") continue;
      const unit = unitForKey(k);
      const isFlag = k === "fmax_warning" || k === "ts_at_endpoint";
      rows.push(
        el(
          "tr",
          { class: isFlag && v ? "warn-row" : "" },
          el("th", { title: k }, labelForKey(k)),
          el("td", {}, `${typeof v === "number" ? fmtNum(v, 9) : v === null ? "–" : String(v)}${unit && v !== null ? ` ${unit}` : ""}`),
        ),
      );
    }
    if ("stress_GPa" in r) {
      rows.push(
        el(
          "tr",
          {},
          el("th", {}, "Stress (Voigt xx yy zz yz xz xy)"),
          el("td", {}, r.stress_GPa ? `${r.stress_GPa.map((x) => fmtNum(x, 5)).join(", ")} GPa` : `not available${r.stress_unavailable_reason ? ` (${r.stress_unavailable_reason})` : ""}`),
        ),
      );
    }
    const children = [el("h3", {}, "Result"), el("table", { class: "kv result-table" }, rows)];
    if (Array.isArray(r.energies_eV)) {
      children.push(
        el("h4", {}, "Band energies"),
        el(
          "table",
          { class: "data-table" },
          el("thead", {}, el("tr", {}, el("th", {}, "Image"), el("th", {}, "E (eV)"), el("th", {}, "E − E₀ (eV)"))),
          el(
            "tbody",
            {},
            r.energies_eV.map((e, i) =>
              el(
                "tr",
                { class: i === r.ts_image_index ? "highlight-row" : "" },
                el("td", {}, i === r.ts_image_index ? `${i} (TS)` : String(i)),
                el("td", {}, fmtNum(e, 9)),
                el("td", {}, fmtNum((r.relative_energies_eV || [])[i], 6)),
              ),
            ),
          ),
        ),
      );
    }
    if (r.record_results && typeof r.record_results === "object") {
      children.push(el("details", {}, el("summary", {}, "Full record results (mliprun_run.json)"), el("pre", { class: "json" }, JSON.stringify(r.record_results, null, 2))));
    }
    clear(this.resultPanel).append(...children);

    // Tables
    clear(this.tablesHost);
    if (Array.isArray(r.forces)) this.tablesHost.append(this._forcesTable(r.forces));
    if (Array.isArray(r.modes)) this.tablesHost.append(this._freqTable(r));
  }

  _forcesTable(forces) {
    const LIMIT = 2000;
    const shown = forces.slice(0, LIMIT);
    return el(
      "section",
      { class: "card" },
      el("h3", {}, "Forces (eV/Å)"),
      forces.length > LIMIT ? el("div", { class: "muted small" }, `Showing the first ${LIMIT} of ${forces.length} atoms; the full table is in singlepoint_forces.csv.`) : null,
      el(
        "div",
        { class: "table-scroll" },
        el(
          "table",
          { class: "data-table", testid: "forces-table" },
          el("thead", {}, el("tr", {}, ["#", "Element", "Fx", "Fy", "Fz", "|F|", "Free"].map((h) => el("th", {}, h)))),
          el(
            "tbody",
            {},
            shown.map((f) =>
              el(
                "tr",
                { class: f.free ? "" : "muted" },
                el("td", {}, String(f.index)),
                el("td", {}, f.symbol),
                el("td", {}, f.fx.toFixed(6)),
                el("td", {}, f.fy.toFixed(6)),
                el("td", {}, f.fz.toFixed(6)),
                el("td", {}, f.f_norm.toFixed(6)),
                el("td", {}, f.free ? "yes" : "fixed"),
              ),
            ),
          ),
        ),
      ),
    );
  }

  _freqTable(r) {
    const floor = r.imaginary_floor_cm1 ?? 0;
    const summary = [
      `${r.n_modes ?? r.modes.length} modes`,
      `${r.n_imaginary} imaginary with |ν| ≥ ${floor} cm⁻¹`,
      r.n_imaginary_raw !== r.n_imaginary ? `${r.n_imaginary_raw} imaginary in total` : null,
      r.zpe_eV !== undefined ? `ZPE ${fmtNum(r.zpe_eV, 6)} eV` : null,
    ].filter(Boolean);
    const modeFiles = Array.isArray(r.mode_files) && r.mode_files.length
      ? el("div", { class: "small" }, "Mode animations: ", r.mode_files.map((p, i) => [i ? ", " : "", el("a", { href: fileUrl(this.detail.job_id, p), download: p.split("/").pop() }, p)]))
      : null;
    return el(
      "section",
      { class: "card" },
      el("h3", {}, "Vibrational modes"),
      el("div", { class: "muted small" }, summary.join(" · ")),
      r.fmax_warning ? el("div", { class: "warn-box small" }, `The input geometry is not at a stationary point (fmax ${fmtNum(r.fmax_at_input_free_eV_per_A, 4)} eV/Å); frequencies may be unreliable.`) : null,
      modeFiles,
      el(
        "div",
        { class: "table-scroll" },
        el(
          "table",
          { class: "data-table", testid: "freq-table" },
          el("thead", {}, el("tr", {}, ["Mode", "Frequency (cm⁻¹)", "Energy (meV)", "Imaginary"].map((h) => el("th", {}, h)))),
          el(
            "tbody",
            {},
            r.modes.map((m) => {
              const counted = m.imaginary && Math.abs(m.frequency_cm1) >= floor;
              return el(
                "tr",
                { class: m.imaginary ? (counted ? "imag-row" : "imag-soft-row") : "", dataset: { imaginary: String(!!m.imaginary) } },
                el("td", {}, String(m.index)),
                el("td", {}, m.imaginary ? `${Math.abs(m.frequency_cm1).toFixed(2)}i` : m.frequency_cm1.toFixed(2)),
                el("td", {}, fmtNum(m.energy_meV, 5)),
                el("td", {}, m.imaginary ? (counted ? "yes" : `yes (below ${floor} cm⁻¹ floor)`) : ""),
              );
            }),
          ),
        ),
      ),
    );
  }

  // ------------------------------------------------------------- files & reproduce

  async _loadFiles() {
    const jobId = this.detail.job_id;
    try {
      const { files } = await api.files(jobId);
      if (this.jobId !== jobId) return;
      clear(this.filesList);
      if (!files.length) this.filesList.append(el("li", { class: "muted" }, "No files yet."));
      for (const f of files) {
        const name = f.path;
        const remoteOnly = !!f.remote_only;
        const where = remoteOnly ? el("span", { class: "muted small remote-only", testid: "file-remote-only" }, ` on ${this.detail.target ? this.detail.target.machine_name : "the remote machine"}, not copied `) : null;
        const view = TEXT_EXT.test(name) ? el("a", { href: fileUrl(jobId, name, true), target: "_blank", rel: "noopener", class: "small" }, "view") : null;
        this.filesList.append(
          el(
            "li",
            { testid: "file-item", dataset: { path: name, remoteOnly: String(remoteOnly) } },
            el("a", { href: fileUrl(jobId, name), download: name.split("/").pop() }, name),
            el("span", { class: "muted small" }, ` ${fmtBytes(f.size)} `),
            where,
            view,
          ),
        );
      }
    } catch (err) {
      if (this.jobId !== jobId) return;
      clear(this.filesList).append(el("li", { class: "error-text" }, `Could not list files: ${err.message}`));
    }
  }

  _renderReproduce() {
    const rp = this.detail.reproduce;
    const p = clear(this.reproducePanel);
    p.append(el("h3", {}, "Reproduce with the mlip CLI"));
    if (!rp) {
      p.append(el("div", { class: "muted small" }, "reproduce.sh is written when the job starts."));
      return;
    }
    const meaning = {
      exact: "The script re-runs this calculation with the mlip command line.",
      approximate: "The script runs an equivalent CLI calculation, with the differences listed below.",
      no: "There is no CLI equivalent (see notes); the script explains why and exits.",
    };
    p.append(
      el("div", {}, el("span", { class: `badge repro-${rp.status}` }, rp.status), " ", el("span", { class: "small" }, meaning[rp.status] || "")),
      rp.notes && rp.notes.length ? el("ul", { class: "small" }, rp.notes.map((n) => el("li", {}, n))) : null,
      el(
        "div",
        { class: "small" },
        el("a", { href: fileUrl(this.detail.job_id, "reproduce.sh", true), target: "_blank", rel: "noopener" }, "view reproduce.sh"),
        " · ",
        el("a", { href: fileUrl(this.detail.job_id, "reproduce.sh"), download: "reproduce.sh" }, "download"),
      ),
    );
  }
}
