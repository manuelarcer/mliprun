// "Machines" panel: SSH machines and their Python environments (REMOTE_PLAN section 9).
// Everything is stored by the server (machines file); this module only edits it.
import { api, ApiError } from "./api.js";
import { el, clear } from "./util.js";

const SETUP_HELP =
  "Shell lines run before every command on this machine for this environment, e.g. module load ..., export CUDA_VISIBLE_DEVICES=1";

const MACHINE_FIELDS = [
  { name: "name", label: "Name", kind: "text", help: "Letters, digits and . _ + - only." },
  { name: "host", label: "Host", kind: "text", help: "Host name or an alias from ~/.ssh/config." },
  { name: "user", label: "User", kind: "text", optional: true },
  { name: "port", label: "Port", kind: "int", optional: true },
  { name: "identity_file", label: "Identity file", kind: "text", optional: true, help: "Private key path (absolute or ~)." },
  { name: "jump_host", label: "Jump host", kind: "text", optional: true, help: "[user@]host[:port]" },
  { name: "default_location", label: "Default remote location", kind: "text", help: "Absolute path on the machine where run folders are created." },
  { name: "poll_interval_s", label: "Poll interval (s)", kind: "float", defaultValue: 2 },
  { name: "copy_back_max_mb", label: "Copy-back limit (MB)", kind: "int", defaultValue: 50, help: "Files larger than this stay on the machine (0 = copy nothing back)." },
];

/** Same rule as remote.default_env_name: `<machine>-<venv dir>`. */
export function defaultEnvName(machineName, python) {
  const parts = String(python || "")
    .trim()
    .split("/")
    .filter((p) => p !== "");
  if (parts.length < 2) return "";
  const parent = parts[parts.length - 2];
  const grand = parts.length >= 3 ? parts[parts.length - 3] : "";
  let venv = parent === "bin" && grand ? grand : parent;
  venv = venv.replace(/[^A-Za-z0-9._+-]/g, "-") || "python";
  return `${machineName}-${venv}`.slice(0, 60);
}

function fieldKey(d) {
  return String((d && d.field) || "").split(".").pop();
}

export class MachinesPanel {
  constructor({ onChange } = {}) {
    this.onChange = onChange || (() => {});
    this.machines = [];
    this.body = el("div", { class: "machines-body", testid: "machines-list" });
    this.loadError = el("div", { class: "error-text hidden" });
    this.root = el(
      "div",
      { class: "overlay hidden", testid: "machines-panel", role: "dialog", "aria-label": "Machines" },
      el(
        "div",
        { class: "overlay-card" },
        el(
          "div",
          { class: "panel-head" },
          el("h2", {}, "Machines"),
          el("button", { type: "button", class: "btn btn-small", testid: "machines-close", onclick: () => this.close() }, "Close"),
        ),
        el(
          "p",
          { class: "muted small" },
          "SSH machines and their Python environments (one per MLIP venv). Login must work without a password (key or an open ControlMaster). Settings are stored by the server, not in the browser.",
        ),
        this.loadError,
        this.body,
      ),
    );
    document.body.append(this.root);
    this.root.addEventListener("click", (e) => {
      if (e.target === this.root) this.close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !this.root.classList.contains("hidden")) this.close();
    });
  }

  async open() {
    this.root.classList.remove("hidden");
    await this.load();
  }

  close() {
    this.root.classList.add("hidden");
  }

  async load() {
    try {
      const { machines } = await api.listMachines();
      this.machines = machines;
      this.loadError.classList.add("hidden");
    } catch (err) {
      this.loadError.textContent = `Could not load machines: ${err.message}`;
      this.loadError.classList.remove("hidden");
      return;
    }
    this._render();
  }

  /** Reload from the server and tell the form to refresh its "Run on" options. */
  async _changed() {
    await this.load();
    this.onChange(this.machines);
  }

  _render() {
    clear(this.body);
    this.body.append(this._machineCard(null));
    for (const m of this.machines) this.body.append(this._machineCard(m));
    if (!this.machines.length) this.body.append(el("div", { class: "muted small", testid: "machines-empty" }, "No machines yet. Fill in the form above and press Save."));
  }

  // ------------------------------------------------------------ machine card

  _machineCard(m) {
    const isNew = m === null;
    const inputs = {};
    const wraps = {};
    const errBox = el("div", { class: "machine-error error-text small hidden", testid: "machine-error", role: "alert" });
    const form = el("form", { class: "machine-form", testid: "machine-form", novalidate: true, dataset: { machineId: isNew ? "" : m.id } });
    const grid = el("div", { class: "machine-grid" });
    for (const f of MACHINE_FIELDS) {
      const input = el("input", {
        type: f.kind === "text" ? "text" : "number",
        name: f.name,
        id: `machine-${isNew ? "new" : m.id}-${f.name}`,
        testid: `machine-${f.name}`,
        spellcheck: "false",
        step: f.kind === "float" ? "any" : f.kind === "int" ? "1" : null,
        placeholder: f.optional ? "optional" : f.defaultValue !== undefined ? String(f.defaultValue) : "",
      });
      const v = isNew ? null : m[f.name];
      input.value = v === null || v === undefined ? (isNew && f.defaultValue !== undefined ? String(f.defaultValue) : "") : String(v);
      input.addEventListener("input", () => this._setError(wraps[f.name], null));
      inputs[f.name] = input;
      wraps[f.name] = el("div", { class: "field", title: f.help || "" }, el("label", { for: input.id }, f.label), input);
      grid.append(wraps[f.name]);
    }
    const testBox = el("div", { class: "machine-test", testid: "machine-test-result" });
    const saveBtn = el("button", { type: "submit", class: "btn btn-primary", testid: "machine-save-button" }, isNew ? "Add machine" : "Save");
    const testBtn = isNew ? null : el("button", { type: "button", class: "btn", testid: "machine-test-button" }, "Test connection");
    const delBtn = isNew ? null : el("button", { type: "button", class: "btn btn-danger", testid: "machine-delete-button" }, "Delete");
    form.append(grid, el("div", { class: "machine-actions" }, saveBtn, testBtn, delBtn), errBox, testBox);

    const readBody = () => {
      const body = {};
      let ok = true;
      for (const f of MACHINE_FIELDS) {
        const raw = inputs[f.name].value.trim();
        if (f.kind === "text") {
          body[f.name] = raw === "" ? (f.optional ? null : "") : raw;
        } else if (raw === "") {
          if (f.optional) body[f.name] = null;
          else if (f.defaultValue !== undefined) body[f.name] = f.defaultValue;
        } else {
          const n = Number(raw);
          if (!Number.isFinite(n) || (f.kind === "int" && !Number.isInteger(n))) {
            this._setError(wraps[f.name], f.kind === "int" ? "must be a whole number" : "must be a number");
            ok = false;
          } else body[f.name] = n;
        }
      }
      return ok ? body : null;
    };

    const showErr = (err) => {
      errBox.classList.add("hidden");
      if (!(err instanceof ApiError)) {
        errBox.textContent = err.message;
        errBox.classList.remove("hidden");
        return;
      }
      let placed = 0;
      for (const d of err.details || []) {
        const key = fieldKey(d);
        if (wraps[key]) {
          this._setError(wraps[key], d.message);
          placed++;
        }
      }
      if (!placed || !(err.details || []).length) {
        errBox.textContent = err.message;
        errBox.classList.remove("hidden");
      }
    };

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      errBox.classList.add("hidden");
      for (const w of Object.values(wraps)) this._setError(w, null);
      const body = readBody();
      if (!body) return;
      saveBtn.disabled = true;
      try {
        if (isNew) await api.addMachine(body);
        else await api.updateMachine(m.id, body);
        await this._changed();
      } catch (err) {
        showErr(err);
      } finally {
        saveBtn.disabled = false;
      }
    });

    if (!isNew) {
      testBtn.addEventListener("click", async () => {
        testBtn.disabled = true;
        testBtn.textContent = "Testing…";
        clear(testBox);
        try {
          this._renderTest(testBox, await api.testMachine(m.id));
        } catch (err) {
          this._renderTest(testBox, { ok: false, message: err.message });
        } finally {
          testBtn.disabled = false;
          testBtn.textContent = "Test connection";
        }
      });
      delBtn.addEventListener("click", async () => {
        if (!window.confirm(`Delete machine ${m.name} and its environments?`)) return;
        errBox.classList.add("hidden");
        try {
          await api.deleteMachine(m.id);
          await this._changed();
        } catch (err) {
          errBox.textContent = err.message; // e.g. 409: still has unfinished jobs
          errBox.classList.remove("hidden");
        }
      });
    }

    const card = el("section", { class: "card machine-card", testid: "machine-card", dataset: { machineId: isNew ? "" : m.id } });
    card.append(el("h3", {}, isNew ? "Add machine" : m.name), form);
    if (!isNew) card.append(this._envSection(m, () => inputs.name.value.trim() || m.name));
    return card;
  }

  _setError(wrap, message) {
    if (!wrap) return;
    const old = wrap.querySelector('[data-testid="machine-field-error"]');
    if (old) old.remove();
    wrap.classList.toggle("has-error", !!message);
    if (message) wrap.append(el("div", { class: "field-error", testid: "machine-field-error" }, message));
  }

  _renderTest(box, r) {
    clear(box);
    if (!r.ok) {
      box.append(el("div", { class: "error-text", testid: "machine-test-error" }, `Connection failed: ${r.message}`));
      return;
    }
    box.append(
      el(
        "div",
        { class: "test-ok", testid: "machine-test-ok" },
        `Connected to `,
        el("strong", { testid: "machine-test-hostname" }, r.hostname),
        ` in ${r.elapsed_s} s`,
      ),
    );
    if (r.message && r.message.includes(";")) box.append(el("div", { class: "muted small" }, r.message));
    const gpus = r.gpus || [];
    if (gpus.length) {
      box.append(
        el(
          "table",
          { class: "data-table gpu-table", testid: "gpu-table" },
          el("thead", {}, el("tr", {}, ["Index", "Name", "Memory used (MB)", "Memory total (MB)", "Utilization (%)"].map((h) => el("th", {}, h)))),
          el(
            "tbody",
            {},
            gpus.map((g) =>
              el("tr", {}, el("td", {}, String(g.index)), el("td", {}, g.name), el("td", {}, String(g.memory_used_mb)), el("td", {}, String(g.memory_total_mb)), el("td", {}, String(g.utilization_pct))),
            ),
          ),
        ),
      );
    } else {
      box.append(el("div", { class: "muted small" }, "No GPUs reported."));
    }
  }

  // ------------------------------------------------------------ environments

  _envSection(m, getMachineName) {
    const sec = el("div", { class: "env-section" }, el("h4", {}, "Environments"));
    if (!m.envs.length) sec.append(el("div", { class: "muted small" }, "No environments yet."));
    for (const env of m.envs) sec.append(this._envRow(m, env));
    sec.append(this._envAdd(m, getMachineName));
    return sec;
  }

  _probeSummary(probe) {
    if (!probe) return el("span", { class: "muted" }, "not probed yet");
    if (!probe.ok) return el("span", { class: "error-text" }, `probe failed: ${probe.error}`);
    const cal = probe.calculators || {};
    const names = (cal.calculators || []).map((c) => c.name).join(", ") || "none";
    const cuda = cal.devices && cal.devices.cuda_available;
    return el(
      "span",
      {},
      `ok: mliprun ${probe.mliprun_version}, worker ok, CUDA ${cuda ? "yes" : "no"}, calculators: ${names}`,
    );
  }

  _envRow(m, env) {
    const err = el("div", { class: "error-text small hidden", testid: "env-error", role: "alert" });
    const nameIn = el("input", { type: "text", testid: "env-name", spellcheck: "false" });
    nameIn.value = env.name;
    const pyIn = el("input", { type: "text", testid: "env-python", spellcheck: "false" });
    pyIn.value = env.python;
    const setupIn = el("textarea", { rows: "3", testid: "env-setup", spellcheck: "false", title: SETUP_HELP, placeholder: "module load ..." });
    setupIn.value = env.setup || "";
    const status = el("div", { class: "env-probe small", testid: "env-probe-status", dataset: { ok: env.probe ? String(!!env.probe.ok) : "none" } }, this._probeSummary(env.probe));
    const wrapOf = (label, input) => el("div", { class: "field" }, el("label", {}, label), input);
    const nameW = wrapOf("Name", nameIn);
    const pyW = wrapOf("Python", pyIn);
    const setW = wrapOf("Setup", setupIn);
    const fieldWraps = { name: nameW, python: pyW, setup: setW };
    const probeBtn = el("button", { type: "button", class: "btn btn-small", testid: "env-probe-button" }, "Probe");
    const saveBtn = el("button", { type: "button", class: "btn btn-small", testid: "env-save-button" }, "Save");
    const delBtn = el("button", { type: "button", class: "btn btn-small btn-danger", testid: "env-delete-button" }, "Delete");
    const fail = (e) => {
      err.classList.add("hidden");
      let placed = 0;
      for (const d of (e instanceof ApiError && e.details) || []) {
        const w = fieldWraps[fieldKey(d)];
        if (w) {
          this._setError(w, d.message);
          placed++;
        }
      }
      if (!placed) {
        err.textContent = e.message;
        err.classList.remove("hidden");
      }
    };
    const clearErrs = () => {
      err.classList.add("hidden");
      for (const w of Object.values(fieldWraps)) this._setError(w, null);
    };
    probeBtn.addEventListener("click", async () => {
      clearErrs();
      probeBtn.disabled = true;
      probeBtn.textContent = "Probing…";
      try {
        await api.probeEnv(m.id, env.id);
        await this._changed();
      } catch (e) {
        fail(e);
        probeBtn.disabled = false;
        probeBtn.textContent = "Probe";
      }
    });
    saveBtn.addEventListener("click", async () => {
      clearErrs();
      saveBtn.disabled = true;
      try {
        await api.updateEnv(m.id, env.id, { name: nameIn.value.trim(), python: pyIn.value.trim(), setup: setupIn.value });
        await this._changed();
      } catch (e) {
        fail(e);
      } finally {
        saveBtn.disabled = false;
      }
    });
    delBtn.addEventListener("click", async () => {
      clearErrs();
      try {
        await api.deleteEnv(m.id, env.id);
        await this._changed();
      } catch (e) {
        fail(e);
      }
    });
    return el(
      "div",
      { class: "env-row", testid: "env-row", dataset: { envId: env.id, envName: env.name } },
      el("div", { class: "env-grid" }, nameW, pyW),
      setW,
      el("div", { class: "muted small" }, SETUP_HELP),
      status,
      el("div", { class: "machine-actions" }, probeBtn, saveBtn, delBtn),
      err,
    );
  }

  _envAdd(m, getMachineName) {
    const pyIn = el("input", { type: "text", testid: "env-python", id: `envadd-py-${m.id}`, spellcheck: "false", placeholder: "/path/to/venv/bin/python" });
    const nameIn = el("input", { type: "text", testid: "env-name", id: `envadd-name-${m.id}`, spellcheck: "false" });
    const setupIn = el("textarea", { rows: "2", testid: "env-setup", title: SETUP_HELP, placeholder: "module load ... (optional)" });
    const err = el("div", { class: "error-text small hidden", testid: "env-error", role: "alert" });
    const pyW = el("div", { class: "field" }, el("label", { for: pyIn.id }, "Python path"), pyIn);
    const nameW = el("div", { class: "field" }, el("label", { for: nameIn.id }, "Name"), nameIn);
    const setW = el("div", { class: "field" }, el("label", {}, "Setup"), setupIn);
    const wraps = { python: pyW, name: nameW, setup: setW };
    const addBtn = el("button", { type: "button", class: "btn btn-primary btn-small", testid: "env-add-button" }, "Add environment");
    let nameEdited = false;
    nameIn.addEventListener("input", () => {
      nameEdited = true;
      this._setError(nameW, null);
    });
    pyIn.addEventListener("input", () => {
      this._setError(pyW, null);
      if (!nameEdited) nameIn.value = defaultEnvName(getMachineName(), pyIn.value);
    });
    addBtn.addEventListener("click", async () => {
      err.classList.add("hidden");
      for (const w of Object.values(wraps)) this._setError(w, null);
      const body = { python: pyIn.value.trim() };
      if (nameIn.value.trim()) body.name = nameIn.value.trim();
      if (setupIn.value.trim()) body.setup = setupIn.value;
      addBtn.disabled = true;
      addBtn.textContent = "Adding and probing…";
      try {
        await api.addEnv(m.id, body); // the server probes before returning
        await this._changed();
      } catch (e) {
        let placed = 0;
        for (const d of (e instanceof ApiError && e.details) || []) {
          if (wraps[fieldKey(d)]) {
            this._setError(wraps[fieldKey(d)], d.message);
            placed++;
          }
        }
        if (!placed) {
          err.textContent = e.message;
          err.classList.remove("hidden");
        }
        addBtn.disabled = false;
        addBtn.textContent = "Add environment";
      }
    });
    return el("div", { class: "env-add", testid: "env-add" }, el("h4", {}, "Add environment"), el("div", { class: "env-grid" }, pyW, nameW), setW, addBtn, err);
  }
}
