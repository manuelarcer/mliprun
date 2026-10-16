// "New calculation" form, generated from GET /api/calc-types (PLAN 3.1) and
// GET /api/calculators (PLAN 7.1). No calc parameter names are hard-coded:
// everything below `fields` comes from the FieldSpec list.
import { api, ApiError } from "./api.js";
import { el, clear, debounce } from "./util.js";

// ---------------------------------------------------------------- FieldSpec logic

/** Parse an indices string ("0,1,5", "12-30", mixed). Returns {ok, values|error}. */
export function parseIndices(text) {
  const s = String(text).replace(/\s+/g, "");
  if (!s) return { ok: false, error: "enter at least one index" };
  const out = [];
  for (const tok of s.split(",")) {
    if (tok === "") return { ok: false, error: "empty item (check the commas)" };
    let m = tok.match(/^(\d+)$/);
    if (m) {
      out.push(parseInt(m[1], 10));
      continue;
    }
    m = tok.match(/^(\d+)-(\d+)$/);
    if (m) {
      const a = parseInt(m[1], 10);
      const b = parseInt(m[2], 10);
      if (b < a) return { ok: false, error: `range ${tok} is reversed` };
      for (let i = a; i <= b; i++) out.push(i);
      continue;
    }
    return { ok: false, error: `"${tok}" is not an index or a range like 12-30` };
  }
  return { ok: true, values: out };
}

function boundText(f) {
  const u = f.unit ? ` ${f.unit}` : "";
  const parts = [];
  if (f.min !== null && f.min !== undefined) parts.push(`${f.min_exclusive ? ">" : "≥"} ${f.min}${u}`);
  if (f.max !== null && f.max !== undefined) parts.push(`${f.max_exclusive ? "<" : "≤"} ${f.max}${u}`);
  return parts.join(" and ");
}

/** Validate one value against its FieldSpec. Returns an error string or null. */
export function validateValue(f, v, ctx = {}) {
  if (v === null) return f.nullable ? null : "required";
  switch (f.type) {
    case "float":
    case "int": {
      if (typeof v !== "number" || Number.isNaN(v)) return "must be a number";
      if (f.type === "int" && !Number.isInteger(v)) return "must be a whole number";
      const lo = f.min,
        hi = f.max;
      const badLo = lo !== null && lo !== undefined && (f.min_exclusive ? !(v > lo) : !(v >= lo));
      const badHi = hi !== null && hi !== undefined && (f.max_exclusive ? !(v < hi) : !(v <= hi));
      if (badLo || badHi) return `must be ${boundText(f)}`;
      return null;
    }
    case "bool":
      return typeof v === "boolean" ? null : "must be true or false";
    case "choice":
      return (f.choices || []).includes(v) ? null : `must be one of ${(f.choices || []).join(", ")}`;
    case "indices": {
      const r = parseIndices(v);
      if (!r.ok) return r.error;
      if (ctx.natoms) {
        const bad = r.values.filter((i) => i >= ctx.natoms);
        if (bad.length) return `index ${bad[0]} is out of range (structure has ${ctx.natoms} atoms: 0–${ctx.natoms - 1})`;
      }
      return null;
    }
    case "mask3":
      return /^[01],[01],[01]$/.test(String(v).replace(/\s+/g, "")) ? null : 'must be three 0/1 values, e.g. "1,1,0"';
    default:
      return null;
  }
}

function sameValue(a, b) {
  if (a === b) return true;
  if (typeof a === "string" && typeof b === "string") return a.trim() === b.trim();
  return false;
}

/** Evaluate visible_if / input_modes in field order (PLAN 3.1 "Effective value"). */
export function computeVisibility(fields, readValue, mode) {
  const eff = {};
  const visible = {};
  for (const f of fields) {
    let vis = true;
    if (f.input_modes && !f.input_modes.includes(mode)) vis = false;
    if (vis && f.visible_if) {
      vis = f.visible_if.some((clause) =>
        Object.entries(clause).every(([name, allowed]) => allowed.some((a) => sameValue(a, eff[name]))),
      );
    }
    visible[f.name] = vis;
    let v = f.default;
    if (vis) {
      const r = readValue(f);
      if (r.ok) v = r.value;
    }
    eff[f.name] = v;
  }
  return visible;
}

// ---------------------------------------------------------------- run naming (PLAN 15.1)

/** Server slug rule for run folder names (PLAN 15.1). */
export function slugify(text) {
  return String(text ?? "")
    .replace(/[^A-Za-z0-9._+-]/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[.-]+/, "")
    .slice(0, 100);
}

/** Zero-padded suffix for job k of n (width max(2, len(str(n-1)))). */
export function jobSuffix(k, n) {
  return String(k).padStart(Math.max(2, String(n - 1).length), "0");
}

/**
 * Per-job names for one submission (PLAN 15.1).
 * files: list of default labels (one per job; NEB is one job).
 * multi: true for a multi-structure submit (N >= 2 single-input jobs).
 * Returns [{defaultLabel, label, folder}] where folder null = job_id (assigned by the server).
 */
export function planNames(files, { label = "", folder = "", multi = false } = {}) {
  const lab = String(label || "").trim();
  const fol = String(folder || "").trim();
  const n = files.length;
  if (!multi || n < 2) {
    return files.map((f) => ({
      defaultLabel: f,
      label: lab || f,
      folder: fol ? slugify(fol) : lab ? slugify(lab) : null,
    }));
  }
  const base = fol || (lab ? slugify(lab) : "");
  return files.map((f, k) => {
    const kk = jobSuffix(k, n);
    return {
      defaultLabel: f,
      label: lab ? `${lab}-${kk}` : f,
      folder: base ? slugify(`${base}-${kk}`) : null,
    };
  });
}

function joinPath(dir, name) {
  if (!dir) return name;
  return dir.endsWith("/") ? dir + name : `${dir}/${name}`;
}

function lsGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function lsSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable (private window, blocked site data) */
  }
}

const LOCATION_KEY = "mliprun.lastLocation";

// ---------------------------------------------------------------- widgets

function makeInput(f) {
  switch (f.type) {
    case "bool": {
      const i = el("input", { type: "checkbox", name: f.name, id: `param-${f.name}` });
      i.checked = !!f.default;
      return i;
    }
    case "choice": {
      const s = el(
        "select",
        { name: f.name, id: `param-${f.name}` },
        (f.choices || []).map((c, k) => el("option", { value: String(k) }, String(c))),
      );
      const k = (f.choices || []).findIndex((c) => c === f.default);
      s.value = String(Math.max(0, k));
      return s;
    }
    case "float":
    case "int": {
      const attrs = { type: "number", name: f.name, id: `param-${f.name}`, step: f.type === "int" ? "1" : "any", inputmode: "decimal" };
      if (f.min !== null && f.min !== undefined && !f.min_exclusive) attrs.min = String(f.min);
      if (f.max !== null && f.max !== undefined && !f.max_exclusive) attrs.max = String(f.max);
      if (f.nullable) attrs.placeholder = "auto (empty)";
      const i = el("input", attrs);
      i.value = f.default === null || f.default === undefined ? "" : String(f.default);
      return i;
    }
    default: {
      const i = el("input", {
        type: "text",
        name: f.name,
        id: `param-${f.name}`,
        spellcheck: "false",
        placeholder: f.type === "indices" ? (f.nullable ? "empty = default; e.g. 0,1,5 or 12-30" : "e.g. 0,1,5 or 12-30") : "x,y,z",
      });
      i.value = f.default === null || f.default === undefined ? "" : String(f.default);
      return i;
    }
  }
}

function readInput(f, input) {
  switch (f.type) {
    case "bool":
      return { ok: true, value: input.checked };
    case "choice": {
      const k = parseInt(input.value, 10);
      return { ok: true, value: (f.choices || [])[k] };
    }
    case "float":
    case "int": {
      const raw = input.value.trim();
      if (raw === "") {
        // A type=number input reports "" for unparsable text too.
        if (input.validity && input.validity.badInput) return { ok: false, error: "must be a number" };
        return { ok: true, value: null };
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) return { ok: false, error: "must be a number" };
      return { ok: true, value: n };
    }
    default: {
      const raw = input.value.trim();
      return { ok: true, value: raw === "" ? null : raw };
    }
  }
}

function writeInput(f, input, value) {
  if (f.type === "bool") input.checked = !!value;
  else if (f.type === "choice") input.value = String(Math.max(0, (f.choices || []).findIndex((c) => c === value)));
  else input.value = value === null || value === undefined ? "" : String(value);
}

// ---------------------------------------------------------------- the form

export class CalcForm {
  constructor(root, { calcTypes, calculators, structures, onSubmitted }) {
    this.root = root;
    this.calcTypes = calcTypes.calc_types;
    this.calcInfo = calculators;
    this.structures = structures;
    this.onSubmitted = onSubmitted;
    this.saved = {}; // calc_type -> {field: value} remembered across type switches
    this.widgets = new Map(); // name -> {spec, wrap, input}
    this.nebMode = "pair";
    this.showAdvanced = false;
    this._build();
    structures.onChange(() => this._renderInputs());
  }

  get calc() {
    return this.calcTypes.find((c) => c.name === this.typeSelect.value);
  }

  get calculator() {
    return this.calcInfo.calculators.find((c) => c.name === this.calcSelect.value);
  }

  get inputMode() {
    const kind = this.calc.inputs;
    if (kind === "single") return "single";
    if (kind === "pair") return "pair";
    return this.nebMode;
  }

  _build() {
    this.typeSelect = el(
      "select",
      { testid: "calc-type-select", id: "calc-type-select" },
      this.calcTypes.map((c) => el("option", { value: c.name }, c.label)),
    );
    this.typeDesc = el("div", { class: "muted small" });
    this.typeSelect.addEventListener("change", () => this._onTypeChange());

    // Calculator block (the only hard-coded part besides structure_every).
    this.calcSelect = el(
      "select",
      { testid: "calculator-select", id: "calculator-select" },
      this.calcInfo.calculators.map((c) => el("option", { value: c.name }, c.label || c.name)),
    );
    const def = this.calcInfo.default;
    if (this.calcInfo.calculators.some((c) => c.name === def)) this.calcSelect.value = def;
    this.calcNote = el("div", { class: "muted small" });
    this.taskSelect = el("select", { testid: "task-select", id: "task-select" });
    this.taskWrap = el("div", { class: "field hidden" }, el("label", { for: "task-select" }, "Task / head"), this.taskSelect);
    const dev = this.calcInfo.devices || { choices: ["auto", "cpu", "cuda"], default: "auto" };
    this.deviceSelect = el(
      "select",
      { testid: "device-select", id: "device-select" },
      dev.choices.map((d) => {
        let label = d;
        let disabled = false;
        if (d === "cuda" && dev.cuda_available === false) {
          label = "cuda (not available)";
          disabled = true;
        } else if (d === "cuda" && dev.cuda_available === null) label = "cuda (detection pending)";
        if (d === "auto") label = "auto (GPU if available)";
        return el("option", { value: d, disabled }, label);
      }),
    );
    this.deviceSelect.value = dev.default || "auto";
    this.calcSelect.addEventListener("change", () => this._onCalculatorChange());

    this.inputsBox = el("div", { class: "inputs-box" });
    this.fieldsBox = el("div", { class: "fields" });
    this.advToggle = el("button", { type: "button", class: "btn-link advanced-toggle", testid: "advanced-toggle", "aria-expanded": "false" }, "");
    this.advBox = el("div", { class: "fields advanced hidden" });
    this.advToggle.addEventListener("click", () => {
      this.showAdvanced = !this.showAdvanced;
      this._applyAdvanced();
    });

    this.everyInput = el("input", {
      type: "number",
      min: "0",
      max: "100000",
      step: "1",
      value: "5",
      id: "structure-every",
      testid: "structure-every",
    });
    this.labelInput = el("input", { type: "text", maxlength: "80", id: "job-label", testid: "job-name", spellcheck: "false", placeholder: "default: file name" });
    this.locationInput = el("input", { type: "text", id: "job-location", testid: "job-location", spellcheck: "false", placeholder: "server runs directory" });
    this.folderInput = el("input", { type: "text", maxlength: "100", id: "job-folder", testid: "job-folder", spellcheck: "false", placeholder: "job id (automatic)" });
    this.nameLabel = el("label", { for: "job-label" }, "Job name");
    this.nameWrap = el("div", { class: "field", testid: "field-job-name" }, this.nameLabel, this.labelInput);
    this.locationStatus = el("div", { class: "muted small path-status" });
    this.locationWrap = el(
      "div",
      { class: "field", testid: "field-job-location", title: "Parent directory of the run folder (absolute, or starting with ~). Created if missing." },
      el("label", { for: "job-location" }, "Location"),
      this.locationInput,
      this.locationStatus,
    );
    this.folderStatus = el("div", { class: "muted small path-status" });
    this.folderWrap = el(
      "div",
      { class: "field", testid: "field-job-folder", title: "Name of the run folder inside Location. Existing folders are never overwritten." },
      el("label", { for: "job-folder" }, "Folder name"),
      this.folderInput,
      this.folderStatus,
    );
    this.namePreview = el("div", { class: "name-preview hidden", testid: "name-preview" });
    const storedLoc = lsGet(LOCATION_KEY);
    if (storedLoc) this.locationInput.value = storedLoc;
    this.pathCheck = { seq: 0, data: null, key: null, unavailable: false };
    this.checkPathsSoon = debounce(() => this._checkPaths(), 300);
    for (const i of [this.labelInput, this.locationInput, this.folderInput]) {
      i.addEventListener("input", () => {
        this._setJobFieldError(i, null);
        this._updateNaming();
      });
    }

    this.submitBtn = el("button", { type: "submit", class: "btn btn-primary", testid: "submit-button" }, "Submit");
    this.submitReason = el("div", { class: "submit-reason small hidden", testid: "submit-blocked-reason", role: "status" });
    this.submitError = el("div", { class: "submit-error hidden", testid: "submit-error", role: "alert" });
    this.submitOk = el("div", { class: "submit-ok muted small", "aria-live": "polite" });

    this.form = el(
      "form",
      { class: "calc-form", novalidate: true },
      el("div", { class: "field" }, el("label", { for: "calc-type-select" }, "Calculation"), this.typeSelect, this.typeDesc),
      el(
        "fieldset",
        {},
        el("legend", {}, "Calculator"),
        el("div", { class: "field" }, el("label", { for: "calculator-select" }, "Model"), this.calcSelect, this.calcNote),
        this.taskWrap,
        el("div", { class: "field" }, el("label", { for: "device-select" }, "Device"), this.deviceSelect),
      ),
      el("fieldset", {}, el("legend", {}, "Input structures"), this.inputsBox),
      el("fieldset", {}, el("legend", {}, "Parameters"), this.fieldsBox, this.advToggle, this.advBox),
      el(
        "fieldset",
        {},
        el("legend", {}, "Job"),
        el(
          "div",
          { class: "field", title: "Attach atom positions to every N-th live event (0 = never). Smaller = smoother live view, bigger event log." },
          el("label", { for: "structure-every" }, "Live structure every"),
          el("div", { class: "input-unit" }, this.everyInput, el("span", { class: "unit" }, "events")),
        ),
        this.nameWrap,
        this.locationWrap,
        this.folderWrap,
        this.namePreview,
      ),
      this.submitError,
      this.submitReason,
      el("div", { class: "submit-row" }, this.submitBtn, this.submitOk),
    );
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.submit();
    });
    this.root.append(el("h2", {}, "New calculation"), this.form);
    this._onCalculatorChange();
    this._onTypeChange(true);
  }

  _onCalculatorChange() {
    const c = this.calculator;
    clear(this.taskSelect);
    if (c && c.task_field) {
      if (!c.task_required) this.taskSelect.append(el("option", { value: "" }, "(none)"));
      for (const t of c.task_choices || []) this.taskSelect.append(el("option", { value: t }, t));
      this.taskWrap.classList.remove("hidden");
      this.taskWrap.querySelector("label").textContent = c.task_field.replace(/_/g, " ");
    } else {
      this.taskWrap.classList.add("hidden");
    }
    const notes = [];
    if (c && c.note) notes.push(c.note);
    if (c && c.elements) notes.push(`Elements: ${c.elements.join(", ")}.`);
    if (c && c.cli_reproducible === false) notes.push("No mlip CLI equivalent, so reproduce.sh will not be runnable.");
    if (this.calcInfo.note) notes.push(this.calcInfo.note);
    this.calcNote.textContent = notes.join(" ");
  }

  _currentValues() {
    const out = {};
    for (const [name, w] of this.widgets) {
      const r = readInput(w.spec, w.input);
      if (r.ok) out[name] = r.value;
    }
    return out;
  }

  _onTypeChange(initial = false) {
    if (!initial && this.prevType) this.saved[this.prevType] = this._currentValues();
    const calc = this.calc;
    this.prevType = calc.name;
    this.typeDesc.textContent = calc.description || "";
    this.widgets.clear(); // before removing the DOM: removal can fire "change"
    clear(this.fieldsBox);
    clear(this.advBox);
    const saved = this.saved[calc.name] || {};
    for (const f of calc.fields) {
      const input = makeInput(f);
      if (f.name in saved) writeInput(f, input, saved[f.name]);
      const tipParts = [f.help, f.cli_flag ? `CLI: ${f.cli_flag}` : null, boundText(f) ? `Allowed: ${boundText(f)}` : null];
      const tip = tipParts.filter(Boolean).join("\n");
      const labelEl = el("label", { for: `param-${f.name}`, title: tip }, f.label || f.name);
      const infoEl = el("span", { class: "info", title: tip, "aria-label": tip }, "ⓘ");
      const body =
        f.type === "bool"
          ? el("div", { class: "input-unit checkbox-row" }, input, labelEl, infoEl)
          : el("div", { class: "input-unit" }, input, f.unit ? el("span", { class: "unit" }, f.unit) : null, infoEl);
      const wrap = el(
        "div",
        { class: `field field-${f.type}`, testid: `field-${f.name}`, title: tip },
        f.type === "bool" ? null : labelEl,
        body,
      );
      // Removing a focused input during a rebuild fires "change"; ignore stale widgets.
      const onChange = () => {
        if (this.widgets.get(f.name)?.input === input) this._onFieldChange(f.name);
      };
      input.addEventListener("input", onChange);
      input.addEventListener("change", onChange);
      this.widgets.set(f.name, { spec: f, wrap, input });
      (f.advanced ? this.advBox : this.fieldsBox).append(wrap);
    }
    this._renderInputs();
    this._refreshVisibility();
    this._clearErrors();
  }

  _onFieldChange(name) {
    this._refreshVisibility();
    const w = this.widgets.get(name);
    if (w) this._setFieldError(name, null);
  }

  _refreshVisibility() {
    const calc = this.calc;
    const vis = computeVisibility(
      calc.fields,
      (f) => {
        const w = this.widgets.get(f.name);
        return w ? readInput(f, w.input) : { ok: false };
      },
      this.inputMode,
    );
    let nAdv = 0;
    for (const [name, w] of this.widgets) {
      w.visible = vis[name];
      w.wrap.classList.toggle("hidden", !vis[name]);
      if (vis[name] && w.spec.advanced) nAdv++;
    }
    this.nAdvancedVisible = nAdv;
    this._applyAdvanced();
  }

  _applyAdvanced() {
    const n = this.nAdvancedVisible || 0;
    this.advToggle.classList.toggle("hidden", n === 0);
    this.advToggle.textContent = `${this.showAdvanced ? "▾ Hide" : "▸ Show"} advanced parameters (${n})`;
    this.advToggle.setAttribute("aria-expanded", String(this.showAdvanced));
    this.advBox.classList.toggle("hidden", !this.showAdvanced || n === 0);
  }

  // ----------------------------------------------------------- input structures

  _renderInputs() {
    this._renderInputsInner();
    this._updateNaming();
  }

  _renderInputsInner() {
    const calc = this.calc;
    if (!calc) return;
    const prev = {
      initial: this.initialSelect?.value,
      final: this.finalSelect?.value,
      band: this.bandSelect?.value,
    };
    clear(this.inputsBox);
    const ups = this.structures.okUploads();
    if (calc.inputs === "single") {
      const sel = this.structures.selectedIds().map((id) => this.structures.get(id));
      this.inputsBox.append(
        el(
          "div",
          { class: "muted small" },
          sel.length
            ? `${sel.length} structure${sel.length > 1 ? "s" : ""} selected (one job each): ${sel.map((u) => u.formula).join(", ")}`
            : "Tick one or more structures in the list above. One job is submitted per ticked structure.",
        ),
      );
      this.initialSelect = this.finalSelect = this.bandSelect = null;
      return;
    }
    const opt = (u) => el("option", { value: u.id }, `${u.formula} — ${u.filename}${u.n_frames > 1 ? ` (${u.n_frames} frames)` : ""}`);
    const mkSelect = (testid, list, placeholder) =>
      el("select", { testid, id: testid, onchange: () => this._updateNaming() }, el("option", { value: "" }, placeholder), list.map(opt));
    if (calc.inputs === "pair_or_band") {
      const mkRadio = (mode, testid, text) => {
        const r = el("input", { type: "radio", name: "neb-mode", value: mode, testid, id: testid });
        r.checked = this.nebMode === mode;
        r.addEventListener("change", () => {
          if (r.checked) {
            this.nebMode = mode;
            this._renderInputs();
            this._refreshVisibility();
          }
        });
        return el("label", { class: "radio", for: testid }, r, text);
      };
      this.inputsBox.append(
        el(
          "div",
          { class: "radio-row" },
          mkRadio("pair", "neb-mode-pair", "Initial + final (interpolate)"),
          mkRadio("band", "neb-mode-band", "Starting band (multi-frame file)"),
        ),
      );
    }
    if (this.inputMode === "band") {
      const bands = ups.filter((u) => u.n_frames >= 3);
      this.bandSelect = mkSelect("neb-band-select", bands, bands.length ? "choose a band…" : "no upload with ≥ 3 frames");
      if (prev.band && bands.some((u) => u.id === prev.band)) this.bandSelect.value = prev.band;
      this.initialSelect = this.finalSelect = null;
      this.inputsBox.append(
        el("div", { class: "field" }, el("label", { for: "neb-band-select" }, "Band"), this.bandSelect),
        el("div", { class: "muted small" }, "All frames are used as the starting band: first = initial, last = final (3–52 frames). No interpolation or endpoint optimization."),
      );
    } else {
      this.initialSelect = mkSelect("neb-initial-select", ups, "choose initial…");
      this.finalSelect = mkSelect("neb-final-select", ups, "choose final…");
      if (prev.initial && ups.some((u) => u.id === prev.initial)) this.initialSelect.value = prev.initial;
      if (prev.final && ups.some((u) => u.id === prev.final)) this.finalSelect.value = prev.final;
      this.bandSelect = null;
      this.inputsBox.append(
        el("div", { class: "field" }, el("label", { for: "neb-initial-select" }, "Initial"), this.initialSelect),
        el("div", { class: "field" }, el("label", { for: "neb-final-select" }, "Final"), this.finalSelect),
        el("div", { class: "muted small" }, "For a multi-frame file the last frame is used."),
      );
    }
  }


  // ----------------------------------------------------------- job name / location / folder

  /** Default location from GET /api/health (used when nothing was remembered). */
  setDefaultLocation(runsDir) {
    this.runsDir = runsDir || null;
    if (runsDir) this.locationInput.placeholder = runsDir;
    if (runsDir && !this.locationInput.value.trim()) this.locationInput.value = runsDir;
    this._updateNaming();
  }

  /** Jobs this submission would create: [{file, defaultLabel}] and whether it is a multi submit. */
  _namingJobs() {
    const mode = this.inputMode;
    const ctx = this._structureContext();
    if (mode === "single") {
      const files = ctx.ups.filter(Boolean).map((u) => u.filename);
      return { files, defaults: files, multi: files.length >= 2 };
    }
    if (mode === "band") {
      return ctx.band ? { files: [ctx.band.filename], defaults: [ctx.band.filename], multi: false } : { files: [], defaults: [], multi: false };
    }
    if (ctx.initial && ctx.final) {
      return { files: [`${ctx.initial.filename} → ${ctx.final.filename}`], defaults: [`${ctx.initial.filename} -> ${ctx.final.filename}`], multi: false };
    }
    return { files: [], defaults: [], multi: false };
  }

  _locationValue() {
    return this.locationInput.value.trim();
  }

  _clientLocationError() {
    const loc = this._locationValue();
    if (loc && !/^(\/|~)/.test(loc)) return "Location must be an absolute path (start with / or ~).";
    return null;
  }

  _updateNaming() {
    if (!this.labelInput || !this.calc) return;
    const jobs = this._namingJobs();
    this.naming = { ...jobs, plan: planNames(jobs.defaults, { label: this.labelInput.value, folder: this.folderInput.value, multi: jobs.multi }) };
    const label = this.labelInput.value.trim();
    // Labels and placeholders
    this.nameLabel.textContent = jobs.multi ? "Common name" : "Job name";
    if (jobs.multi) {
      this.labelInput.placeholder = "none: each job is named after its file";
      this.folderInput.placeholder = label ? `${slugify(label)}-${jobSuffix(0, jobs.files.length)}, -${jobSuffix(1, jobs.files.length)}, …` : "job id of each job (automatic)";
    } else {
      this.labelInput.placeholder = jobs.defaults.length ? jobs.defaults[0] : "default: file name";
      this.folderInput.placeholder = label ? slugify(label) || "job id (automatic)" : "job id (automatic)";
    }
    this._renderNaming();
    this.checkPathsSoon();
  }

  async _checkPaths() {
    const plan = (this.naming && this.naming.plan) || [];
    const folders = plan.map((p) => p.folder).filter((f) => f);
    const location = this._locationValue() || null;
    if (this._clientLocationError()) {
      this.pathCheck.data = null;
      this._renderNaming();
      return;
    }
    if (this.pathCheck.unavailable) return;
    const key = JSON.stringify([location, folders]);
    const seq = ++this.pathCheck.seq;
    try {
      const data = await api.checkPaths(location, folders);
      if (seq !== this.pathCheck.seq) return; // a newer check is on its way
      this.pathCheck.data = data;
      this.pathCheck.key = key;
    } catch (err) {
      if (seq !== this.pathCheck.seq) return;
      this.pathCheck.data = null;
      // Older server without /api/paths/check: no live feedback, submit still works.
      if (err instanceof ApiError && (err.status === 404 || err.status === 405)) this.pathCheck.unavailable = true;
    }
    this._renderNaming();
  }

  /** Current check data if it matches the inputs on screen, else null. */
  _freshCheck() {
    const d = this.pathCheck.data;
    if (!d) return null;
    const plan = (this.naming && this.naming.plan) || [];
    const key = JSON.stringify([this._locationValue() || null, plan.map((p) => p.folder).filter((f) => f)]);
    return key === this.pathCheck.key ? d : null;
  }

  _locationProblem(check) {
    const client = this._clientLocationError();
    if (client) return client;
    if (!check) return null;
    if (check.error) return check.error;
    if (check.exists === true && check.writable === false) return `${check.location} is not writable.`;
    if (check.exists === false && check.writable === false) return `${check.location} does not exist and cannot be created.`;
    return null;
  }

  /** Rows with resolved paths and conflict flags. */
  _namingRows(check) {
    const plan = (this.naming && this.naming.plan) || [];
    const files = (this.naming && this.naming.files) || [];
    const loc = check ? check.location : this._locationValue() || this.runsDir || "";
    let k = 0;
    const rows = plan.map((p, i) => {
      const row = { file: files[i], label: p.label, folder: p.folder, path: null, exists: false, duplicate: false };
      if (p.folder) {
        const info = check && Array.isArray(check.folders) ? check.folders[k] : null;
        k++;
        if (info) {
          row.folder = info.folder;
          row.path = info.path;
          row.exists = !!info.exists;
        } else row.path = joinPath(loc, p.folder);
      } else row.path = joinPath(loc, "<job id>");
      return row;
    });
    const count = new Map();
    for (const r of rows) if (r.folder) count.set(r.folder, (count.get(r.folder) || 0) + 1);
    for (const r of rows) if (r.folder && count.get(r.folder) > 1) r.duplicate = true;
    return rows;
  }

  _renderNaming() {
    if (!this.naming) return;
    const check = this._freshCheck();
    const locProblem = this._locationProblem(check);
    const rows = this._namingRows(check);

    // Location status line
    this.locationWrap.classList.toggle("has-warning", !!locProblem);
    if (locProblem) {
      this.locationStatus.className = "small path-status path-bad";
      this.locationStatus.textContent = locProblem;
    } else if (check) {
      this.locationStatus.className = "muted small path-status";
      this.locationStatus.textContent = check.exists ? `→ ${check.location}` : `→ ${check.location} (will be created)`;
    } else {
      this.locationStatus.className = "muted small path-status";
      this.locationStatus.textContent = "";
    }

    // Single job: one path line under the folder input.
    const multi = this.naming.multi;
    if (!multi && rows.length === 1) {
      const r = rows[0];
      const bad = r.exists;
      this.folderStatus.className = `small path-status${bad ? " path-bad" : " muted"}`;
      this.folderStatus.textContent = bad ? `${r.path} already exists; choose another folder name (runs are never overwritten).` : `Run folder: ${r.path}`;
    } else {
      this.folderStatus.className = "muted small path-status";
      this.folderStatus.textContent = "";
    }

    // Multi: preview table
    clear(this.namePreview);
    this.namePreview.classList.toggle("hidden", !multi);
    if (multi) {
      const body = rows.map((r) => {
        const flag = r.exists ? "exists" : r.duplicate ? "duplicate" : "";
        return el(
          "tr",
          { class: flag ? "conflict-row" : "", dataset: { conflict: flag || "none" } },
          el("td", { class: "np-file" }, r.file),
          el("td", { class: "np-label" }, r.label),
          el("td", { class: "np-path", title: r.path }, r.path, flag ? el("span", { class: "badge status-failed np-flag" }, flag === "exists" ? "already exists" : "duplicate") : null),
        );
      });
      this.namePreview.append(
        el("div", { class: "muted small" }, `${rows.length} jobs will be created:`),
        el(
          "div",
          { class: "table-scroll" },
          el("table", { class: "data-table" }, el("thead", {}, el("tr", {}, el("th", {}, "File"), el("th", {}, "Job name"), el("th", {}, "Folder"))), el("tbody", {}, body)),
        ),
      );
    }

    // Gate the submit button.
    const reasons = [];
    if (locProblem) reasons.push(`Location: ${locProblem}`);
    const nExist = rows.filter((r) => r.exists).length;
    const nDup = rows.filter((r) => r.duplicate).length;
    if (nExist) reasons.push(nExist === 1 && !multi ? "The run folder already exists. Choose another folder name." : `${nExist} run folder${nExist > 1 ? "s" : ""} already exist${nExist > 1 ? "" : "s"}. Choose another name or folder.`);
    if (nDup) reasons.push(`${nDup} jobs would share a folder name.`);
    this.blockReason = reasons.length ? reasons.join(" ") : null;
    this._applyGate();
  }

  _applyGate() {
    this.submitReason.textContent = this.blockReason ? `Cannot submit: ${this.blockReason}` : "";
    this.submitReason.classList.toggle("hidden", !this.blockReason);
    if (!this.submitting) this.submitBtn.disabled = !!this.blockReason;
  }

  _jobFieldWrap(name) {
    return { label: this.nameWrap, location: this.locationWrap, folder: this.folderWrap }[name] || null;
  }

  _setJobFieldError(inputOrName, message) {
    const name = typeof inputOrName === "string" ? inputOrName : inputOrName === this.labelInput ? "label" : inputOrName === this.locationInput ? "location" : "folder";
    const wrap = this._jobFieldWrap(name);
    if (!wrap) return false;
    const old = wrap.querySelector('[data-testid="field-error"]');
    if (old) old.remove();
    wrap.classList.toggle("has-error", !!message);
    if (message) wrap.append(el("div", { class: "field-error", testid: "field-error" }, message));
    return true;
  }

  // ----------------------------------------------------------- errors

  _clearErrors() {
    for (const [name] of this.widgets) this._setFieldError(name, null);
    if (this.nameWrap) for (const n of ["label", "location", "folder"]) this._setJobFieldError(n, null);
    this.submitError.classList.add("hidden");
    clear(this.submitError);
  }

  _setFieldError(name, message) {
    const w = this.widgets.get(name);
    if (!w) return false;
    const old = w.wrap.querySelector('[data-testid="field-error"]');
    if (old) old.remove();
    w.wrap.classList.toggle("has-error", !!message);
    if (message) w.wrap.append(el("div", { class: "field-error", testid: "field-error" }, message));
    return true;
  }

  _showSubmitErrors(title, items) {
    clear(this.submitError);
    this.submitError.append(el("div", { class: "submit-error-title" }, title));
    if (items.length) this.submitError.append(el("ul", {}, items.map((t) => el("li", {}, t))));
    this.submitError.classList.remove("hidden");
  }

  // ----------------------------------------------------------- submission

  _structureContext() {
    const calc = this.calc;
    const mode = this.inputMode;
    if (mode === "single") {
      const ids = this.structures.selectedIds();
      const ups = ids.map((id) => this.structures.get(id));
      return { ids, ups, natoms: ups.length ? Math.min(...ups.map((u) => u.natoms)) : null };
    }
    if (mode === "band") {
      const u = this.bandSelect && this.bandSelect.value ? this.structures.get(this.bandSelect.value) : null;
      return { ups: u ? [u] : [], band: u, natoms: u ? u.natoms : null };
    }
    const ini = this.initialSelect && this.initialSelect.value ? this.structures.get(this.initialSelect.value) : null;
    const fin = this.finalSelect && this.finalSelect.value ? this.structures.get(this.finalSelect.value) : null;
    return { ups: [ini, fin].filter(Boolean), initial: ini, final: fin, natoms: ini ? ini.natoms : null, calc };
  }

  /** Build the POST /api/jobs body; returns {payload} or {fieldErrors, errors}. */
  buildPayload() {
    const calc = this.calc;
    const ctx = this._structureContext();
    const fieldErrors = {};
    const errors = [];
    const params = {};
    for (const f of calc.fields) {
      const w = this.widgets.get(f.name);
      if (!w.visible) continue; // hidden fields are never sent (PLAN D4)
      const r = readInput(f, w.input);
      if (!r.ok) {
        fieldErrors[f.name] = r.error;
        continue;
      }
      const err = validateValue(f, r.value, { natoms: ctx.natoms });
      if (err) {
        fieldErrors[f.name] = err;
        continue;
      }
      // Only values that differ from the default are sent, so the server
      // records them as "user" and the rest as "default" in param_sources.
      const isDefault = r.value === f.default || (typeof r.value === "string" && r.value === String(f.default ?? ""));
      if (!isDefault) params[f.name] = r.value;
    }

    const c = this.calculator;
    const calculator = { name: c.name, device: this.deviceSelect.value };
    if (c.task_field) {
      const t = this.taskSelect.value;
      if (t) calculator[c.task_field] = t;
      else if (c.task_required) errors.push(`${c.name} needs a ${c.task_field.replace(/_/g, " ")}.`);
    }

    const every = this.everyInput.value.trim() === "" ? 5 : Number(this.everyInput.value);
    if (!Number.isInteger(every) || every < 0 || every > 100000) errors.push("Live structure every: whole number 0–100000.");

    const payload = { calc_type: calc.name, calculator, params, structure_every: every };
    const label = this.labelInput.value.trim();
    if (label) payload.label = label;
    const location = this._locationValue();
    if (location) payload.location = location;
    const folder = this.folderInput.value.trim();
    if (folder) payload.folder = folder;
    const locErr = this._clientLocationError();
    if (locErr) errors.push(locErr);

    const mode = this.inputMode;
    if (mode === "single") {
      if (!ctx.ids.length) errors.push("Select at least one structure (tick it in the Structures list).");
      if (ctx.ids.length > 200) errors.push("At most 200 structures per submission.");
      payload.structure_ids = ctx.ids;
    } else if (mode === "band") {
      if (!ctx.band) errors.push("Choose a band upload (a multi-frame file with ≥ 3 frames).");
      else {
        payload.band_id = ctx.band.id;
        if (ctx.band.n_frames > 52) errors.push(`The band has ${ctx.band.n_frames} frames; at most 52 are allowed.`);
      }
    } else {
      if (!ctx.initial) errors.push("Choose the initial structure.");
      if (!ctx.final) errors.push("Choose the final structure.");
      if (ctx.initial && ctx.final) {
        payload.initial_id = ctx.initial.id;
        payload.final_id = ctx.final.id;
        if (ctx.initial.id === ctx.final.id) errors.push("Initial and final are the same upload.");
        if (ctx.initial.natoms !== ctx.final.natoms)
          errors.push(`Initial has ${ctx.initial.natoms} atoms but final has ${ctx.final.natoms}.`);
        else if (ctx.initial.symbols && ctx.final.symbols && ctx.initial.symbols.join() !== ctx.final.symbols.join())
          errors.push("Initial and final have different chemical symbols or atom order.");
      }
    }

    if (c.elements) {
      const allowed = new Set(c.elements);
      const bad = new Set();
      for (const u of ctx.ups) for (const s of u.symbols || []) if (!allowed.has(s)) bad.add(s);
      if (bad.size) errors.push(`${c.name} does not support element(s) ${[...bad].join(", ")}.`);
    }
    return { payload, fieldErrors, errors };
  }

  async submit() {
    this._clearErrors();
    this.submitOk.textContent = "";
    const { payload, fieldErrors, errors } = this.buildPayload();
    const nField = Object.keys(fieldErrors).length;
    for (const [name, msg] of Object.entries(fieldErrors)) this._setFieldError(name, msg);
    if (nField && Object.keys(fieldErrors).some((n) => this.widgets.get(n).spec.advanced)) {
      this.showAdvanced = true;
      this._applyAdvanced();
    }
    if (nField || errors.length) {
      this._showSubmitErrors(
        `Please fix ${nField + errors.length} problem${nField + errors.length > 1 ? "s" : ""} before submitting.`,
        nField ? [...errors, `${nField} parameter${nField > 1 ? "s are" : " is"} invalid (marked in red).`] : errors,
      );
      return;
    }
    if (this.blockReason) return;
    this.submitting = true;
    this.submitBtn.disabled = true;
    this.submitBtn.textContent = "Submitting…";
    try {
      const { jobs } = await api.submitJob(payload);
      this.submitOk.textContent = `Submitted ${jobs.length} job${jobs.length > 1 ? "s" : ""}.`;
      if (payload.location) lsSet(LOCATION_KEY, payload.location);
      // The folders just created now exist on disk: clear Job name and Folder
      // name (Location is kept) so the next submission starts from fresh,
      // non-conflicting defaults instead of flagging its own folders.
      this.labelInput.value = "";
      this.folderInput.value = "";
      this._updateNaming();
      if (this.onSubmitted) this.onSubmitted(jobs);
    } catch (err) {
      this._handleSubmitError(err, payload);
    } finally {
      this.submitting = false;
      this.submitBtn.textContent = "Submit";
      this._applyGate();
      // The folders just used now exist: refresh the live check.
      this.checkPathsSoon();
      setTimeout(() => this.checkPathsSoon(), 1500);
    }
  }

  /** Place a 422 detail for label/location/folder next to its input. Returns true if handled. */
  _jobFieldError(d, payload) {
    const m = String(d.field || "").match(/^(?:structure_ids\[(\d+)\]\.)?(label|location|folder)$/);
    if (!m) return false;
    let msg = d.message;
    if (m[1] !== undefined && payload.structure_ids) {
      const u = this.structures.get(payload.structure_ids[parseInt(m[1], 10)]);
      if (u) msg = `${u.filename}: ${msg}`;
    }
    const wrap = this._jobFieldWrap(m[2]);
    const prev = wrap && wrap.querySelector('[data-testid="field-error"]');
    if (prev) msg = `${prev.textContent}; ${msg}`;
    this._setJobFieldError(m[2], msg);
    return true;
  }

  _handleSubmitError(err, payload) {
    if (!(err instanceof ApiError)) {
      this._showSubmitErrors(`Submission failed: ${err.message}`, []);
      return;
    }
    const other = [];
    for (const d of err.details || []) {
      const m = String(d.field || "").match(/^(?:structure_ids\[(\d+)\]\.)?params\.([A-Za-z0-9_]+)$/);
      if (m && this.widgets.has(m[2])) {
        let msg = d.message;
        if (m[1] !== undefined && payload.structure_ids) {
          const u = this.structures.get(payload.structure_ids[parseInt(m[1], 10)]);
          if (u) msg = `${u.filename}: ${msg}`;
        }
        this._setFieldError(m[2], msg);
        if (this.widgets.get(m[2]).spec.advanced) {
          this.showAdvanced = true;
          this._applyAdvanced();
        }
      } else if (this._jobFieldError(d, payload)) {
        /* shown next to the Job name / Location / Folder input */
      } else {
        other.push(d.field ? `${d.field}: ${d.message}` : d.message);
      }
    }
    this._showSubmitErrors(err.message || "Submission rejected", other);
  }
}

