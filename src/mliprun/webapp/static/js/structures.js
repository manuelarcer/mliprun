// Upload panel, structure list and previews (PLAN 4.2, 4.3, 11.1 "Structures panel").
import { api } from "./api.js";
import { el, clear, fmtNum, cellParams, cellVolume, pbcText, splitXyzFrames } from "./util.js";
import { viewerWithSlider } from "./viewer.js";

const FORMATS = [
  ["", "auto-detect"],
  ["vasp", "vasp (POSCAR/CONTCAR)"],
  ["extxyz", "extxyz"],
  ["cif", "cif"],
  ["traj", "traj (ASE trajectory)"],
  ["vasp-out", "vasp-out (OUTCAR)"],
];

export class StructuresPanel {
  constructor(root) {
    this.root = root;
    this.uploads = new Map(); // id -> UploadInfo (may lack xyz until fetched)
    this.failed = []; // failure entries from this session
    this.order = []; // ids, newest first
    this.selected = new Set();
    this.listeners = new Set();
    this.previewId = null;
    this.fullCache = new Map();
    this._build();
  }

  onChange(fn) {
    this.listeners.add(fn);
  }

  _emit() {
    for (const fn of this.listeners) fn();
  }

  _build() {
    const input = el("input", {
      type: "file",
      multiple: true,
      id: "upload-input",
      testid: "upload-input",
      class: "visually-hidden-file",
    });
    input.addEventListener("change", () => {
      if (input.files && input.files.length) this.uploadFiles([...input.files]);
      input.value = "";
    });
    this.formatSelect = el(
      "select",
      { testid: "upload-format", id: "upload-format", title: "ASE format applied to every file in this upload" },
      FORMATS.map(([v, l]) => el("option", { value: v }, l)),
    );
    const drop = el(
      "div",
      { class: "dropzone", testid: "upload-dropzone", tabindex: "0" },
      el("div", { class: "dropzone-title" }, "Drop structure files here"),
      el("div", { class: "muted" }, "POSCAR, CONTCAR, .xyz/.extxyz, .cif, .traj, OUTCAR … (anything ase.io.read understands)"),
      el("label", { class: "btn", for: "upload-input" }, "Choose files…"),
      input,
    );
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("dragging");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("dragging");
      const files = [...(e.dataTransfer?.files || [])];
      if (files.length) this.uploadFiles(files);
    });
    drop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") input.click();
    });

    this.status = el("div", { class: "upload-status muted", "aria-live": "polite" });
    this.list = el("ul", { class: "structure-list", testid: "structure-list" });
    this.previewBox = el("div", { class: "preview hidden" });

    this.root.append(
      el("h2", {}, "Structures"),
      drop,
      el("div", { class: "row-inline" }, el("label", { for: "upload-format" }, "Format"), this.formatSelect),
      this.status,
      this.list,
      this.previewBox,
    );
  }

  async loadExisting() {
    try {
      const { uploads } = await api.listUploads();
      for (const u of uploads) {
        if (!u.ok || !u.id) continue;
        if (!this.uploads.has(u.id)) {
          this.uploads.set(u.id, u);
          this.order.push(u.id);
        }
      }
      this._render();
      this._emit();
    } catch (err) {
      this.status.textContent = `Could not list previous uploads: ${err.message}`;
    }
  }

  async uploadFiles(files) {
    this.status.textContent = `Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`;
    this.status.classList.remove("error-text");
    try {
      const { uploads } = await api.upload(files, this.formatSelect.value || null);
      const ok = uploads.filter((u) => u.ok);
      const newIds = [];
      for (const u of uploads) {
        if (u.ok && u.id) {
          this.uploads.set(u.id, u);
          this.fullCache.set(u.id, u); // POST returns the full UploadInfo
          this.order = [u.id, ...this.order.filter((x) => x !== u.id)];
          this.selected.add(u.id);
          newIds.push(u.id);
        } else {
          this.failed.unshift({ ...u, _key: `${Date.now()}-${Math.random()}` });
        }
      }
      const nFail = uploads.length - ok.length;
      this.status.textContent =
        `${ok.length} structure${ok.length === 1 ? "" : "s"} uploaded` + (nFail ? `, ${nFail} failed (see below)` : "") + ".";
      if (nFail) this.status.classList.add("error-text");
      this._render();
      this._emit();
      if (newIds.length) this.preview(newIds[0]);
    } catch (err) {
      this.status.textContent = `Upload failed: ${err.message}`;
      this.status.classList.add("error-text");
    }
  }

  okUploads() {
    return this.order.map((id) => this.uploads.get(id)).filter(Boolean);
  }

  selectedIds() {
    return this.order.filter((id) => this.selected.has(id));
  }

  get(id) {
    return this.uploads.get(id) || null;
  }

  async getFull(id) {
    if (this.fullCache.has(id)) return this.fullCache.get(id);
    const full = await api.getUpload(id);
    this.fullCache.set(id, full);
    this.uploads.set(id, { ...this.uploads.get(id), ...full });
    return full;
  }

  _render() {
    clear(this.list);
    if (!this.order.length && !this.failed.length) {
      this.list.append(el("li", { class: "muted empty" }, "No structures yet. Upload one to start."));
    }
    for (const f of this.failed) {
      const dismiss = el("button", { type: "button", class: "btn-link", title: "Dismiss" }, "×");
      dismiss.addEventListener("click", (e) => {
        e.stopPropagation();
        this.failed = this.failed.filter((x) => x._key !== f._key);
        this._render();
      });
      this.list.append(
        el(
          "li",
          { class: "structure-item failed", testid: "structure-item", dataset: { uploadId: "", ok: "false" } },
          el("div", { class: "si-main" }, el("span", { class: "si-name" }, f.filename || "(unnamed)"), dismiss),
          el("div", { class: "structure-error error-text", testid: "structure-error" }, f.error || "upload failed"),
        ),
      );
    }
    for (const id of this.order) {
      const u = this.uploads.get(id);
      const cb = el("input", {
        type: "checkbox",
        testid: "structure-select",
        title: "Use this structure for the next submission (single-structure calculations)",
        "aria-label": `select ${u.filename}`,
      });
      cb.checked = this.selected.has(id);
      cb.addEventListener("click", (e) => e.stopPropagation());
      cb.addEventListener("change", () => {
        if (cb.checked) this.selected.add(id);
        else this.selected.delete(id);
        this._emit();
      });
      const meta = [
        `${u.natoms} atoms`,
        u.n_frames > 1 ? `${u.n_frames} frames` : null,
        `pbc ${pbcText(u.pbc)}`,
        u.format,
      ].filter(Boolean);
      const li = el(
        "li",
        {
          class: "structure-item" + (this.previewId === id ? " active" : ""),
          testid: "structure-item",
          dataset: { uploadId: id, ok: "true" },
          title: "Click to preview",
        },
        el(
          "div",
          { class: "si-main" },
          cb,
          el("span", { class: "si-formula" }, u.formula),
          el("span", { class: "si-name muted" }, u.filename),
        ),
        el("div", { class: "si-meta muted" }, meta.join(" · ")),
      );
      li.addEventListener("click", () => this.preview(id));
      this.list.append(li);
    }
  }

  async preview(id) {
    this.previewId = id;
    this._render();
    let u;
    try {
      u = await this.getFull(id);
    } catch (err) {
      clear(this.previewBox).append(el("div", { class: "error-text" }, `Preview failed: ${err.message}`));
      this.previewBox.classList.remove("hidden");
      return;
    }
    if (this.previewId !== id) return;
    if (this.previewViewer) this.previewViewer.viewer.destroy();
    clear(this.previewBox);
    this.previewBox.classList.remove("hidden");
    const close = el("button", { type: "button", class: "btn-link", title: "Close preview" }, "×");
    close.addEventListener("click", () => {
      this.previewId = null;
      if (this.previewViewer) this.previewViewer.viewer.destroy();
      this.previewViewer = null;
      clear(this.previewBox);
      this.previewBox.classList.add("hidden");
      this._render();
    });
    this.previewBox.append(el("div", { class: "preview-head" }, el("strong", {}, `${u.formula} — ${u.filename}`), close));
    const vhost = el("div", { class: "preview-viewer" });
    this.previewBox.append(vhost);
    this.previewViewer = viewerWithSlider(vhost, { height: 240, testid: "structure-preview" });
    const frames = u.frames_xyz ? splitXyzFrames(u.frames_xyz) : null;
    if (frames && frames.length > 1) {
      this.previewViewer.setFrames(frames, (i, n) => `Frame ${i + 1} / ${n}`);
      this.previewViewer.viewer.goto(frames.length - 1);
      this.previewViewer.refreshCaption();
    } else {
      this.previewViewer.viewer.showXyz(u.xyz);
    }
    const cp = cellParams(u.cell);
    const rows = [
      ["Formula", u.formula],
      ["Atoms", u.natoms],
      ["Frames", u.n_frames > 1 ? `${u.n_frames} (last frame is used by non-NEB calculations)` : "1"],
      ["Format", u.format],
      ["PBC", pbcText(u.pbc)],
      cp
        ? ["Cell a, b, c", `${fmtNum(cp.a, 5)}, ${fmtNum(cp.b, 5)}, ${fmtNum(cp.c, 5)} Å`]
        : ["Cell", "none (zero cell)"],
      cp ? ["α, β, γ", `${fmtNum(cp.alpha, 5)}°, ${fmtNum(cp.beta, 5)}°, ${fmtNum(cp.gamma, 5)}°`] : null,
      cp ? ["Volume", `${fmtNum(cellVolume(u.cell), 6)} Å³`] : null,
      ["Constraints", u.has_constraints ? "yes (e.g. FixAtoms)" : "none"],
    ].filter(Boolean);
    this.previewBox.append(
      el(
        "table",
        { class: "kv" },
        rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, String(v)))),
      ),
    );
  }
}
