// Small shared helpers: DOM building, number formatting, xyz strings, throttling.

/** Create an element: el("div", {class: "x", dataset: {testid: "y"}}, child, "text", ...). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "testid") node.dataset.testid = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** Format a number for a scientist: fixed for moderate magnitudes, scientific otherwise. */
export function fmtNum(x, digits = 6) {
  if (x === null || x === undefined) return "–";
  if (typeof x === "boolean") return x ? "yes" : "no";
  if (typeof x !== "number") return String(x);
  if (!Number.isFinite(x)) return String(x);
  if (Number.isInteger(x) && Math.abs(x) < 1e9) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(Math.max(2, digits - 2));
  // Up to `digits` significant figures, without trailing zeros beyond 1 decimal.
  const s = x.toPrecision(digits);
  return s.includes("e") ? x.toExponential(digits - 2) : String(parseFloat(s));
}

/** Guess a display unit from a result key suffix (PLAN 6.4 naming convention). */
export function unitForKey(key) {
  const rules = [
    [/_eV_per_A$/, "eV/Å"],
    [/_eV_per_A2$/, "eV/Å²"],
    [/_meV$/, "meV"],
    [/_eV$/, "eV"],
    [/_cm1$/, "cm⁻¹"],
    [/_GPa$/, "GPa"],
    [/_K$/, "K"],
    [/_A3$/, "Å³"],
    [/_A$/, "Å"],
    [/_fs$/, "fs"],
    [/_ps$/, "ps"],
    [/_s$/, "s"],
  ];
  for (const [re, unit] of rules) if (re.test(key)) return unit;
  return "";
}

/** Human label from a snake_case result key, unit suffix removed. */
export function labelForKey(key) {
  const stripped = key.replace(/_(eV_per_A2?|meV|eV|cm1|GPa|K|A3|A|fs|ps|s)$/, "");
  const s = stripped.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function fmtTime(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { hour12: false });
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m} min ${s} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

/** Lattice parameters a, b, c (Å) and alpha, beta, gamma (deg) from a 3x3 cell. */
export function cellParams(cell) {
  if (!cell) return null;
  const len = (v) => Math.hypot(v[0], v[1], v[2]);
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const [a, b, c] = cell.map(len);
  if (a === 0 && b === 0 && c === 0) return null;
  const ang = (u, v, lu, lv) => (lu && lv ? (Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (lu * lv)))) * 180) / Math.PI : 0);
  return {
    a,
    b,
    c,
    alpha: ang(cell[1], cell[2], b, c),
    beta: ang(cell[0], cell[2], a, c),
    gamma: ang(cell[0], cell[1], a, b),
  };
}

export function cellVolume(cell) {
  const [a, b, c] = cell;
  return Math.abs(
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]),
  );
}

export function pbcText(pbc) {
  if (!pbc) return "–";
  return pbc.map((p) => (p ? "T" : "F")).join(" ");
}

/** Extended-XYZ block (PLAN 4.3 format) from symbols + positions + cell. */
export function toXyz(symbols, positions, cell, pbc) {
  const lines = [String(symbols.length)];
  const c = cell || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const p = pbc || [true, true, true];
  lines.push(
    `Lattice="${c.flat().map((v) => v.toFixed(6)).join(" ")}" Properties=species:S:1:pos:R:3 pbc="${p
      .map((x) => (x ? "T" : "F"))
      .join(" ")}"`,
  );
  for (let i = 0; i < symbols.length; i++) {
    const r = positions[i];
    lines.push(`${symbols[i]} ${r[0].toFixed(6)} ${r[1].toFixed(6)} ${r[2].toFixed(6)}`);
  }
  return lines.join("\n") + "\n";
}

/** Split a concatenation of xyz blocks (frames_xyz) into individual blocks. */
export function splitXyzFrames(text) {
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  const frames = [];
  let i = 0;
  while (i < lines.length) {
    const head = lines[i].trim();
    if (head === "") {
      i++;
      continue;
    }
    const n = parseInt(head, 10);
    if (!Number.isFinite(n) || n < 0) break;
    frames.push(lines.slice(i, i + n + 2).join("\n") + "\n");
    i += n + 2;
  }
  return frames;
}

/** Parse the Lattice="..." of an extended-xyz comment line into a 3x3 cell. */
export function latticeFromXyz(xyz) {
  const line = xyz.split(/\r?\n/)[1] || "";
  const m = line.match(/Lattice="([^"]*)"/i);
  if (!m) return null;
  const v = m[1].trim().split(/\s+/).map(Number);
  if (v.length !== 9 || v.some((x) => !Number.isFinite(x))) return null;
  return [v.slice(0, 3), v.slice(3, 6), v.slice(6, 9)];
}

/**
 * Coalesce calls into at most one per animation frame, and no more often
 * than `minIntervalMs`. Used to keep plot/viewer redraws cheap on fast streams.
 */
export function frameThrottle(fn, minIntervalMs = 120) {
  let scheduled = false;
  let last = 0;
  let timer = null;
  const run = () => {
    scheduled = false;
    last = performance.now();
    fn();
  };
  const trigger = () => {
    if (scheduled) return;
    scheduled = true;
    const wait = Math.max(0, minIntervalMs - (performance.now() - last));
    const go = () => {
      timer = null;
      requestAnimationFrame(run);
    };
    if (wait > 0) timer = setTimeout(go, wait);
    else go();
  };
  trigger.flush = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    scheduled = false;
    run();
  };
  trigger.cancel = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    scheduled = false;
  };
  return trigger;
}

export function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export const TERMINAL = new Set(["done", "failed", "cancelled"]);

export function statusBadge(status, testid = "job-status") {
  return el("span", { class: `badge status-${status}`, testid, dataset: { status } }, status);
}

export function libsAvailable() {
  return { mol: typeof window.$3Dmol !== "undefined", plotly: typeof window.Plotly !== "undefined" };
}
