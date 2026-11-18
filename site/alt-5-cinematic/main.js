// mliprun front page, alternative 5 ("cinematic"). One fixed 3D stage behind
// the page; each use-case card that reaches the middle of the viewport swaps
// the scene. Every number is read from data/*.json at load time.

// ---------------------------------------------------------------- config
// Hero source. The spec wants the water MD as the hero, but md.json is
// "pending" (no frames). Set heroUseMD to true once md.json names a frames
// file: the hero then plays the MD. While md.json has no frames the hero falls
// back to the NEB band even when this is true.
const CONFIG = {
  heroUseMD: false,
};

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const narrow = () => window.matchMedia("(max-width: 900px)").matches;
const $ = (sel, root = document) => root.querySelector(sel);
const SVGNS = "http://www.w3.org/2000/svg";

// ../shared/viewer.js in the repo; ./shared/viewer.js where the page is
// published on its own (artifact preview), since "../" cannot be published.
async function importViewer() {
  try {
    return await import("../shared/viewer.js");
  } catch (e) {
    return await import("./shared/viewer.js");
  }
}

// ---------------------------------------------------------------- data helpers
async function getJSON(name) {
  const r = await fetch(`data/${name}`);
  if (!r.ok) throw new Error(`data/${name}: HTTP ${r.status}`);
  return r.json();
}
async function getText(name) {
  const r = await fetch(`data/${name}`);
  if (!r.ok) throw new Error(`data/${name}: HTTP ${r.status}`);
  return r.text();
}
/** Key number whose pack label contains `sub` (throws if the pack lacks it). */
function kn(j, sub) {
  const k = j.key_numbers.find((x) => x.label.includes(sub));
  if (!k) throw new Error(`${j.use_case}.json has no key number matching "${sub}"`);
  return k;
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function fmtNum(v, unit = "") {
  const a = Math.abs(v);
  if (["steps", "models", "atoms"].includes(unit)) return String(Math.round(v));
  if (unit === "cm⁻¹") return v.toFixed(1);
  if (a !== 0 && a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    const m = (v / 10 ** e).toFixed(1);
    return `${m}×10${String(e).split("").map((c) => SUP[c]).join("")}`;
  }
  if (a >= 100) return v.toFixed(3);
  if (a >= 10) return v.toFixed(1);
  return Number(v.toPrecision(3)).toString();
}
function unitText(u) {
  return u === "dimensionless" ? "" : u;
}
function valueHTML(k) {
  const u = unitText(k.unit);
  return `${esc(fmtNum(k.value, k.unit))}${u ? `<span class="u">${esc(u)}</span>` : ""}`;
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}
/** Strip the environment path from mlip and the cluster paths from inputs. */
function shortCmd(c) {
  return c.replace(/^\S*\/mlip\s/, "mlip ").replace(/(?:\/[^\s/]+)+\/([^\s/]+)/g, "$1");
}
function modelOf(cmd) {
  const m = cmd.match(/--mlip\s+(\S+)/);
  const t = cmd.match(/--uma-task\s+(\S+)/);
  return m ? `${m[1]}${t ? "/" + t[1] : ""}` : "";
}

// ---------------------------------------------------------------- card pieces
function statusHTML(j, detail = "") {
  if (j.status === "pending") {
    return `<p class="status status-wait">Pending<span class="status-detail">${esc(detail)}</span></p>`;
  }
  return j.converged
    ? `<p class="status status-ok">Converged<span class="status-detail">${esc(detail)}</span></p>`
    : `<p class="status status-wait">Not converged<span class="status-detail">${esc(detail)}</span></p>`;
}
function cmdHTML(cmd) {
  return `<div class="cmd"><code title="${esc(cmd)}">${esc(shortCmd(cmd))}</code>` +
    `<button type="button" class="copy" data-copy="${esc(shortCmd(cmd))}">Copy</button></div>`;
}
// Two key numbers on the card (visuals first); the rest go to the run notes.
let moreNums = [];
function numsHTML(items) {
  moreNums = items.slice(2).map(([label, k]) => `${label}: ${fmtNum(k.value, k.unit)}${unitText(k.unit) ? " " + unitText(k.unit) : ""}`);
  return `<dl class="nums">${items.slice(0, 2).map(([label, k, lead]) =>
    `<div class="${lead ? "lead" : ""}"><dt>${esc(label)}</dt><dd>${valueHTML(k)}</dd></div>`).join("")}</dl>`;
}
function notesHTML(j, extra = []) {
  const notes = [...moreNums, ...(j.notes || []), ...extra].map(h2o);
  moreNums = [];
  return `<details class="notes"><summary>Run notes from ${esc(j.use_case)}.json</summary><ul>` +
    notes.map((n) => `<li>${esc(n)}</li>`).join("") +
    `<li>Full command as run for this showcase (its paths are where these runs happened, not paths to use): <code>${esc(j.command)}</code></li></ul></details>`;
}

// ---------------------------------------------------------------- charts (plain SVG)
function niceTicks(min, max, n = 4) {
  const span = max - min || 1;
  const raw = span / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * p).find((s) => s >= raw) || p * 10;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(+v.toPrecision(10));
  return out;
}
function tickLabel(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 0.01) return v.toExponential(0);
  return Number(v.toPrecision(3)).toString();
}
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
/**
 * Line chart with labelled axes. x/y: {min, max, label, log?, ticks?}.
 * series: [{xs, ys, cls, dots}]. hlines: [{y, cls, label}].
 * Returns {node, cursor(i)} where cursor marks index i of series[cursorSeries].
 */
function lineChart({ width = 380, height = 168, x, y, series, hlines = [], marks = [], cursorSeries = 0, title }) {
  const m = { l: 50, r: 14, t: 10, b: 36 };
  const W = width - m.l - m.r, H = height - m.t - m.b;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
  const ylog = !!y.log;
  const ty = (v) => {
    if (ylog) {
      const a = Math.log10(y.min), b = Math.log10(y.max);
      return m.t + H - ((Math.log10(v) - a) / (b - a)) * H;
    }
    return m.t + H - ((v - y.min) / (y.max - y.min)) * H;
  };
  const tx = (v) => m.l + ((v - x.min) / (x.max - x.min)) * W;
  const grid = el("g", { class: "grid" }, svg);
  const ax = el("g", { class: "ax" }, svg);
  const yt = y.ticks || (ylog ? decades(y.min, y.max) : niceTicks(y.min, y.max));
  for (const v of yt) {
    el("line", { x1: m.l, x2: m.l + W, y1: ty(v), y2: ty(v) }, grid);
    const t = el("text", { x: m.l - 6, y: ty(v) + 3.5, "text-anchor": "end" }, ax);
    t.textContent = tickLabel(v);
  }
  const xt = x.ticks || niceTicks(x.min, x.max, 5);
  for (const v of xt) {
    el("line", { x1: tx(v), x2: tx(v), y1: m.t + H, y2: m.t + H + 4 }, ax);
    const t = el("text", { x: tx(v), y: m.t + H + 15, "text-anchor": "middle" }, ax);
    t.textContent = tickLabel(v);
  }
  el("line", { x1: m.l, x2: m.l + W, y1: m.t + H, y2: m.t + H }, ax);
  el("line", { x1: m.l, x2: m.l, y1: m.t, y2: m.t + H }, ax);
  const xl = el("text", { x: m.l + W / 2, y: height - 4, "text-anchor": "middle", class: "ax-label" }, svg);
  xl.textContent = x.label;
  const yl = el("text", { x: 11, y: m.t + H / 2, "text-anchor": "middle", class: "ax-label",
    transform: `rotate(-90 11 ${m.t + H / 2})` }, svg);
  yl.textContent = y.label;
  for (const h of hlines) {
    el("line", { x1: m.l, x2: m.l + W, y1: ty(h.y), y2: ty(h.y), class: h.cls }, svg);
    if (h.label) {
      const t = el("text", { x: m.l + 4, y: ty(h.y) + 11, "text-anchor": "start", class: "lbl-muted" }, svg);
      t.textContent = h.label;
    }
  }
  for (const s of series) {
    const d = s.xs.map((xv, i) => `${i ? "L" : "M"}${tx(xv).toFixed(1)},${ty(s.ys[i]).toFixed(1)}`).join("");
    el("path", { d, class: s.cls }, svg);
    if (s.dots) s.xs.forEach((xv, i) => el("circle", { cx: tx(xv), cy: ty(s.ys[i]), r: 2.6, class: s.dots }, svg));
  }
  for (const mk of marks) {
    el("circle", { cx: tx(mk.x), cy: ty(mk.y), r: 4.2, class: "dot-hollow" }, svg);
    const t = el("text", { x: tx(mk.x), y: ty(mk.y) - 9, "text-anchor": "middle", class: "lbl" }, svg);
    t.textContent = mk.label;
  }
  const cs = series[cursorSeries];
  const cl = el("line", { class: "cursor", y1: m.t, y2: m.t + H, visibility: "hidden" }, svg);
  const cd = el("circle", { r: 4, class: "cursor-dot", visibility: "hidden" }, svg);
  return {
    node: svg,
    cursor(i) {
      if (!cs || i == null || i < 0 || i >= cs.xs.length) {
        cl.setAttribute("visibility", "hidden");
        cd.setAttribute("visibility", "hidden");
        return;
      }
      const X = tx(cs.xs[i]), Y = ty(cs.ys[i]);
      cl.setAttribute("x1", X); cl.setAttribute("x2", X); cl.setAttribute("visibility", "visible");
      cd.setAttribute("cx", X); cd.setAttribute("cy", Y); cd.setAttribute("visibility", "visible");
    },
  };
}
function stepTicks(steps) {
  const last = steps[steps.length - 1];
  const every = last > 20 ? 10 : last > 8 ? 2 : 1;
  const out = [];
  for (let v = 0; v <= last; v += every) out.push(v);
  return out;
}
function decades(min, max) {
  const out = [];
  for (let e = Math.ceil(Math.log10(min)); e <= Math.floor(Math.log10(max)); e++) out.push(10 ** e);
  return out;
}
function figure(chart, caption) {
  const f = document.createElement("figure");
  f.className = "chart";
  f.appendChild(chart.node);
  if (caption) {
    const c = document.createElement("figcaption");
    c.textContent = caption;
    f.appendChild(c);
  }
  return f;
}
function legend(items) {
  return `<ul class="legend">${items.map(([cls, t]) => `<li class="${cls}">${esc(t)}</li>`).join("")}</ul>`;
}

// ---------------------------------------------------------------- geometry from frames
/** O height above the mean z of the top Pt layer, per frame (Å). */
function oHeights(frames) {
  return frames.map((f) => {
    const zPt = [], zO = [];
    f.symbols.forEach((s, a) => {
      const z = f.positions[3 * a + 2];
      if (s === "Pt") zPt.push(z);
      else if (s === "O") zO.push(z);
    });
    const top = Math.max(...zPt);
    const layer = zPt.filter((z) => z > top - 0.8);
    const mean = layer.reduce((p, q) => p + q, 0) / layer.length;
    return zO.length ? Math.max(...zO) - mean : NaN;
  });
}
function firstFrameText(text) {
  const lines = text.split(/\r?\n/);
  const n = parseInt(lines[0], 10);
  return lines.slice(0, n + 2).join("\n") + "\n";
}
function lastFrameText(text) {
  const lines = text.split(/\r?\n/);
  const n = parseInt(lines[0], 10);
  let i = 0, start = 0;
  while (i < lines.length && lines[i].trim()) { start = i; i += n + 2; }
  return lines.slice(start, start + n + 2).join("\n") + "\n";
}
function hexMix(a, b, t) {
  const p = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
  const A = p(a), B = p(b);
  return "#" + A.map((v, k) => Math.round(v + (B[k] - v) * t).toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------- main
const state = { spare: [], V: null, D: null, specs: {}, cursors: {}, current: null, token: 0, chapter: null, mode: "band" };

async function main() {
  const [V, relax, committee, neb, md, webapp, relaxXYZ] = await Promise.all([
    importViewer(), getJSON("relax.json"), getJSON("committee.json"), getJSON("neb.json"),
    getJSON("md.json"), getJSON("webapp.json"), getText("relax.xyz"),
  ]);
  const [nebText, tsText] = await Promise.all([getText(neb.frames), getText(neb.frames_extra.ts_mode)]);
  state.V = V;
  state.softGL = softwareGL();
  const D = { relax, committee, neb, md, webapp, relaxXYZ, relaxFrames: V.parseExtXYZ(relaxXYZ),
    nebText, nebFrames: V.parseExtXYZ(nebText), tsText, tsFrames: V.parseExtXYZ(tsText) };
  state.D = D;
  buildRelax(D);
  buildCommittee(D);
  buildNEB(D);
  buildMD(D);
  buildWebapp(D);
  buildCoda(D);
  defineScenes(D);
  wireControls();
  watchChapters();
}

// ---------------------------------------------------------------- cards
function buildRelax(D) {
  const j = D.relax;
  const s = j.series;
  const fmax = kn(j, "final fmax, O at fcc hollow");
  const target = kn(j, "fmax target");
  const body = $('[data-fill="relax"]');
  body.innerHTML =
    statusHTML(j, ` fmax ${fmtNum(fmax.value)} eV/Å, target ${fmtNum(target.value)} eV/Å`) +
    cmdHTML(j.command) +
    numsHTML([
      ["O height above the top Pt layer", kn(j, "O height above top Pt layer, fcc"), true],
      ["BFGS steps", kn(j, "BFGS steps, O at fcc hollow")],
      ["Shortest O–Pt distance", kn(j, "shortest O–Pt distance, fcc")],
      ["E(hcp) − E(fcc), both relaxed", kn(j, "E(hcp) − E(fcc)")],
    ]);
  const xr = { min: 0, max: s.step[s.step.length - 1], label: "BFGS step", ticks: stepTicks(s.step) };
  const fmin = Math.min(...s.fmax_eV_per_A, target.value);
  const c2 = lineChart({
    title: "Largest force against BFGS step, log scale, with the fmax target",
    x: xr, y: { min: 10 ** Math.floor(Math.log10(fmin)), max: Math.max(...s.fmax_eV_per_A) * 1.6,
      label: "fmax (eV/Å)", log: true },
    series: [{ xs: s.step, ys: s.fmax_eV_per_A, cls: "s-key", dots: "dot-key" }],
    hlines: [{ y: target.value, cls: "s-target", label: `target ${fmtNum(target.value)} eV/Å` }],
  });
  body.appendChild(figure(c2, "Log scale."));
  body.insertAdjacentHTML("beforeend", notesHTML(j));
  state.cursors.relax = (i) => c2.cursor(i);
}

function buildCommittee(D) {
  const j = D.committee;
  const s = j.series;
  const target = kn(j, "fmax target");
  const meanF = kn(j, "final fmax of the committee-mean force");
  const members = j.key_numbers.filter((k) => k.label.startsWith("final energy, member "))
    .map((k) => k.label.replace("final energy, member ", "").replace(/\s*\(.*$/, ""));
  const body = $('[data-fill="committee"]');
  const nMembers = kn(j, "committee members");
  $("#card-committee .card-sub").textContent = `${fmtNum(nMembers.value, nMembers.unit)} models relax together and report how far their forces disagree.`;
  body.innerHTML =
    statusHTML(j, ` mean-force fmax ${fmtNum(meanF.value)} eV/Å, target ${fmtNum(target.value)} eV/Å`) +
    `<ul class="members" aria-label="Committee members">${members.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` +
    cmdHTML(j.command) +
    numsHTML([
      ["Largest force disagreement ÷ final fmax", kn(j, "sigma_max_free / fmax, final step"), true],
      ["Largest per-atom force σ, final step", kn(j, "sigma_max_free, final step")],
      ["Energy spread, final step (σ)", kn(j, "energy spread across members, final step")],
      ["BFGS steps", kn(j, "BFGS steps")],
    ]);
  const all = [...s.fmax_eV_per_A, ...s.sigma_max_free_eV_per_A, ...s.sigma_mean_free_eV_per_A, target.value];
  const c = lineChart({
    title: "Committee-mean fmax and force disagreement against BFGS step, log scale",
    x: { min: 0, max: s.step[s.step.length - 1], label: "BFGS step", ticks: stepTicks(s.step) },
    y: { min: 10 ** Math.floor(Math.log10(Math.min(...all))), max: Math.max(...all) * 1.6,
      label: "force (eV/Å)", log: true },
    series: [
      { xs: s.step, ys: s.fmax_eV_per_A, cls: "s-key", dots: "dot-key" },
      { xs: s.step, ys: s.sigma_max_free_eV_per_A, cls: "s-cool" },
      { xs: s.step, ys: s.sigma_mean_free_eV_per_A, cls: "s-cool2" },
    ],
    hlines: [{ y: target.value, cls: "s-target", label: `target ${fmtNum(target.value)} eV/Å` }],
  });
  body.insertAdjacentHTML("beforeend", legend([
    ["l-key", "fmax of the mean force"], ["l-cool", "σ max, free atoms"],
    ["l-cool2", "σ mean, free atoms"], ["l-target", "fmax target"]]));
  body.appendChild(figure(c, "Log scale."));

  body.insertAdjacentHTML("beforeend", notesHTML(j, [
    "Scene: the committee's start structure (relax.xyz step 0, the same POSCAR), Pt coloured by its final-step force σ. The committee's own final geometry is not in the data pack.",
  ]));
}
const SIGMA_LO = "#3b5675";
const SIGMA_HI = "#f1c98a";
const FIXED = "#1f2a38";

function buildNEB(D) {
  const j = D.neb;
  const s = j.series;
  const ts = j.ts_image_index;
  const target = kn(j, "fmax target (band and endpoints)");
  const bandF = kn(j, "final band fmax");
  const freq = kn(j, "TS imaginary frequency");
  const ov = kn(j, "overlap of imaginary mode with NEB tangent");
  const ovT = kn(j, "overlap threshold");
  const body = $('[data-fill="neb"]');
  const verify = [];
  if (j.flags.includes("ts_verified")) {
    verify.push(["✓", `One imaginary mode, ${fmtNum(freq.value, freq.unit)}i cm⁻¹`]);
    verify.push(["✓", `Mode overlap with the NEB tangent ${fmtNum(ov.value)} (threshold ${fmtNum(ovT.value)})`]);
    if (/--verify-ts\s+both/.test(j.command)) verify.push(["✓", "Displacing the TS along the mode and relaxing reaches both endpoints"]);
  }
  const drift = kn(j, "band check: largest drift from seeded band");
  const allow = kn(j, "band check: allowance for that atom");
  verify.push([drift.value < allow.value ? "✓" : "!", `Band intact: largest drift ${fmtNum(drift.value)} Å, allowance ${fmtNum(allow.value)} Å`]);
  if (j.flags.includes("ts_verify_endpoint_fmax_above_relax_fmax")) {
    const is = kn(j, "ts_verify endpoint fmax, IS"), fs = kn(j, "ts_verify endpoint fmax, FS"), inner = kn(j, "ts_verify internal relax fmax");
    verify.push(["!", `Flag: endpoint fmax ${fmtNum(is.value)} (IS) and ${fmtNum(fs.value)} (FS) eV/Å, above the check's internal ${fmtNum(inner.value)} eV/Å. The verdict is unaffected (run notes).`]);
  }
  body.innerHTML =
    statusHTML(j, ` band fmax ${fmtNum(bandF.value)} eV/Å, target ${fmtNum(target.value)} eV/Å`) +
    cmdHTML(j.command) +
    numsHTML([
      ["Forward barrier, fcc → hcp", kn(j, "forward barrier"), true],
      ["Reaction energy, E(hcp) − E(fcc)", kn(j, "reaction energy")],
      ["Reverse barrier, hcp → fcc", kn(j, "reverse barrier")],
      ["NEB optimizer steps", kn(j, "NEB optimizer steps")],
    ]) +
    `<details class="verify-box"><summary>TS checks: ${verify.filter((v) => v[0] === "✓").length} passed` +
    `${verify.some((v) => v[0] === "!") ? `, ${verify.filter((v) => v[0] === "!").length} flag` : ""}</summary>` +
    `<ul class="verify" aria-label="TS checks">${verify.map(([m, t]) =>
      `<li><span class="mark ${m === "!" ? "mark-flag" : ""}" aria-hidden="true">${m}</span><span>${esc(t)}</span></li>`).join("")}</ul></details>`;
  const x = s.reaction_coordinate_A, yv = s.relative_energy_eV;
  const c = lineChart({
    title: "NEB energy profile: energy relative to the initial state against reaction coordinate",
    x: { min: 0, max: Math.max(...x), label: "reaction coordinate (Å)" },
    y: { min: Math.min(0, ...yv) - 0.05, max: Math.max(...yv) * 1.25, label: "E − E(fcc) (eV)" },
    series: [{ xs: x, ys: yv, cls: "s-key", dots: "dot-key" }],
    marks: [{ x: x[ts], y: yv[ts], label: `TS, image ${ts}` }],
  });
  body.appendChild(figure(c));
  body.insertAdjacentHTML("beforeend", notesHTML(j));
  state.cursors.neb = (i) => c.cursor(i);
}

function buildMD(D) {
  const j = D.md;
  const body = $('[data-fill="md"]');
  const live = j.status !== "pending" && !!j.frames; // one condition for the card, the scene and the hero switch
  if (!live) {
    const built = j.structures && j.structures.built;
    body.innerHTML =
      statusHTML(j, " MD not run; next step pending a decision") +
      notesHTML(j);
    return;
  }
  // Future: md.json with frames and series. Plot every series against the first key.
  body.innerHTML = statusHTML(j) + cmdHTML(j.command) +
    (j.key_numbers.length ? numsHTML(j.key_numbers.slice(0, 4).map((k, i) => [k.label, k, i === 0])) : "");
  const keys = Object.keys(j.series || {});
  if (keys.length > 1) {
    const xs = j.series[keys[0]];
    for (const k of keys.slice(1)) {
      const ys = j.series[k];
      if (!Array.isArray(ys) || ys.length !== xs.length) continue;
      const c = lineChart({ title: `${k} against ${keys[0]}`,
        x: { min: Math.min(...xs), max: Math.max(...xs), label: keys[0] },
        y: { min: Math.min(...ys), max: Math.max(...ys), label: k },
        series: [{ xs, ys, cls: "s-key" }] });
      body.appendChild(figure(c));
    }
  }
  body.insertAdjacentHTML("beforeend", notesHTML(j));
}

function buildWebapp(D) {
  const j = D.webapp;
  // The finished-job view is on screen in the last ~6 s of the recording (poster cut at the same time).
  state.webappResultT = Math.max(0, kn(j, "video length").value - 6);
  const target = kn(j, "fmax target");
  const fmax = kn(j, "final fmax");
  const speed = kn(j, "video speed-up of the waiting part");
  const len = kn(j, "video length");
  const body = $('[data-fill="webapp"]');
  body.innerHTML =
    `<div class="webapp-grid"><div>` +
    `<video id="webapp-video" src="data/${esc(j.video)}" muted loop playsinline controls preload="none" poster="data/webapp-poster.jpg" ` +
    `aria-label="Screen recording of the web app submitting an optimize job to ${esc(j.target.machine)}"></video>` +
    `<p class="video-cap">Screen recording, ${fmtNum(len.value)} s; waiting sped up ×${fmtNum(speed.value)}.</p>` +
    `</div><div>` +
    statusHTML(j, ` fmax ${fmtNum(fmax.value)} eV/Å, target ${fmtNum(target.value)} eV/Å`) +
    `<p class="card-sub">Showcase run on <code>${esc(j.target.machine)}</code> (not a default).</p>` +
    numsHTML([
      ["Job wall time on the cluster, model load included", kn(j, "job wall time on cos-cluster"), true],
      ["Relaxation only", kn(j, "relaxation wall time")],
      ["BFGS steps", kn(j, "BFGS steps")],
      ["Energy difference, web app − CLI run", kn(j, "energy difference, web app − CLI")],
    ]) +
    `</div></div>` + notesHTML(j);
}

function buildCoda(D) {
  const slab = (D.relax.notes || []).find((n) => n.startsWith("Slab:"));
  const model = modelOf(D.relax.command);
  $("#prov-slab").textContent = [slab, model ? `Model for single-model runs: ${model}.` : ""].filter(Boolean).join(" ");
}

// ---------------------------------------------------------------- scenes
/** True when WebGL runs in software (SwiftShader, llvmpipe): the live hero would stall. */
function softwareGL() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2") || document.createElement("canvas").getContext("webgl");
    if (!gl) return true;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const r = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
    return /swiftshader|llvmpipe|softpipe|software|basic render/i.test(r);
  } catch {
    return true;
  }
}

/** Scene coordinates of source atom `src` (central cell) in frame k of `frames`. */
function sceneAt(ctrl, frames, src, k) {
  const atoms = ctrl.viewer.getModel().selectedAtoms({});
  const a = atoms.find((x) => x.properties && x.properties.src === src && !x.properties.image);
  if (!a) return null;
  const cur = frames[ctrl.frame].positions, p = frames[k].positions;
  return { x: a.x + p[3 * src] - cur[3 * src], y: a.y + p[3 * src + 1] - cur[3 * src + 1], z: a.z + p[3 * src + 2] - cur[3 * src + 2] };
}
const KEY = "#f1c98a";

function sigmaScaleSVG(smax) {
  const W = 300, H = 40;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": `Colour scale of per-atom force sigma from 0 to ${fmtNum(smax)} eV/Å` });
  const defs = el("defs", {}, svg);
  const g = el("linearGradient", { id: "sigramp", x1: "0", x2: "1" }, defs);
  el("stop", { offset: "0", "stop-color": SIGMA_LO }, g);
  el("stop", { offset: "1", "stop-color": SIGMA_HI }, g);
  el("rect", { x: 8, y: 2, width: W - 24, height: 9, fill: "url(#sigramp)", rx: 2 }, svg);
  const ax = el("g", { class: "ax" }, svg);
  for (const v of niceTicks(0, smax, 3).filter((v) => (smax - v) / smax * (W - 24) > 64)) {
    const X = 8 + (v / smax) * (W - 24);
    el("line", { x1: X, x2: X, y1: 11, y2: 15 }, ax);
    const t = el("text", { x: X, y: 25, "text-anchor": "middle" }, ax);
    t.textContent = tickLabel(v);
  }
  const end = el("text", { x: W - 16, y: 25, "text-anchor": "end", class: "lbl" }, svg);
  end.textContent = `max ${fmtNum(smax)}`;
  const lab = el("text", { x: 8, y: 38, class: "ax-label" }, svg);
  lab.textContent = "Pt force σ, final step (eV/Å); fixed atoms dark";
  return svg;
}

function defineScenes(D) {
  const small = narrow();
  const rep3 = [3, 3];
  const nebModel = modelOf(D.neb.command);
  const s = D.neb.series;
  const nImg = s.relative_energy_eV.length;
  const nebReadout = (i) => {
    if (i < 0 || i >= nImg) return "";
    const tag = i === D.neb.ts_image_index ? ", climbing image (TS)" : i === 0 ? ", fcc" : i === nImg - 1 ? ", hcp" : "";
    return `Image ${i} of ${nImg}${tag}: E − E(fcc) = ${fmtNum(s.relative_energy_eV[i])} eV`;
  };
  const oIdx = D.nebFrames[0].symbols.indexOf("O");
  // NEB overlay: every image's O as a translucent ghost at its real position,
  // joined by a thin line through the true O centres.
  const pathMarkers = (ctrl) => {
    const pts = D.nebFrames.map((_, k) => sceneAt(ctrl, D.nebFrames, oIdx, k));
    if (pts.some((p) => !p)) return;
    pts.forEach((c, k) => {
      ctrl.viewer.addSphere({ center: c, radius: 0.25, color: KEY, alpha: 0.5 });
      if (k) ctrl.viewer.addCylinder({ start: pts[k - 1], end: c, radius: 0.03, color: KEY, fromCap: 1, toCap: 1 });
    });
    ctrl.viewer.render();
    state.tags = [{ text: "fcc", at: pts[0] }, { text: "hcp", at: pts[pts.length - 1] }];
    placeTags();
  };

  const nebBand = (zoom) => ({
    url: `data/${D.neb.frames}`,
    opts: { xyzText: D.nebText, fps: 4, bounce: true, hold: 0.9, replicate: rep3, imageTint: 0.72,
      restFrame: D.neb.ts_image_index, view: { elevation: 44, azimuth: -24, zoom } },
    caption: `O hops fcc → hcp on Pt(111); NEB, barrier ${fmtNum(kn(D.neb, "forward barrier").value)} eV. Images, not dynamics.`,
    readout: nebReadout,
    decorate: pathMarkers,
  });
  const mdLive = D.md.status !== "pending" && !!D.md.frames;
  const mdSpec = (repl) => ({ url: `data/${D.md.frames}`, opts: { fps: 20, loop: true, replicate: repl, imageTint: 0.6,
    view: { elevation: 22, azimuth: -28, zoom: 1.2 }, style: "ballstick" },
  caption: `MD of water on Pt(111), ${modelOf(D.md.command)}.`, readout: () => "" });
  state.specs.hero = CONFIG.heroUseMD && mdLive ? mdSpec(rep3) : { ...nebBand(small ? 2.6 : 2.0), poster: true };

  const relaxH = oHeights(D.relaxFrames);
  const rs = D.relax.series;
  state.specs.relax = {
    url: "data/relax.xyz", opts: { xyzText: D.relaxXYZ, fps: 3, loop: true, hold: 1.4, replicate: rep3,
      imageTint: 0.72, view: { elevation: 22, azimuth: -30, zoom: 1.5 }, restFrame: "last" },
    caption: `O relaxing into the fcc hollow, ${rs.step.length - 1} BFGS steps.`,
    readout: (i) => i < rs.step.length
      ? `Step ${i}: O height ${fmtNum(relaxH[i])} Å, fmax ${fmtNum(rs.fmax_eV_per_A[i])} eV/Å` : "",
    cursor: "relax",
  };

  const sig = D.committee.series.peratom_final_sigma_free_eV_per_A;
  const smax = Math.max(...sig);
  const noteWorst = (D.committee.notes || []).map((n) => n.match(/Worst free atom[^#]*#(\d+)/)).find(Boolean);
  const worst = noteWorst ? +noteWorst[1] : sig.indexOf(smax);
  const ties = sig.map((v, k) => k).filter((k) => k !== worst && Math.abs(sig[k] - sig[worst]) < 1e-6);
  const f0 = D.relaxFrames[0];
  const zPt = f0.symbols.map((e, a) => (e === "Pt" ? f0.positions[3 * a + 2] : -Infinity));
  const worstLayer = Math.max(...zPt) - f0.positions[3 * worst + 2] > 1.0 ? "subsurface" : "surface";
  const bg = "#0e1724";
  const colour = (atom) => {
    const sym = atom.elem;
    let c;
    if (sym === "Pt") {
      const v = sig[atom.properties.src];
      c = v ? hexMix(SIGMA_LO, SIGMA_HI, v / smax) : FIXED;
    } else {
      c = (state.V.ELEMENT_STYLE[sym] || { color: "#ff66cc" }).color;
    }
    return atom.properties.image ? hexMix(c, bg, 0.65) : c;
  };
  state.specs.committee = {
    url: "data/relax.xyz", opts: { xyzText: firstFrameText(D.relaxXYZ), replicate: rep3,
      view: { elevation: 36, azimuth: -30, zoom: 1.45 },
      restyle: (viewer) => {
        for (const e of ["Pt", "O"]) {
          const st = state.V.ELEMENT_STYLE[e];
          viewer.setStyle({ elem: e }, { sphere: { radius: st.space, colorfunc: colour } });
        }
      } },
    caption: `σ (final step) on the start structure; pin: worst atom Pt #${worst}, ${worstLayer}${ties.length ? ` (tied ${ties.map((k) => `#${k}`).join(", ")})` : ""}`,
    readout: () => `Pt #${worst}: σ ${fmtNum(sig[worst])} eV/Å`,
    legend: () => sigmaScaleSVG(smax),
    decorate: (ctrl) => {
      const atoms = ctrl.viewer.getModel().selectedAtoms({});
      const a = atoms.find((x) => x.properties.src === worst && !x.properties.image);
      if (!a) return;
      const zTop = Math.max(...atoms.filter((x) => x.elem === "Pt").map((x) => x.z));
      // The worst atom may sit below the surface layer: mark it with a pin that
      // rises from the atom centre through the surface.
      const top = { x: a.x, y: a.y, z: zTop + 3.2 };
      ctrl.viewer.addCylinder({ start: { x: a.x, y: a.y, z: a.z }, end: top, radius: 0.08, color: KEY, fromCap: 1, toCap: 1 });
      ctrl.viewer.addSphere({ center: top, radius: 0.3, color: KEY });
      ctrl.viewer.render();
    },
  };

  state.specs.neb = { ...nebBand(small ? 2.6 : 2.4), cursor: "neb" };

  const tsF = D.tsFrames;
  const tsO = tsF[0].symbols.indexOf("O");
  let kMax = 0, dMax = 0;
  tsF.forEach((f, k) => {
    const d = Math.hypot(...[0, 1, 2].map((c) => f.positions[3 * tsO + c] - tsF[0].positions[3 * tsO + c]));
    if (d > dMax) { dMax = d; kMax = k; }
  });
  state.specs.ts = {
    url: `data/${D.neb.frames_extra.ts_mode}`,
    opts: { xyzText: D.tsText, fps: 20, loop: true, hold: 0, replicate: rep3, imageTint: 0.72, restFrame: "first",
      view: { elevation: 34, azimuth: -24, zoom: small ? 1.7 : 1.9 } },
    caption: `Imaginary mode at the TS, ${fmtNum(kn(D.neb, "TS imaginary frequency").value, "cm⁻¹")}i cm⁻¹; mode, not dynamics.`,
    readout: () => "",
    decorate: (ctrl) => {
      const c0 = sceneAt(ctrl, tsF, tsO, 0), c1 = sceneAt(ctrl, tsF, tsO, kMax);
      if (!c0 || !c1) return;
      const d = [c1.x - c0.x, c1.y - c0.y, c1.z - c0.z];
      const n = Math.hypot(...d) || 1;
      const L = 2.6;
      for (const sgn of [1, -1]) {
        ctrl.viewer.addArrow({ start: c0, end: { x: c0.x + sgn * L * d[0] / n, y: c0.y + sgn * L * d[1] / n, z: c0.z + sgn * L * d[2] / n },
          radius: 0.13, radiusRatio: 2.2, mid: 0.8, color: KEY });
      }
      ctrl.viewer.render();
    },
  };

  const built = D.md.structures && D.md.structures.built;
  state.specs.md = mdLive ? mdSpec(small ? [1, 1] : [2, 2])
    : { url: `data/${built.file}`, opts: { replicate: [2, 2], imageTint: 0.5, style: "ballstick",
          view: { elevation: 22, azimuth: -30, zoom: 1.0 } },
        caption: `${h2o(built.label)}. Static; MD not run, next step pending a decision.`, readout: () => "" };
  state.specs.webapp = {
    url: "data/relax.xyz", opts: { xyzText: lastFrameText(D.relaxXYZ), replicate: rep3, imageTint: 0.6,
      view: { elevation: 30, azimuth: -30, zoom: 1.0 } },
    caption: "Behind: the relaxed O/Pt(111) from the CLI run.",
    readout: () => "",
  };
}
// In-scene "fcc" / "hcp" tags, placed from the projected screen position of the
// endpoint O (re-placed after every frame, rotation and resize).
function placeTags() {
  const box = $("#scene-tags");
  const c = state.current && state.current.ctrl;
  if (!state.tags || !c || !state.current.key.match(/^(hero|neb)$/)) { box.replaceChildren(); return; }
  const r = state.current.layer.getBoundingClientRect();
  // Project both endpoints, then push each tag away from the other along their
  // screen separation (the two hollows are only ~2 Å apart, so tags would collide).
  const P = state.tags.map((t) => c.viewer.modelToScreen(t.at)).map((p) => ({ x: p.x - window.scrollX, y: p.y - window.scrollY }));
  const mx = P.reduce((a, p) => a + p.x, 0) / P.length, my = P.reduce((a, p) => a + p.y, 0) / P.length;
  // Push length scales with the projected hop (short hop on a phone), and a
  // leader line ties each tag to its endpoint O.
  const hop = P.length > 1 ? Math.hypot(P[1].x - P[0].x, P[1].y - P[0].y) : 0;
  const push = Math.max(46, Math.min(58, 0.5 * hop + 40));
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "tag-leaders");
  const spans = state.tags.map((t, k) => {
    const p = P[k];
    let dx = p.x - mx, dy = p.y - my;
    const n = Math.hypot(dx, dy) || 1;
    dx /= n; dy /= n;
    const tx = p.x + dx * push, ty = p.y + dy * push - 10;
    const e = document.createElement("span");
    e.className = "scene-tag";
    e.textContent = t.text;
    e.style.left = `${tx}px`;
    e.style.top = `${ty}px`;
    const off = p.x < r.left || p.x > r.right || p.y < r.top || p.y > r.bottom;
    e.hidden = off;
    if (!off) el("line", { x1: p.x, y1: p.y, x2: tx, y2: ty, class: "tag-leader" }, svg);
    if (!off) el("circle", { cx: p.x, cy: p.y, r: 2.5, class: "tag-dot" }, svg);
    return e;
  });
  box.replaceChildren(svg, ...spans);
}
function h2o(s) {
  return String(s).replace(/H2O/g, "H₂O");
}

function sceneKey(chapter) {
  return chapter === "neb" && state.mode === "ts" ? "ts" : chapter;
}

function showPoster(spec) {
  const stage = $("#stage");
  const fig = document.createElement("div");
  fig.className = "poster";
  fig.innerHTML = `<img src="media/hero-poster.jpg" alt="O on Pt(111) at the transition state of the fcc to hcp hop, still from the NEB band">` +
    `<button type="button" class="ctl poster-btn">Load the live 3D scene</button>`;
  stage.appendChild(fig);
  $("#cap-text").textContent = "Still picture (3D off: software rendering). " + spec.caption;
  $("#cap-live").textContent = "";
  window.__cinematicKey = "poster";
  fig.querySelector("button").addEventListener("click", () => {
    state.forceLive = true;
    fig.remove();
    showScene(state.current ? state.current.key : sceneKey(state.chapter || "hero"));
  });
  state.poster = fig;
}

// A destroyed scene's container stays in the DOM and is reused: removing it
// makes 3Dmol redraw at zero size (GL_INVALID_FRAMEBUFFER_OPERATION warnings).
function retire(layer) {
  layer.style.visibility = "hidden";
  state.spare.push(layer);
}

async function showScene(key) {
  const t = ++state.token;
  const stage = $("#stage");
  const spec = state.specs[key];
  if (!spec) return;
  if (spec.poster && state.softGL && !state.forceLive) {
    if (state.current) { state.current.ctrl.destroy(); retire(state.current.layer); state.current = null; }
    if (!state.poster || !state.poster.isConnected) showPoster(spec);
    updatePlayButton();
    return;
  }
  if (state.poster) { state.poster.remove(); state.poster = null; }
  if (state.current) {
    if (!reducedMotion) {
      stage.classList.add("is-swapping");
      await new Promise((r) => setTimeout(r, 260));
      if (t !== state.token) return;
    }
    const old = state.current;
    state.current = null;
    old.ctrl.destroy();
    retire(old.layer);
  }
  state.tags = null;
  placeTags();
  $("#cap-text").textContent = spec.caption;
  $("#cap-live").textContent = "";
  const extra = $("#cap-extra");
  extra.replaceChildren();
  if (spec.legend) extra.appendChild(spec.legend());
  extra.hidden = !spec.legend;
  stage.setAttribute("aria-label", `3D scene: ${spec.caption}`);
  const layer = state.spare.pop() || stage.appendChild(Object.assign(document.createElement("div"), { className: "scene-layer" }));
  layer.style.visibility = "";
  try {
    const ctrl = await state.V.loadScene(layer, spec.url, {
      background: getComputedStyle(document.documentElement).getPropertyValue("--stage").trim() || "#0e1724",
      ambientOcclusion: 0.55,
      fill: narrow() ? 0.74 : 0.9,
      ...spec.opts,
      onFrame: (i) => {
        if (t !== state.token) return;
        if (spec.readout) $("#cap-live").textContent = spec.readout(i);
        if (spec.cursor && state.cursors[spec.cursor]) state.cursors[spec.cursor](i);
        if (state.tags && state.current && state.current.layer === layer) placeTags(); // follows any refit
      },
    });
    if (t !== state.token) { ctrl.destroy(); retire(layer); return; }
    state.current = { ctrl, layer, key };
    if (spec.decorate) spec.decorate(ctrl);
    if (state.touchRotate) ctrl.setTouchRotate(true);
    window.__cinematic = ctrl; // for check.py
    window.__cinematicKey = key;
    window.__alt5 = state; // read-only hook for build_media.py (README frames)
    const i = ctrl.frame;
    if (spec.readout) $("#cap-live").textContent = spec.readout(i);
    if (spec.cursor && state.cursors[spec.cursor]) state.cursors[spec.cursor](i);
    if (state.chapter !== "coda" && !reducedMotion) ctrl.play();
    updatePlayButton();
  } catch (err) {
    retire(layer);
    $("#cap-text").textContent = `The 3D scene did not load (${err.message}). The numbers on the cards are unaffected.`;
  } finally {
    if (t === state.token) stage.classList.remove("is-swapping");
  }
}

function updatePlayButton() {
  const b = $("#btn-play");
  const c = state.current && state.current.ctrl;
  if (!c || c.nFrames < 2) { b.hidden = true; return; }
  b.hidden = false;
  b.textContent = c.playing ? "Pause" : "Play";
}

function wireControls() {
  let tagRaf = 0;
  const retag = () => { if (!tagRaf) tagRaf = requestAnimationFrame(() => { tagRaf = 0; placeTags(); }); };
  $("#stage").addEventListener("pointermove", (e) => { if (e.buttons) retag(); });
  $("#stage").addEventListener("pointerup", retag);
  window.addEventListener("resize", () => setTimeout(retag, 400));
  // Phone: the stage starts below the caption bar, so nothing runs under the controls.
  const bar = $("#scene-bar");
  let barTimer = 0;
  const setBar = () => {
    const h = Math.ceil(bar.getBoundingClientRect().bottom + 6);
    if (document.documentElement.style.getPropertyValue("--bar-h") === `${h}px`) return;
    document.documentElement.style.setProperty("--bar-h", `${h}px`);
    clearTimeout(barTimer);
    barTimer = setTimeout(() => { if (state.current && narrow()) { state.current.ctrl.resize(); placeTags(); } }, 120);
  };
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(setBar).observe(bar);
  setBar();
  $("#btn-play").addEventListener("click", () => {
    const c = state.current && state.current.ctrl;
    if (!c) return;
    if (c.playing) c.pause(); else c.play({ force: true });
    updatePlayButton();
  });
  const mode = $("#btn-mode");
  mode.addEventListener("click", () => {
    state.mode = state.mode === "ts" ? "band" : "ts";
    mode.textContent = state.mode === "ts" ? "Show NEB band" : "Show TS mode";
    if (state.mode === "ts" && state.cursors.neb) state.cursors.neb(state.D.neb.ts_image_index);
    showScene(sceneKey("neb"));
  });
  const rot = $("#btn-rotate");
  rot.addEventListener("click", () => {
    const on = rot.getAttribute("aria-pressed") !== "true";
    rot.setAttribute("aria-pressed", String(on));
    rot.textContent = on ? "Done rotating" : "Rotate scene";
    state.touchRotate = on;
    if (state.current) state.current.ctrl.setTouchRotate(on);
  });
  document.addEventListener("click", async (e) => {
    const b = e.target.closest(".copy");
    if (!b) return;
    const text = b.dataset.copy;
    try {
      await navigator.clipboard.writeText(text);
      b.textContent = "Copied";
    } catch {
      const code = b.previousElementSibling;
      const r = document.createRange();
      r.selectNodeContents(code);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      b.textContent = "Selected";
    }
    setTimeout(() => { b.textContent = "Copy"; }, 1600);
  });
}

// ---------------------------------------------------------------- chapter tracking
function setChapter(ch) {
  if (ch === state.chapter) return;
  const prev = state.chapter;
  state.chapter = ch;
  document.body.dataset.chapter = ch;
  document.querySelectorAll(".chapters a").forEach((a) => {
    if (a.dataset.for === ch) a.setAttribute("aria-current", "true");
    else a.removeAttribute("aria-current");
  });
  $("#btn-mode").hidden = ch !== "neb";
  const video = $("#webapp-video");
  if (video) {
    if (ch === "webapp" && !reducedMotion) {
      if (!video.dataset.started) {
        // Open on the finished job, then loop the whole clip. The seek must wait
        // for metadata: on a preload="none" video an earlier seek is dropped.
        video.dataset.started = "1";
        const start = () => {
          video.currentTime = state.webappResultT;
          if (state.chapter === "webapp") video.play().catch(() => {});
        };
        if (video.readyState >= 1) start();
        else {
          video.addEventListener("loadedmetadata", start, { once: true });
          video.preload = "metadata";
          video.load();
        }
      } else {
        video.play().catch(() => {});
      }
    }
    else video.pause();
  }
  if (ch === "coda") {
    if (state.current) { state.current.ctrl.pause(); updatePlayButton(); }
    return;
  }
  const key = sceneKey(ch);
  if (prev === "coda" && state.current && state.current.key === key) {
    if (!reducedMotion) state.current.ctrl.play();
    updatePlayButton();
    return;
  }
  showScene(key);
}

function watchChapters() {
  const sections = [...document.querySelectorAll("[data-scene]"), $("#install")];
  let raf = 0;
  const pick = () => {
    raf = 0;
    const mid = window.innerHeight * (narrow() ? 0.7 : 0.5);
    let found = "hero";
    for (const s of sections) {
      const r = s.getBoundingClientRect();
      if (r.top <= mid && r.bottom > mid) found = s.dataset.scene || "coda";
    }
    // The install/footer band is opaque: switch to it as soon as it takes the lower fifth.
    if ($("#install").getBoundingClientRect().top < window.innerHeight * 0.8) found = "coda";
    setChapter(found);
  };
  window.addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(pick); }, { passive: true });
  let lastNarrow = narrow();
  window.addEventListener("resize", () => {
    if (narrow() !== lastNarrow) {
      lastNarrow = narrow();
      defineScenes(state.D);
      if (state.chapter && state.chapter !== "coda") showScene(sceneKey(state.chapter));
    }
    pick();
  });
  pick();
}

main().catch((err) => {
  console.error(err);
  $("#cap-text").textContent = `The page could not load its data (${err.message}).`;
});
