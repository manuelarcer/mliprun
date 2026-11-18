// mliprun front page, alternative 2 ("story").
// One pinned stage (3D scene + live plot) beside six chapters. The reader's scroll
// position is the playhead: it picks the chapter, swaps the scene, scrubs frames and
// draws the plot up to the current step. Every number is read from data/*.json at load.
//
// Scene swaps follow the viewer.js contract: the old controller is destroy()ed, and the
// new one is built in a fresh child element. Loads are serialised (one at a time), so a
// fast scroll can never leave two viewers in the stage.


const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const viewerPath = document.querySelector('meta[name="viewer-path"]')?.content || "../shared/viewer.js";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

let V; // viewer module
let pack; // {relax, committee, neb, md, webapp}
const xyz = {}; // raw xyz text per scene
const parsed = {}; // parsed frames per scene

// ------------------------------------------------------------------ formatting
const MINUS = "−";
const NB = " ";
function num(v, dp) {
  const s = Math.abs(v).toFixed(dp);
  return (v < 0 && +s !== 0 ? MINUS : "") + s;
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function sci(v, dp = 1) {
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  return `${num(m, dp)} × 10${String(e).split("").map((c) => SUP[c]).join("")}`;
}
function unitText(unit, override) {
  const u = override ?? unit;
  if (!u || u === "dimensionless") return "";
  return NB + u;
}
function withUnit(v, dp, unit, override) {
  const s = Math.abs(v) > 0 && Math.abs(v) < 10 ** -Math.max(dp, 3) ? sci(v) : num(v, dp);
  return s + unitText(unit, override);
}
function kn(name, label) {
  const list = pack[name]?.key_numbers || [];
  const hit = list.find((k) => k.label === label) || list.find((k) => k.label.startsWith(label));
  if (!hit) console.warn(`pack ${name}: no key number "${label}"`);
  return hit || null;
}
function knv(name, label) { const k = kn(name, label); return k ? k.value : NaN; }
function knText(name, label, dp, override) {
  const k = kn(name, label);
  return k ? withUnit(k.value, dp, k.unit, override) : "n/a";
}

// ------------------------------------------------------------------ commands
function cmdTokens(cmd) {
  return cmd.trim().split(/\s+/).map((t, i) => {
    if (i === 0) return t.split("/").pop(); // the env's mlip binary, shown as `mlip`
    return t.startsWith("/") ? t.split("/").pop() : t; // absolute paths -> file name
  });
}
function cmdPretty(cmd) {
  const t = cmdTokens(cmd);
  const lines = [];
  let cur = [];
  for (const tok of t) {
    if (tok.startsWith("--") && cur.length) { lines.push(cur.join(" ")); cur = []; }
    cur.push(tok);
  }
  if (cur.length) lines.push(cur.join(" "));
  return lines.map((l, i) => (i ? "  " : "") + l).join(" \\\n");
}
function cmdFlag(cmd, flag) {
  const t = cmd.trim().split(/\s+/);
  const i = t.indexOf("--" + flag);
  return i >= 0 ? t[i + 1] : "";
}
function renderCommand(host, name) {
  const cmd = pack[name].command;
  const pretty = cmdPretty(cmd);
  host.innerHTML = "";
  const row = document.createElement("div");
  row.className = "cmd-row";
  const pre = document.createElement("pre");
  pre.textContent = pretty;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "Copy";
  btn.setAttribute("aria-label", "Copy this command");
  btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(pretty); btn.textContent = "Copied"; }
    catch { const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Selected"; }
    setTimeout(() => { btn.textContent = "Copy"; }, 1600);
  });
  row.append(pre, btn);
  const det = document.createElement("details");
  const sum = document.createElement("summary");
  sum.textContent = "Exact command as run for this showcase (its paths are those of the machine where it ran, not yours)";
  const full = document.createElement("pre");
  full.textContent = cmd + (pack[name].cwd ? `\n(cwd ${pack[name].cwd})` : "");
  det.append(sum, full);
  host.append(row, det);
}

// ------------------------------------------------------------------ geometry from frames
function oHeights(frames) {
  return frames.map((f) => {
    const zPt = [], zO = [];
    f.symbols.forEach((s, a) => {
      const z = f.positions[3 * a + 2];
      if (s === "Pt") zPt.push(z); else if (s === "O") zO.push(z);
    });
    const top = Math.max(...zPt);
    const layer = zPt.filter((z) => z > top - 0.6);
    const zTop = layer.reduce((s, z) => s + z, 0) / layer.length;
    return Math.max(...zO) - zTop; // the adsorbate is the highest O
  });
}
function maxPtShift(f0, f1) {
  let m = 0;
  f0.symbols.forEach((s, a) => {
    if (s !== "Pt") return;
    const d = Math.hypot(...[0, 1, 2].map((k) => f1.positions[3 * a + k] - f0.positions[3 * a + k]));
    m = Math.max(m, d);
  });
  return m;
}

// ------------------------------------------------------------------ SVG plotting
const NS = "http://www.w3.org/2000/svg";
function S(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function lin(d0, d1, r0, r1) { return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0); }
function logs(d0, d1, r0, r1) {
  const a = Math.log10(d0), b = Math.log10(d1);
  return (v) => r0 + ((Math.log10(Math.max(v, d0)) - a) / (b - a)) * (r1 - r0);
}
function niceStep(range, n) {
  const raw = range / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
function ticks(lo, hi, n = 4) {
  const st = niceStep(hi - lo, n);
  const out = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10));
  return out;
}
function dp(step) { return Math.max(0, -Math.floor(Math.log10(step) + 1e-9)); }

/** One panel: axes, grid, labels. Returns {x, y, g, box}. */
function panel(svg, box, o) {
  const g = S("g", {}, svg);
  const { x0, y0, w, h } = box;
  const x = lin(o.xd[0], o.xd[1], x0, x0 + w);
  const y = o.ylog ? logs(o.yd[0], o.yd[1], y0 + h, y0) : lin(o.yd[0], o.yd[1], y0 + h, y0);
  const ax = S("g", { class: "ax" }, g);
  const xt = o.xticks || ticks(o.xd[0], o.xd[1], o.nx || 4);
  const yt = o.yticks || ticks(o.yd[0], o.yd[1], o.ny || 3);
  const xdp = o.xdp ?? (xt.length > 1 ? dp(xt[1] - xt[0]) : 0);
  const ydp = o.ydp ?? (yt.length > 1 ? dp(Math.abs(yt[1] - yt[0])) : 2);
  for (const v of yt) {
    S("line", { class: "gridl", x1: x0, x2: x0 + w, y1: y(v), y2: y(v) }, g);
    S("text", { x: x0 - 6, y: y(v) + 3.5, "text-anchor": "end" }, ax, o.yfmt ? o.yfmt(v) : num(v, ydp));
  }
  for (const v of xt) {
    S("line", { x1: x(v), x2: x(v), y1: y0 + h, y2: y0 + h + 4 }, ax);
    S("text", { x: x(v), y: y0 + h + 15, "text-anchor": "middle" }, ax, num(v, xdp));
  }
  S("path", { d: `M${x0},${y0}V${y0 + h}H${x0 + w}` }, ax);
  S("text", { class: "lab", x: x0 + w, y: y0 + h + 29, "text-anchor": "end" }, ax, o.xlabel);
  S("text", { class: "lab", x: x0 - 6, y: y0 - 6, "text-anchor": "start", transform: "" }, ax, o.ylabel)
    .setAttribute("x", Math.max(2, x0 - (o.ml || 40) + 4));
  return { x, y, g };
}
function path(pts) { return pts.map(([a, b], i) => `${i ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join(""); }

function plotFrame() {
  const svg = $("#plot");
  svg.innerHTML = "";
  svg.style.display = "";
  $("#plot-empty").hidden = true;
  const W = svg.clientWidth || 600, H = svg.clientHeight || 220;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const narrow = W < 520;
  const ml = narrow ? 38 : 50, mb = 34, mt = 20, gap = narrow ? 44 : 64;
  const split = (n) => {
    const w = (W - ml - 8 - gap * (n - 1)) / n;
    return Array.from({ length: n }, (_, i) => ({ x0: ml + i * (w + gap), y0: mt, w, h: H - mt - mb }));
  };
  return { svg, W, H, split, ml, narrow };
}

// --- per-chapter plots. `k` is the current step / image index.
function plotRelax(k) {
  const { svg, split, ml, narrow } = plotFrame();
  const s = pack.relax.series;
  const E0 = s.energy_eV[0];
  const dE = s.energy_eV.map((e) => e - E0);
  const n = s.step.length - 1;
  const [b1, b2, b3] = split(narrow ? 2 : 3);
  const emin = Math.min(...dE);
  const p1 = panel(svg, b1, { xd: [0, n], yd: [emin * 1.08, 0.02], xlabel: "BFGS step", ylabel: "E − E₀ (eV)", ml });
  const p2 = panel(svg, b2, { xd: [0, n], yd: [0.005, 3], ylog: true, yticks: [0.01, 0.1, 1], yfmt: (v) => String(v), xlabel: "BFGS step", ylabel: "fmax (eV/Å)", ml });
  const target = knv("relax", "fmax target");
  S("line", { class: "target", x1: b2.x0, x2: b2.x0 + b2.w, y1: p2.y(target), y2: p2.y(target) }, svg);
  S("text", { class: "target-t", x: b2.x0 + 4, y: p2.y(target) + 13 }, svg, `target ${num(target, 2)}`);
  const pe = s.step.map((st, i) => [p1.x(st), p1.y(dE[i])]);
  const pf = s.step.map((st, i) => [p2.x(st), p2.y(s.fmax_eV_per_A[i])]);
  S("path", { class: "ghost", d: path(pe) }, svg);
  S("path", { class: "ghost", d: path(pf) }, svg);
  S("path", { class: "ln", d: path(pe.slice(0, k + 1)) }, svg);
  S("path", { class: "ln oxy", d: path(pf.slice(0, k + 1)) }, svg);
  S("circle", { class: "dot", cx: pe[k][0], cy: pe[k][1], r: 4 }, svg);
  S("circle", { class: "dot oxy", cx: pf[k][0], cy: pf[k][1], r: 4 }, svg);
  if (b3) { // O height above the top Pt layer, from the frames (phones: in the readout strip)
    const H = parsed.relaxH, lo = Math.min(...H), hi = Math.max(...H), pad = (hi - lo) * 0.15;
    const p3 = panel(svg, b3, { xd: [0, n], yd: [lo - pad, hi + pad], yticks: [+lo.toFixed(2), +hi.toFixed(2)], ydp: 2, xlabel: "BFGS step", ylabel: "O height (Å)", ml });
    const ph = H.map((z, i) => [p3.x(i), p3.y(z)]);
    S("path", { class: "ghost", d: path(ph) }, svg);
    S("path", { class: "ln oxy", d: path(ph.slice(0, k + 1)) }, svg);
    S("circle", { class: "dot oxy", cx: ph[k][0], cy: ph[k][1], r: 4 }, svg);
  }
  $("#plot-cap").textContent = `Energy, fmax${b3 ? " and O height" : ""} per BFGS step (step ${k} of ${n})`;
}

function plotCommittee(k) {
  const { svg, split, ml } = plotFrame();
  const s = pack.committee.series;
  const n = s.step.length - 1;
  const [b1, b2] = split(2);
  const floor = 0.005;
  const p1 = panel(svg, b1, { xd: [0, n], yd: [floor, 3], ylog: true, yticks: [0.01, 0.1, 1], yfmt: (v) => String(v), xlabel: "BFGS step", ylabel: "force (eV/Å)", ml });
  const target = knv("committee", "fmax target");
  const cut = k + 1;
  const sig = s.sigma_max_free_eV_per_A, fm = s.fmax_eV_per_A;
  // zone from the axis floor up to sigma_max: forces inside it are below what the
  // three models can tell apart
  const top = s.step.map((st, i) => [p1.x(st), p1.y(sig[i])]);
  const bot = [[p1.x(s.step[cut - 1]), p1.y(floor)], [p1.x(s.step[0]), p1.y(floor)]];
  S("path", { class: "band", d: path(top.slice(0, cut)) + path(bot).replace(/^M/, "L") + "Z" }, svg);
  S("path", { class: "ghost", d: path(top) }, svg);
  S("path", { class: "ln cool", d: path(top.slice(0, cut)) }, svg);
  const pf = s.step.map((st, i) => [p1.x(st), p1.y(fm[i])]);
  S("path", { class: "ghost", d: path(pf) }, svg);
  S("path", { class: "ln oxy", d: path(pf.slice(0, cut)) }, svg);
  S("circle", { class: "dot oxy", cx: pf[k][0], cy: pf[k][1], r: 4 }, svg);
  S("circle", { class: "dot cool", cx: top[k][0], cy: top[k][1], r: 3.5 }, svg);
  S("line", { class: "target", x1: b1.x0, x2: b1.x0 + b1.w, y1: p1.y(target), y2: p1.y(target) }, svg);
  S("text", { class: "target-t", x: b1.x0 + 4, y: p1.y(target) + 13 }, svg, `target ${num(target, 2)}`);
  // colour key, written next to each curve (every width)
  S("text", { class: "ann", x: p1.x(s.step[1]) + 4, y: p1.y(fm[1]) - 5, style: "fill:var(--oxy)" }, svg, "fmax");
  S("text", { class: "ann", x: b1.x0 + 4, y: p1.y(sig[0]) - 6, style: "fill:var(--cool)" }, svg, "σmax");
  const cross = fm.findIndex((f, i) => f < sig[i]);
  if (cross >= 0 && k >= cross) {
    const xc = p1.x(s.step[cross]);
    S("line", { class: "cursor", x1: xc, x2: xc, y1: p1.y(3), y2: p1.y(floor) }, svg);
    if (b1.w >= 220) { // phones: the marker line stays; "from step" is in the chapter facts
      S("text", { class: "ann", x: xc - 4, y: p1.y(3) + 12, "text-anchor": "end" }, svg, `fmax < σmax`);
      S("text", { class: "ann m", x: xc - 4, y: p1.y(3) + 25, "text-anchor": "end" }, svg, `from step ${s.step[cross]}`);
    }
  }

  const sp = s.energy_spread_aligned_eV;
  const smax = Math.max(...sp);
  const p2 = panel(svg, b2, { xd: [0, n], yd: [0, smax * 1.25], ny: 3, xlabel: "BFGS step", ylabel: "energy spread, std. dev. (eV)", ml });
  const ps = s.step.map((st, i) => [p2.x(st), p2.y(sp[i])]);
  S("path", { class: "ghost", d: path(ps) }, svg);
  S("path", { class: "ln cool", d: path(ps.slice(0, cut)) }, svg);
  S("circle", { class: "dot cool", cx: ps[k][0], cy: ps[k][1], r: 4 }, svg);
  $("#plot-cap").textContent = `fmax and σmax per step; energy spread (step ${k} of ${n})`;
}

function nebProfile(svg, box, ml, upto, ringTs) {
  const s = pack.neb.series;
  const rc = s.reaction_coordinate_A, e = s.relative_energy_eV;
  const emax = Math.max(...e);
  const roomy = box.w > 300;
  const gutter = roomy ? 104 : 0; // right gutter for the reverse-barrier and ΔE brackets
  const pb = { ...box, w: box.w - gutter };
  const p = panel(svg, pb, { xd: [0, rc[rc.length - 1]], yd: [-0.05, emax * 1.22], ny: 3, xlabel: "reaction coordinate (Å)", ylabel: "E − E(fcc) (eV)", ml });
  const pts = rc.map((r, i) => [p.x(r), p.y(e[i])]);
  S("path", { class: "ghost", d: path(pts) }, svg);
  S("path", { class: "ln", d: path(pts.slice(0, upto + 1)) }, svg);
  pts.forEach((q, i) => {
    const hot = i === upto && !ringTs;
    if (i <= upto) S("circle", { class: hot ? "dot oxy" : "dot", cx: q[0], cy: q[1], r: hot ? 4.5 : 3 }, svg);
  });
  const ts = pack.neb.ts_image_index;
  if (upto >= ts || ringTs) {
    const fwd = knv("neb", "forward barrier"), rev = knv("neb", "reverse barrier"), rxn = knv("neb", "reaction energy");
    const [tx, ty] = pts[ts];
    if (ringTs) S("circle", { cx: tx, cy: ty, r: 8, fill: "none", stroke: "var(--oxy)", "stroke-width": 1.5 }, svg);
    S("text", { class: "ann", x: tx, y: ty - 12, "text-anchor": "middle" }, svg, `TS, image ${ts}`);
    if (!roomy) return p;
    const xr = pb.x0 + pb.w;
    const lab = (x, y, t) => S("text", { class: "ann m", x, y }, svg, t);
    const bracket = (x, ya, yb) => S("path", { class: "brk", d: `M${x - 3},${ya}H${x + 3}M${x},${ya}V${yb}M${x - 3},${yb}H${x + 3}` }, svg);
    // TS level guide, forward barrier on the left
    S("line", { class: "gridl", x1: pb.x0, x2: xr + 14, y1: p.y(fwd), y2: p.y(fwd), "stroke-dasharray": "3 3" }, svg);
    bracket(pb.x0 + 12, p.y(0), p.y(fwd));
    lab(pb.x0 + 19, p.y(fwd * 0.62), `Ea ${num(fwd, 3)} eV`);
    if (upto >= rc.length - 1 || ringTs) {
      S("line", { class: "gridl", x1: pts[pts.length - 1][0], x2: xr + 14, y1: p.y(rxn), y2: p.y(rxn), "stroke-dasharray": "3 3" }, svg);
      bracket(xr + 14, p.y(rxn), p.y(fwd));
      lab(xr + 21, p.y((fwd + rxn) / 2) + 4, `rev ${num(rev, 3)} eV`);
      bracket(xr + 14, p.y(0), p.y(rxn));
      lab(xr + 21, p.y(rxn / 2) + 4, `ΔE ${num(rxn, 3)} eV`);
    }
  }
  return p;
}
function plotNeb(k) {
  const { svg, W, H, ml } = plotFrame();
  nebProfile(svg, { x0: ml, y0: 20, w: W - ml - 10, h: H - 54 }, ml, k, false);
  const n = pack.neb.series.image_index.length - 1;
  $("#plot-cap").textContent = `NEB energy profile (image ${k} of ${n})`;
}
function plotTs() {
  const { svg, split, ml, W, narrow } = plotFrame();
  const n = pack.neb.series.image_index.length - 1;
  const [b1, b2] = narrow ? split(2) : (() => { const bb = split(2); const w1 = (W - ml - 8 - 64) * 0.62; return [{ ...bb[0], w: w1 }, { ...bb[1], x0: ml + w1 + 64, w: W - ml - 8 - 64 - w1 }]; })();
  nebProfile(svg, b1, ml, n, true);
  const ov = knv("neb", "overlap of imaginary mode"), thr = knv("neb", "overlap threshold");
  const p = panel(svg, b2, { xd: [0, 1], yd: [0, 1], xticks: [0, 0.5, 1], xdp: 1, yticks: [], xlabel: "overlap with NEB tangent", ylabel: "mode check", ml });
  const yb = b2.y0 + b2.h * 0.35, hb = b2.h * 0.3;
  S("rect", { class: "bar-a", x: p.x(0), y: yb, width: p.x(1) - p.x(0), height: hb }, svg);
  S("rect", { class: "bar-c", x: p.x(0), y: yb, width: p.x(ov) - p.x(0), height: hb }, svg);
  S("line", { class: "target", x1: p.x(thr), x2: p.x(thr), y1: yb - 8, y2: yb + hb + 8 }, svg);
  S("text", { class: "target-t", x: p.x(thr), y: yb - 11, "text-anchor": "middle" }, svg, `threshold ${num(thr, 1)}`);
  S("text", { class: "ann", x: p.x(ov), y: yb + hb + 20, "text-anchor": "end" }, svg, num(ov, 3));
  $("#plot-cap").textContent = "NEB profile with the TS; mode overlap with the path";
}
function plotWater() {
  const md = pack.md;
  const keys = Object.keys(md.series || {});
  if (keys.length >= 2) { // the real MD dropped in: plot every series against the first
    const { svg, split, ml } = plotFrame();
    const xk = keys[0], ys = keys.slice(1, 3);
    split(ys.length).forEach((b, i) => {
      const xv = md.series[xk], yv = md.series[ys[i]];
      const lo = Math.min(...yv), hi = Math.max(...yv), pad = (hi - lo) * 0.1 || 1;
      const p = panel(svg, b, { xd: [xv[0], xv[xv.length - 1]], yd: [lo - pad, hi + pad], xlabel: xk, ylabel: ys[i], ml });
      S("path", { class: "ln", d: path(xv.map((x, j) => [p.x(x), p.y(yv[j])])) }, svg);
    });
    $("#plot-cap").textContent = "MD series from md.json.";
    return;
  }
  $("#plot").innerHTML = "";
  $("#plot").style.display = "none"; // one line instead of an empty chart: the scene gets the height
  const empty = $("#plot-empty");
  empty.hidden = false;
  empty.textContent = "MD not run: no T(t) or E(t) yet";
  $("#plot-cap").textContent = "";
}
function plotRemote() {
  const { svg, split, ml, narrow } = plotFrame();
  const [b1, b2] = split(2);
  const cli = pack.relax.series, web = pack.webapp.series;
  const E0 = cli.energy_eV[0];
  const n = cli.step.length - 1;
  const emin = Math.min(...cli.energy_eV) - E0;
  const p1 = panel(svg, b1, { xd: [0, n], yd: [emin * 1.08, 0.02], xlabel: "BFGS step", ylabel: "E − E₀ (eV)", ml });
  S("path", { class: "ln", d: path(cli.step.map((st, i) => [p1.x(st), p1.y(cli.energy_eV[i] - E0)])) }, svg);
  web.step.forEach((st, i) => S("circle", { class: "dot oxy", cx: p1.x(st), cy: p1.y(web.energy_eV[i] - E0), r: 3.2 }, svg));
  S("text", { class: "ann m", x: b1.x0 + b1.w, y: p1.y(emin * 0.35), "text-anchor": "end" }, svg, "line: CLI");
  S("text", { class: "ann", x: b1.x0 + b1.w, y: p1.y(emin * 0.35) + 14, "text-anchor": "end", style: "fill:var(--oxy)" }, svg, "dots: web app");

  const wall = knv("webapp", "job wall time"), opt = knv("webapp", "relaxation wall time");
  const p2 = panel(svg, b2, { xd: [0, Math.ceil(wall / 5) * 5], yd: [0, 1], yticks: [], nx: narrow ? 2 : 3, xlabel: "wall time on cos-cluster (s)", ylabel: "one web app job", ml });
  const yb = b2.y0 + b2.h * 0.3, hb = b2.h * 0.34;
  S("rect", { class: "bar-a", x: p2.x(0), y: yb, width: p2.x(wall - opt) - p2.x(0), height: hb }, svg);
  S("rect", { class: "bar-b", x: p2.x(wall - opt), y: yb, width: p2.x(wall) - p2.x(wall - opt), height: hb }, svg);
  S("text", { class: "ann m", x: p2.x(0) + 4, y: yb - 6 }, svg, `${narrow ? "model load" : "model load and setup"} ${num(wall - opt, 1)} s`);
  S("text", { class: "ann", x: p2.x(wall), y: yb + hb + 16, "text-anchor": "end" }, svg, `relaxation ${num(opt, 1)} s`);
  $("#plot-cap").textContent = "Web app and CLI energies; job wall time";
}

// ------------------------------------------------------------------ HUD + legend + captions
function hud(rows) {
  const d = $("#hud");
  d.innerHTML = "";
  for (const [t, v, cls] of rows) {
    const dt = document.createElement("dt"); dt.textContent = t;
    const dd = document.createElement("dd"); dd.textContent = v; if (cls) dd.className = cls;
    d.append(dt, dd);
  }
}
// σ colour ramps: σ = 0 is the quietest colour in both themes.
const RAMP_LIGHT = ["#d3d9e0", "#9fb8cc", "#5d8fb3", "#2a6593", "#123f66"];
const RAMP_DARK = ["#5a6470", "#46708f", "#4f95c4", "#80c3ea", "#d2ecfb"];
function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
const rampColours = () => (isDark() ? RAMP_DARK : RAMP_LIGHT);
function ramp(t) {
  const R = rampColours();
  t = Math.max(0, Math.min(1, t)) * (R.length - 1);
  const i = Math.min(R.length - 2, Math.floor(t)), f = t - i;
  const a = R[i], b = R[i + 1];
  const c = [1, 3, 5].map((k) => Math.round(parseInt(a.slice(k, k + 2), 16) * (1 - f) + parseInt(b.slice(k, k + 2), 16) * f));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}
function legend(show) {
  const L = $("#legend");
  L.hidden = !show;
  if (!show) return;
  const smax = Math.max(...pack.committee.series.peratom_final_sigma_free_eV_per_A);
  L.innerHTML = `<span><span class="lg-long">per-atom σ, final step (eV/Å)</span><span class="lg-short">σ (eV/Å)</span></span><span>0</span><span class="bar" style="background:linear-gradient(90deg,${rampColours().join(",")})"></span><span>${num(smax, 3)}</span>`;
}

// ------------------------------------------------------------------ chapter 1 slice
// The relaxation moves O by 0.07 Å, invisible on a whole slab. Chapter 1 therefore shows
// a cross-section: the O atom plus every Pt atom (all layers, periodic images) within
// SLICE_HALF_W of the O along the screen axis (x) and from SLICE_FRONT in front of the O
// to SLICE_DEPTH behind it along the view axis (+y is away from the camera at azimuth
// 0). No Pt sits in front of the O, so the whole O sphere stays visible on its hollow.
// Positions are the real ones in every frame; nothing is scaled.
const SLICE_HALF_W = 6.5; // Å
const SLICE_FRONT = 0.5; // Å in front of the O
const SLICE_DEPTH = 6.0; // Å behind the O
function cropRelax(frames) {
  const f0 = frames[0];
  const n = f0.symbols.length;
  const oIdx = f0.symbols.lastIndexOf("O");
  const [a, b] = [f0.cell[0], f0.cell[1]];
  const picks = []; // [atom index, shift vector]
  for (let i = 0; i < n; i++) {
    for (let u = -2; u <= 2; u++) for (let v = -2; v <= 2; v++) {
      if (i === oIdx) { if (u === 0 && v === 0) picks.push([i, [0, 0, 0]]); continue; }
      const sh = [0, 1, 2].map((k) => u * a[k] + v * b[k]);
      const dx = f0.positions[3 * i] + sh[0] - f0.positions[3 * oIdx];
      const dy = f0.positions[3 * i + 1] + sh[1] - f0.positions[3 * oIdx + 1];
      if (Math.abs(dx) <= SLICE_HALF_W && dy >= -SLICE_FRONT && dy <= SLICE_DEPTH) picks.push([i, sh]);
    }
  }
  const text = frames.map((f, k) => {
    const lines = picks.map(([i, sh]) => `${f.symbols[i]} ${[0, 1, 2].map((c) => (f.positions[3 * i + c] + sh[c]).toFixed(6)).join(" ")}`);
    return `${picks.length}\nProperties=species:S:1:pos:R:3 step=${k}\n${lines.join("\n")}`;
  }).join("\n") + "\n";
  return { text, nAtoms: picks.length };
}

// ------------------------------------------------------------------ scenes
// Each scene is built once and kept; changing chapter toggles visibility. Building runs
// the viewer's pixel auto-fit, which blocks the main thread (about 0.5 s on a GPU), so
// a build paints a loading state first, and the next chapter's scene is prebuilt when
// the reader pauses. On software WebGL (no GPU) idle prebuilds are skipped, because a
// build there blocks for tens of seconds, and ambient occlusion is turned off.
const SCENES = {
  relax: () => ({ text: parsed.crop.text, opts: { view: { elevation: 18, azimuth: 0 }, replicate: [1, 1], restFrame: "first", fill: 0.8 } }),
  committee: () => ({ text: xyz.relax, opts: { view: "tilted", restFrame: "last" }, sigma: true }),
  neb: () => ({ text: xyz.neb, opts: { view: { elevation: 64, azimuth: -30 }, restFrame: "first" } }),
  ts: () => ({ text: xyz.ts, opts: { view: "tilted", restFrame: "first", fps: 24, hold: 0, loop: true } }),
  water: () => ({ text: xyz.water, opts: { view: "tilted", style: "ballstick", restFrame: "first" } }),
};
const ORDER = ["relax", "committee", "neb", "ts", "water"];
const sceneHost = $("#scene");
const clip = $("#clip");
const cur = { key: null, ctrl: null };
const built = new Map(); // key -> {box, ctrl, viewer, spec}
let buildChain = Promise.resolve();
let wantKey = null;
let software = false;

function webglRenderer() {
  try {
    const g = document.createElement("canvas").getContext("webgl2") || document.createElement("canvas").getContext("webgl");
    const e = g && g.getExtension("WEBGL_debug_renderer_info");
    return e ? String(g.getParameter(e.UNMASKED_RENDERER_WEBGL)) : "";
  } catch { return ""; }
}

// restyle hook (viewer.js runs it after the default style on every rebuild)
function sigmaRestyle(viewer) {
  const s = pack.committee.series.peratom_final_sigma_free_eV_per_A;
  const smax = Math.max(...s);
  for (const e of ["Pt", "O"]) {
    const st = V.ELEMENT_STYLE[e];
    viewer.setStyle({ elem: e }, { sphere: { radius: st.space, colorfunc: (a) => (e === "O" ? st.color : ramp((s[a.properties.src] || 0) / smax)) } });
  }
}

// Fixed scene aspect per layout, so a framing camera baked once per scene and layout
// (data/cameras.json, made by bake_cameras.py) can be reused: loads then skip the
// viewer's pixel auto-fit, the slow part of a build.
const ASPECT = { desk: 1.5, mobile: 1.0 }; // phones: near-square box, ~45svh with the readouts
const layoutKey = () => (window.innerWidth <= 900 ? "mobile" : "desk");
const BAKE = new URLSearchParams(location.search).has("bake");
let cameras = null;
let builtLayout = null;
function sizeScene() {
  const vp = $("#viewport");
  const top = layoutKey() === "mobile" ? 24 : 28; // the readout strip sits above the atoms
  const W = vp.clientWidth, H = Math.max(10, vp.clientHeight - top);
  const A = ASPECT[layoutKey()];
  const w = Math.min(W, H * A), h = w / A;
  Object.assign(sceneHost.style, {
    left: `${Math.round((W - w) / 2)}px`, top: `${Math.round(top + (H - h) / 2)}px`,
    width: `${Math.round(w)}px`, height: `${Math.round(h)}px`, right: "auto", bottom: "auto",
  });
  sceneSize = `${Math.round(w)}x${Math.round(h)}`;
}
let sceneSize = "";
// keep the visible canvas matched to its box (the caption height differs per chapter)
function syncSceneSize() {
  const e = built.get(cur.key);
  if (e && e.ctrl && e.size !== sceneSize) { e.size = sceneSize; e.ctrl.resize(); }
}
const nextPaint = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

function build(key) {
  if (built.has(key)) return Promise.resolve(built.get(key));
  const job = buildChain.then(async () => {
    if (built.has(key)) return built.get(key);
    const spec = SCENES[key]();
    const box = document.createElement("div");
    box.className = "scene-box";
    box.style.visibility = "hidden";
    sceneHost.appendChild(box);
    const entry = { key, box, spec, ctrl: null, viewer: null };
    const cam = !BAKE && cameras?.[builtLayout]?.[key];
    try {
      entry.ctrl = await V.loadScene(box, null, {
        xyzText: spec.text, background: "transparent", replicate: [2, 2], fill: layoutKey() === "mobile" ? 0.95 : 0.86,
        ambientOcclusion: software ? false : 0.4,
        ...spec.opts,
        ...(cam ? { fit: false, camera: cam } : {}),
        restyle: spec.sigma ? sigmaRestyle : null,
        onFrame: (i) => { if (cur.key === key) onSceneFrame(key, i); },
      });
      entry.viewer = entry.ctrl.viewer;
      entry.baked = !!cam;
      entry.size = sceneSize;
    } catch (err) {
      console.warn("scene failed to load:", err);
      entry.failed = true;
    }
    built.set(key, entry);
    return entry;
  });
  buildChain = job.catch(() => {});
  return job;
}

async function showScene(key) {
  wantKey = key;
  if (key === "video") {
    for (const e of built.values()) { e.box.style.visibility = "hidden"; e.ctrl?.pause(); }
    sceneHost.classList.remove("swapping");
    cur.key = key; cur.ctrl = null;
    clip.hidden = false;
    if (!clip.getAttribute("src")) { clip.preload = "metadata"; clip.src = "data/" + pack.webapp.video; }
    if (!reduced) clip.play().catch(() => {});
    return;
  }
  clip.pause();
  clip.hidden = true;
  if (!built.has(key)) {
    sceneHost.classList.add("swapping");
    $("#loading").hidden = false;
    await nextPaint(); // let the loading state paint before the blocking build
    await build(key);
    if (wantKey !== key) return;
  }
  const entry = built.get(key);
  for (const e of built.values()) {
    e.box.style.visibility = e === entry ? "visible" : "hidden";
    if (e !== entry) e.ctrl?.pause();
  }
  $("#fallback").hidden = !entry.failed;
  $("#loading").hidden = true;
  sceneHost.classList.remove("swapping");
  cur.key = key;
  cur.ctrl = entry.ctrl;
  syncSceneSize();
  applyProgress(true);
  schedulePrebuild();
}
function requestScene(key) { if (key !== wantKey) showScene(key); }

// Prebuild the next unbuilt scene once the reader has paused for a moment.
let idleTimer = 0;
function schedulePrebuild() {
  clearTimeout(idleTimer);
  if (software) return;
  idleTimer = setTimeout(() => {
    const from = Math.max(0, ORDER.indexOf(cur.key));
    const next = [...ORDER.slice(from + 1), ...ORDER.slice(0, from)].find((k) => !built.has(k));
    if (!next) return;
    const go = () => build(next).then(schedulePrebuild);
    if ("requestIdleCallback" in window) requestIdleCallback(go, { timeout: 2000 }); else go();
  }, 1200);
}
function dropScenes(all) {
  // scenes framed for another layout (breakpoint crossed) are rebuilt; without a baked
  // camera, a resize would refit every kept scene, so keep only the visible one then
  for (const [k, e] of [...built]) {
    if (!all && (k === cur.key || e.baked)) continue;
    e.ctrl?.destroy();
    e.box.remove();
    built.delete(k);
  }
}

// HUD readouts that follow the frame shown.
function onSceneFrame(key, i) {
  if (key === "relax" && (active === "relax" || active === "intro")) relaxHud(i);
  if (key === "neb" && active === "neb") nebHud(i);
}
function relaxHud(i) {
  const s = pack.relax.series;
  hud([
    ["step", `${i} of ${s.step.length - 1}`],
    ["O height", withUnit(parsed.relaxH[i], 3, "Å")],
    ["fmax", withUnit(s.fmax_eV_per_A[i], 4, "eV/Å")],
    ["E − E₀", withUnit(s.energy_eV[i] - s.energy_eV[0], 3, "eV")],
  ]);
}
function nebHud(i) {
  const s = pack.neb.series;
  hud([
    ["image", `${i} of ${s.image_index.length - 1}`],
    ["E − E(fcc)", withUnit(s.relative_energy_eV[i], 3, "eV")],
    ["path length", withUnit(s.reaction_coordinate_A[i], 2, "Å")],
    ["O height", withUnit(parsed.nebH[i], 2, "Å")],
  ]);
}

// ------------------------------------------------------------------ chapters + scroll
const CHAPTERS = {
  intro: { scene: "relax" },
  relax: { scene: "relax" },
  committee: { scene: "committee" },
  neb: { scene: "neb" },
  ts: { scene: "ts" },
  water: { scene: "water" },
  remote: { scene: "video" },
};
let active = null;
let lastIdx = -1;
const sections = $$("[data-ch]").filter((e) => e.tagName === "SECTION");
const railLinks = $$("#rail a");
const playBtn = $("#play");

function probeY() {
  const stage = $("#stage").getBoundingClientRect();
  const mobile = window.innerWidth <= 900;
  return mobile ? stage.bottom + (window.innerHeight - stage.bottom) * 0.3 : window.innerHeight * 0.5;
}
function progressOf(sec) {
  const r = sec.getBoundingClientRect();
  return Math.max(0, Math.min(1, (probeY() - r.top) / Math.max(1, r.height)));
}
const ease = (p, a, b) => Math.max(0, Math.min(1, (p - a) / (b - a)));

function enterChapter(ch) {
  active = ch;
  lastIdx = -1;
  const order = Object.keys(CHAPTERS).slice(1);
  const at = order.indexOf(ch);
  railLinks.forEach((a, i) => {
    if (i === at) a.setAttribute("aria-current", "step"); else a.removeAttribute("aria-current");
    a.classList.toggle("done", at >= 0 && i < at);
  });
  legend(ch === "committee");
  $("#rail-now").textContent = at >= 0 ? `${at + 1} ${railLinks[at].querySelector(".rail-t").textContent}` : "";
  sceneHost.setAttribute("aria-label", ch === "water"
    ? `3D view, static structure, not a simulation: ${pack.md.structures?.built?.label || "water bilayer as built"}`
    : SCENE_LABELS[ch] || "");
  playBtn.hidden = ch !== "ts";
  sceneCaption(ch);
  requestScene(CHAPTERS[ch].scene);
  if (ch === "committee") commHud();
  if (ch === "ts") tsHud();
  if (ch === "water") waterHud();
  if (ch === "remote") hud([]);
  if (ch !== "remote" && clip && !clip.paused) clip.pause();
}
const SCENE_LABELS = {
  intro: "3D side view of the O atom at the fcc hollow of Pt(111) and the Pt atoms near it",
  relax: "3D side view of the O atom and nearby Pt atoms at the current BFGS step",
  committee: "3D view of the Pt(111) slab with each atom coloured by the committee's force disagreement",
  neb: "3D view from above of the NEB band image for the current scroll position",
  ts: "3D view of the transition state moving along its imaginary mode",
  water: "",
  remote: "",
};
function sceneCaption(ch) {
  // one rough line per scene (ruling AG); numbers still come from the pack
  const t = {
    intro: "O on Pt(111), fcc hollow; slice through the slab",
    relax: "BFGS relaxation, step by step",
    committee: "Committee force disagreement σ, on ch. 1 geometry",
    neb: `O hops fcc → hcp; barrier ${knText("neb", "forward barrier", 2)}`,
    ts: `Imaginary mode, ${parsed.nuText}; mode, not dynamics`,
    water: "Water bilayer as built; static, not a simulation",
    remote: `Web app to a remote GPU; waiting part ×${num(knv("webapp", "video speed-up"), 0)}`,
  }[ch];
  const cap = $("#scene-cap");
  cap.textContent = t || "";
  // the water caption is short (ruling AG); the md.json label stays reachable verbatim
  if (ch === "water" && pack.md.structures?.built?.label) cap.title = pack.md.structures.built.label;
  else cap.removeAttribute("title");
}
function commHud() {
  hud([
    ["members", num(knv("committee", "committee members"), 0)],
    ["σmax, final", knText("committee", "sigma_max_free, final step", 3)],
    ["σmax / fmax", num(knv("committee", "sigma_max_free / fmax"), 1)],
    ["worst atom", (pack.committee.notes.find((n) => n.startsWith("Worst free atom")) || "").replace(/^.*?:\s*/, "").replace(/\.$/, "")],
  ]);
}
function tsHud() {
  hud([
    ["imaginary mode", parsed.nuText],
    ["overlap", `${num(knv("neb", "overlap of imaginary mode"), 3)} (≥ ${num(knv("neb", "overlap threshold"), 1)})`],
    ["TS fmax", knText("neb", "TS fmax", 4)],
    ["verdict", pack.neb.flags.includes("ts_verified") ? "TS verified" : "not verified"],
  ]);
}
function waterHud() {
  const pending = pack.md.status === "pending";
  hud([
    ["status", pending ? "MD pending" : pack.md.status || "", pending ? "pending" : ""],
    ["water molecules", String(parsed.nWater)],
    ["frames", pack.md.frames ? "from md.json" : "none (static structure)"],
  ]);
}

function applyProgress(force = false) {
  const y = probeY();
  let ch = "intro";
  for (const s of sections) {
    const r = s.getBoundingClientRect();
    if (r.top <= y) ch = s.dataset.ch;
  }
  if (ch !== active) enterChapter(ch);
  const sec = sections.find((s) => s.dataset.ch === ch);
  const p = progressOf(sec);
  const c = cur.key === CHAPTERS[ch].scene ? cur.ctrl : null;
  if (ch === "intro") {
    if (force || lastIdx !== 0) { plotRelax(0); lastIdx = 0; }
    if (c && c.frame !== 0) c.setFrame(0); else relaxHud(0);
  } else if (ch === "relax") {
    const n = pack.relax.series.step.length - 1;
    const k = Math.round(ease(p, 0.04, 0.72) * n);
    if (force || k !== lastIdx) { plotRelax(k); lastIdx = k; }
    if (c && c.frame !== k) c.setFrame(k); else relaxHud(k);
  } else if (ch === "committee") {
    const n = pack.committee.series.step.length - 1;
    const k = Math.round(ease(p, 0.02, 0.6) * n);
    if (force || k !== lastIdx) { plotCommittee(k); lastIdx = k; }
  } else if (ch === "neb") {
    const n = pack.neb.series.image_index.length - 1;
    const k = Math.round(ease(p, 0.04, 0.7) * n);
    if (force || k !== lastIdx) { plotNeb(k); lastIdx = k; }
    if (c && c.frame !== k) c.setFrame(k); else if (c) nebHud(k);
  } else if (ch === "ts") {
    if (force || lastIdx !== 0) { plotTs(); lastIdx = 0; }
    if (c && !c.playing && !reduced && !userPaused) { c.play(); playBtn.textContent = "Pause the mode"; }
    if (c && reduced && !c.playing) playBtn.textContent = "Play the mode";
  } else if (ch === "water") {
    if (force || lastIdx !== 0) { plotWater(); lastIdx = 0; }
    if (c && pack.md.frames && !c.playing && !reduced) c.play();
  } else if (ch === "remote") {
    if (force || lastIdx !== 0) { plotRemote(); lastIdx = 0; }
  }
}

let userPaused = false;
playBtn.addEventListener("click", () => {
  const c = cur.key === "ts" ? cur.ctrl : null;
  if (!c) return;
  if (c.playing) { c.pause(); userPaused = true; playBtn.textContent = "Play the mode"; }
  else { c.play({ force: true }); userPaused = false; playBtn.textContent = "Pause the mode"; }
});

let ticking = false;
function onScroll() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => { ticking = false; applyProgress(); });
}

// ------------------------------------------------------------------ bindings
function bindText() {
  for (const e of $$("[data-k]")) {
    const [name, label] = e.dataset.k.split(/:(.*)/s);
    e.textContent = knText(name, label, +(e.dataset.dp ?? 3), e.dataset.unit);
  }
  for (const e of $$("[data-cmd]")) {
    const [name, flag] = e.dataset.cmd.split(":");
    e.textContent = cmdFlag(pack[name].command, flag);
  }
  for (const e of $$("[data-note]")) {
    const [name, i] = e.dataset.note.split(":");
    e.textContent = pack[name].notes?.[+i] || "";
  }
  for (const e of $$("[data-state]")) {
    const d = pack[e.dataset.state];
    if (e.dataset.verdict === "ts") {
      const v = d.flags.includes("ts_verified");
      e.textContent = v ? "TS verified" : "TS not verified";
      e.classList.add(v ? "ok" : "no");
    } else if (d.status === "pending") { e.textContent = "pending"; e.classList.add("pending"); }
    else if (d.converged) { e.textContent = "converged"; e.classList.add("ok"); }
    else { e.textContent = "not converged"; e.classList.add("no"); }
  }
  for (const e of $$("[data-command]")) renderCommand(e, e.dataset.command);

  const set = (sel, t) => { const e = $(sel); if (e) e.textContent = t; };
  // relax, computed from the frames and series
  const rs = pack.relax.series;
  const E = rs.energy_eV;
  set("#relax-de", withUnit(E[E.length - 1] - E[0], 3, "eV"));
  set("#relax-h", `${num(parsed.relaxH[0], 2)} → ${num(parsed.relaxH[parsed.relaxH.length - 1], 2)}${NB}Å`);

  // committee
  const cs = pack.committee.series;
  set("#comm-n", num(knv("committee", "committee members"), 0));
  const members = pack.committee.key_numbers.filter((k) => k.label.startsWith("final energy, member "))
    .map((k) => k.label.replace(/^final energy, member /, "").replace(/ \(.*$/, ""));
  set("#comm-members", members.join(", "));
  const cross = cs.step.findIndex((_, i) => cs.fmax_eV_per_A[i] < cs.sigma_max_free_eV_per_A[i]);
  set("#comm-cross", cross >= 0 ? `from step ${num(cs.step[cross], 0)}` : "never");

  // ts
  const nu = kn("neb", "TS imaginary frequency");
  parsed.nuText = nu ? `${num(nu.value, 1)}i${NB}${nu.unit}` : "n/a";
  set("#ts-nu", parsed.nuText);
  set("#ts-overlap", `${num(knv("neb", "overlap of imaginary mode"), 3)} (≥${NB}${num(knv("neb", "overlap threshold"), 1)})`);
  const flagged = pack.neb.flags.includes("ts_verify_endpoint_fmax_above_relax_fmax");
  set("#ts-flag", flagged ? pack.neb.notes[0] : "");
  const more = $("#ts-flag")?.closest("details");
  if (more) more.hidden = !flagged;

  // water
  set("#water-n", `${parsed.nWater} H₂O`);
  if (pack.md.status !== "pending") set("#water-lede", `MD status: ${pack.md.status || (pack.md.converged ? "done" : "not converged")}.`);

  // remote
  set("#remote-machine", `${pack.webapp.target?.machine || ""} (showcase)`);
  set("#remote-wall", `${knText("webapp", "job wall time", 1)} (relax ${knText("webapp", "relaxation wall time", 1)})`);
  set("#remote-de", knText("webapp", "energy difference, web app − CLI", 1));
}

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    V = await import(new URL(viewerPath, document.baseURI).href);
    const names = ["relax", "committee", "neb", "md", "webapp"];
    const got = await Promise.all(names.map((n) => fetch(`data/${n}.json`).then((r) => {
      if (!r.ok) throw new Error(`data/${n}.json: HTTP ${r.status}`);
      return r.json();
    })));
    pack = Object.fromEntries(names.map((n, i) => [n, got[i]]));
    const files = { relax: pack.relax.frames, neb: pack.neb.frames, ts: pack.neb.frames_extra.ts_mode,
      water: pack.md.frames || pack.md.structures.built.file };
    await Promise.all(Object.entries(files).map(async ([k, f]) => {
      const r = await fetch("data/" + f);
      if (!r.ok) throw new Error(`data/${f}: HTTP ${r.status}`);
      xyz[k] = await r.text();
    }));
  } catch (err) {
    console.warn(err);
    $("#scene-cap").textContent = `The data pack could not be loaded (${err.message}).`;
    return;
  }
  software = /swiftshader|llvmpipe|software|basic render/i.test(webglRenderer());
  parsed.relax = V.parseExtXYZ(xyz.relax);
  parsed.crop = cropRelax(parsed.relax);
  parsed.relaxH = oHeights(parsed.relax);
  parsed.nebH = oHeights(V.parseExtXYZ(xyz.neb));
  parsed.nWater = V.parseExtXYZ(xyz.water)[0].symbols.filter((s) => s === "O").length;
  bindText();
  try { const r = await fetch("data/cameras.json"); if (r.ok) cameras = await r.json(); } catch { /* fit instead */ }
  builtLayout = layoutKey();
  sizeScene();
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => { if (layoutKey() === builtLayout) { sizeScene(); syncSceneSize(); } }).observe($("#viewport"));
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  let rt = 0;
  window.addEventListener("resize", () => {
    sizeScene();
    if (layoutKey() !== builtLayout) {
      builtLayout = layoutKey();
      dropScenes(true);
      const k = cur.key;
      cur.key = null; wantKey = null;
      if (k && k !== "video") requestScene(k);
    } else {
      dropScenes(false);
    }
    clearTimeout(rt);
    rt = setTimeout(() => { if (active) sceneCaption(active); applyProgress(true); schedulePrebuild(); }, 160);
  });
  applyProgress(true);
  window.__story = {
    get ctrl() { return cur.ctrl; }, get scene() { return cur.key; }, get chapter() { return active; },
    get built() { return [...built.keys()]; }, get software() { return software; },
    get playingAny() { return [...built.values()].some((e) => e.ctrl && e.ctrl.playing); },
    clip, reduced,
    get baked() { return [...built.values()].filter((e) => e.baked).map((e) => e.key); },
  };
  if (BAKE) { // bake_cameras.py: build every scene with the auto-fit, report the cameras
    for (const k of ORDER) await build(k);
    window.__cameras = { [builtLayout]: Object.fromEntries([...built].map(([k, e]) => [k, e.ctrl.getCamera()])) };
  }
}
boot();
