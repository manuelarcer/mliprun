// mliprun front page, alternative 1 (terminal).
// Every number comes from data/*.json (the showcase pack) and every terminal line from
// data/logs/* (the real run logs, copied verbatim by build.py). Nothing is typed in here.
import { loadScene } from "../shared/viewer.js";

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const DARK = () => getComputedStyle(document.documentElement).colorScheme.includes("dark");
const $ = (s, r = document) => r.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const MODE_VIEW = { elevation: 90, azimuth: -30, zoom: 4.5 }; // same zoom as build.py MODE_ZOOM

// ------------------------------------------------------------------ data
async function getJSON(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}
async function getText(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.text();
}

/** Key number by exact label, else by unique label prefix; throws if the pack changed. */
function kn(d, prefix) {
  const exact = d.key_numbers.filter((k) => k.label === prefix);
  const hits = exact.length ? exact : d.key_numbers.filter((k) => k.label.startsWith(prefix));
  if (hits.length !== 1) throw new Error(`${d.use_case}: ${hits.length} key numbers match "${prefix}"`);
  return hits[0];
}
const fx = (v, nd) => Number(v).toFixed(nd);
const withUnit = (k, nd) => `${fx(k.value, nd)} ${k.unit === "dimensionless" ? "" : k.unit}`.trim();

/** Ruling AC: the MD state in words, bound to the md.json flags. */
function mdStatus(md) {
  const p = [];
  if ((md.flags || []).includes("md_not_run")) p.push("MD not run");
  if ((md.flags || []).includes("pending_scientist_decision")) p.push("next step pending a decision");
  return p.join("; ") || md.status;
}

/** '/a/b/c/d/e' -> '…/c/d/e' (same rule as build.py). */
function elide(tok) {
  if (!tok.startsWith("/")) return tok;
  const parts = tok.split("/").filter(Boolean);
  return parts.length > 3 ? "…/" + parts.slice(-3).join("/") : tok;
}
const elideText = (s) => s.replace(/(^|[\s(=])(\/[^\s)]+)/g, (m, a, p) => a + elide(p));

/** Command -> shell continuation lines: head, then flags packed to `width` chars. */
function cmdLines(cmd, width = 54) {
  const toks = cmd.split(/\s+/).map(elide);
  let i = 0;
  const head = [];
  while (i < toks.length && !toks[i].startsWith("--")) head.push(toks[i++]);
  const groups = [];
  while (i < toks.length) {
    const g = [toks[i++]];
    while (i < toks.length && !toks[i].startsWith("--")) g.push(toks[i++]);
    // a flag + value too long for one line goes on two lines (still valid shell)
    if (g.join(" ").length > width && g.length > 1) groups.push(g[0], g.slice(1).join(" "));
    else groups.push(g.join(" "));
  }
  const lines = [head.join(" ")];
  let cur = "";
  for (const g of groups) {
    if (cur && cur.length + 1 + g.length > width) { lines.push(cur); cur = g; }
    else cur = (cur + " " + g).trim();
  }
  if (cur) lines.push(cur);
  return lines.map((l, k) => l + (k < lines.length - 1 ? " \\" : ""));
}

const hms = (s) => { const [h, m, x] = s.split(":").map(Number); return h * 3600 + m * 60 + x; };
const dur = (s) => (s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`);

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function sv(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
/** Append text; the superscript "⁻¹" (missing from the mono face) is drawn as <sup>−1</sup>. */
function txt(node, s) {
  const parts = s.split("⁻¹");
  parts.forEach((p, i) => {
    if (p) node.append(document.createTextNode(p));
    if (i < parts.length - 1) node.append(el("sup", { class: "u" }, "−1"));
  });
  return node;
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo;
  let step = 10 ** Math.floor(Math.log10(span / n));
  for (const m of [1, 2, 2.5, 5, 10]) if (span / (m * step) <= n) { step *= m; break; }
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
const tickLabel = (v) => (Math.abs(v) < 1e-12 ? "0" : String(+v.toPrecision(3)));

// ------------------------------------------------------------------ command rendering
/** Width of one monospace character inside `box`, measured in its real (loaded) font. */
function chWidth(box) {
  const probe = el("span", { style: "position:absolute;visibility:hidden;white-space:pre" }, "0".repeat(40));
  box.append(probe);
  const w = probe.getBoundingClientRect().width / 40;
  probe.remove();
  return w || 7.5;
}
/** Continuation lines that fit `box`: 4 leading spaces + flags + " \" must fit one line. */
function fitChars(box) {
  const cs = getComputedStyle(box);
  const avail = box.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  return Math.max(16, Math.floor(avail / chWidth(box)) - 7);
}
/** Render `cmd` into `box` as continuation lines. CSS hangs and wraps any overlong token. */
function renderCmd(box, cmd) {
  box.replaceChildren();
  cmdLines(cmd, Math.min(60, fitChars(box))).forEach((t, k) => {
    const d = el("div", { class: "l" + (k ? " cont" : "") });
    if (!k) d.append(el("span", { class: "p" }, "$ "));
    d.append(document.createTextNode((k ? "    " : "") + t));
    box.append(d);
  });
}
const cmdRegistry = []; // [box, cmd] pairs re-wrapped when fonts load or the window resizes
function registerCmd(box, cmd) {
  cmdRegistry.push([box, cmd]);
  renderCmd(box, cmd);
}
function rewrapAll() {
  for (const [box, cmd] of cmdRegistry) renderCmd(box, cmd);
  hero?.rewrap();
}

// ------------------------------------------------------------------ charts
const fitW = (box, max = 640) => Math.round(Math.max(330, Math.min(max, box.clientWidth || max)));

/** Line chart with labelled axes. Returns {svg, X, Y, plot, m, pw, ph} for overlays. */
function lineChart({ w = 640, h = 240, x, series, logy = false, xlabel, ylabel, target, ymin, ymax }) {
  const m = { l: 56, r: 14, t: w < 560 && series.length > 1 ? 46 : 28, b: 42 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const ys = series.flatMap((s) => s.y).concat(target ? [target.y] : []);
  const T = logy ? Math.log10 : (v) => v;
  let lo = ymin ?? (logy ? Math.floor(Math.log10(Math.min(...ys))) : Math.min(...ys));
  let hi = ymax ?? (logy ? Math.log10(Math.max(...ys) * 1.4) : Math.max(...ys));
  if (!logy && ymin === undefined) { const p = (hi - lo) * 0.08; lo -= p; hi += p; }
  const x0 = Math.min(...x), x1 = Math.max(...x);
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * pw;
  const Y = (v) => m.t + ((hi - T(v)) / (hi - lo)) * ph;
  const svg = sv("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  const yt = logy
    ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, k) => 10 ** (Math.ceil(lo) + k))
    : niceTicks(lo, hi, 4);
  for (const v of yt) {
    svg.append(sv("line", { class: "grid", x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v) }));
    svg.append(sv("text", { x: m.l - 7, y: Y(v) + 4, "text-anchor": "end" }, tickLabel(v)));
  }
  for (const v of niceTicks(x0, x1, 6)) {
    svg.append(sv("text", { x: X(v), y: m.t + ph + 16, "text-anchor": "middle" }, tickLabel(v)));
  }
  svg.append(sv("line", { class: "axis", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }));
  svg.append(sv("text", { x: m.l + pw / 2, y: h - 6, "text-anchor": "middle" }, xlabel));
  svg.append(sv("text", { transform: `translate(14 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle" }, ylabel));
  const plot = sv("g");
  svg.append(plot);
  if (target) {
    // label at the left, under the line: the data are far above the target there
    svg.append(sv("line", { class: "target", x1: m.l, x2: m.l + pw, y1: Y(target.y), y2: Y(target.y) }));
    svg.append(sv("text", { class: "t-ok", x: m.l + 6, y: Y(target.y) + 14 }, target.label));
  }
  let lx = m.l, ly = 10;
  for (const s of series) {
    if (lx + 34 + 6.2 * s.label.length > w) { lx = m.l; ly += 16; }
    svg.append(sv("polyline", { class: s.cls, points: x.map((v, i) => `${X(v)},${Y(s.y[i])}`).join(" ") }));
    if (s.dot) for (let i = 0; i < x.length; i++) svg.append(sv("circle", { class: s.dot, cx: X(x[i]), cy: Y(s.y[i]), r: 2.4 }));
    svg.append(sv("line", { class: s.cls, x1: lx, x2: lx + 18, y1: ly, y2: ly }));
    svg.append(sv("text", { class: "ink", x: lx + 23, y: ly + 4 }, s.label));
    lx += 34 + 6.2 * s.label.length;
  }
  return { svg, X, Y, plot, m, pw, ph };
}

/** NEB profile on the converged band's scale; a live "highest image" line while the log
 *  streams (pinned at the top edge with its value while it is above the axis), then the band. */
function profileChart(box, neb) {
  const w = fitW(box, 480), h = 290, m = { l: 54, r: 18, t: 22, b: 46 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const xs = neb.series.reaction_coordinate_A, ys = neb.series.relative_energy_eV;
  const ts = neb.ts_image_index;
  const xmax = Math.ceil(Math.max(...xs) * 2) / 2;
  const ymin = -0.1, ymax = Math.ceil(Math.max(...ys) * 10) / 10 + 0.05;
  const X = (v) => m.l + (v / xmax) * pw;
  const Y = (v) => m.t + ((ymax - v) / (ymax - ymin)) * ph;
  const svg = sv("svg", { viewBox: `0 0 ${w} ${h}`, role: "img",
    "aria-label": "Energy of each NEB image relative to the initial state, against the reaction coordinate" });
  for (const v of niceTicks(ymin, ymax, 5)) {
    svg.append(sv("line", { class: "grid", x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v) }));
    svg.append(sv("text", { x: m.l - 7, y: Y(v) + 4, "text-anchor": "end" }, tickLabel(v)));
  }
  for (const v of niceTicks(0, xmax, 4)) svg.append(sv("text", { x: X(v), y: m.t + ph + 17, "text-anchor": "middle" }, tickLabel(v)));
  svg.append(sv("line", { class: "axis", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }));
  svg.append(sv("text", { x: m.l + pw / 2, y: h - 6, "text-anchor": "middle" }, "reaction coordinate (Å)"));
  const yl = sv("text", { transform: `translate(15 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle" });
  yl.append("E − E", sv("tspan", { dy: 3, "font-size": 9 }, "IS"), sv("tspan", { dy: -3 }, " (eV)"));
  svg.append(yl);

  const live = sv("g");
  const liveLine = sv("line", { class: "live", x1: m.l, x2: m.l + pw, y1: Y(0), y2: Y(0) });
  const liveText = sv("text", { class: "lbl", x: m.l + pw - 2, y: Y(0) - 6, "text-anchor": "end" }, "");
  live.append(liveLine, liveText);
  live.style.opacity = 0;
  svg.append(live);

  const band = sv("g");
  band.style.opacity = 0;
  band.append(sv("line", { class: "zero", x1: X(0), x2: X(xs[ts]), y1: Y(0), y2: Y(0) }));
  band.append(sv("line", { class: "brk", x1: X(xs[ts]), x2: X(xs[ts]), y1: Y(0), y2: Y(ys[ts]) + 7 }));
  const fwd = kn(neb, "forward barrier"), dE = kn(neb, "reaction energy");
  const ea = sv("text", { class: "lbl", x: X(xs[ts]) - 8, y: Y(ys[ts] / 2) + 4, "text-anchor": "end" });
  ea.append("E", sv("tspan", { dy: 3, "font-size": 9 }, "a"), sv("tspan", { dy: -3 }, ` = ${withUnit(fwd, 3)}`));
  band.append(ea);
  const n = xs.length - 1;
  band.append(sv("text", { class: "lbl", x: X(xs[n]), y: Y(ys[n]) + 34, "text-anchor": "end" }, `ΔE = +${withUnit(dE, 3)}`));
  band.append(sv("text", { x: X(xs[0]) + 4, y: Y(0) + 18 }, "O fcc (IS)"));
  band.append(sv("text", { x: X(xs[n]), y: Y(ys[n]) + 18, "text-anchor": "end" }, "O hcp (FS)"));
  band.append(sv("text", { x: X(xs[ts]), y: Y(ys[ts]) - 12, "text-anchor": "middle" }, `TS, image ${ts}`));
  svg.append(band);
  const path = sv("polyline", { class: "path", points: xs.map((v, i) => `${X(v)},${Y(ys[i])}`).join(" ") });
  let len = 0;
  for (let i = 1; i < xs.length; i++) len += Math.hypot(X(xs[i]) - X(xs[i - 1]), Y(ys[i]) - Y(ys[i - 1]));
  path.style.strokeDasharray = `${len + 4}`;
  path.style.strokeDashoffset = `${len + 4}`;
  svg.append(path);
  const pts = sv("g");
  pts.style.opacity = 0;
  xs.forEach((v, i) => pts.append(sv("circle", { class: i === ts ? "pt ts" : "pt", cx: X(v), cy: Y(ys[i]), r: i === ts ? 4.5 : 3.2 })));
  svg.append(pts);
  box.replaceChildren(svg);

  return {
    live(step, barrier) {
      live.style.opacity = 1;
      const off = barrier > ymax;
      const y = off ? m.t : Y(barrier);
      liveLine.setAttribute("y1", y); liveLine.setAttribute("y2", y);
      liveText.setAttribute("y", off ? m.t - 6 : y - 6);
      liveText.textContent = off
        ? `↑ highest image at step ${step}: ${fx(barrier, 3)} eV (above the axis)`
        : `highest image at step ${step}: ${fx(barrier, 3)} eV`;
    },
    reset() {
      live.style.opacity = 0; band.style.opacity = 0; pts.style.opacity = 0;
      path.style.transition = "none"; path.style.strokeDashoffset = `${len + 4}`;
    },
    reveal(animate) {
      live.style.opacity = 0;
      if (animate) {
        path.getBoundingClientRect();
        path.style.transition = "stroke-dashoffset 1.4s cubic-bezier(.3,.1,.2,1)";
        band.style.transition = pts.style.transition = "opacity .5s 1.1s";
      } else {
        path.style.transition = band.style.transition = pts.style.transition = "none";
      }
      path.style.strokeDashoffset = "0";
      band.style.opacity = 1; pts.style.opacity = 1;
    },
  };
}

/** Barrier per band step (neb.log Step lines), drawn point by point as the log streams. */
function traceChart(box, steps, finalLabel) {
  const w = fitW(box, 480), h = 150, m = { l: 54, r: 18, t: 10, b: 40 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const ks = steps.map((s) => s.k), bs = steps.map((s) => s.barrier);
  const kmax = Math.max(...ks);
  const t = niceTicks(Math.min(...bs), Math.max(...bs), 3);
  const lo = Math.min(t[0], Math.min(...bs)), hi = Math.max(t.at(-1), Math.max(...bs));
  const X = (k) => m.l + (k / kmax) * pw;
  const Y = (v) => m.t + ((hi - v) / (hi - lo)) * ph;
  const svg = sv("svg", { viewBox: `0 0 ${w} ${h}`, role: "img",
    "aria-label": "Barrier reported by neb.log at each band step" });
  for (const v of t) {
    svg.append(sv("line", { class: "grid", x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v) }));
    svg.append(sv("text", { x: m.l - 7, y: Y(v) + 4, "text-anchor": "end" }, tickLabel(v)));
  }
  for (const v of niceTicks(0, kmax, 5)) svg.append(sv("text", { x: X(v), y: m.t + ph + 16, "text-anchor": "middle" }, tickLabel(v)));
  svg.append(sv("line", { class: "axis", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }));
  svg.append(sv("text", { x: m.l + pw / 2, y: h - 5, "text-anchor": "middle" }, "band step"));
  svg.append(sv("text", { transform: `translate(15 ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle" }, "barrier (eV)"));
  const line = sv("polyline", { class: "s-acc-solid", points: "" });
  const dot = sv("circle", { class: "d-acc", r: 3, cx: X(0), cy: Y(steps[0].barrier), visibility: "hidden" });
  const end = sv("text", { class: "lbl", x: X(steps.at(-1).k), y: Y(steps.at(-1).barrier) - 8, "text-anchor": "end",
    visibility: "hidden" }, finalLabel);
  svg.append(line, dot, end);
  box.replaceChildren(svg);
  return {
    upto(i) {
      line.setAttribute("points", steps.slice(0, i + 1).map((s) => `${X(s.k)},${Y(s.barrier)}`).join(" "));
      dot.setAttribute("visibility", i >= 0 ? "visible" : "hidden");
      if (i >= 0) { dot.setAttribute("cx", X(steps[i].k)); dot.setAttribute("cy", Y(steps[i].barrier)); }
      end.setAttribute("visibility", i === steps.length - 1 ? "visible" : "hidden");
    },
  };
}

// ------------------------------------------------------------------ hero terminal
function buildTranscript(neb, logs) {
  const stdout = logs.stdout.split("\n");
  const nebLog = logs.neblog.split("\n");
  const runlog = logs.runlog.split("\n");
  const L = [];
  const add = (cls, text, extra = {}) => L.push({ cls, text, ...extra });

  const fire = nebLog.filter((s) => s.startsWith("FIRE:"));
  const tFirst = hms(fire[0].split(/\s+/)[2]);
  const tLast = hms(fire[fire.length - 1].split(/\s+/)[2]);
  const tStart = hms(runlog.find((s) => s.includes("neb run (")).match(/T(\d\d:\d\d:\d\d)/)[1]);
  const tExit = hms(runlog.find((s) => s.includes("neb run exited")).match(/T(\d\d:\d\d:\d\d)/)[1]);

  const iUma = stdout.findIndex((s) => s.includes("UMA task"));
  const iInterp = stdout.findIndex((s) => s.includes("Interpolating"));
  for (const s of stdout.slice(0, iUma + 1)) add("out", s);
  const nOmit = iInterp - iUma - 1;
  if (nOmit > 0) add("note", `${nOmit} lines omitted: an ASE UserWarning about the NEB method default, printed twice`);
  add("out", stdout[iInterp].trim());
  add("note", `not replayed: ${dur(tFirst - tStart)} from the start of the run to band step 0 (model load, endpoint relaxation, IDPP)`, { pause: true });
  const iRun = stdout.findIndex((s) => s.startsWith("Running NEB"));
  for (const s of stdout.slice(iInterp + 1, iRun + 1)) add("out", s.trim());

  const stepLines = nebLog.filter((s) => s.startsWith("Step"));
  add("note", `neb.log, its ${stepLines.length} "Step" lines; the interleaved FIRE lines are left out`);
  const steps = [];
  for (const s of stepLines) {
    const m = s.match(/Step\s+(\d+)\s+Fmax:\s+([\d.]+).*Barrier:\s+([\d.]+)/);
    const st = { k: +m[1], fmax: +m[2], barrier: +m[3], i: steps.length };
    steps.push(st);
    add("step", s, { step: st });
  }
  const iProc = stdout.findIndex((s) => s.startsWith("Processing results"));
  const iVerify = stdout.findIndex((s) => s.startsWith("Verifying image"));
  const block = stdout.slice(iProc, iVerify);
  const paths = block.filter((s) => s.startsWith(" - "));
  for (const s of block.filter((s) => !s.startsWith(" - "))) add("out", s);
  add("note", `${paths.length} output paths omitted (CSV, trajectories, plots, POSCARs of the images)`);
  add("note", `not replayed: ${dur(tExit - tLast)} between the last band step and the end of the run (results, TS verification)`, { pause: true });
  for (const s of stdout.slice(iVerify)) {
    if (!s.trim()) { add("out", ""); continue; }
    const ok = /verified|-> pass|connects/.test(s);
    add(ok ? "ok" : "out", elideText(s.replace(/^\s+/, (w) => w.slice(0, 3))));
  }
  while (L.length && L[L.length - 1].text === "") L.pop();
  return { L, steps, nSteps: steps.at(-1).k, realBand: tLast - tFirst };
}

let hero = null;
function heroTerminal(neb, logs) {
  const body = $("#term-body"), head = $("#term-cmd"), state = $("#term-state");
  const { L, steps, nSteps, realBand } = buildTranscript(neb, logs);
  const prof = profileChart($("#profile"), neb);
  const trace = traceChart($("#trace"), steps, `forward barrier ${withUnit(kn(neb, "forward barrier"), 3)}`);
  const target = kn(neb, "fmax target (band");
  $("#m-steps").textContent = nSteps;
  $("#m-target").textContent = withUnit(target, 2);
  $("#term-host").textContent = `ran on cos-cluster  ${elide(neb.cwd)}`;
  const TYPE_MS = 6, STEP_MS = 55;
  const speed = realBand / ((nSteps + 1) * STEP_MS / 1000);
  const note = REDUCED
    ? "Real output, excerpted. Shown at its final state (reduced motion)."
    : `Real output, excerpted. Band steps replay at about ${Math.round(speed)}× the run (${nSteps} steps took ${dur(realBand)}).`;

  const meter = (s) => {
    $("#m-step").textContent = s.k;
    $("#m-fmax").textContent = fx(s.fmax, 4);
    $("#m-barrier").textContent = fx(s.barrier, 3);
    $("#m-bar").style.width = `${(100 * s.k) / nSteps}%`;
  };
  const line = (l) => {
    const d = el("div", { class: "l " + l.cls });
    txt(d, l.text);
    body.append(d);
    return d;
  };
  let typing = false;
  const finish = (animate) => {
    if (!typing) renderCmd(head, neb.command);
    state.textContent = "done: TS verified";
    state.classList.add("done");
    prof.reveal(animate);
    trace.upto(steps.length - 1);
    meter(steps.at(-1));
    $("#replay-note").textContent = note;
    heroDone = true;
    if (modeCtl && !REDUCED) { modeCtl.play(); $("#btn-mode").textContent = "Pause"; }
  };

  let run = 0;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function play() {
    const me = ++run;
    body.replaceChildren();
    prof.reset();
    trace.upto(-1);
    state.textContent = "running";
    state.classList.remove("done");
    $("#replay-note").textContent = note;
    // type the command into the pinned header
    typing = true;
    renderCmd(head, neb.command);
    const lines = [...head.children];
    const full = lines.map((d) => d.lastChild.textContent);
    lines.forEach((d) => { d.lastChild.textContent = ""; d.hidden = true; });
    const caret = el("span", { class: "caret" });
    for (let k = 0; k < lines.length; k++) {
      lines[k].hidden = false;
      lines[k].append(caret);
      for (let i = 1; i <= full[k].length; i++) {
        if (me !== run) return;
        lines[k].childNodes[k ? 0 : 1].textContent = full[k].slice(0, i);
        await sleep(TYPE_MS);
      }
    }
    caret.remove();
    typing = false;
    renderCmd(head, neb.command); // re-wrap in case the font arrived while typing
    await sleep(120);
    for (const l of L) {
      if (me !== run) return;
      line(l);
      if (l.step) { meter(l.step); prof.live(l.step.k, l.step.barrier); trace.upto(l.step.i); }
      body.scrollTop = body.scrollHeight;
      await sleep(l.step ? STEP_MS : l.pause ? 450 : l.cls === "note" ? 140 : 45);
    }
    if (me === run) finish(true);
  }
  function showEnd() {
    run++;
    typing = false;
    renderCmd(head, neb.command);
    body.replaceChildren();
    for (const l of L) line(l);
    body.scrollTop = body.scrollHeight;
    finish(false);
  }
  hero = { rewrap() { if (!typing) renderCmd(head, neb.command); } };
  $("#btn-replay").onclick = () => (REDUCED ? showEnd() : play());
  $("#btn-skip").onclick = showEnd;
  fullToggle($("#btn-full"), $("#full-neb"), $("#full-neb-text"), neb.command);
  $("#profile-note").textContent =
    `Straight lines join the ${neb.series.reaction_coordinate_A.length} images of the converged band (no interpolation). ` +
    `Reverse barrier ${withUnit(kn(neb, "reverse barrier"), 3)}. ` +
    `Band fmax ${withUnit(kn(neb, "final band fmax"), 4)} after ${fx(kn(neb, "NEB optimizer steps").value, 0)} FIRE steps ` +
    `(target ${withUnit(target, 2)}). While the log runs, the dashed line is the highest image of the current band.`;
  if (REDUCED) showEnd(); else play();
}

function fullToggle(btn, box, pre, cmd) {
  pre.textContent = cmd;
  btn.onclick = () => {
    const open = box.hidden;
    box.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "Hide full command" : "Full command";
  };
}
document.addEventListener("click", async (e) => {
  const b = e.target.closest("button.copy");
  if (!b) return;
  const src = document.getElementById(b.dataset.copy);
  try {
    await navigator.clipboard.writeText(src.textContent);
    b.textContent = "Copied";
  } catch {
    const r = document.createRange(); r.selectNodeContents(src);
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    b.textContent = "Selected";
  }
  setTimeout(() => (b.textContent = "Copy"), 1600);
});

// ------------------------------------------------------------------ pairs
function cmdBox(box, cmd, tag, tagCls = "") {
  box.replaceChildren();
  if (tag) box.append(el("span", { class: "tag " + tagCls }, tag));
  const lines = el("div", { class: "lines cmdlines" });
  box.append(lines);
  const id = "full-" + box.dataset.cmd;
  const row = el("div", { class: "row" });
  const btn = el("button", { type: "button", "aria-expanded": "false", "aria-controls": id }, "Full command");
  const copy = el("button", { type: "button", class: "copy", "data-copy": id + "-text" }, "Copy");
  row.append(btn, copy);
  const full = el("div", { class: "full", id, hidden: "" });
  const pre = el("pre", { id: id + "-text" });
  full.append(pre);
  box.append(row, full);
  registerCmd(lines, cmd);
  fullToggle(btn, full, pre, cmd);
}

function statusPills(box, d, extra = []) {
  box.replaceChildren();
  if (d.status === "pending") box.append(el("span", { class: "pill pending" }, mdStatus(d)));
  else box.append(el("span", { class: "pill " + (d.converged ? "ok" : "flag") }, d.converged ? "converged" : "not converged"));
  for (const [cls, t] of extra) box.append(el("span", { class: "pill " + cls }, t));
}

function nums(box, d, rows) {
  box.replaceChildren();
  for (const [prefix, label, nd, fmt] of rows) {
    const k = kn(d, prefix);
    const div = el("div");
    div.append(el("dt", {}, label));
    div.append(txt(el("dd"), fmt ? fmt(k) : withUnit(k, nd)));
    box.append(div);
  }
}

/** Output lines into a <pre>, one hanging-indented span per line (wraps, never clips). */
function outLines(pre, lines, noteIdx = new Set()) {
  pre.replaceChildren(...lines.map((s, i) => txt(el("span", { class: noteIdx.has(i) ? "ol on" : "ol" }), elideText(s))));
}
function logPre(pre, text, range) {
  let lines = text.replace(/\n+$/, "").split("\n");
  if (range) { const [a, b] = range.split("-").map(Number); lines = lines.slice(a, b + 1); }
  outLines(pre, lines);
}

function fallbackGif(sceneBox, stem, alt) {
  const img = el("img", { src: `media/${stem}-${DARK() ? "dark" : "light"}.gif`, alt });
  sceneBox.replaceChildren(img);
  sceneBox.classList.add("fallback");
}

async function relaxPair(rel, logs) {
  cmdBox($('[data-cmd="relax"]'), rel.command);
  statusPills($('[data-status="relax"]'), rel);
  // opt.log without its wall-clock column, so the energy and fmax columns fit a phone
  const raw = logs.relaxopt.replace(/\n+$/, "").split("\n");
  const iTime = raw[0].indexOf("     Time");
  const shown = [raw[0].slice(0, iTime) + raw[0].slice(iTime + 9),
    ...raw.slice(1).map((s) => s.replace(/^(\S+:\s+\d+) \d\d:\d\d:\d\d/, "$1"))];
  shown.push("(opt.log; its Time column is left out here)");
  outLines($('[data-log="relax_opt.log"]'), shown, new Set([shown.length - 1]));
  const s = rel.series;
  const tgt = kn(rel, "fmax target");
  const c = lineChart({ w: fitW($("#relax-chart")), h: 230, x: s.step, xlabel: "BFGS step", ylabel: "fmax (eV/Å)", logy: true,
    series: [{ y: s.fmax_eV_per_A, cls: "s-ink", dot: "d-ink", label: "largest force on a free atom" }],
    target: { y: tgt.value, label: `target ${withUnit(tgt, 2)}` } });
  const cursor = sv("line", { class: "cursor", x1: c.X(0), x2: c.X(0), y1: c.m.t, y2: c.m.t + c.ph });
  c.svg.append(cursor);
  c.svg.setAttribute("aria-label", "Largest force per BFGS step, log scale, with the target line");
  $("#relax-chart").replaceChildren(c.svg);
  const label = $("#relax-frame");
  const onFrame = (i) => {
    const j = Math.min(i, s.step.length - 1);
    cursor.setAttribute("x1", c.X(s.step[j])); cursor.setAttribute("x2", c.X(s.step[j]));
    label.textContent = `step ${s.step[j]}, E = ${fx(s.energy_eV[j], 3)} eV`;
  };
  nums($('[data-nums="relax"]'), rel, [
    ["total energy, O at fcc", "total energy, O at fcc", 3],
    ["final fmax, O at fcc", "final fmax", 4],
    ["BFGS steps", "BFGS steps", 0, (k) => `${fx(k.value, 0)}`],
    ["O height above top Pt layer, fcc", "O height above the top Pt layer", 2],
    ["shortest O–Pt distance", "shortest O–Pt distance", 2],
    ["E(hcp) − E(fcc)", "E(hcp) − E(fcc), the two hollows", 3],
    ["Pt lattice constant", "Pt lattice constant a (relaxed bulk)", 3],
  ]);
  const btn = $("#btn-relax");
  try {
    const ctl = await loadScene($("#relax-scene"), "data/relax.xyz", {
      view: { elevation: 12, azimuth: 0, zoom: 1 }, replicate: [2, 2], fps: 6, hold: 1.2, background: sceneBg(), onFrame });
    onFrame(ctl.frame);
    scenes.relax = ctl;
    btn.onclick = () => {
      if (ctl.playing) { ctl.pause(); btn.textContent = "Play"; }
      else { ctl.play({ force: true }); btn.textContent = "Pause"; }
    };
  } catch (e) {
    console.warn("relax 3D view unavailable:", e.message);
    fallbackGif($("#relax-scene"), "relax3d", "Recorded animation of the 13 BFGS steps, side view, 2×2 periodic images");
    label.textContent = "3D view unavailable; recorded animation";
    btn.hidden = true;
  }
}

function committeePair(com, logs) {
  cmdBox($('[data-cmd="committee"]'), com.command);
  statusPills($('[data-status="committee"]'), com);
  const out = logs.comstdout.replace(/\n+$/, "").split("\n");
  const i0 = out.findIndex((s) => s.includes("Committee of"));
  const iLast = out.findIndex((s) => s.includes("Committee disagreement"));
  const members = out.slice(i0, i0 + 1 + kn(com, "committee members").value);
  const omitted = iLast - (i0 + members.length);
  const shown = [...members, `   ⋯ ${omitted} lines omitted (optimizer settings, output files)`, out[iLast]];
  outLines($("#com-out"), shown, new Set([members.length]));
  const s = com.series, tgt = kn(com, "fmax target");
  const c = lineChart({ w: fitW($("#com-chart")), h: 270, x: s.step, xlabel: "BFGS step", ylabel: "force (eV/Å)", logy: true,
    series: [
      { y: s.fmax_eV_per_A, cls: "s-ink", dot: "d-ink", label: "fmax of the mean force" },
      { y: s.sigma_max_free_eV_per_A, cls: "s-acc", dot: "d-acc", label: "σ max (free atoms)" },
      { y: s.sigma_mean_free_eV_per_A, cls: "s-mut", label: "σ mean" },
    ],
    target: { y: tgt.value, label: `target ${withUnit(tgt, 2)}` } });
  const k = s.step.findIndex((_, i) => s.sigma_max_free_eV_per_A[i] > s.fmax_eV_per_A[i]);
  if (k >= 0) {
    c.plot.append(sv("rect", { class: "shade", x: c.X(s.step[k]), y: c.m.t, width: c.X(s.step.at(-1)) - c.X(s.step[k]), height: c.ph }));
  }
  c.svg.setAttribute("aria-label", "Committee relaxation: force on the mean and member disagreement per step, log scale");
  $("#com-chart").replaceChildren(c.svg);
  const ratio = kn(com, "sigma_max_free / fmax");
  $("#com-note").textContent = k >= 0
    ? `Shaded: from step ${s.step[k]} on, the largest disagreement between members (σ max) exceeds the force the optimizer is minimizing. ` +
      `At the last step it is ${fx(ratio.value, 1)}× larger. σ is a standard deviation over the ${fx(kn(com, "committee members").value, 0)} members.`
    : "";
  nums($('[data-nums="committee"]'), com, [
    ["sigma_max_free, final step", "σ max, free atoms, final step", 4],
    ["sigma_mean_free, final step", "σ mean, free atoms, final step", 4],
    ["final fmax of the committee-mean force", "fmax of the mean force, final", 4],
    ["energy spread across members", "energy spread (std. dev., step-0 aligned)", 4],
    ["BFGS steps", "BFGS steps", 0, (k) => fx(k.value, 0)],
    ["free atoms", "free atoms in σ", 0, (k) => fx(k.value, 0)],
  ]);
}

function nebPair(neb, logs) {
  cmdBox($('[data-cmd="neb"]'), neb.command);
  const flags = neb.flags.includes("ts_verified") ? [["ok", "TS verified"]] : [];
  statusPills($('[data-status="neb"]'), neb, flags);
  const pre = $('[data-log="ts_verify.txt"]');
  logPre(pre, logs.tsverify, pre.dataset.lines);
  const im = kn(neb, "TS imaginary frequency");
  nums($('[data-nums="neb"]'), neb, [
    ["forward barrier", "forward barrier, fcc → hcp", 3],
    ["reverse barrier", "reverse barrier, hcp → fcc", 3],
    ["reaction energy", "reaction energy", 3],
    ["TS imaginary frequency", "imaginary mode at the TS", 1, (k) => `${fx(k.value, 1)}i ${k.unit}`],
    ["overlap of imaginary mode", `overlap with the NEB tangent (threshold ${kn(neb, "overlap threshold").value})`, 3],
    ["TS fmax", "TS fmax, free components", 4],
    ["final band fmax", "final band fmax", 4],
    ["band check: largest drift", `largest drift from the seed band (allowance ${withUnit(kn(neb, "band check: allowance"), 2)})`, 2],
    ["O height above top Pt layer at TS", "O height above the top Pt layer at the TS", 2],
  ]);
  $("#neb-flag").textContent = `Flag ${neb.flags.filter((f) => f !== "ts_verified").join(", ")}: ${neb.notes[0]}`;
  const cap = $("#mode-cap");
  cap.replaceChildren();
  txt(cap, `Imaginary mode at the TS, ${fx(im.value, 1)}i ${im.unit}; overlap with the path ${fx(kn(neb, "overlap of imaginary mode").value, 3)}. ` +
    `Drawn at ${withUnit(kn(neb, "TS mode animation peak displacement"), 1)} peak displacement: an illustration of the mode, not dynamics. ` +
    `Top view of the periodic slab (3×3 images), centred on the O.`);
}

async function mdPair(md, plan) {
  const pending = md.status === "pending";
  cmdBox($('[data-cmd="md"]'), plan.command,
    pending ? `planned, has not run (from ${plan.source.split(" (")[0]})` : "", "pending");
  statusPills($('[data-status="md"]'), md);
  const box = $("#md-result");
  const keys = Object.keys(md.series || {});
  if (pending || !keys.length) {
    const p = el("div", { class: "pending-box" });
    const s = mdStatus(md);
    p.append(el("p", {}, s.charAt(0).toUpperCase() + s.slice(1) + "."));
    p.append(el("p", {}, "There is no output here, and nothing is simulated in its place. " +
      "When the run exists, this slot fills from md.json: its time series and frames."));
    box.replaceChildren(p);
    return;
  }
  // Drop-in path: plot every series against the first time-like key.
  const xk = keys.find((k) => /time|step/.test(k)) || keys[0];
  for (const k of keys.filter((k) => k !== xk)) {
    const c = lineChart({ w: fitW(box), h: 200, x: md.series[xk], xlabel: xk, ylabel: k,
      series: [{ y: md.series[k], cls: "s-ink", label: k }] });
    const d = el("div", { class: "svgbox" });
    d.append(c.svg);
    box.append(d);
  }
}

function webPair(web) {
  cmdBox($('[data-cmd="webapp"]'), web.command);
  statusPills($('[data-status="webapp"]'), web);
  $("#web-path").textContent =
    `No terminal: the input of the first row, uploaded in the browser and sent to a remote GPU machine. ` +
    `In this recording that machine was ${web.target.machine}, where the showcase runs ran (environment ${web.target.env}).`;
  const v = $("#web-video");
  v.src = "data/webapp.mp4";
  v.setAttribute("aria-label", "Screen recording of the mliprun web app submitting a relaxation to cos-cluster");
  const speed = kn(web, "video speed-up");
  $("#web-cap").textContent =
    `Screen recording of the real job, ${fx(kn(web, "video length").value, 1)} s. ` +
    `The waiting part is sped up ×${fx(speed.value, 0)} and labelled in the video; the form and the result play at normal speed.`;
  if (!REDUCED) { v.autoplay = true; v.play().catch(() => {}); }
  nums($('[data-nums="webapp"]'), web, [
    ["total energy, O at fcc hollow (relaxed, web app job)", "total energy, web app job", 3],
    ["energy difference, web app − CLI", "difference to the CLI run", 1, (k) => `${k.value.toExponential(1)} ${k.unit}`],
    ["final fmax", "final fmax", 4],
    ["BFGS steps", "BFGS steps", 0, (k) => fx(k.value, 0)],
    ["job wall time", "job wall time, incl. model load", 1],
    ["relaxation wall time", "optimizer wall time", 1],
  ]);
}

const sceneBg = () => getComputedStyle(document.documentElement).getPropertyValue("--scene-bg").trim() || "#ffffff";

let modeCtl = null;
// Scene controllers, readable by check.py (reduced-motion test); nothing else uses this.
const scenes = (window.__mliprunScenes = {});
let heroDone = false;
async function modeScene() {
  const btn = $("#btn-mode");
  try {
    // ts_mode_centred.xyz: the pack's ts_mode.xyz with the cell origin shifted in-plane so the
    // O sits mid-cell (a rigid periodic translation, made by build.py); 3×3 images fill the view.
    modeCtl = await loadScene($("#mode-scene"), "data/ts_mode_centred.xyz", {
      view: MODE_VIEW, replicate: [3, 3], fps: 20, hold: 0, restFrame: "first", background: sceneBg() });
    scenes.mode = modeCtl;
    btn.onclick = () => {
      if (modeCtl.playing) { modeCtl.pause(); btn.textContent = "Play the mode"; }
      else { modeCtl.play({ force: true }); btn.textContent = "Pause"; }
    };
    if (heroDone && !REDUCED) { modeCtl.play(); btn.textContent = "Pause"; }
  } catch (e) {
    console.warn("TS-mode 3D view unavailable:", e.message);
    fallbackGif($("#mode-scene"), "ts-mode", "Recorded animation of the imaginary mode at the TS, top view");
    btn.hidden = true;
  }
}

// ------------------------------------------------------------------ boot
async function main() {
  const [rel, com, neb, md, web, plan, stdout, neblog, runlog, tsverify, relaxopt, comstdout] = await Promise.all([
    getJSON("data/relax.json"), getJSON("data/committee.json"), getJSON("data/neb.json"),
    getJSON("data/md.json"), getJSON("data/webapp.json"), getJSON("data/md_planned.json"),
    getText("data/logs/neb_stdout.txt"), getText("data/logs/neb.log"), getText("data/logs/neb_run.log"),
    getText("data/logs/ts_verify.txt"), getText("data/logs/relax_opt.log"), getText("data/logs/committee_stdout.txt"),
  ]);
  const logs = { stdout, neblog, runlog, tsverify, relaxopt, comstdout };
  const lvl = rel.command.match(/--mlip (\S+) --uma-task (\S+)/);
  $("#foot-level").textContent = `${lvl[1]}/${lvl[2]}, fmax target ${withUnit(kn(rel, "fmax target"), 2)}, run on cos-cluster`;
  nebPair(neb, logs);
  committeePair(com, logs);
  mdPair(md, plan);
  webPair(web);
  heroTerminal(neb, logs);
  // Re-wrap every command once the real mono face is in (its glyphs are wider than the fallback).
  document.fonts?.ready.then(() => { rewrapAll(); document.documentElement.dataset.fonts = "1"; });
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(rewrapAll, 150); });
  document.documentElement.dataset.ready = "1";
  await Promise.all([modeScene(), relaxPair(rel, logs)]);
  document.documentElement.dataset.scenes = "1";
}
main().catch((e) => {
  console.error(e);
  const s = $("#term-state");
  if (s) s.textContent = "data failed to load: " + e.message;
});
