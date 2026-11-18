// mliprun gallery (alternative 3). Every number on the page is read at runtime
// from data/*.json (copies of docs/showcase/data/pack). Nothing numeric is typed
// here except display choices (decimals, chart sizes, views, frame rates).

const DATA = "data/";
const REPLICATE = [2, 2]; // periodic images shown in the slab panels (display choice)
const IMAGE_TINT = 0.55; // fade of the periodic images, so the central cell reads as the subject
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ loading
async function loadViewer() {
  // GitHub Pages layout: site/alt-3-gallery/ next to site/shared/.
  // Artifact preview: a copy published at shared/viewer.js beside the page.
  try {
    return await import("../shared/viewer.js");
  } catch (e) {
    return await import("./shared/viewer.js");
  }
}

async function getJSON(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}

// ------------------------------------------------------------------ formatting
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function fx(v, d) { return Number(v).toFixed(d); }
function sci(v, d = 1) {
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  return `${m.toFixed(d)} × 10${String(e).split("").map((c) => SUP[c]).join("")}`;
}
function signed(s, v) { return v > 0 ? "+" + s : s; }
function intv(v) { return String(Math.round(v)); }

function K(j, prefix) {
  const k = j.key_numbers.find((x) => x.label.startsWith(prefix));
  if (!k) throw new Error(`${j.use_case}: no key number "${prefix}"`);
  return k;
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// The command as run, with the environment's absolute path to `mlip` shortened.
function cmdTokens(cmd) {
  const t = cmd.trim().split(/\s+/);
  const i = t.findIndex((x) => x === "mlip" || x.endsWith("/mlip"));
  return i < 0 ? t : t.slice(i + 1);
}
function cmdFull(cmd) { return ["mlip", ...cmdTokens(cmd)]; }
function cmdShort(cmd, keep) {
  const t = cmdTokens(cmd);
  const out = ["mlip", t[0], t[1]];
  for (let i = 2; i < t.length; i++) {
    if (!keep.includes(t[i])) continue;
    out.push(t[i]);
    if (t[i + 1] && !t[i + 1].startsWith("--")) out.push(t[i + 1]);
  }
  return out;
}
function flagValue(cmd, flag) {
  const t = cmdTokens(cmd);
  const i = t.indexOf(flag);
  return i >= 0 ? t[i + 1] : null;
}
// Command as HTML: a flag and its value form one unbreakable group; lines
// break only at the spaces between groups (never inside "--uma-task").
function cmdHTML(tokens) {
  const groups = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.startsWith("--") && tokens[i + 1] && !tokens[i + 1].startsWith("--")) {
      groups.push(`${t} ${tokens[i + 1]}`);
      i++;
    } else groups.push(t);
  }
  return groups.map((g) => {
    const sp = g.indexOf(" ");
    if (sp > 0 && g.includes("/", sp)) { // flag + path: keep the flag whole, let the path break after "/"
      return `<span class="tok">${esc(g.slice(0, sp))}</span> <span class="path">${esc(g.slice(sp + 1)).replace(/\//g, "/<wbr>")}</span>`;
    }
    return `<span class="tok">${esc(g)}</span>`;
  }).join(" ");
}

// ------------------------------------------------------------------ charts
const SVGNS = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || Math.abs(hi) || 1;
  let step = 10 ** Math.floor(Math.log10(span / n));
  const err = span / n / step;
  if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return { ticks: out, step };
}
function tickLabel(v, step) {
  const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(d).replace(/^-0(\.0+)?$/, "0");
}
// Log axis clipped to the data: 1-2-5 ticks inside [lo, hi], decades only if crowded.
function logTicks(lo, hi, maxN = 6) {
  const all = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) all.push(v); }
  }
  const dec = all.filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
  return all.length > maxN && dec.length >= 2 ? dec : all;
}
const logLabel = (v) => (v >= 1 ? String(+v.toPrecision(3)) : String(+v.toPrecision(2)));

/**
 * Labelled chart. spec: {x, xLabel, yLabel, yLog, series:[{y, cls, points, ptCls, r, end}],
 * band:{lo, hi}, refs:[{y, label, at:"left"|"gutter"}], marks:[{i, label}], ptexts:[{i, s, text, dx, dy, anchor}],
 * texts:[{fx, fy, text, anchor}], bracket:{i, a, b, label}, height, gutter, fit, empty}
 * "end" labels and gutter refs sit in a right gutter, pushed apart so they never overlap.
 * Returns {setCursor(i)}; redraws at the real pixel size on resize.
 */
function chart(fig, spec) {
  const M = { l: 40, r: spec.gutter ?? 10, t: 18, b: 26 };
  let host = fig;
  if (spec.fit) {
    host = document.createElement("div");
    host.className = "plotbox";
    fig.appendChild(host);
  }
  let cursorI = null;
  let svg = null;
  const draw = () => {
    const W = Math.max(160, Math.round(host.clientWidth));
    const H = spec.fit ? Math.max(spec.height || 160, Math.round(host.clientHeight)) : (spec.height || 104);
    if (svg) svg.remove();
    svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": spec.aria || "" });
    if (spec.fit) svg.classList.add("fitsvg");
    host.appendChild(svg);
    const pw = W - M.l - M.r, ph = H - M.t - M.b;
    if (spec.empty) {
      svg.appendChild(el("rect", { class: "emptybox", x: 0.5, y: 0.5, width: W - 1, height: H - 1, rx: 2 }));
      spec.empty.forEach((s, k) => svg.appendChild(el("text", { class: "empty", x: 12, y: H / 2 - (spec.empty.length - 1) * 8 + k * 16 + 4 }, s)));
      return;
    }
    const xs = spec.x;
    const allY = [];
    for (const s of spec.series) allY.push(...s.y);
    if (spec.band) allY.push(...spec.band.lo, ...spec.band.hi);
    for (const r of spec.refs || []) allY.push(r.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let yt, ymap, ylab;
    if (spec.yLog) {
      const lo = Math.min(...allY.filter((v) => v > 0)) / 1.25, hi = Math.max(...allY) * 1.25;
      yt = logTicks(lo, hi, Math.max(3, Math.floor(ph / 16)));
      ymap = (v) => M.t + ph - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * ph;
      ylab = logLabel;
    } else {
      let lo = Math.min(...allY), hi = Math.max(...allY);
      const pad = (hi - lo) * 0.08 || 0.01;
      lo -= pad; hi += pad;
      const nt = niceTicks(lo, hi, 3);
      yt = nt.ticks;
      lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);
      ymap = (v) => M.t + ph - ((v - lo) / (hi - lo)) * ph;
      ylab = (v) => tickLabel(v, nt.step);
    }
    const xmap = (v) => M.l + ((v - xmin) / (xmax - xmin || 1)) * pw;
    const xt = niceTicks(xmin, xmax, Math.max(2, Math.min(6, Math.round(pw / 70))));

    const g = el("g");
    svg.appendChild(g);
    for (const v of yt) {
      const y = ymap(v);
      if (y < M.t - 1 || y > M.t + ph + 1) continue;
      g.appendChild(el("line", { class: "gl", x1: M.l, x2: M.l + pw, y1: y, y2: y }));
      g.appendChild(el("text", { x: M.l - 5, y: y + 3.5, "text-anchor": "end" }, ylab(v)));
    }
    for (const v of xt.ticks) {
      const x = xmap(v);
      g.appendChild(el("line", { class: "ax", x1: x, x2: x, y1: M.t + ph, y2: M.t + ph + 3 }));
      g.appendChild(el("text", { x, y: M.t + ph + 13, "text-anchor": "middle" }, tickLabel(v, xt.step)));
    }
    g.appendChild(el("path", { class: "ax", d: `M${M.l},${M.t} V${M.t + ph} H${M.l + pw}` }));
    g.appendChild(el("text", { class: "lab", x: M.l - 5, y: 10, "text-anchor": "start" }, spec.yLabel));
    g.appendChild(el("text", { class: "lab", x: M.l + pw, y: H - 2, "text-anchor": "end" }, spec.xLabel));

    if (spec.band) {
      const top = xs.map((x, i) => `${xmap(x)},${ymap(spec.band.hi[i])}`);
      const bot = xs.map((x, i) => `${xmap(x)},${ymap(spec.band.lo[i])}`).reverse();
      g.appendChild(el("polygon", { class: spec.band.cls || "band", points: top.concat(bot).join(" ") }));
    }
    const gutter = [];
    for (const r of spec.refs || []) {
      const y = ymap(r.y);
      g.appendChild(el("line", { class: "ref", x1: M.l, x2: M.l + pw, y1: y, y2: y }));
      if (r.at === "gutter") gutter.push({ y, text: r.label, cls: "reflab" });
      else g.appendChild(el("text", { class: "reflab", x: M.l + 4, y: y - 4, "text-anchor": "start" }, r.label));
    }
    for (const s of spec.series) {
      const d = xs.map((x, i) => `${i ? "L" : "M"}${xmap(x).toFixed(1)},${ymap(s.y[i]).toFixed(1)}`).join("");
      if (s.cls) g.appendChild(el("path", { class: s.cls, d }));
      if (s.points) xs.forEach((x, i) => g.appendChild(el("circle", { class: s.ptCls || "pt", cx: xmap(x), cy: ymap(s.y[i]), r: s.r || 2.2 })));
      if (s.end) gutter.push({ y: ymap(s.y[s.y.length - 1]), text: s.end, cls: s.endCls || "lab" });
    }
    // gutter labels: sort, push apart (>= 11 px), keep inside the plot height
    gutter.sort((a, b) => a.y - b.y);
    for (let k = 1; k < gutter.length; k++) gutter[k].y = Math.max(gutter[k].y, gutter[k - 1].y + 11);
    const over = gutter.length ? gutter[gutter.length - 1].y - (M.t + ph + 4) : 0;
    if (over > 0) gutter.forEach((q) => { q.y -= over; });
    for (const q of gutter) g.appendChild(el("text", { class: q.cls, x: M.l + pw + 6, y: q.y + 3.5 }, q.text));

    if (spec.bracket) {
      const b = spec.bracket, x = xmap(xs[b.i]) - 10, ya = ymap(b.a), yb = ymap(b.b);
      g.appendChild(el("path", { class: "brk", d: `M${x + 4},${ya} H${x} V${yb} H${x + 4}` }));
      g.appendChild(el("text", { class: "brklab", x: x - 4, y: (ya + yb) / 2 + 4, "text-anchor": "end" }, b.label));
    }
    for (const m of spec.marks || []) {
      const x = xmap(xs[m.i]), y = ymap(spec.series[0].y[m.i]);
      g.appendChild(el("circle", { class: "ts", cx: x, cy: y, r: 3.8 }));
      if (m.label) g.appendChild(el("text", { class: "tslab", x, y: y - 9, "text-anchor": "middle" }, m.label));
    }
    for (const p of spec.ptexts || []) {
      const s = spec.series[p.s || 0];
      g.appendChild(el("text", { class: p.cls || "lab", x: xmap(xs[p.i]) + (p.dx || 0), y: ymap(s.y[p.i]) + (p.dy || 0), "text-anchor": p.anchor || "middle" }, p.text));
    }
    for (const t of spec.texts || []) {
      g.appendChild(el("text", { class: t.cls || "lab", x: M.l + t.fx * pw, y: M.t + t.fy * ph, "text-anchor": t.anchor || "start" }, t.text));
    }
    if (cursorI != null) {
      const x = xmap(xs[cursorI]);
      const cg = el("g", { class: "cursor" });
      cg.appendChild(el("line", { class: "cur", x1: x, x2: x, y1: M.t, y2: M.t + ph }));
      cg.appendChild(el("circle", { class: "curdot", cx: x, cy: ymap(spec.series[0].y[cursorI]), r: 4.2 }));
      svg.appendChild(cg);
    }
  };
  draw();
  let tm = 0;
  new ResizeObserver(() => { clearTimeout(tm); tm = setTimeout(draw, 60); }).observe(host);
  return { setCursor(i) { if (i === cursorI) return; cursorI = i; draw(); } };
}

// ------------------------------------------------------------------ tiles
function fill(tile, { status, statusCls, cmd, cmdFullTokens, keyV, keyL, conv }) {
  const q = (s) => tile.querySelector(s);
  const st = q("[data-status]");
  st.textContent = status;
  st.classList.add(statusCls);
  q("[data-cmd]").innerHTML = Array.isArray(cmd) ? cmdHTML(cmd) : cmd;
  q("[data-cmd-full]").innerHTML = cmdHTML(cmdFullTokens);
  q("[data-key-v]").innerHTML = keyV;
  q("[data-key-l]").textContent = keyL;
  q("[data-conv]").innerHTML = conv;
}
const bind = (name, html) => document.querySelectorAll(`[data-bind="${name}"]`).forEach((n) => { n.innerHTML = html; });
const num = (v, unit) => `${v}<small>${esc(unit)}</small>`;

// Heights of O above the top Pt layer, per frame, from the xyz positions.
function oHeights(frames) {
  return frames.map((f) => {
    const z = [], zo = [];
    f.symbols.forEach((s, a) => (s === "Pt" ? z : s === "O" ? zo : []).push(f.positions[3 * a + 2]));
    const top = Math.max(...z);
    const layer = z.filter((v) => v > top - 1.0);
    const mean = layer.reduce((a, b) => a + b, 0) / layer.length;
    return Math.max(...zo) - mean;
  });
}

const scenes = {};
const css = (name, fb) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fb;
const sceneOpts = () => ({ background: css("--tile", "#ffffff"), replicate: REPLICATE, pauseOffscreen: true });
// The 3D scenes bake the panel colour into their fog; a theme switch reloads them.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => location.reload());
new MutationObserver(() => location.reload()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

async function mountScene(V, id, url, opts) {
  const host = document.getElementById(id);
  let lastErr = null;
  for (let attempt = 0; attempt < 2; attempt++) { // one retry: a CDN hiccup should not blank a panel
    try {
      const ctrl = await V.loadScene(host, url, { ...sceneOpts(), ...opts });
      scenes[id] = ctrl;
      let tm = 0;
      new ResizeObserver(() => {
        clearTimeout(tm);
        tm = setTimeout(() => ctrl.resize(), 120);
      }).observe(host);
      return ctrl;
    } catch (e) {
      lastErr = e;
      if (attempt === 0) await new Promise((r) => setTimeout(r, 800));
    }
  }
  host.classList.add("failed");
  host.textContent = `3D view could not load (${lastErr.message}).`;
  return null;
}

function playAll() { for (const c of Object.values(scenes)) if (c && c.loops) c.play({ force: true }); }
function pauseAll() { for (const c of Object.values(scenes)) if (c && c.loops) c.pause(); }

// ------------------------------------------------------------------ main
async function main() {
  const [V, relax, committee, neb, md, webapp, caps] = await Promise.all([
    loadViewer(), getJSON("relax.json"), getJSON("committee.json"), getJSON("neb.json"),
    getJSON("md.json"), getJSON("webapp.json"), getJSON("capabilities.json"),
  ]);
  window.__gallery = { scenes, data: { relax, committee, neb, md, webapp }, ready: false };

  const level = `${flagValue(relax.command, "--mlip")}/${flagValue(relax.command, "--uma-task")}`;
  const target = K(relax, "fmax target");
  bind("level", `${esc(level)}, fmax target ${target.value} ${esc(target.unit)}`);

  // ---------- (a) NEB + TS
  {
    const tile = document.getElementById("neb");
    const s = neb.series;
    const fwd = K(neb, "forward barrier"), rev = K(neb, "reverse barrier"), dE = K(neb, "reaction energy");
    const bf = K(neb, "final band fmax"), nt = K(neb, "fmax target"), ns = K(neb, "NEB optimizer steps");
    const nu = K(neb, "TS imaginary frequency"), ov = K(neb, "overlap of imaginary mode"), ovt = K(neb, "overlap threshold");
    const isF = K(neb, "ts_verify endpoint fmax, IS"), fsF = K(neb, "ts_verify endpoint fmax, FS"), inF = K(neb, "ts_verify internal relax fmax");
    const amp = K(neb, "TS mode animation peak displacement");
    const tsH = K(neb, "O height above top Pt layer at TS"), tsF = K(neb, "TS fmax");
    const ts = neb.ts_image_index;
    const nImg = s.image_index.length;
    const verified = neb.flags.includes("ts_verified");
    const flagged = neb.flags.includes("ts_verify_endpoint_fmax_above_relax_fmax");
    fill(tile, {
      status: neb.converged ? (verified ? "converged, TS verified" : "converged") : "not converged",
      statusCls: neb.converged ? "ok" : "pend",
      cmd: cmdShort(neb.command, ["--num-images", "--climb", "--verify-ts"]),
      cmdFullTokens: cmdFull(neb.command),
      keyV: num(fx(fwd.value, 3), fwd.unit),
      keyL: `barrier, fcc → hcp`,
      conv: `${neb.converged ? "Converged" : "Not converged"}: band fmax ${fx(bf.value, 4)} ${esc(bf.unit)} against a target of ${nt.value} ${esc(nt.unit)}, ${intv(ns.value)} optimizer steps, ${nImg} images (${nImg - 2} intermediate + 2 endpoints), climbing image at index ${ts} (images counted 0–${nImg - 1}, as in the run record). Reverse barrier ${fx(rev.value, 3)} ${esc(rev.unit)}. TS fmax ${fx(tsF.value, 4)} ${esc(tsF.unit)}; O ${fx(tsH.value, 2)} ${esc(tsH.unit)} above the top Pt layer; mode drawn at ${amp.value} ${esc(amp.unit)} peak in one cell. ` +
        (verified ? `TS verified: one imaginary mode, ${fx(nu.value, 1)}i ${esc(nu.unit)}, overlap with the band tangent ${fx(ov.value, 3)} (threshold ${ovt.value}); the displace-and-relax check connects to both endpoints. ` : "") +
        (flagged ? `<span class="flag">Flag: ts_verify measured endpoint fmax ${fx(isF.value, 4)} (IS) and ${fx(fsF.value, 4)} (FS) ${esc(isF.unit)}, above its internal ${inF.value} ${esc(inF.unit)}; the verdict is unaffected.</span>` : ""),
    });
    bind("ts-cap", `<span class="ts-h">TS mode <b>${fx(nu.value, 1)}i ${esc(nu.unit)}</b></span>overlap ${fx(ov.value, 3)} (≥ ${ovt.value}); mode, not dynamics`);
    document.querySelector('[data-bind-aria="ts-aria"]').setAttribute("aria-label",
      `Imaginary mode at the transition state, ${fx(nu.value, 1)}i ${nu.unit}, drawn as an oscillation of ${amp.value} ${amp.unit} peak displacement in one cell. An illustration of the mode, not dynamics.`);
    const cP = chart(tile.querySelector('[data-chart="neb-profile"]'), {
      x: s.reaction_coordinate_A, xLabel: "reaction coordinate / Å", yLabel: "E − E(fcc) / eV", height: 200, fit: true,
      series: [{ y: s.relative_energy_eV, cls: "ln", points: true, ptCls: "pt2", r: 3 }],
      marks: [{ i: ts, label: `TS ${fx(fwd.value, 3)} eV` }],
      ptexts: [
        { i: 0, text: "fcc", dx: 4, dy: -9, anchor: "start" },
        { i: nImg - 1, text: `hcp, ΔE ${signed(fx(dE.value, 3), dE.value)} eV`, dx: 0, dy: 18, anchor: "end" },
      ],
      aria: `Energy along the final band. Forward barrier ${fx(fwd.value, 3)} eV, reaction energy ${fx(dE.value, 3)} eV.`,
    });
    chart(tile.querySelector('[data-chart="neb-conv"]'), {
      x: s.neb_step, xLabel: "NEB step", yLabel: "band fmax / eV Å⁻¹", yLog: true, height: 200,
      series: [{ y: s.neb_fmax_eV_per_A, cls: "ln" }],
      refs: [{ y: nt.value, label: `target ${nt.value}` }],
      aria: "Band maximum force against NEB optimizer step, log scale.",
    });
    const tag = tile.querySelector('[data-overlay="neb"]');
    const onFrame = (i) => {
      cP.setCursor(i);
      tag.innerHTML = `image <b>${i}</b> (0–${nImg - 1}) · E − E(fcc) <b>${fx(s.relative_energy_eV[i], 3)} eV</b>` +
        (i === ts ? ` · <span class="ts">climbing image (TS)</span>` : i === 0 ? " · fcc" : i === nImg - 1 ? " · hcp" : "");
    };
    const band = await mountScene(V, "scene-neb", DATA + neb.frames, {
      fps: 4, bounce: true, hold: 1.0, view: { elevation: 52, azimuth: -30 }, fill: 0.9, imageTint: IMAGE_TINT, onFrame,
    });
    if (band) {
      band.loops = true;
      markSites(band, nImg, document.getElementById("scene-neb"));
      onFrame(band.frame);
    }
    const mode = await mountScene(V, "scene-ts", DATA + neb.frames_extra.ts_mode, {
      fps: 20, hold: 0, replicate: [1, 1], view: { elevation: 30, azimuth: -30 }, restFrame: "first",
      interactive: false,
    });
    if (mode) mode.loops = true;
  }

  // ---------- (b) relaxation
  {
    const tile = document.getElementById("relax");
    const steps = K(relax, "BFGS steps, O at fcc"), ff = K(relax, "final fmax, O at fcc");
    const hO = K(relax, "O height above top Pt layer, fcc");
    fill(tile, {
      status: relax.converged ? "converged" : "not converged", statusCls: relax.converged ? "ok" : "pend",
      cmd: cmdShort(relax.command, ["--mlip", "--uma-task", "--fmax"]),
      cmdFullTokens: cmdFull(relax.command),
      keyV: num(intv(steps.value), "BFGS steps"),
      keyL: `to fmax ${fx(ff.value, 4)} ${ff.unit} (target ${target.value})`,
      conv: `${relax.converged ? "Converged" : "Not converged"}: final fmax ${fx(ff.value, 4)} ${esc(ff.unit)} against a target of ${target.value} ${esc(target.unit)}, ${intv(steps.value)} BFGS steps. Relaxed O height above the top Pt layer ${fx(hO.value, 2)} ${esc(hO.unit)}.`,
    });
    const s = relax.series;
    const eF = s.energy_eV[s.energy_eV.length - 1];
    const cE = chart(tile.querySelector('[data-chart="relax-e"]'), {
      x: s.step, xLabel: "optimizer step", yLabel: "E − E(final) / eV",
      series: [{ y: s.energy_eV.map((e) => e - eF), cls: "ln", points: true }],
      aria: "Energy relative to the final energy against optimizer step.",
    });
    const cF = chart(tile.querySelector('[data-chart="relax-f"]'), {
      x: s.step, xLabel: "optimizer step", yLabel: "fmax / eV Å⁻¹", yLog: true,
      series: [{ y: s.fmax_eV_per_A, cls: "ln", points: true }],
      refs: [{ y: target.value, label: `target ${target.value}` }],
      aria: "Maximum force against optimizer step, log scale, with the fmax target line.",
    });
    let heights = null;
    const tag = tile.querySelector('[data-overlay="relax"]');
    const onFrame = (i) => {
      cE.setCursor(i); cF.setCursor(i);
      const h = heights ? ` · O height <b>${fx(heights[i], 2)} Å</b>` : "";
      tag.innerHTML = `step <b>${i}</b> of ${s.step.length - 1}${h}`;
    };
    const text = await (await fetch(DATA + relax.frames)).text();
    heights = oHeights(V.parseExtXYZ(text));
    const ctrl = await mountScene(V, "scene-relax", null, {
      xyzText: text, fps: 5, hold: 1.4, view: { elevation: 20, azimuth: -30 }, imageTint: IMAGE_TINT, onFrame,
    });
    if (ctrl) { ctrl.loops = true; onFrame(ctrl.frame); }
  }

  // ---------- (c) committee
  {
    const tile = document.getElementById("committee");
    const s = committee.series;
    const members = Object.keys(s).filter((k) => k.startsWith("E_")).map((k) => k.slice(2, -3));
    const ratio = K(committee, "sigma_max_free / fmax, final");
    const smax = K(committee, "sigma_max_free, final");
    const smean = K(committee, "sigma_mean_free, final");
    const fm = K(committee, "final fmax of the committee-mean");
    const ct = K(committee, "fmax target");
    const nsteps = K(committee, "BFGS steps");
    const spread = K(committee, "energy spread across members, final");
    fill(tile, {
      status: committee.converged ? "converged" : "not converged", statusCls: committee.converged ? "ok" : "pend",
      cmd: cmdShort(committee.command, ["--committee", "--fmax"]),
      cmdFullTokens: cmdFull(committee.command),
      keyV: num(`${fx(ratio.value, 1)}×`, "fmax"),
      keyL: `largest force σ over the remaining fmax, last step`,
      conv: `${committee.converged ? "Converged" : "Not converged"}: committee-mean fmax ${fx(fm.value, 4)} ${esc(fm.unit)} against a target of ${ct.value} ${esc(ct.unit)}, ${intv(nsteps.value)} BFGS steps. Members: ${members.map(esc).join(", ")}. The picture is the start geometry (the committee run saves no trajectory), free atoms coloured by force σ at the last step, O red. Largest σ ${fx(smax.value, 3)} ${esc(smax.unit)}. Energy spread at the last step ${fx(spread.value * 1000, 1)} meV (standard deviation, step-0 offsets removed).`,
    });
    const last = s.step.length - 1;
    chart(tile.querySelector('[data-chart="committee"]'), {
      x: s.step, xLabel: "optimizer step", yLabel: "force / eV Å⁻¹", yLog: true, height: 150, gutter: 92,
      band: { lo: s.sigma_mean_free_eV_per_A, hi: s.sigma_max_free_eV_per_A, cls: "band-sig" },
      series: [
        { y: s.fmax_eV_per_A, cls: "ln", points: true, end: `fmax ${fx(fm.value, 3)}`, endCls: "lab-acc" },
        { y: s.sigma_max_free_eV_per_A, cls: "ln-sig", end: `σ max ${fx(smax.value, 3)}`, endCls: "lab-sig" },
        { y: s.sigma_mean_free_eV_per_A, cls: "ln-sig2", end: `σ mean ${fx(smean.value, 3)}`, endCls: "lab-sig" },
      ],
      refs: [{ y: ct.value, label: `target ${ct.value}`, at: "gutter" }],
      bracket: { i: last, a: s.fmax_eV_per_A[last], b: s.sigma_max_free_eV_per_A[last], label: `σ max = ${fx(ratio.value, 1)}× fmax` },
      aria: `Committee-mean maximum force against step, with the largest and the mean per-atom force standard deviation over free atoms. At the last step the largest sigma is ${fx(ratio.value, 1)} times the remaining fmax.`,
    });
    const sig = s.peratom_final_sigma_free_eV_per_A;
    const smaxAtom = Math.max(...sig);
    bind("ramp-lo", "0");
    bind("ramp-hi", `${fx(smaxAtom, 3)} eV/Å`);
    bind("committee-cap", "Start geometry, σ at last step");
    // relax.xyz frame 0 is the shared unrelaxed input (committee.json notes).
    const text = await (await fetch(DATA + relax.frames)).text();
    const first = text.split(/\r?\n/);
    const nat = parseInt(first[0], 10);
    const frame0 = first.slice(0, nat + 2).join("\n") + "\n";
    const ramp = (t) => {
      const stops = [[0xd8, 0xe2, 0xff], [0x6b, 0x7c, 0xf0], [0x45, 0x27, 0xb8]];
      const u = Math.max(0, Math.min(1, t)) * 2, k = Math.min(1, Math.floor(u)), f = u - k;
      const c = stops[k].map((v, j) => Math.round(v + (stops[k + 1][j] - v) * f));
      return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
    };
    // Per-atom sigma colouring lives in the viewer's restyle hook, which runs
    // after the default style on every model rebuild (frame, fit, resize).
    const restyle = (viewer, model) => {
      const colorfunc = (a) => {
        if (a.elem === "O") return V.ELEMENT_STYLE.O.color;
        const v = sig[a.properties.src];
        return v > 0 ? ramp(v / smaxAtom) : "#9da2ab";
      };
      for (const [e, st] of Object.entries(V.ELEMENT_STYLE)) {
        model.setStyle({ elem: e }, { sphere: { radius: st.space, colorfunc } });
      }
    };
    const ctrl = await mountScene(V, "scene-committee", null, {
      xyzText: frame0, view: { elevation: 90, azimuth: -30 }, restyle,
    });
    if (ctrl) ctrl.sigmaColour = (src) => ramp(sig[src] / smaxAtom); // for check.py
  }

  // ---------- (d) MD (data-driven: pending now, frames/series later)
  {
    const tile = document.getElementById("md");
    const pending = md.status === "pending" || !md.frames;
    const seriesKeys = Object.keys(md.series || {});
    fill(tile, {
      status: pending ? "pending" : md.converged ? "complete" : "not converged", statusCls: pending ? "pend" : "ok",
      cmd: pending ? ["mlip", "md", "run"] : cmdShort(md.command, ["--ensemble", "--temperature", "--timestep"]),
      cmdFullTokens: cmdFull(md.command),
      keyV: pending ? "pending" : num(intv(md.series[seriesKeys[0]].length), "frames"),
      keyL: pending ? "MD not run; next step pending a decision" : "",
      conv: pending
        ? `MD not run; next step pending a decision. Status: ${esc(md.status)} (${md.flags.map(esc).join(", ")}). The command above is the bilayer relaxation that ran before the MD stage, not an MD command. The 3D view: ${esc(md.structures.built.label)}; static, not a simulation.`
        : `MD complete.`,
    });
    const tsK = seriesKeys.find((k) => /time/i.test(k));
    const tK = seriesKeys.find((k) => /^T|temp/i.test(k));
    if (!pending && tsK && tK) {
      chart(tile.querySelector('[data-chart="md"]'), {
        x: md.series[tsK], xLabel: tsK.replace(/_/g, " "), yLabel: tK.replace(/_/g, " "),
        series: [{ y: md.series[tK], cls: "ln" }],
      });
    } else {
      tile.querySelector(".strip").hidden = true; // nothing to plot until the MD has run
    }
    if (pending) {
      bind("md-tag", "As built; static, not a simulation");
      await mountScene(V, "scene-md", DATA + md.structures.built.file, {
        style: "ballstick", view: { elevation: 26, azimuth: -30 }, fill: 0.8,
      });
    } else {
      bind("md-tag", "MD trajectory");
      const c = await mountScene(V, "scene-md", DATA + md.frames, { style: "ballstick", fps: 20, view: { elevation: 26, azimuth: -30 } });
      if (c) c.loops = true;
    }
  }

  // ---------- (e) web app: three stills; the recording plays on request
  {
    const tile = document.getElementById("webapp");
    const s = webapp.series;
    const wall = K(webapp, "job wall time"), relaxT = K(webapp, "relaxation wall time");
    const dE = K(webapp, "energy difference, web app"), ff = K(webapp, "final fmax"), wt = K(webapp, "fmax target");
    const steps = K(webapp, "BFGS steps"), speed = K(webapp, "video speed-up"), vlen = K(webapp, "video length");
    const machine = webapp.target.machine, env = webapp.target.env;
    fill(tile, {
      status: webapp.converged ? "converged" : "not converged", statusCls: webapp.converged ? "ok" : "pend",
      cmd: `<span class="tok">mlip webapp</span> <span class="tok">→ Machines → Add machine</span>`,
      cmdFullTokens: cmdFull(webapp.command),
      keyV: num(fx(wall.value, 1), wall.unit),
      keyL: `job wall time on ${machine}`,
      conv: `${webapp.converged ? "Converged" : "Not converged"}: fmax ${fx(ff.value, 4)} ${esc(ff.unit)} against a target of ${wt.value} ${esc(wt.unit)}, ${intv(steps.value)} BFGS steps, run on ${esc(machine)} (env ${esc(env)}). Same input as panel b; the final energy differs from the CLI run by ${signed(sci(dE.value), dE.value)} ${esc(dE.unit)}. Job wall time ${fx(wall.value, 1)} ${esc(wall.unit)} (model load + relaxation; optimizer ${fx(relaxT.value, 1)} ${esc(relaxT.unit)}).`,
    });
    bind("still-submit", "Submit");
    bind("still-running", "Running");
    bind("still-done", "Done");
    bind("vplay", `Play the recording (${fx(vlen.value, 1)} s; waiting part ×${intv(speed.value)})`);
    const eCLI = relax.series.energy_eV, eRef = eCLI[eCLI.length - 1];
    chart(tile.querySelector('[data-chart="webapp"]'), {
      x: s.step, xLabel: "optimizer step", yLabel: "E − E(final, CLI) / eV", height: 120,
      series: [
        { y: relax.series.energy_eV.map((e) => e - eRef), cls: "ln2" },
        { y: s.energy_eV.map((e) => e - eRef), points: true, r: 2.6 },
      ],
      texts: [{ text: `final difference ${signed(sci(dE.value), dE.value)} eV`, fx: 1, fy: 0.3, anchor: "end" }],
      aria: "Energy against step for the web app job and the CLI run on the same input; the two coincide.",
    });
    const v = document.getElementById("webapp-video");
    const vbox = tile.querySelector(".vbox");
    const vplay = document.getElementById("vplay");
    const showVideo = (on) => {
      vbox.hidden = !on;
      tile.classList.toggle("playing-video", on);
      if (on) {
        if (!v.src) v.src = DATA + webapp.video;
        v.play().catch(() => {});
      } else v.pause();
    };
    vplay.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!tile.classList.contains("expanded")) tile.querySelector(".grow").click();
      showVideo(true);
    });
    tile.addEventListener("tile-collapse", () => showVideo(false));
    window.__gallery.video = v;
  }

  // ---------- below the plate: caption, cos-cluster, capabilities
  {
    bind("machine", esc(webapp.target.machine));

    // Panel b's command with the run's own structure path replaced by a plain
    // file name: the path is where the showcase run happened, not a path to use.
    const runCmd = cmdFull(relax.command).map((t, i, a) => (a[i - 1] === "--structure" ? "POSCAR" : t));
    document.getElementById("cluster").innerHTML =
      `<p>No machines are preconfigured: add your own.</p>` +
      `<p><b>CLI</b>, on your GPU machine:</p><code class="cmd-full block">${cmdHTML(runCmd)}</code>` +
      `<p><b>Web app</b> (<code>mlip webapp</code>): <i>Machines</i> → <i>Add machine</i> (<i>Host</i> <code>&lt;your ssh alias&gt;</code>, <i>Default remote location</i> <code>&lt;your remote folder&gt;</code>) → <i>Add environment</i> (<i>Python path</i> <code>&lt;your uma env&gt;/bin/python</code>) → <i>Test connection</i>. ` +
      `<a href="https://github.com/manuelarcer/mliprun/blob/main/webapp/README.md#running-on-another-machine-ssh">Guide</a>.</p>`;

    document.getElementById("caps").innerHTML = caps.items
      .map((c) => `<li>${c.cmd ? `<code>${esc(c.cmd)}</code> ` : ""}${esc(c.text)}</li>`).join("");
  }

  // ---------- loops: start (not under reduced motion), global toggle
  const loopsBtn = document.getElementById("loops");
  const setBtn = (on) => { loopsBtn.textContent = on ? "Pause loops" : "Play loops"; loopsBtn.setAttribute("aria-pressed", on ? "false" : "true"); };
  let looping = !reducedMotion;
  if (looping) for (const c of Object.values(scenes)) if (c && c.loops) c.play();
  setBtn(looping);
  loopsBtn.addEventListener("click", () => {
    looping = !looping;
    if (looping) playAll(); else pauseAll();
    setBtn(looping);
  });

  setupDetails();
  setupEnlarge();
  // Fonts and the grid settle after the first fits: refit every scene once.
  await document.fonts.ready;
  await new Promise((r) => setTimeout(r, 150));
  for (const c of Object.values(scenes)) if (c) c.resize();
  const nb = scenes["scene-neb"];
  if (nb && nb.replaceSites) nb.replaceSites();
  window.__gallery.ready = true;
}

// fcc / hcp markers on the two hollows of the central cell (frame 0 = fcc,
// last frame = hcp). The hollows are only 1.6 A apart, so each one gets a ring
// at its projected screen position plus a leader line to a label placed AWAY
// from the other hollow (and a little upward). Positions come from
// viewer.modelToScreen (page coordinates) and are recomputed after every
// resize and user rotation, so a label can never point at the wrong hollow.
function markSites(ctrl, nImg, host) {
  const keep = ctrl.frame;
  const pos = (k) => {
    ctrl.setFrame(k);
    const o = ctrl.viewer.getModel().selectedAtoms({ elem: "O" }).find((a) => !a.properties.image);
    return o ? { x: o.x, y: o.y, z: o.z } : null;
  };
  const sites = { fcc: pos(0), hcp: pos(nImg - 1) };
  ctrl.setFrame(keep);
  if (!sites.fcc || !sites.hcp) return;
  const svg = el("svg", { class: "sites", "aria-hidden": "true" });
  host.parentElement.appendChild(svg);
  const R = 46; // px from hollow to label centre
  const place = () => {
    const r = host.getBoundingClientRect();
    const ox = r.left + window.scrollX, oy = r.top + window.scrollY;
    const scr = {};
    for (const k of ["fcc", "hcp"]) {
      const p = ctrl.viewer.modelToScreen(sites[k]);
      scr[k] = { x: p.x - ox + host.offsetLeft, y: p.y - oy + host.offsetTop };
    }
    svg.replaceChildren();
    const out = {};
    for (const [k, other] of [["fcc", "hcp"], ["hcp", "fcc"]]) {
      let ux = scr[k].x - scr[other].x, uy = scr[k].y - scr[other].y;
      const n0 = Math.hypot(ux, uy) || 1;
      ux = ux / n0; uy = uy / n0 - 1.1; // away from the other hollow, biased upward
      const n1 = Math.hypot(ux, uy) || 1;
      const L = { x: scr[k].x + (ux / n1) * R, y: scr[k].y + (uy / n1) * R };
      out[k] = { site: scr[k], label: L };
      svg.appendChild(el("line", { class: "site-lead", x1: scr[k].x, y1: scr[k].y, x2: L.x, y2: L.y }));
      svg.appendChild(el("circle", { class: "site-ring", cx: scr[k].x, cy: scr[k].y, r: 7 }));
      svg.appendChild(el("rect", { class: "site-box", x: L.x - 15, y: L.y - 9, width: 30, height: 18, rx: 3 }));
      svg.appendChild(el("text", { class: "site-lab", x: L.x, y: L.y + 4, "text-anchor": "middle" }, k));
    }
    window.__gallery.sites = out; // read by check.py
  };
  // Expected ring positions right now (for check.py): same projection, no caching.
  window.__gallery.siteCheck = () => {
    const r = host.getBoundingClientRect();
    const ox = r.left + window.scrollX, oy = r.top + window.scrollY;
    const exp = {};
    for (const k of ["fcc", "hcp"]) {
      const p = ctrl.viewer.modelToScreen(sites[k]);
      exp[k] = { x: p.x - ox + host.offsetLeft, y: p.y - oy + host.offsetTop };
    }
    return { expected: exp, drawn: { fcc: window.__gallery.sites.fcc.site, hcp: window.__gallery.sites.hcp.site } };
  };
  place();
  let tm = 0;
  const later = () => { clearTimeout(tm); tm = setTimeout(place, 80); };
  new ResizeObserver(() => setTimeout(later, 220)).observe(host);
  host.addEventListener("pointerup", later);
  host.addEventListener("pointermove", (e) => { if (e.buttons) later(); });
  ctrl.replaceSites = place;
  // The viewer rebuilds and re-fits on resize (which also resets a user
  // rotation); its restyle hook runs after every rebuild, so the rings follow.
  let raf = 0;
  ctrl.setRestyle(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(place); });
}

// ------------------------------------------------------------------ details
// The full command and convergence label live INSIDE the tile, as an overlay
// directly above the command line (pointer-events: none, so it never covers
// another panel or traps the pointer); when enlarged it sits in the flow. Shown while the pointer is on the command line,
// on keyboard focus of the tile, and when enlarged.
function setupDetails() {
  for (const t of document.querySelectorAll(".tile")) {
    const d = t.querySelector(".detail");
    const foot = t.querySelector(".tile-foot");
    if (d && foot) foot.appendChild(d); // sits just above the command it expands
    // Hovering the command line (not the whole tile) reveals the full command,
    // so the picture stays visible while the pointer is on it.
    const fitDetail = () => {
      const top = foot.getBoundingClientRect().top;
      d.style.maxHeight = `${Math.max(80, Math.min(top - 8, window.innerHeight * 0.6))}px`;
    };
    foot.addEventListener("mouseenter", () => { fitDetail(); t.classList.add("show-detail"); });
    t.addEventListener("focusin", fitDetail);
    foot.addEventListener("mouseleave", () => t.classList.remove("show-detail"));
  }
}

// ------------------------------------------------------------------ enlarge
function setupEnlarge() {
  const backdrop = document.getElementById("backdrop");
  let open = null, placeholder = null, lastFocus = null;
  const others = () => [document.querySelector(".masthead"), document.querySelector(".below"), document.querySelector(".plate-title"),
    ...document.querySelectorAll(".tile")].filter((n) => n !== open);
  const close = () => {
    if (!open) return;
    const t = open;
    t.classList.remove("expanded");
    const b = t.querySelector(".grow");
    b.setAttribute("aria-expanded", "false");
    b.querySelector(".vh").textContent = b.querySelector(".vh").textContent.replace("Reduce", "Enlarge");
    t.dispatchEvent(new Event("tile-collapse"));
    placeholder.remove();
    backdrop.hidden = true;
    others().forEach((n) => { n.inert = false; });
    open = null;
    (lastFocus || t).focus({ preventScroll: true });
  };
  const expand = (t) => {
    if (open) close();
    lastFocus = document.activeElement;
    placeholder = document.createElement("div");
    placeholder.className = "placeholder " + [...t.classList].filter((c) => c.startsWith("t-")).join(" ");
    placeholder.style.height = t.offsetHeight + "px";
    t.before(placeholder);
    t.classList.add("expanded");
    t.scrollTop = 0;
    open = t;
    backdrop.hidden = false;
    others().forEach((n) => { n.inert = true; });
    const b = t.querySelector(".grow");
    b.setAttribute("aria-expanded", "true");
    b.querySelector(".vh").textContent = b.querySelector(".vh").textContent.replace("Enlarge", "Reduce");
    b.focus({ preventScroll: true });
  };
  for (const t of document.querySelectorAll(".tile")) {
    t.querySelector(".grow").addEventListener("click", (e) => { e.stopPropagation(); open === t ? close() : expand(t); });
    t.addEventListener("click", (e) => {
      if (open === t) return;
      if (e.target.closest("button, a, .scene, video")) return;
      expand(t);
    });
    t.addEventListener("keydown", (e) => {
      if (e.target === t && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open === t ? close() : expand(t); }
    });
  }
  backdrop.addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
}

main().catch((e) => {
  console.error(e);
  const p = document.createElement("p");
  p.className = "conv";
  p.textContent = `The gallery could not load its data: ${e.message}`;
  document.querySelector(".plate").before(p);
});
