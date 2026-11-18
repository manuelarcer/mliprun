// mliprun handbook: binds every number on the page to the data pack in data/.
// Nothing numeric is typed here except display choices (decimal places, chart sizes).
// The 3D viewer is the shared site/shared/viewer.js (not edited).

const DATA = "data/";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const state = { scenes: {}, videos: [], charts: {} };
window.__handbook = state; // read by check.py

/* ------------------------------------------------------------ loading */
async function getJSON(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}
async function getText(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.text();
}
async function getViewer() {
  // ../shared/ on GitHub Pages and locally; ./shared/ in the published preview copy.
  try { return await import("../shared/viewer.js"); }
  catch { return await import("./shared/viewer.js"); }
}

/* ------------------------------------------------------------ numbers */
function findNum(d, prefix) {
  const k = d.key_numbers.find((n) => n.label.startsWith(prefix));
  if (!k) console.warn(`key number not found: ${d.use_case} | ${prefix}`);
  return k;
}
const sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
function fmtValue(v, unit) {
  const a = Math.abs(v);
  if (["steps", "models", "atoms"].includes(unit)) return String(Math.round(v));
  if (a !== 0 && a < 1e-3) {
    const [m, e] = v.toExponential(1).split("e");
    return `${m} × 10${String(Number(e)).split("").map((c) => sup[c]).join("")}`;
  }
  if (unit === "eV" && a >= 10) return v.toFixed(4);
  if (unit === "Å") return v.toFixed(3);
  if (unit === "cm⁻¹") return v.toFixed(1);
  if (unit === "s") return v.toFixed(1);
  return String(Number(v.toPrecision(3)));
}
function unitText(unit) { return unit === "dimensionless" ? "" : unit; }
function numHTML(k, { imaginary = false } = {}) {
  if (!k) return '<span class="missing">missing</span>';
  const u = unitText(k.unit);
  return `${fmtValue(k.value, k.unit)}${imaginary ? "i" : ""}${u ? `<span class="u">${u}</span>` : ""}`;
}
function numText(k) {
  if (!k) return "missing";
  const u = unitText(k.unit);
  return `${fmtValue(k.value, k.unit)}${u ? " " + u : ""}`;
}
// Readable names for pack keys; the pack label stays in the title attribute.
const READABLE = [["sigma_max_free / fmax", "max force σ on free atoms / fmax"],
  ["sigma_max_free", "max force σ on free atoms"], ["sigma_mean_free", "mean force σ on free atoms"]];
const readable = (label) => READABLE.reduce((t, [a, b]) => t.split(a).join(b), label);
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ------------------------------------------------------------ status */
function statusHTML(d) {
  if (d.status === "pending") return '<span class="pill pill-pending">pending</span>';
  let h = d.converged
    ? '<span class="pill pill-ok">converged</span>'
    : '<span class="pill pill-pending">not converged</span>';
  if ((d.flags || []).includes("ts_verified")) h += ' <span class="pill pill-ok">TS verified</span>';
  return h;
}

/* ------------------------------------------------------------ recipes */
// Turn the command that ran into a copyable recipe: the binary path becomes `mlip`
// (run inside the env), and only the PATH values listed in `subs` become placeholders.
// Every flag and every non-path value stays exactly as run.
function buildRecipe(cmd, subs = {}, pre = []) {
  const t = cmd.trim().split(/\s+/);
  if (/(^|\/)mlip$/.test(t[0])) t[0] = "mlip";
  const groups = [];
  let head = [];
  let i = 0;
  while (i < t.length && !t[i].startsWith("--")) head.push(t[i++]);
  while (i < t.length) {
    const g = [{ s: t[i] }];
    const flag = t[i++];
    while (i < t.length && !t[i].startsWith("--")) {
      const ph = subs[flag];
      g.push(ph ? { s: ph, ph: true } : { s: t[i] });
      i++;
    }
    groups.push(g);
  }
  // pack groups into lines of at most ~52 characters
  const lines = [head.map((s) => ({ s }))];
  let cur = null;
  for (const g of groups) {
    const len = g.reduce((n, p) => n + p.s.length + 1, 0);
    if (!cur || cur.len + len > 52) { cur = { parts: [], len: 0 }; lines.push(cur); }
    cur.parts.push(...g); cur.len += len;
  }
  const rows = lines.map((l) => (Array.isArray(l) ? l : l.parts));
  const html = [], text = [];
  for (const p of pre) {
    const h = p.cm ? `<span class="cm">${esc(p.s)}</span>` : esc(p.s).replace(/&lt;[^&]+&gt;/g, (m) => `<span class="ph">${m}</span>`);
    html.push(h); text.push(p.s);
  }
  rows.forEach((parts, j) => {
    const last = j === rows.length - 1;
    const indent = j === 0 ? "" : "  ";
    html.push(indent + parts.map((p) => (p.ph ? `<span class="ph">${esc(p.s)}</span>` : esc(p.s))).join(" ") + (last ? "" : " \\"));
    text.push(indent + parts.map((p) => p.s).join(" ") + (last ? "" : " \\"));
  });
  return { html: html.join("\n"), text: text.join("\n") };
}
function flagValue(cmd, flag) {
  const t = cmd.trim().split(/\s+/);
  const i = t.indexOf(flag);
  return i >= 0 ? t[i + 1] : null;
}
function setRecipe(key, r) {
  const pre = document.querySelector(`[data-recipe="${key}"]`);
  pre.querySelector("code").innerHTML = r.html;
  pre.dataset.text = r.text;
}
function fillAsRun(key, d, extra = []) {
  const el = document.querySelector(`[data-asrun="${key}"]`);
  const rows = [];
  if (d.cwd) rows.push(["Folder", d.cwd]);
  if (d.run_dir) rows.push(["Pack run dir", d.run_dir]);
  for (const [a, b] of extra) rows.push([a, b]);
  el.insertAdjacentHTML("beforeend",
    `<div class="raw">${esc(d.command)}</div>` +
    (rows.length ? `<dl>${rows.map(([a, b]) => `<dt>${esc(a)}</dt><dd>${esc(b)}</dd>`).join("")}</dl>` : ""));
}

/* ------------------------------------------------------------ copy buttons */
function wireCopy() {
  document.querySelectorAll("pre[data-copy]").forEach((pre) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = "Copy";
    b.setAttribute("aria-label", "Copy this block");
    b.addEventListener("click", () => {
      const text = pre.dataset.text || pre.querySelector("code").textContent;
      const ok = () => { b.textContent = "Copied"; b.classList.add("done"); setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1600); };
      const fallback = () => {
        const r = document.createRange(); r.selectNodeContents(pre.querySelector("code"));
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        b.textContent = "Selected: press Ctrl+C";
      };
      try { navigator.clipboard.writeText(text).then(ok, fallback); } catch { fallback(); }
    });
    pre.appendChild(b);
  });
}

/* ------------------------------------------------------------ charts (plain SVG) */
const NS = "http://www.w3.org/2000/svg";
function niceTicks(min, max, n = 5) {
  const span = max - min || Math.abs(max) || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}
function tickLabel(v) { return String(Number(v.toPrecision(6))); }

function chart(el, cfg) {
  // viewBox width follows the rendered width, so 11 px text stays about 11 px on phones
  const avail = el.clientWidth || (el.parentElement && el.parentElement.clientWidth) || 520;
  const W = cfg.w || Math.round(Math.min(560, Math.max(300, avail))), H = cfg.h || 210;
  const m = { l: 50, r: 14, t: 24, b: 40 };
  const x = cfg.x;
  const ys = cfg.series.flatMap((s) => s.y).concat(cfg.thr ? [cfg.thr.y] : []);
  let xmin = Math.min(...x), xmax = Math.max(...x);
  let y0, y1, yTicks, ty;
  if (cfg.log) {
    const lo = Math.floor(Math.log10(Math.min(...ys.filter((v) => v > 0))));
    const hi = Math.ceil(Math.log10(Math.max(...ys)));
    y0 = lo; y1 = hi;
    yTicks = []; for (let e = lo; e <= hi; e++) yTicks.push(10 ** e);
    ty = (v) => m.t + (H - m.t - m.b) * (1 - (Math.log10(v) - y0) / (y1 - y0));
  } else {
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (cfg.yZero) lo = Math.min(lo, 0);
    const pad = (hi - lo) * 0.08; lo -= cfg.yZero && lo === 0 ? 0 : pad; hi += pad;
    yTicks = niceTicks(lo, hi, 4);
    const st = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
    while (yTicks[yTicks.length - 1] < Math.max(...ys)) yTicks.push(Number((yTicks[yTicks.length - 1] + st).toPrecision(12)));
    y0 = Math.min(lo, yTicks[0]); y1 = Math.max(hi, yTicks[yTicks.length - 1]);
    ty = (v) => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
  }
  const xTicks = cfg.xTicks || niceTicks(xmin, xmax, 6);
  const tx = (v) => m.l + (W - m.l - m.r) * ((v - xmin) / (xmax - xmin));
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", cfg.aria || `${cfg.yLabel} against ${cfg.xLabel}`);
  const add = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    svg.appendChild(e); return e;
  };
  // grid and ticks
  for (const v of yTicks) {
    const y = ty(v);
    add("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridl" });
    add("text", { x: m.l - 6, y: y + 3.5, "text-anchor": "end" }, cfg.log ? tickLabel(v) : tickLabel(v));
  }
  for (const v of xTicks) {
    const xx = tx(v);
    add("line", { x1: xx, x2: xx, y1: H - m.b, y2: H - m.b + 4, class: "axis" });
    add("text", { x: xx, y: H - m.b + 16, "text-anchor": "middle" }, tickLabel(v));
  }
  add("path", { d: `M${m.l},${m.t - 4}V${H - m.b}H${W - m.r}`, class: "axis" });
  add("text", { x: m.l, y: 12, class: "ttl" }, cfg.yLabel);
  add("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, cfg.xLabel);
  if (cfg.thr) {
    const y = ty(cfg.thr.y);
    add("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "thr" });
    add("text", { x: m.l + 8, y: y - 5, class: "thr-t" }, cfg.thr.label);
  }
  cfg.series.forEach((s, j) => {
    const d = s.y.map((v, i) => `${i ? "L" : "M"}${tx(x[i]).toFixed(1)},${ty(v).toFixed(1)}`).join("");
    add("path", { d, class: s.cls || `s${j + 1}` });
    if (s.dots) s.y.forEach((v, i) => add("circle", { cx: tx(x[i]), cy: ty(v), r: 3.2, class: `d${j + 1}` }));
  });
  if (cfg.mark) {
    const { i, label, si = 0 } = cfg.mark;
    const cx = tx(x[i]), cy = ty(cfg.series[si].y[i]);
    add("circle", { cx, cy, r: 5, class: "ts" });
    add("text", { x: cx, y: cy - 10, "text-anchor": "middle", class: "ttl" }, label);
  }
  const cur = add("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "cursor", visibility: "hidden" });
  const cdots = cfg.series.map(() => add("circle", { r: 4, class: "cdot", visibility: "hidden" }));
  el.replaceChildren(svg);
  if (cfg.series.length > 1 || cfg.legend) {
    const lg = document.createElement("p");
    lg.className = "legend";
    lg.innerHTML = cfg.series.map((s, j) => `<span><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" class="${s.cls || "s" + (j + 1)}"/></svg>${esc(s.label)}</span>`).join("") +
      (cfg.thr ? `<span><svg width="18" height="8" aria-hidden="true"><line x1="0" x2="18" y1="4" y2="4" class="thr"/></svg>${esc(cfg.thr.label)}</span>` : "");
    el.appendChild(lg);
  }
  return {
    cursor(i) {
      if (i == null || i < 0 || i >= x.length) return;
      const xx = tx(x[i]);
      cur.setAttribute("x1", xx); cur.setAttribute("x2", xx); cur.setAttribute("visibility", "visible");
      cfg.series.forEach((s, j) => { cdots[j].setAttribute("cx", xx); cdots[j].setAttribute("cy", ty(s.y[i])); cdots[j].setAttribute("visibility", "visible"); });
    },
  };
}

/* ------------------------------------------------------------ scenes */
async function load3d(viewer, el, url, opts) {
  return viewer.loadScene(el, url, opts);
}
// When the 3D viewer cannot load (CDN down), show the README GIF of the same run.
const GIF = { relax: ["relax", "Relaxation of O on Pt(111), one frame per BFGS step"],
  neb: ["neb-band", "Converged NEB band, image by image"], mode: ["ts-mode", "TS imaginary mode, illustration"] };
function gifFallback(el, key, err) {
  const g = GIF[key];
  if (!g) { el.innerHTML = '<p class="scene-fail">3D view could not load.</p>'; return; }
  el.parentElement?.classList.add("fallback");
  el.innerHTML = `<figure class="gif-fallback"><img src="media/${g[0]}-${isDark() ? "dark" : "light"}.gif" alt="${esc(g[1])}"><figcaption>Recorded GIF of the same run; the live 3D viewer could not load.</figcaption></figure>`;
  console.warn("3D viewer failed:", err && err.message);
}
const tint = () => (isDark() ? 0.3 : 0.55);
const sceneBg = () => getComputedStyle(document.documentElement).getPropertyValue("--scene-bg").trim() || "#ffffff";
async function makeScene(viewer, key, el, url, opts, onFrame) {
  const btn = document.querySelector(`.playbtn[data-scene="${key}"]`);
  try {
    const sc = await load3d(viewer, el, url, { background: sceneBg(), scrollZoom: false, onFrame, ...opts });
    state.scenes[key] = sc;
    onFrame && onFrame(sc.frame);
    if (!reducedMotion) sc.play();
    if (btn) {
      const label = () => { btn.textContent = sc.playing ? "Pause" : "Play"; };
      label();
      btn.addEventListener("click", () => { sc.playing ? sc.pause() : sc.play({ force: true }); label(); });
      el.addEventListener("pointerdown", () => setTimeout(label, 0));
      setInterval(label, 500);
    }
    return sc;
  } catch (e) {
    gifFallback(el, key, e);
    if (btn) btn.hidden = true;
    return null;
  }
}
function setupVideo(v, src) {
  v.src = src;
  state.videos.push(v);
  if (reducedMotion) { v.controls = true; v.removeAttribute("loop"); return; }
  v.controls = true;
  v.autoplay = true;
  v.play().catch(() => {});
}

/* ------------------------------------------------------------ main */
async function main() {
  const [relax, committee, neb, md, webapp, mdPlan, meta, yaml] = await Promise.all([
    getJSON("relax.json"), getJSON("committee.json"), getJSON("neb.json"), getJSON("md.json"),
    getJSON("webapp.json"), getJSON("md_planned.json"), getJSON("meta.json"), getText("committee.yaml"),
  ]);
  const P = { relax, committee, neb, md, webapp, md_planned: mdPlan, meta };

  // simple bindings
  document.querySelectorAll("[data-k]").forEach((el) => {
    const [uc, prefix] = el.dataset.k.split("|");
    el.innerHTML = numHTML(findNum(P[uc], prefix));
  });
  document.querySelectorAll("[data-meta]").forEach((el) => { el.textContent = meta[el.dataset.meta]; });
  document.querySelectorAll("[data-path]").forEach((el) => {
    const v = el.dataset.path.split(".").reduce((o, k) => (o ? o[k] : undefined), P);
    el.textContent = v ?? "missing";
  });
  document.querySelectorAll("[data-note]").forEach((el) => {
    const [uc, prefix] = el.dataset.note.split("|");
    el.textContent = (P[uc].notes || []).find((n) => n.startsWith(prefix)) || "";
  });
  document.querySelectorAll("[data-count]").forEach((el) => {
    const v = el.dataset.count.split(".").reduce((o, k) => (o ? o[k] : undefined), P);
    el.textContent = Array.isArray(v) ? String(v.length) : "missing";
  });
  document.querySelectorAll("[data-flag]").forEach((el) => {
    const [uc, flag] = el.dataset.flag.split("|");
    el.textContent = flagValue(P[uc].command, flag) ?? "missing";
  });
  document.querySelectorAll("[data-status]").forEach((el) => { el.innerHTML = statusHTML(P[el.dataset.status]); });

  // key-number lists + the full list in a disclosure
  document.querySelectorAll("dl[data-nums]").forEach((dl) => {
    const d = P[dl.dataset.nums];
    const picks = dl.dataset.pick.split("|");
    dl.innerHTML = picks.map((p) => {
      const k = findNum(d, p);
      if (!k) return "";
      const label = readable(k.label.replace(/\s+\([^()]*\)$/, ""));
      return `<div><dt title="${esc(k.label)}">${esc(label)}</dt><dd>${numHTML(k, { imaginary: k.label.includes("the mode is imaginary") })}</dd></div>`;
    }).join("");
    const det = document.createElement("details");
    det.className = "all-nums";
    det.innerHTML = `<summary>All ${d.key_numbers.length} numbers in ${esc(d.use_case)}.json</summary><div class="table-wrap"><table><tbody>${
      d.key_numbers.map((k) => `<tr><td title="${esc(k.label)}">${esc(readable(k.label))}</td><td>${numHTML(k)}</td></tr>`).join("")}</tbody></table></div>`;
    dl.after(det);
    // key numbers sit under the command, so the text column is not left short
    const text = dl.closest(".recipe-grid")?.querySelector(".recipe-side");
    if (text) { text.append(dl, det); dl.classList.add("nums-text"); }
  });

  // recipes
  const act = [{ s: "source <your uma env>/bin/activate", cm: false }];
  setRecipe("relax", buildRecipe(relax.command, { "--structure": "<your POSCAR>" }, act));
  setRecipe("committee", buildRecipe(committee.command, { "--structure": "<your POSCAR>" },
    [{ s: "# members run in the envs named in committee.yaml", cm: true }, ...act]));
  setRecipe("neb", buildRecipe(neb.command, { "--initial": "<IS POSCAR>", "--final": "<FS POSCAR>" },
    [...act, { s: "# NEB writes into the current folder", cm: true }, { s: "cd <neb run folder>", cm: false }]));
  setRecipe("md", buildRecipe(mdPlan.command, { "--structure": "<relaxed bilayer POSCAR>" },
    [{ s: "# planned: this command has NOT run yet", cm: true }, ...act]));
  setRecipe("webapp", buildRecipe(webapp.command, {}, []));
  fillAsRun("relax", relax);
  fillAsRun("committee", committee);
  fillAsRun("neb", neb);
  fillAsRun("md", md, [["Note", (md.notes || []).find((n) => n.startsWith("command is")) || ""]]);
  fillAsRun("webapp", webapp, [["Machine", webapp.target.machine], ["Environment", webapp.target.env], ["Python", webapp.target.python]]);
  document.getElementById("md-source").innerHTML =
    `From <code>${esc(mdPlan.source)}</code>, status <strong>${esc(mdPlan.status)}</strong>. ${esc(mdPlan.structure_note)}`;

  // committee.yaml with env paths as placeholders
  const yamlBody = yaml.slice(yaml.indexOf("members:")).trimEnd();
  const yamlHTML = esc(yamlBody).replace(/(env:\s+)(\S+)/g, (_, a, path) => `${a}<span class="ph">&lt;your ${esc(path.split("/").pop())} env&gt;</span>`);
  const yamlPre = document.querySelector("pre[data-yaml]");
  yamlPre.querySelector("code").innerHTML = yamlHTML;
  yamlPre.dataset.text = yamlBody.replace(/(env:\s+)(\S+)/g, (_, a, path) => `${a}<your ${path.split("/").pop()} env>`);

  // environment table, from committee.yaml and the commands that ran
  const members = [];
  for (const block of yamlBody.split(/\n\s*-\s+/).slice(1)) {
    const get = (k) => (block.match(new RegExp(`${k}:\\s*(\\S+)`)) || [])[1];
    members.push({ env: get("env"), mlip: get("mlip"), head: get("uma_task") || get("sevennet_task") || get("mace_head") });
  }
  const pkg = (t) => (t.startsWith("uma") ? "fairchem-core" : t.startsWith("7net") ? "sevenn" : t.startsWith("mace") ? "mace-torch" : "?");
  const names = { relax: "relax", neb: "NEB", md: "bilayer relax", webapp: "web app job" };
  document.getElementById("env-rows").innerHTML = members.map((mm) => {
    const users = Object.keys(names).filter((k) => P[k].command.startsWith(mm.env + "/bin/") ||
      (k === "webapp" && webapp.target.python.startsWith(mm.env + "/bin/"))).map((k) => names[k]);
    users.push("committee member");
    return `<tr><td data-label="MLIP and head"><code>${esc(mm.mlip)}</code> / <code>${esc(mm.head)}</code></td><td data-label="Package">${pkg(mm.mlip)}</td><td data-label="Used by">${users.join(", ")}</td><td data-label="Where these runs happened"><code>${esc(mm.env)}</code></td></tr>`;
  }).join("");

  // NEB flag
  if ((neb.flags || []).some((f) => f !== "ts_verified")) {
    const f = document.getElementById("neb-flag");
    document.getElementById("neb-flag-text").textContent = neb.notes[0];
    f.hidden = false;
  }

  // webapp caption
  document.getElementById("webapp-cap").innerHTML =
    `Laptop submits, remote GPU machine runs; recording ${numText(findNum(webapp, "video length"))}, waiting part ×${fmtValue(findNum(webapp, "video speed-up").value, "dimensionless")}.`;

  // charts
  const rs = relax.series;
  const fT = findNum(relax, "fmax target");
  state.charts.relaxF = chart(document.getElementById("chart-relax-f"), {
    x: rs.step, series: [{ y: rs.fmax_eV_per_A, label: "fmax", dots: true }], log: true,
    thr: { y: fT.value, label: `target ${numText(fT)}` }, yLabel: "fmax (eV/Å, log scale)", xLabel: "BFGS step", h: 190,
  });
  const e0 = rs.energy_eV[0];
  state.charts.relaxE = chart(document.getElementById("chart-relax-e"), {
    x: rs.step, series: [{ y: rs.energy_eV.map((v) => v - e0), label: "E − E(step 0)", dots: true }],
    yLabel: "E − E(step 0) (eV)", xLabel: "BFGS step", h: 170,
  });
  const cs = committee.series;
  const cT = findNum(committee, "fmax target");
  const comF = (el) => chart(el, {
    x: cs.step, log: true, h: 210,
    series: [
      { y: cs.fmax_eV_per_A, label: "fmax of the committee-mean force" },
      { y: cs.sigma_max_free_eV_per_A, label: "max force σ on free atoms" },
      { y: cs.sigma_mean_free_eV_per_A, label: "mean force σ on free atoms" },
    ],
    thr: { y: cT.value, label: `fmax target ${numText(cT)}` }, yLabel: "force (eV/Å, log scale)", xLabel: "BFGS step",
  });
  state.charts.comF = comF(document.getElementById("chart-com-f"));
  state.charts.comE = chart(document.getElementById("chart-com-e"), {
    x: cs.step, series: [{ y: cs.energy_spread_aligned_eV, label: "energy spread", dots: true }], yZero: true,
    yLabel: "energy spread across members, σ (eV)", xLabel: "BFGS step", h: 170,
  });
  const ns = neb.series;
  const tsI = neb.ts_image_index;
  const nebChart = (el) => chart(el, {
    x: ns.reaction_coordinate_A, series: [{ y: ns.relative_energy_eV, label: "image energy", dots: true }],
    yLabel: "E − E(IS) (eV)", xLabel: "reaction coordinate (Å)", h: 210, yZero: true,
    mark: { i: tsI, label: `TS, image ${tsI} (0–${ns.image_index.length - 1}): ${numText(findNum(neb, "forward barrier"))}` },
  });
  state.charts.neb = nebChart(document.getElementById("chart-neb"));

  // MD slot: real data drops in here by data alone
  const mdHasSeries = md.series && Object.values(md.series).some((v) => Array.isArray(v) && v.length);

  // 3D scenes
  const viewer = await getViewer();
  const relaxHud = document.getElementById("hud-relax");
  const nSteps = rs.step.length - 1;
  makeScene(viewer, "relax", document.getElementById("scene-relax"), DATA + "relax.xyz",
    { view: { elevation: 14, azimuth: -30 }, fps: 4, hold: 1.2, restFrame: "last" },
    (i) => {
      relaxHud.textContent = `BFGS step ${rs.step[i]} of ${nSteps}, fmax ${fmtValue(rs.fmax_eV_per_A[i], "eV/Å")} eV/Å`;
      state.charts.relaxF.cursor(i); state.charts.relaxE.cursor(i);
    });
  const nebHud = document.getElementById("hud-neb");
  const nImg = ns.image_index.length;
  makeScene(viewer, "neb", document.getElementById("scene-neb"), DATA + "neb_band.xyz",
    { view: { elevation: 62, azimuth: -30 }, fps: 3, bounce: true, hold: 0.8, imageTint: tint(), restFrame: tsI },
    (i) => {
      nebHud.textContent = `image ${ns.image_index[i]} (0–${nImg - 1})${i === tsI ? ", TS" : ""}, E − E(IS) ${fmtValue(ns.relative_energy_eV[i], "eV")} eV`;
      state.charts.neb.cursor(i);
    });
  const fI = findNum(neb, "TS imaginary frequency"), amp = findNum(neb, "TS mode animation peak");
  document.getElementById("hud-mode").textContent = `TS mode ${fmtValue(fI.value, fI.unit)}i ${fI.unit}; mode, not dynamics`;
  makeScene(viewer, "mode", document.getElementById("scene-mode"), DATA + "ts_mode.xyz",
    { view: "tilted", fps: 20, hold: 0, imageTint: tint(), restFrame: "first" });

  if (md.frames) {
    const box = document.getElementById("md-result");
    box.innerHTML = '<div class="scene-box"><div class="scene" id="scene-md"></div></div>';
    makeScene(viewer, "md", document.getElementById("scene-md"), DATA + md.frames, {});
  }
  if (mdHasSeries) {
    const box = document.getElementById("md-result");
    const keys = Object.keys(md.series);
    const xk = keys[0];
    for (const k of keys.slice(1)) {
      const c = document.createElement("div"); c.className = "chart"; box.appendChild(c);
      chart(c, { x: md.series[xk], series: [{ y: md.series[k], label: k }], yLabel: k, xLabel: xk });
    }
  }

  const playRec = document.getElementById("webapp-play");
  playRec.addEventListener("click", () => {
    const box = document.getElementById("webapp-box");
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.controls = true;
    v.setAttribute("aria-label", "Screen recording of the web app submitting an optimize job to cos-cluster and showing the result");
    box.replaceChildren(v); box.classList.add("video-box");
    v.src = DATA + webapp.video; state.videos.push(v);
    v.play().catch(() => {});
    playRec.hidden = true;
    document.getElementById("webapp-still-cap").textContent = "Screen recording of the real job";
  });

  // at-a-glance preview
  const stage = document.getElementById("preview-stage");
  const cap = document.getElementById("preview-cap");
  const link = document.getElementById("preview-link");
  let pv = null;
  const target = { relax: "#relax", committee: "#committee", neb: "#neb", md: "#md", webapp: "#remote" };
  const caps = {
    relax: () => `O relaxes in the fcc hollow; final fmax ${numText(findNum(relax, "final fmax"))}.`,
    committee: () => `Force disagreement between three MLIPs, per BFGS step.`,
    neb: () => `O hops fcc → hcp; barrier ${numText(findNum(neb, "forward barrier"))}, TS mode ${fmtValue(fI.value, fI.unit)}i cm⁻¹.`,
    md: () => "MD not run; next step pending a decision.",
    webapp: () => `Web app on a laptop, job on a remote GPU machine; same energy as the CLI within ${numText(findNum(webapp, "energy difference"))}.`,
  };
  async function pick(key) {
    document.querySelectorAll(".row").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.pick === key)));
    if (pv && pv.destroy) pv.destroy();
    pv = null;
    stage.querySelectorAll("video").forEach((v) => { v.pause(); state.videos = state.videos.filter((x) => x !== v); });
    stage.replaceChildren();
    cap.innerHTML = caps[key]();
    link.href = target[key];
    if (key === "relax" || key === "neb") {
      const el = document.createElement("div"); el.className = "scene"; el.setAttribute("role", "img");
      el.setAttribute("aria-label", key === "relax" ? "Relaxation of O on Pt(111)" : "Final NEB band");
      stage.appendChild(el);
      const opts = key === "relax"
        ? { view: { elevation: 14, azimuth: -30 }, fps: 4, hold: 1.2 }
        : { view: { elevation: 62, azimuth: -30 }, fps: 3, bounce: true, hold: 0.8, imageTint: tint(), restFrame: tsI };
      try {
        const sc = await load3d(viewer, el, DATA + (key === "relax" ? "relax.xyz" : "neb_band.xyz"), { background: sceneBg(), ...opts });
        if (!el.isConnected) { sc.destroy(); return; }
        pv = sc; state.scenes.preview = sc;
        if (!reducedMotion) sc.play();
      } catch (e) { gifFallback(el, key === "relax" ? "relax" : "neb", e); }
    } else if (key === "committee") {
      const el = document.createElement("div"); el.className = "chart"; stage.appendChild(el);
      comF(el);
    } else if (key === "md") {
      stage.innerHTML = '<div class="pending-slot"><p class="pending-h">MD — pending</p><p>MD not run; next step pending a decision. No trajectory, no temperature trace.</p></div>';
    } else if (key === "webapp") {
      const img = document.createElement("img");
      img.className = "still"; img.src = "data/webapp_result.png";
      img.alt = "Web app at the end of the job: result panel and live energy and fmax plots";
      stage.appendChild(img);
    }
  }
  document.querySelectorAll(".row").forEach((b) => b.addEventListener("click", () => pick(b.dataset.pick)));
  await pick("neb");
}

/* ------------------------------------------------------------ chrome */
function tocSpy() {
  const links = [...document.querySelectorAll(".toc a")];
  const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const targets = [...map.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  let queued = false;
  const update = () => {
    queued = false;
    const line = Math.min(160, innerHeight * 0.25);
    let cur = targets[0];
    for (const t of targets) if (t.getBoundingClientRect().top <= line) cur = t;
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) cur = targets[targets.length - 1];
    links.forEach((a) => a.classList.toggle("on", a === map.get(cur.id)));
  };
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
  update();
  if (window.matchMedia("(max-width: 860px)").matches) document.getElementById("toc").open = false;
}

wireCopyLater();
function wireCopyLater() {
  tocSpy();
  main().then(wireCopy).catch((e) => {
    console.warn(e);
    const p = document.getElementById("load-error");
    p.textContent = `Some data could not load: ${e.message}. Numbers on this page come only from data/; serve the folder over HTTP.`;
    p.hidden = false;
    wireCopy();
  });
}
