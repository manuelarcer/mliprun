// Shared 3D trajectory viewer for the mliprun front-page showcase.
//
//   import { loadScene } from "../shared/viewer.js";
//   const scene = await loadScene(el, "relax.xyz", { fps: 20, replicate: [2, 2] });
//   scene.play();
//
// Input: multi-frame extended XYZ (ASE `write(..., format="extxyz")`), lattice in
// the comment line. Renders with 3Dmol.js 2.4.2 (loaded on demand from the
// pinned CDN URL when the page has not loaded it already).
//
// Every scene uses the same element colours and radii (ELEMENT_STYLE), the same
// camera convention (VIEWS) and the same lighting, so the five design
// alternatives look like one system.

export const THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js";

// Colours and radii (Angstrom). Pt: cool metallic grey; O: red; H: white.
// `space` = spacefill radius (close to covalent/metallic radii, so Pt spheres
// touch at the Pt-Pt distance); `ball` = ball-and-stick radius. In ball-and-stick
// the metal keeps a large radius on purpose: a slab drawn as small balls reads
// as a sparse cluster, not as a surface.
export const ELEMENT_STYLE = {
  Pt: { color: "#b4b9c3", space: 1.39, ball: 1.15 },
  O: { color: "#e0352b", space: 0.74, ball: 0.42 },
  H: { color: "#f7f7f7", space: 0.4, ball: 0.24 },
  C: { color: "#4a4a4a", space: 0.76, ball: 0.4 },
  N: { color: "#3756d6", space: 0.71, ball: 0.4 },
};
const FALLBACK = { color: "#ff66cc", space: 1.0, ball: 0.5 };

// Covalent radii used only to decide which non-metal pairs get a stick.
const COVALENT = { H: 0.31, C: 0.76, N: 0.71, O: 0.66 };
const STICK_RADIUS = 0.12;

// Camera presets. elevation = camera angle above the surface plane (deg);
// azimuth = rotation of the slab about its surface normal (deg).
export const VIEWS = {
  tilted: { elevation: 38, azimuth: -30, zoom: 1.0 },
  top: { elevation: 90, azimuth: -30, zoom: 1.0 },
  side: { elevation: 4, azimuth: 0, zoom: 1.0 },
  hero: { elevation: 18, azimuth: -28, zoom: 1.15 },
};

const DEFAULTS = {
  fps: 20,
  loop: true,
  bounce: false,
  hold: 0.6, // seconds to rest on the last frame (and on frame 0 when bouncing)
  style: "spacefill", // "spacefill" | "ballstick"
  showCell: false, // false | true (full cell) | "base" (in-plane parallelogram)
  replicate: [1, 1],
  replicateElements: null, // null = replicate every atom (true periodic images)
  imageTint: 0, // 0..1: fade periodic images toward the background colour
  spin: 0, // deg/s about the surface normal while playing; true = 10 deg/s
  view: "tilted", // preset name or {elevation, azimuth, zoom}; zoom is relative to the auto-fit
  fill: 0.9, // auto-fit: fraction of the limiting canvas dimension the scene fills
  fitSpin: false, // fit for every azimuth (set automatically when spin is on)
  background: "transparent",
  fog: true,
  ambientOcclusion: 0.4, // strength 0..1 (false = off): darkens crevices between atoms
  fov: 14, // camera field of view (deg); small = little perspective distortion
  orthographic: false,
  autoplay: false,
  restFrame: "last", // frame shown when not playing: "last" | "first" | index
  interactive: true, // mouse rotate; wheel zoom only with scrollZoom
  scrollZoom: false,
  pauseOffscreen: true,
  capture: false, // expose window.__scene for scripted frame stepping
  onFrame: null,
};

// ------------------------------------------------------------------ parsing
function parseComment(line) {
  const out = {};
  const re = /([A-Za-z_][\w-]*)\s*=\s*("([^"]*)"|'([^']*)'|\{([^}]*)\}|\S+)/g;
  let m;
  while ((m = re.exec(line)) !== null) {
    const raw = m[3] ?? m[4] ?? m[5] ?? m[2];
    out[m[1]] = raw;
  }
  return out;
}

function coerce(v) {
  if (v === "T" || v === "True" || v === "true") return true;
  if (v === "F" || v === "False" || v === "false") return false;
  const parts = v.trim().split(/\s+/);
  const nums = parts.map(Number);
  if (nums.every((x) => Number.isFinite(x))) return nums.length === 1 ? nums[0] : nums;
  return v;
}

/** Parse multi-frame (extended) XYZ text into frames {symbols, positions, cell, info}. */
export function parseExtXYZ(text) {
  const lines = text.split(/\r?\n/);
  const frames = [];
  let i = 0;
  while (i < lines.length) {
    if (!lines[i].trim()) {
      i++;
      continue;
    }
    const n = parseInt(lines[i], 10);
    if (!Number.isFinite(n)) throw new Error(`xyz: expected atom count at line ${i + 1}`);
    const kv = parseComment(lines[i + 1] || "");
    let speciesCol = 0;
    let posCol = 1;
    if (kv.Properties) {
      const p = kv.Properties.split(":");
      let col = 0;
      for (let k = 0; k + 2 < p.length; k += 3) {
        const name = p[k].toLowerCase();
        const width = parseInt(p[k + 2], 10);
        if (name === "species") speciesCol = col;
        if (name === "pos") posCol = col;
        col += width;
      }
    }
    let cell = null;
    if (kv.Lattice) {
      const L = kv.Lattice.trim().split(/\s+/).map(Number);
      if (L.length === 9) cell = [L.slice(0, 3), L.slice(3, 6), L.slice(6, 9)];
    }
    const info = {};
    for (const [k, v] of Object.entries(kv)) {
      if (k !== "Lattice" && k !== "Properties") info[k] = coerce(v);
    }
    const symbols = new Array(n);
    const positions = new Float64Array(3 * n);
    for (let a = 0; a < n; a++) {
      const t = (lines[i + 2 + a] || "").trim().split(/\s+/);
      if (t.length < posCol + 3) throw new Error(`xyz: short atom line ${i + 3 + a}`);
      symbols[a] = t[speciesCol].replace(/[^A-Za-z]/g, "");
      positions[3 * a] = +t[posCol];
      positions[3 * a + 1] = +t[posCol + 1];
      positions[3 * a + 2] = +t[posCol + 2];
    }
    frames.push({ symbols, positions, cell, info });
    i += n + 2;
  }
  if (!frames.length) throw new Error("xyz: no frames");
  return frames;
}

// ------------------------------------------------------------------ helpers
let threeDmolPromise = null;
function ensure3Dmol() {
  if (typeof window !== "undefined" && window.$3Dmol) return Promise.resolve(window.$3Dmol);
  if (!threeDmolPromise) {
    threeDmolPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = THREEDMOL_URL;
      s.async = true;
      s.onload = () => (window.$3Dmol ? resolve(window.$3Dmol) : reject(new Error("3Dmol missing")));
      s.onerror = () => reject(new Error(`could not load ${THREEDMOL_URL}`));
      document.head.appendChild(s);
    });
  }
  return threeDmolPromise;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((k) => parseInt(h.slice(k, k + 2), 16));
}
function mix(hexA, hexB, t) {
  const a = hexToRgb(hexA);
  const b = hexToRgb(hexB);
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * t));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}
function cssToHex(css) {
  if (/^#[0-9a-f]{6}$/i.test(css)) return css;
  const c = document.createElement("canvas").getContext("2d");
  c.fillStyle = css;
  const v = c.fillStyle; // normalised to #rrggbb or rgba()
  if (v.startsWith("#")) return v;
  const m = v.match(/\d+(\.\d+)?/g) || ["255", "255", "255"];
  return "#" + m.slice(0, 3).map((x) => (+x).toString(16).padStart(2, "0")).join("");
}

function quatFromView(elevationDeg, azimuthDeg) {
  // Rotate about the model z (surface normal) by azimuth, then about screen x
  // so the surface normal points up on screen, tilted toward the camera.
  // (3Dmol applies the quaternion as the inverse of the three.js convention,
  // hence +90: verified visually, the surface normal points up on screen.)
  const th = ((90 - elevationDeg) * Math.PI) / 180;
  const ph = (azimuthDeg * Math.PI) / 180;
  const sx = Math.sin(th / 2), cx = Math.cos(th / 2);
  const sz = Math.sin(ph / 2), cz = Math.cos(ph / 2);
  return [cz * sx, -sx * sz, cx * sz, cx * cz];
}

function prefersReducedMotion() {
  return typeof window !== "undefined" && !!window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// Replicate one frame into an atom list for 3Dmol (+ bonds between non-metals).
function buildAtoms(frame, rep, repElements, center) {
  const { symbols, positions, cell } = frame;
  const n = symbols.length;
  const [nx, ny] = rep;
  const ox = -Math.floor((nx - 1) / 2);
  const oy = -Math.floor((ny - 1) / 2);
  const atoms = [];
  for (let ix = ox; ix < ox + nx; ix++) {
    for (let iy = oy; iy < oy + ny; iy++) {
      const image = ix !== 0 || iy !== 0;
      const sh = cell && image
        ? [0, 1, 2].map((k) => ix * cell[0][k] + iy * cell[1][k])
        : [0, 0, 0];
      for (let a = 0; a < n; a++) {
        if (image && repElements && !repElements.includes(symbols[a])) continue;
        atoms.push({
          elem: symbols[a],
          x: positions[3 * a] + sh[0] - center[0],
          y: positions[3 * a + 1] + sh[1] - center[1],
          z: positions[3 * a + 2] + sh[2] - center[2],
          properties: { image, src: a },
          bonds: [],
          bondOrder: [],
        });
      }
    }
  }
  // Sticks only between non-metal atoms (O-H, C-O, ...), never to the metal.
  const nm = [];
  atoms.forEach((at, k) => { if (COVALENT[at.elem]) nm.push(k); });
  for (let p = 0; p < nm.length; p++) {
    for (let q = p + 1; q < nm.length; q++) {
      const A = atoms[nm[p]], B = atoms[nm[q]];
      if (A.elem === "H" && B.elem === "H") continue;
      const cut = 1.2 * (COVALENT[A.elem] + COVALENT[B.elem]);
      const dx = A.x - B.x, dy = A.y - B.y, dz = A.z - B.z;
      if (dx * dx + dy * dy + dz * dz < cut * cut) {
        A.bonds.push(nm[q]); A.bondOrder.push(1);
        B.bonds.push(nm[p]); B.bondOrder.push(1);
      }
    }
  }
  atoms.forEach((at, k) => { at.index = k; at.serial = k; });
  return atoms;
}

// Centre of the replicated block, averaged over all frames (fixed camera target).
function sceneCenter(frames, rep) {
  const f = frames[frames.length - 1];
  const n = f.symbols.length;
  const c = [0, 0, 0];
  for (let a = 0; a < n; a++) for (let k = 0; k < 3; k++) c[k] += f.positions[3 * a + k] / n;
  if (f.cell) {
    const [nx, ny] = rep;
    const ox = -Math.floor((nx - 1) / 2) + (nx - 1) / 2;
    const oy = -Math.floor((ny - 1) / 2) + (ny - 1) / 2;
    for (let k = 0; k < 3; k++) c[k] += ox * f.cell[0][k] + oy * f.cell[1][k];
  }
  return c;
}

// ------------------------------------------------------------------ main API
/**
 * Load a multi-frame extended-XYZ trajectory into `el` and return a controller.
 *
 * @param {HTMLElement} el      container; the canvas fills it (give it a size)
 * @param {string} xyzUrl       URL (http, relative or data:) of the .xyz file
 * @param {object} opts         see DEFAULTS above
 * @returns {Promise<{play, pause, setFrame, nFrames, ...}>}
 */
export async function loadScene(el, xyzUrl, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  if (o.spin === true) o.spin = 10;
  const reduced = prefersReducedMotion();

  const [$3Dmol, text] = await Promise.all([
    ensure3Dmol(),
    o.xyzText ? Promise.resolve(o.xyzText) : fetch(xyzUrl).then((r) => {
      if (!r.ok) throw new Error(`fetch ${xyzUrl}: HTTP ${r.status}`);
      return r.text();
    }),
  ]);
  const frames = parseExtXYZ(text);
  const nFrames = frames.length;
  const rep = o.replicate || [1, 1];
  const center = sceneCenter(frames, rep);

  const transparent = o.background === "transparent";
  const bgHex = transparent ? "#ffffff" : cssToHex(o.background);
  const tintTarget = transparent ? "#ffffff" : bgHex;

  if (getComputedStyle(el).position === "static") el.style.position = "relative";
  const viewer = $3Dmol.createViewer(el, {
    backgroundColor: bgHex,
    backgroundAlpha: transparent ? 0 : 1,
    antialias: true,
    nomouse: !o.interactive,
  });
  viewer.setCameraParameters({ fov: o.fov, orthographic: !!o.orthographic });
  viewer.setBackgroundColor(bgHex, transparent ? 0 : 1);
  if (o.ambientOcclusion) {
    viewer.setViewStyle({ style: "ambientOcclusion", strength: +o.ambientOcclusion || 0.4, radius: 4 });
  }
  if (!o.fog) viewer.enableFog(false);

  // Keep the page scrolling over the canvas unless scroll-zoom was asked for.
  const wheelGuard = (e) => { if (!o.scrollZoom) e.stopPropagation(); };
  el.addEventListener("wheel", wheelGuard, { capture: true });

  const colorOf = (atom) => {
    const base = (ELEMENT_STYLE[atom.elem] || FALLBACK).color;
    return atom.properties && atom.properties.image && o.imageTint > 0
      ? mix(base, tintTarget, o.imageTint) : base;
  };
  const elems = [...new Set(frames[0].symbols)];
  const applyStyle = () => {
    for (const e of elems) {
      const s = ELEMENT_STYLE[e] || FALLBACK;
      const style = o.style === "ballstick"
        ? { sphere: { radius: s.ball, colorfunc: colorOf },
            stick: { radius: STICK_RADIUS, colorfunc: colorOf } }
        : { sphere: { radius: s.space, colorfunc: colorOf } };
      viewer.setStyle({ elem: e }, style);
    }
  };

  const drawCell = () => {
    const cell = frames[0].cell;
    if (!o.showCell || !cell) return;
    const [a, b, c] = cell;
    const add = (...vs) => vs.reduce((s, v) => [s[0] + v[0], s[1] + v[1], s[2] + v[2]], [0, 0, 0]);
    const zero = [0, 0, 0];
    // In "base" mode the parallelogram sits at the bottom-layer height.
    const z0 = o.showCell === "base"
      ? [0, 0, Math.min(...frames[0].positions.filter((_, k) => k % 3 === 2))] : zero;
    const C = {
      o: add(zero, z0), a: add(a, z0), b: add(b, z0), ab: add(a, b, z0),
      c: add(c, z0), ac: add(a, c, z0), bc: add(b, c, z0), abc: add(a, b, c, z0),
    };
    let pairs = [["o", "a"], ["o", "b"], ["a", "ab"], ["b", "ab"]];
    if (o.showCell !== "base") {
      pairs = pairs.concat([["o", "c"], ["a", "ac"], ["b", "bc"], ["ab", "abc"],
        ["c", "ac"], ["c", "bc"], ["ac", "abc"], ["bc", "abc"]]);
    }
    const P = (v) => ({ x: v[0] - center[0], y: v[1] - center[1], z: v[2] - center[2] });
    for (const [p, q] of pairs) {
      viewer.addCylinder({ start: P(C[p]), end: P(C[q]), radius: 0.05, color: "#5b6b7f",
        fromCap: 1, toCap: 1 });
    }
  };

  let current = -1;
  let silent = false; // no onFrame callbacks while the camera fit renders frames
  const show = (i) => {
    const k = Math.max(0, Math.min(nFrames - 1, Math.round(i)));
    viewer.removeAllModels();
    const m = viewer.addModel();
    m.addAtoms(buildAtoms(frames[k], rep, o.replicateElements, center));
    applyStyle();
    viewer.render();
    current = k;
    if (o.onFrame && !silent) o.onFrame(k);
    return k;
  };

  // Camera. 3Dmol's zoomTo() fits the untilted model, which crops a tilted
  // slab, so the fit is done on rendered pixels instead: set the orientation,
  // draw the first and last frames with a transparent clear colour, take the
  // bounding box of drawn pixels, then zoom (and centre) so the content fills
  // `fill` of the limiting canvas dimension. With spin, the fit is the union
  // over azimuths and the scene is not shifted (it rotates about its centre).
  let viewState = typeof o.view === "string" ? { ...(VIEWS[o.view] || VIEWS.tilted) }
    : { ...VIEWS.tilted, ...o.view };
  show(0);
  drawCell();
  viewer.zoomTo();
  let fitted = viewer.getView();
  const quat = () => quatFromView(viewState.elevation, viewState.azimuth);
  const setRaw = (q) => viewer.setView([fitted[0], fitted[1], fitted[2], fitted[3], q[0], q[1], q[2], q[3]]);
  const applyView = () => {
    setRaw(quat());
    if (viewState.zoom && viewState.zoom !== 1) viewer.zoom(viewState.zoom);
    viewer.render();
  };

  const probe = document.createElement("canvas");
  const measure = () => {
    // Bounding box of drawn pixels as canvas fractions [x0, y0, x1, y1] (or null).
    viewer.render();
    const src = viewer.getCanvas();
    const w = 200;
    const h = Math.max(1, Math.round((w * src.height) / Math.max(1, src.width)));
    probe.width = w;
    probe.height = h;
    const g = probe.getContext("2d", { willReadFrequently: true });
    g.clearRect(0, 0, w, h);
    g.drawImage(src, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data;
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (d[4 * (y * w + x) + 3] > 8) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    return x1 < 0 ? null : [x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h];
  };
  const fit = ({ spinFit = false } = {}) => {
    const keep = current;
    silent = true;
    const fitFrames = nFrames > 1 ? [0, nFrames - 1] : [0];
    const azimuths = spinFit ? [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
      .map((a) => viewState.azimuth + a) : [viewState.azimuth];
    viewer.setBackgroundColor(bgHex, 0);
    const userZoom = viewState.zoom;
    viewState.zoom = 1;
    for (let iter = 0; iter < 3; iter++) {
      let box = null;
      for (const az of azimuths) {
        setRaw(quatFromView(viewState.elevation, az));
        for (const f of fitFrames) {
          show(f);
          const b = measure();
          if (b) box = box ? [Math.min(box[0], b[0]), Math.min(box[1], b[1]),
            Math.max(box[2], b[2]), Math.max(box[3], b[3])] : b;
        }
      }
      if (!box) break;
      setRaw(quat());
      const fx = box[2] - box[0];
      const fy = box[3] - box[1];
      if (!spinFit) {
        const W = el.clientWidth || 1, H = el.clientHeight || 1;
        const dx = (0.5 - (box[0] + box[2]) / 2) * W;
        const dy = (0.5 - (box[1] + box[3]) / 2) * H;
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) viewer.translateScene(dx, dy); // 3Dmol screen y runs down, like the canvas
      }
      const factor = o.fill / Math.max(fx, fy, 1e-3);
      if (Math.abs(factor - 1) > 0.005) viewer.zoom(factor);
      fitted = viewer.getView();
      if (Math.abs(factor - 1) < 0.01 && spinFit) break;
    }
    viewState.zoom = userZoom;
    viewer.setBackgroundColor(bgHex, transparent ? 0 : 1);
    applyView();
    if (keep >= 0) show(keep);
    silent = false;
  };
  fit({ spinFit: !!(o.spin || o.fitSpin) });

  const restIndex = o.restFrame === "first" ? 0
    : o.restFrame === "last" ? nFrames - 1 : Math.max(0, Math.min(nFrames - 1, o.restFrame | 0));
  show(restIndex);

  // Playback: a time-based sequence so fps is honoured on slow devices too.
  const seq = [];
  const holdN = Math.max(0, Math.round(o.hold * o.fps));
  for (let k = 0; k < nFrames; k++) seq.push(k);
  for (let h = 0; h < holdN; h++) seq.push(nFrames - 1);
  if (o.bounce && nFrames > 1) {
    for (let k = nFrames - 2; k >= 0; k--) seq.push(k);
    for (let h = 0; h < holdN; h++) seq.push(0);
  }

  let raf = 0, playing = false, t0 = 0, seqStart = 0, visible = true, wanted = false;
  let lastSpinT = 0;
  const tick = (t) => {
    if (!playing) return;
    // rAF timestamps can precede the performance.now() taken in start().
    const step = Math.max(0, Math.floor(((t - t0) / 1000) * o.fps));
    let pos = seqStart + step;
    if (pos >= seq.length) {
      if (!o.loop) {
        show(seq[seq.length - 1]);
        playing = false;
        wanted = false;
        return;
      }
      pos %= seq.length;
    }
    if (o.spin && lastSpinT) {
      viewState.azimuth += (o.spin * (t - lastSpinT)) / 1000;
      applyView();
    }
    lastSpinT = t;
    const idx = seq[pos];
    if (idx !== current) show(idx);
    else if (o.spin) viewer.render();
    ctrl.seqPos = pos;
    raf = requestAnimationFrame(tick);
  };

  const start = () => {
    if (playing || nFrames < 2) return;
    playing = true;
    t0 = performance.now();
    lastSpinT = 0;
    raf = requestAnimationFrame(tick);
  };
  const stop = () => {
    playing = false;
    cancelAnimationFrame(raf);
    if (typeof ctrl.seqPos === "number") seqStart = ctrl.seqPos;
  };

  let io = null;
  if (o.pauseOffscreen && typeof IntersectionObserver !== "undefined") {
    io = new IntersectionObserver((entries) => {
      visible = entries.some((e) => e.isIntersecting);
      if (!visible) stop();
      else if (wanted) start();
    });
    io.observe(el);
  }
  const stopSpinOnTouch = () => { o.spin = 0; };
  el.addEventListener("pointerdown", stopSpinOnTouch);

  const onResize = () => {
    viewer.resize();
    fit({ spinFit: !!(o.spin || o.fitSpin) });
  };
  window.addEventListener("resize", onResize);

  const ctrl = {
    nFrames,
    frames: frames.map((f) => f.info), // per-frame extxyz info (energy, step, ...)
    cell: frames[0].cell,
    symbols: frames[0].symbols,
    reducedMotion: reduced,
    viewer, // the raw 3Dmol viewer, for overlays (labels, arrows, shapes)
    seqPos: 0,
    ready: true,
    /** Start playback. Under prefers-reduced-motion this is a no-op unless {force:true}. */
    play({ force = false } = {}) {
      if (reduced && !force) return false;
      wanted = true;
      if (visible) start();
      return true;
    },
    pause() { wanted = false; stop(); },
    /** Show frame i (clamped). Pauses playback. Returns the index shown. */
    setFrame(i) {
      wanted = false;
      stop();
      const k = show(i);
      const p = seq.indexOf(k);
      seqStart = p < 0 ? 0 : p;
      ctrl.seqPos = seqStart;
      return k;
    },
    get frame() { return current; },
    get playing() { return playing; },
    /**
     * Change camera: {elevation, azimuth, zoom} (any subset) or a preset name.
     * Re-fits the framing unless {fit: false} (use that for per-frame spins).
     */
    setView(v, { fit: refit = true } = {}) {
      viewState = typeof v === "string" ? { ...(VIEWS[v] || VIEWS.tilted) } : { ...viewState, ...v };
      if (refit) fit({ spinFit: !!(o.spin || o.fitSpin) });
      else applyView();
    },
    getView() { return { ...viewState }; },
    /** PNG data URL of the current canvas (device pixels; alpha kept if transparent). */
    snapshot() { viewer.render(); return viewer.pngURI(); },
    resize: onResize,
    destroy() {
      stop();
      if (io) io.disconnect();
      window.removeEventListener("resize", onResize);
      el.removeEventListener("wheel", wheelGuard, { capture: true });
      el.removeEventListener("pointerdown", stopSpinOnTouch);
      viewer.clear();
      el.innerHTML = "";
    },
  };

  if (o.autoplay && !reduced) ctrl.play();
  if (o.capture && typeof window !== "undefined") window.__scene = ctrl;
  return ctrl;
}
