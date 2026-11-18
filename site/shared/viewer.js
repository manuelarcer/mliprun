// Shared 3D trajectory viewer for the mliprun front-page showcase.
//
// THIS HEADER IS THE BUILDER CONTRACT. Task 9 builders must not edit
// site/shared/; everything a page needs is described here.
//
//   import { loadScene } from "../shared/viewer.js";
//   const el = document.querySelector("#scene");   // must have a CSS size
//   const scene = await loadScene(el, "../../docs/showcase/data/pack/relax.xyz", {
//     background: "#0d1117", fps: 20 });
//   playButton.onclick = () => scene.play({ force: true });   // user-pressed only
//   if (!scene.reducedMotion) scene.play();
//
// INPUT
//   Multi-frame extended XYZ (ASE `write(..., format="extxyz")`); the lattice is
//   read from `Lattice="..."` in each comment line, columns from `Properties=`.
//   Other comment keys (energy, step, ...) come back per frame in ctrl.frames[i].
//   3Dmol.js is injected from THREEDMOL_URL (exact pinned 2.4.2) unless the page
//   already loaded it. A failed 3Dmol load is not cached: the next loadScene
//   call retries.
//
// OPTIONS (opts; defaults in DEFAULTS below)
//   xyzText            string: xyz content to use instead of fetching xyzUrl
//   fps (20)           playback frames per second (time-based, not per rAF)
//   loop (true)        restart after the last frame; false = stop on the last
//   bounce (false)     play forward then backward (NEB band, vibrational mode)
//   hold (0.6)         seconds resting on the last frame (and frame 0 if bounce)
//   style              "spacefill" (default) | "ballstick" (sticks only between
//                      non-metal atoms; the metal stays large to read as a surface)
//   showCell (false)   false | true (full cell) | "base" (in-plane parallelogram)
//   replicate ([2,2])  periodic images [nx, ny]; [2,2] makes a slab read as a
//                      surface. Default images include adsorbates (true periodic
//                      images, so a 2x2 shows 4 O atoms).
//   replicateElements  null (all atoms) | e.g. ["Pt"] to repeat only the metal;
//                      then the picture is not the periodic system: say so.
//   imageTint (0)      0..1 fade of the periodic images toward the background.
//                      With background "transparent" the target is WHITE: on a
//                      dark theme pass `background` explicitly or images fade
//                      to white.
//   spin (0)           deg/s about the surface normal while playing; true = 10.
//                      Stops when the user presses on the canvas.
//   view ("tilted")    "tilted" | "top" | "side" | "hero" or {elevation,
//                      azimuth, zoom}; zoom is relative to the auto-fit.
//   fill (0.9)         auto-fit: fraction of the limiting canvas side filled
//   fitSpin (false)    fit for all azimuths (automatic when spin is set)
//   background         "transparent" (default) | any CSS colour
//   fog (true)         depth fog toward the background colour
//   ambientOcclusion   strength 0..1 (0.4) or false
//   fov (14)           field of view in degrees; orthographic (false)
//   autoplay (false)   start playing on load (ignored under reduced motion)
//   restFrame ("last") frame shown when not playing: "last" | "first" | index
//   interactive (true) mouse rotation; scrollZoom (false): the wheel scrolls
//                      the page, not the scene, unless scrollZoom is true
//   touchRotate        null (default) = auto: OFF on coarse pointers (phones,
//                      tablets), ON otherwise. When off, touches pass through
//                      the canvas so a swipe on a scene SCROLLS THE PAGE (3Dmol
//                      itself calls preventDefault on touch). Enable rotation
//                      deliberately with touchRotate:true, or at runtime with
//                      ctrl.setTouchRotate(true) (e.g. behind a "rotate" toggle).
//   restyle(viewer, model, frameIndex)
//                      page styling hook. The viewer REBUILDS the model on every
//                      frame (playback, setFrame), on every auto-fit (window
//                      resize, setView, resize()) and at load, and each rebuild
//                      re-applies the default element style. Anything styled
//                      directly on ctrl.viewer is therefore wiped. Put custom
//                      colouring (e.g. per-atom sigma) in restyle: it runs after
//                      the default style on EVERY rebuild, before rendering.
//                      model.selectedAtoms({}) gives atoms with .properties.src
//                      (index in the xyz frame) and .properties.image (periodic
//                      copy). Change it later with ctrl.setRestyle(fn) (redraws).
//                      Errors thrown inside it are logged, not raised.
//   fit (true)         pixel auto-fit of the framing. It is the slow part of a
//                      load under software WebGL (SwiftShader: 15-40 s for a
//                      scene swap). To skip it: load once with fit on, read
//                      ctrl.getCamera(), and pass that array back as `camera`
//                      on later loads with the SAME view, replicate and element
//                      aspect ratio. fit:false without camera uses 3Dmol's
//                      zoomTo framing plus view.zoom (may crop a tilted slab).
//   camera             null | 8-number array from ctrl.getCamera(): fixed
//                      framing, no auto-fit at load or on resize
//   pauseOffscreen (true) pause while the element is out of the viewport
//   capture (false)    set window.__scene = ctrl (used by capture.py)
//   onFrame(i)         called after each frame is drawn. It FIRES BEFORE the
//                      loadScene promise resolves (initial frames); do not
//                      touch the controller inside it until the await returns.
//
// CONTROLLER (resolved value)
//   play({force})  start; returns false and does nothing under
//                  prefers-reduced-motion unless force:true. Wire force ONLY to
//                  a user-pressed control (button), never to page load.
//   pause(), setFrame(i) (clamps, pauses, returns index), frame, playing,
//   nFrames, frames (per-frame info objects), cell, symbols, reducedMotion,
//   setView(v, {fit=true}), getView(), resize(),
//   getCamera()    8-number framing array (pass back as opts.camera)
//   setRestyle(fn) replace the restyle hook and redraw the current frame
//   setTouchRotate(on) switch touch rotation at runtime
//   snapshot()     PNG data URL matching the SCREEN orientation. Do not use
//                  ctrl.viewer.pngURI(): in Chromium it is vertically mirrored.
//   destroy(), viewer (raw 3Dmol viewer for labels/arrows), ready.
//
// CONTRACT NOTES
//   - The promise REJECTS on fetch failure (HTTP error), xyz parse failure or
//     3Dmol load failure. Catch it and show a static fallback.
//   - `el` needs an explicit CSS size (width and height); the canvas fills it.
//   - Call ctrl.destroy() when a scene is replaced: it stops playback, removes
//     listeners and models, and frees the scene's GPU buffers. WebGL contexts:
//     3Dmol 2.4.2 draws ALL viewers on a page through ONE shared OffscreenCanvas
//     WebGL2 context and copies each frame to the visible canvas, so the
//     browser limit of about 16 contexts does not bind for many scenes; that
//     shared context is never lost by destroy() (it would blank the other
//     scenes). Only on browsers without OffscreenCanvas does each scene own a
//     context, and destroy() then releases it with WEBGL_lose_context.
//   - Window resizes refit automatically (debounced). After a CSS-only layout
//     change (no window resize), call ctrl.resize().
//   - prefers-reduced-motion is read ONCE, at load; under it the scene rests on
//     restFrame (the final frame by default) and autoplay is ignored.
//   - Small motions are invisible at the tilted view. Example: the O height in
//     the relaxation changes by under 1 A and is hard to see; use view:"side",
//     or an overlay (label, chart cursor) driven by ctrl.frames[i] in onFrame.
//   - Auto-fit samples about 5 evenly spaced frames; an atom that leaves that
//     envelope in other frames can touch the edge (lower `fill` if so).
//   - A spinning scene is fitted for all azimuths, so it fills less of the
//     frame (about 45% for a 3x3 rhombus).
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
  replicate: [2, 2],
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
  restyle: null, // (viewer, model, frameIndex) => void; runs after every rebuild
  touchRotate: null, // null = auto (off on coarse pointers)
  fit: true, // pixel auto-fit of the framing
  camera: null, // fixed framing from ctrl.getCamera(); skips auto-fit
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
      s.onload = () => {
        if (window.$3Dmol) return resolve(window.$3Dmol);
        threeDmolPromise = null;
        s.remove();
        return reject(new Error("3Dmol missing after load"));
      };
      s.onerror = () => {
        threeDmolPromise = null; // let a later loadScene retry
        s.remove();
        reject(new Error(`could not load ${THREEDMOL_URL}`));
      };
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
  // Verified ON SCREEN (page screenshot, not the 3Dmol readback, which is
  // vertically mirrored): -90 + elevation puts the surface normal up and the
  // adsorbate above the slab; top view shows +x right, +y up.
  const th = ((-90 + elevationDeg) * Math.PI) / 180;
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

// Centre of the replicated block from the LAST frame's atoms (fixed camera target).
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
 * Full contract (options, controller, caveats) in the header of this file.
 *
 * @param {HTMLElement} el  container with an explicit CSS width and height
 * @param {string} xyzUrl   http, relative or data: URL of the .xyz file
 *                          (ignored when opts.xyzText is given)
 * @param {object} [opts]   options; see the OPTIONS list in the file header
 * @param {string} [opts.xyzText]  xyz content to use instead of fetching
 * @param {(i:number)=>void} [opts.onFrame]  fires before the promise resolves
 * @returns {Promise<{play:(o?:{force?:boolean})=>boolean, pause:()=>void,
 *   setFrame:(i:number)=>number, nFrames:number, frames:object[],
 *   setView:Function, getView:Function, snapshot:()=>string,
 *   resize:()=>void, destroy:()=>void, reducedMotion:boolean, viewer:object}>}
 *   Rejects on fetch, parse or 3Dmol load failure.
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

  // Touch: 3Dmol calls preventDefault on touchstart/touchmove, which blocks page
  // scrolling. Unless touch rotation is on, stop touch events before they reach
  // the canvas (capture phase, passive) so the browser scrolls the page.
  const coarse = typeof window !== "undefined" && !!window.matchMedia &&
    window.matchMedia("(any-pointer: coarse)").matches;
  let touchOn = o.touchRotate === null || o.touchRotate === undefined ? !coarse : !!o.touchRotate;
  const touchGuard = (e) => { if (!touchOn) e.stopPropagation(); };
  const TOUCH_EVENTS = ["touchstart", "touchmove", "touchend", "touchcancel"];
  for (const t of TOUCH_EVENTS) el.addEventListener(t, touchGuard, { capture: true, passive: true });

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
  let restyleFn = typeof o.restyle === "function" ? o.restyle : null;
  const show = (i) => {
    const k = Math.max(0, Math.min(nFrames - 1, Math.round(i)));
    viewer.removeAllModels();
    const m = viewer.addModel();
    m.addAtoms(buildAtoms(frames[k], rep, o.replicateElements, center));
    applyStyle();
    if (restyleFn) {
      try {
        restyleFn(viewer, m, k);
      } catch (err) {
        console.error("viewer.js restyle hook failed:", err);
      }
    }
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
    // About 5 evenly spaced frames (always first and last).
    const nFit = Math.min(nFrames, 5);
    const fitFrames = nFit > 1
      ? [...new Set(Array.from({ length: nFit }, (_, j) => Math.round((j * (nFrames - 1)) / (nFit - 1))))]
      : [0];
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
  const fixedCamera = Array.isArray(o.camera) && o.camera.length >= 8;
  const autoFit = !!o.fit && !fixedCamera;
  if (fixedCamera) {
    fitted = o.camera.slice(0, 8);
    applyView();
  } else if (autoFit) {
    fit({ spinFit: !!(o.spin || o.fitSpin) });
  } else {
    applyView();
  }

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

  let destroyed = false;
  const onResize = () => {
    if (destroyed) return;
    viewer.resize();
    if (autoFit) fit({ spinFit: !!(o.spin || o.fitSpin) });
    else { applyView(); show(current); }
  };
  // Window resizes fire in bursts; refit once they settle (~150 ms).
  let resizeTimer = 0;
  const onWindowResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(onResize, 150);
  };
  window.addEventListener("resize", onWindowResize);

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
      if (refit && autoFit) fit({ spinFit: !!(o.spin || o.fitSpin) });
      else applyView();
    },
    getView() { return { ...viewState }; },
    /** Framing array (before view.zoom); pass back as opts.camera to skip auto-fit. */
    getCamera() { return fitted.slice(0, 8); },
    /** Replace the restyle hook (null removes it) and redraw the current frame. */
    setRestyle(fn) {
      restyleFn = typeof fn === "function" ? fn : null;
      show(current);
    },
    /** Turn touch rotation on (touches rotate) or off (touches scroll the page). */
    setTouchRotate(on) { touchOn = !!on; },
    /** PNG data URL of the current canvas (device pixels; alpha kept if transparent). */
    snapshot() {
      // viewer.pngURI() (toDataURL on 3Dmol's bitmaprenderer canvas) returns
      // the image vertically MIRRORED relative to the screen in Chromium.
      // drawImage of the same canvas matches the screen, so read back that way.
      viewer.render();
      const src = viewer.getCanvas();
      const c = document.createElement("canvas");
      c.width = src.width;
      c.height = src.height;
      c.getContext("2d").drawImage(src, 0, 0);
      return c.toDataURL("image/png");
    },
    resize: onResize,
    destroy() {
      stop();
      if (io) io.disconnect();
      destroyed = true;
      clearTimeout(resizeTimer);
      window.removeEventListener("resize", onWindowResize);
      el.removeEventListener("wheel", wheelGuard, { capture: true });
      for (const t of TOUCH_EVENTS) el.removeEventListener(t, touchGuard, { capture: true });
      el.removeEventListener("pointerdown", stopSpinOnTouch);
      const canvas = viewer.getCanvas();
      const gl = viewer.renderer && viewer.renderer._gl;
      viewer.clear();
      // Release the WebGL context, but only one this scene owns. 3Dmol 2.4.2
      // renders every viewer on the page through ONE shared OffscreenCanvas
      // context (whenever OffscreenCanvas exists); losing that one would blank
      // every other scene. Only the fallback path (no OffscreenCanvas) gives
      // each canvas its own context, and that one is released here.
      try {
        if (gl && gl.canvas === canvas) {
          const ext = gl.getExtension("WEBGL_lose_context");
          if (ext) ext.loseContext();
        }
      } catch (e) { /* context already gone */ }
      el.innerHTML = "";
    },
  };

  if (o.autoplay && !reduced) ctrl.play();
  if (o.capture && typeof window !== "undefined") window.__scene = ctrl;
  return ctrl;
}
