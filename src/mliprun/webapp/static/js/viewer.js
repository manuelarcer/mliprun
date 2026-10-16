// 3Dmol.js wrapper: load xyz, update positions, draw the unit cell, play a band.
import { el, clear, toXyz, latticeFromXyz, libsAvailable } from "./util.js";

const BASE_STYLE = { sphere: { scale: 0.32, colorscheme: "Jmol" }, stick: { radius: 0.12, colorscheme: "Jmol" } };
const CELL_COLOR = "#5b6b7f";

function cellEdges(cell) {
  const [a, b, c] = cell;
  const add = (...vs) => vs.reduce((acc, v) => [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]], [0, 0, 0]);
  const o = [0, 0, 0];
  const corners = {
    o,
    a,
    b,
    c,
    ab: add(a, b),
    ac: add(a, c),
    bc: add(b, c),
    abc: add(a, b, c),
  };
  const pairs = [
    ["o", "a"], ["o", "b"], ["o", "c"],
    ["a", "ab"], ["a", "ac"], ["b", "ab"], ["b", "bc"],
    ["c", "ac"], ["c", "bc"], ["ab", "abc"], ["ac", "abc"], ["bc", "abc"],
  ];
  return pairs.map(([p, q]) => [corners[p], corners[q]]);
}

function isZeroCell(cell) {
  return !cell || cell.flat().every((x) => Math.abs(x) < 1e-12);
}

/** Map t in [0,1] to a blue -> yellow -> red colour (for per-atom scalars). */
export function heatColor(t) {
  const x = Math.max(0, Math.min(1, t));
  const stops = [
    [0.0, [49, 104, 196]],
    [0.5, [240, 200, 60]],
    [1.0, [205, 40, 40]],
  ];
  let i = 0;
  while (i < stops.length - 2 && x > stops[i + 1][0]) i++;
  const [t0, c0] = stops[i];
  const [t1, c1] = stops[i + 1];
  const f = (x - t0) / (t1 - t0 || 1);
  const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export class StructureViewer {
  /**
   * @param {HTMLElement} host  element the viewer is created in (gets the canvas)
   */
  constructor(host, { height = 320, testid = null } = {}) {
    this.host = host;
    this.box = el("div", { class: "viewer-box", style: { height: `${height}px` }, testid });
    host.append(this.box);
    this.viewer = null;
    this.cellKey = null;
    this.hasModel = false;
    this.colorValues = null;
    this.colorRange = null;
    this.frames = null;
    this.frameIndex = 0;
    this.playTimer = null;
    if (libsAvailable().mol) {
      try {
        this.viewer = window.$3Dmol.createViewer(this.box, { backgroundColor: "white", antialias: true });
      } catch (err) {
        console.error("3Dmol init failed", err);
      }
    }
    if (!this.viewer) {
      this.box.append(el("div", { class: "viewer-missing" }, "3D viewer unavailable (3Dmol.js did not load)."));
    }
    this._onResize = () => this.viewer && this.viewer.resize();
    window.addEventListener("resize", this._onResize);
  }

  destroy() {
    this.stop();
    window.removeEventListener("resize", this._onResize);
    if (this.viewer) {
      try {
        this.viewer.clear();
      } catch {
        /* ignore */
      }
    }
    this.viewer = null;
    clear(this.host);
  }

  _style() {
    if (!this.viewer) return;
    if (this.colorValues) {
      const vals = this.colorValues;
      const [lo, hi] = this.colorRange;
      const colorfunc = (atom) => heatColor(hi > lo ? (vals[atom.index] - lo) / (hi - lo) : 0);
      this.viewer.setStyle({}, { sphere: { scale: 0.34, colorfunc }, stick: { radius: 0.1, color: "#bbbbbb" } });
    } else {
      this.viewer.setStyle({}, BASE_STYLE);
    }
  }

  _drawCell(cell) {
    const key = cell ? JSON.stringify(cell) : "none";
    if (key === this.cellKey) return;
    this.cellKey = key;
    this.viewer.removeAllShapes();
    if (isZeroCell(cell)) return;
    for (const [p, q] of cellEdges(cell)) {
      this.viewer.addLine({
        start: { x: p[0], y: p[1], z: p[2] },
        end: { x: q[0], y: q[1], z: q[2] },
        color: CELL_COLOR,
        linewidth: 1.5,
      });
    }
  }

  /** Show an xyz block. `zoom` re-centres the camera (first load only by default). */
  showXyz(xyz, { cell = undefined, zoom = undefined } = {}) {
    if (!this.viewer || !xyz) return;
    const doZoom = zoom === undefined ? !this.hasModel : zoom;
    this.viewer.removeAllModels();
    this.viewer.addModel(xyz, "xyz");
    this.hasModel = true;
    this._style();
    this._drawCell(cell === undefined ? latticeFromXyz(xyz) : cell);
    if (doZoom) this.viewer.zoomTo();
    this.viewer.render();
  }

  /** Show one Frame payload (PLAN 6.2) with the job's constant symbols. */
  showFrame(symbols, frame, pbc, opts = {}) {
    if (!frame || !symbols) return;
    this.showXyz(toXyz(symbols, frame.positions, frame.cell, pbc), { cell: frame.cell, ...opts });
  }

  /** Colour atoms by a per-atom scalar (e.g. |F|); null restores element colours. */
  colorBy(values) {
    if (!values) {
      this.colorValues = null;
    } else {
      this.colorValues = values;
      const finite = values.filter((v) => Number.isFinite(v));
      this.colorRange = [Math.min(...finite, 0), Math.max(...finite, 0)];
    }
    if (this.viewer && this.hasModel) {
      this._style();
      this.viewer.render();
    }
  }

  /** Band / trajectory playback. frames = list of xyz strings. */
  setFrames(xyzFrames) {
    this.frames = xyzFrames && xyzFrames.length ? xyzFrames : null;
    this.frameIndex = Math.min(this.frameIndex, this.frames ? this.frames.length - 1 : 0);
    if (this.frames) this.showXyz(this.frames[this.frameIndex]);
  }

  goto(i) {
    if (!this.frames) return;
    this.frameIndex = Math.max(0, Math.min(this.frames.length - 1, i));
    this.showXyz(this.frames[this.frameIndex]);
  }

  play(onTick, intervalMs = 350) {
    if (!this.frames || this.frames.length < 2) return;
    this.stop();
    this.playTimer = setInterval(() => {
      this.goto((this.frameIndex + 1) % this.frames.length);
      if (onTick) onTick(this.frameIndex);
    }, intervalMs);
  }

  stop() {
    if (this.playTimer) clearInterval(this.playTimer);
    this.playTimer = null;
  }

  get playing() {
    return this.playTimer !== null;
  }
}

/**
 * Viewer plus a frame slider and play button for multi-frame data.
 * Returns {viewer, setFrames(xyzList, labelFn), root}.
 */
export function viewerWithSlider(host, { height = 320, testid = null, label = "Frame" } = {}) {
  const viewerHost = el("div", { class: "viewer-host" });
  const slider = el("input", { type: "range", min: "0", max: "0", value: "0", class: "frame-slider", "aria-label": label });
  const caption = el("span", { class: "frame-caption" });
  const playBtn = el("button", { type: "button", class: "btn btn-small" }, "Play");
  const controls = el("div", { class: "frame-controls hidden" }, playBtn, slider, caption);
  host.append(viewerHost, controls);
  const viewer = new StructureViewer(viewerHost, { height, testid });
  let labelFn = (i, n) => `${label} ${i + 1} / ${n}`;
  const update = () => {
    const n = viewer.frames ? viewer.frames.length : 0;
    slider.value = String(viewer.frameIndex);
    caption.textContent = n ? labelFn(viewer.frameIndex, n) : "";
  };
  slider.addEventListener("input", () => {
    viewer.stop();
    playBtn.textContent = "Play";
    viewer.goto(parseInt(slider.value, 10));
    update();
  });
  playBtn.addEventListener("click", () => {
    if (viewer.playing) {
      viewer.stop();
      playBtn.textContent = "Play";
    } else {
      viewer.play(update);
      playBtn.textContent = "Pause";
    }
  });
  return {
    viewer,
    root: host,
    setFrames(xyzList, fn) {
      if (fn) labelFn = fn;
      viewer.setFrames(xyzList);
      const n = xyzList ? xyzList.length : 0;
      slider.max = String(Math.max(0, n - 1));
      controls.classList.toggle("hidden", n < 2);
      update();
    },
    refreshCaption: update,
  };
}
