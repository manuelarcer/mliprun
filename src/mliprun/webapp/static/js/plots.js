// Plotly charts per calc type (PLAN 11.1 "Job detail").
// Each plot set accumulates event data and redraws through frameThrottle, so a
// fast event stream costs at most one Plotly.react per animation frame.
// Two measures of different scale go into stacked subplots that share the x
// axis, never onto two y-scales of one plot.
import { frameThrottle, libsAvailable } from "./util.js";

const C = {
  s1: "#2a78d6", // blue
  s2: "#eb6834", // orange
  s3: "#1baf7a", // aqua
  ref: "#8a8984", // reference / target lines
  ghost: "#c9c8c2",
  imag: "#e34948", // imaginary modes
  imagSoft: "#f2a9a8",
  text: "#52514e",
  grid: "#e8e7e3",
};

const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function baseLayout(extra = {}) {
  return {
    margin: { l: 64, r: 16, t: 54, b: 46 },
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: C.text },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#fcfcfb",
    hovermode: "x unified",
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.01, yanchor: "bottom", font: { size: 11 } },
    xaxis: { gridcolor: C.grid, zeroline: false, automargin: true },
    yaxis: { gridcolor: C.grid, zeroline: false, automargin: true, exponentformat: "power" },
    ...extra,
  };
}

function line(x, y, name, color, extra = {}) {
  return { type: "scatter", mode: "lines", x, y, name, line: { color, width: 2 }, ...extra };
}

function hline(y, text, axis = "y") {
  return {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    yref: axis,
    y0: y,
    y1: y,
    line: { color: C.ref, width: 1.5, dash: "dash" },
    label: text ? { text, textposition: "end", font: { size: 10, color: C.text } } : undefined,
  };
}


/** Two stacked subplots sharing one x range (top: x/y, bottom: x2/y2). */
function stacked(L, xTitle, topTitle, bottomTitle, split = 0.42, extra = {}) {
  return {
    xaxis: { ...L.xaxis, anchor: "y", showticklabels: false },
    xaxis2: { ...L.xaxis, anchor: "y2", matches: "x", title: { text: xTitle } },
    yaxis: { ...L.yaxis, domain: [split + 0.08, 1], ...(extra.yaxis || {}), title: { text: topTitle } },
    yaxis2: { ...L.yaxis, domain: [0, split], ...(extra.yaxis2 || {}), title: { text: bottomTitle } },
  };
}

class PlotSet {
  constructor(divs, ctx) {
    this.divs = divs; // {main, secondary, extra}
    this.ctx = ctx; // {params, calcType}
    this.ok = libsAvailable().plotly;
    this.drawn = new Set();
    this.redraw = frameThrottle(() => this.draw(), 150);
  }

  ingest() {}
  setResult() {}
  usesSecondary() {
    return true;
  }
  usesExtra() {
    return false;
  }

  react(div, data, layout) {
    if (!this.ok || !div) return;
    try {
      // Data arrays are mutated in place, so bump datarevision to make react redraw.
      this.rev = (this.rev || 0) + 1;
      window.Plotly.react(div, data, { ...layout, datarevision: this.rev }, CONFIG);
      this.drawn.add(div);
    } catch (err) {
      console.error("Plotly", err);
    }
  }

  destroy() {
    this.redraw.cancel();
    if (!this.ok) return;
    for (const d of this.drawn) {
      try {
        window.Plotly.purge(d);
      } catch {
        /* ignore */
      }
    }
  }
}

// --------------------------------------------------------------- optimize

class OptimizePlots extends PlotSet {
  constructor(divs, ctx) {
    super(divs, ctx);
    this.step = [];
    this.energy = [];
    this.fmax = [];
  }
  ingest(ev) {
    if (ev.type !== "opt_step") return;
    this.step.push(ev.step);
    this.energy.push(ev.energy);
    this.fmax.push(ev.fmax);
    this.redraw();
  }
  draw() {
    this.react(
      this.divs.main,
      [line(this.step, this.energy, "Energy", C.s1, { mode: "lines+markers", marker: { size: 5 } })],
      baseLayout({
        title: { text: "Energy vs step", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        showlegend: false,
        xaxis: { ...baseLayout().xaxis, title: { text: "Optimizer step" } },
        yaxis: { ...baseLayout().yaxis, title: { text: "Energy (eV)" }, tickformat: ".4~f" },
      }),
    );
    const target = this.ctx.params && this.ctx.params.fmax;
    this.react(
      this.divs.secondary,
      [line(this.step, this.fmax, "fmax", C.s2, { mode: "lines+markers", marker: { size: 5 } })],
      baseLayout({
        title: { text: "Max force vs step", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        showlegend: false,
        xaxis: { ...baseLayout().xaxis, title: { text: "Optimizer step" } },
        yaxis: { ...baseLayout().yaxis, type: "log", title: { text: "fmax (eV/Å)" } },
        shapes: target ? [hline(target, `target ${target} eV/Å`)] : [],
      }),
    );
  }
}

// --------------------------------------------------------------- md

class MdPlots extends PlotSet {
  constructor(divs, ctx) {
    super(divs, ctx);
    this.t = [];
    this.epot = [];
    this.ekin = [];
    this.etot = [];
    this.temp = [];
    this.p = [];
    this.v = [];
    this.npt = (ctx.params && ctx.params.ensemble) === "npt";
  }
  usesExtra() {
    return this.npt;
  }
  ingest(ev) {
    if (ev.type !== "md_step") return;
    this.t.push(ev.time_fs);
    this.epot.push(ev.epot);
    this.ekin.push(ev.ekin);
    this.etot.push(ev.etot);
    this.temp.push(ev.temperature);
    if (ev.pressure_GPa !== undefined || ev.volume_A3 !== undefined) this.npt = true;
    this.p.push(ev.pressure_GPa ?? null);
    this.v.push(ev.volume_A3 ?? null);
    this.redraw();
  }
  draw() {
    const L = baseLayout();
    // Energies: Etot and Epot share a scale; Ekin in its own subplot below.
    this.react(
      this.divs.main,
      [
        line(this.t, this.etot, "E total", C.s1),
        line(this.t, this.epot, "E potential", C.s2),
        line(this.t, this.ekin, "E kinetic", C.s3, { xaxis: "x2", yaxis: "y2" }),
      ],
      baseLayout({
        title: { text: "Energies vs time", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        ...stacked(L, "Time (fs)", "Etot, Epot (eV)", "Ekin (eV)", 0.34),
        height: 380,
      }),
    );
    const target = this.ctx.params && this.ctx.params.ensemble !== "nve" ? this.ctx.params.temperature : null;
    this.react(
      this.divs.secondary,
      [line(this.t, this.temp, "Temperature", C.s2)],
      baseLayout({
        title: { text: "Temperature vs time", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        showlegend: false,
        xaxis: { ...L.xaxis, title: { text: "Time (fs)" } },
        yaxis: { ...L.yaxis, title: { text: "T (K)" } },
        shapes: target ? [hline(target, `target ${target} K`)] : [],
      }),
    );
    if (this.npt) {
      this.divs.extra.classList.remove("hidden");
      this.react(
        this.divs.extra,
        [line(this.t, this.p, "Pressure", C.s1), line(this.t, this.v, "Volume", C.s3, { xaxis: "x2", yaxis: "y2" })],
        baseLayout({
          title: { text: "Pressure and volume vs time", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
          ...stacked(L, "Time (fs)", "P (GPa)", "V (Å³)", 0.44),
          height: 360,
        }),
      );
    }
  }
}

// --------------------------------------------------------------- neb / autoneb

class NebPlots extends PlotSet {
  constructor(divs, ctx) {
    super(divs, ctx);
    this.iter = [];
    this.fmax = [];
    this.barrier = [];
    this.first = null;
    this.current = null;
    this.final = null;
  }
  ingest(ev) {
    if (ev.type !== "neb_iter") return;
    const prof = { rel: ev.rel_energies, highest: ev.highest_image, iteration: ev.iteration, sub: ev.autoneb };
    if (!this.first) this.first = prof;
    this.current = prof;
    this.iter.push(ev.iteration);
    this.fmax.push(ev.fmax);
    this.barrier.push(ev.barrier);
    this.redraw();
  }
  setResult(r) {
    if (r && Array.isArray(r.relative_energies_eV)) {
      this.final = { rel: r.relative_energies_eV, highest: r.ts_image_index };
      this.redraw();
    }
  }
  draw() {
    const L = baseLayout();
    const traces = [];
    const idx = (arr) => arr.map((_, i) => i);
    if (this.first && this.first !== this.current && !this.ctx.autoneb) {
      traces.push(
        line(idx(this.first.rel), this.first.rel, `iteration ${this.first.iteration}`, C.ghost, {
          mode: "lines+markers",
          line: { color: C.ghost, width: 2, dash: "dot" },
          marker: { size: 6 },
        }),
      );
    }
    const prof = this.final || this.current;
    if (prof) {
      const name = this.final ? "final band" : `iteration ${prof.iteration}${prof.sub ? ` (sub-step ${prof.sub.sub_step})` : ""}`;
      traces.push(
        line(idx(prof.rel), prof.rel, name, C.s1, {
          mode: "lines+markers",
          marker: { size: 8, line: { color: "#fcfcfb", width: 2 } },
          connectgaps: true,
        }),
      );
      const h = prof.highest;
      if (h !== null && h !== undefined && prof.rel[h] !== null && prof.rel[h] !== undefined) {
        traces.push({
          type: "scatter",
          mode: "markers+text",
          x: [h],
          y: [prof.rel[h]],
          name: "highest image",
          marker: { size: 13, color: C.s2, symbol: "diamond", line: { color: "#fcfcfb", width: 2 } },
          text: [`${prof.rel[h].toFixed(3)} eV`],
          textposition: "top center",
          textfont: { color: C.text },
          hoverinfo: "skip",
        });
      }
    }
    this.react(
      this.divs.main,
      traces,
      baseLayout({
        title: { text: "Energy profile along the band", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        hovermode: "closest",
        xaxis: { ...L.xaxis, title: { text: "Image index (0 = initial)" }, dtick: 1 },
        yaxis: { ...L.yaxis, title: { text: "E − E(initial) (eV)" } },
      }),
    );
    const target = this.ctx.params && this.ctx.params.fmax;
    this.react(
      this.divs.secondary,
      [
        line(this.iter, this.fmax, "fmax", C.s2),
        line(this.iter, this.barrier, "barrier", C.s1, { xaxis: "x2", yaxis: "y2" }),
      ],
      baseLayout({
        title: { text: "Convergence vs iteration", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        ...stacked(L, this.ctx.autoneb ? "Optimizer call (all sub-bands)" : "NEB iteration", "fmax (eV/Å)", "Barrier (eV)", 0.36, {
          yaxis: { type: "log" },
        }),
        shapes: target ? [hline(target, `target ${target} eV/Å`)] : [],
        height: 380,
      }),
    );
  }
}

// --------------------------------------------------------------- freq

class FreqPlots extends PlotSet {
  usesSecondary() {
    return false;
  }
  setResult(r) {
    this.result = r;
    this.redraw();
  }
  draw() {
    const r = this.result;
    if (!r || !Array.isArray(r.modes)) return;
    const floor = r.imaginary_floor_cm1 ?? 0;
    const real = r.modes.filter((m) => !m.imaginary);
    const imag = r.modes.filter((m) => m.imaginary && Math.abs(m.frequency_cm1) >= floor);
    const soft = r.modes.filter((m) => m.imaginary && Math.abs(m.frequency_cm1) < floor);
    const bar = (ms, name, color) => ({
      type: "bar",
      x: ms.map((m) => m.index),
      y: ms.map((m) => m.frequency_cm1),
      name,
      marker: { color },
      customdata: ms.map((m) => [m.energy_meV, m.imaginary ? `${Math.abs(m.frequency_cm1).toFixed(2)}i` : m.frequency_cm1.toFixed(2)]),
      hovertemplate: "mode %{x}<br>%{customdata[1]} cm⁻¹<br>%{customdata[0]:.3f} meV<extra>" + name + "</extra>",
    });
    const traces = [bar(real, "real", C.s1)];
    if (imag.length) traces.push(bar(imag, "imaginary", C.imag));
    if (soft.length) traces.push(bar(soft, `imaginary, |ν| < ${floor} cm⁻¹ (not counted)`, C.imagSoft));
    const L = baseLayout();
    this.react(
      this.divs.main,
      traces,
      baseLayout({
        title: { text: "Vibrational frequencies (imaginary shown as negative)", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        hovermode: "closest",
        bargap: 0.35,
        xaxis: { ...L.xaxis, title: { text: "Mode index" } },
        yaxis: { ...L.yaxis, title: { text: "Frequency (cm⁻¹)" }, zeroline: true, zerolinecolor: C.ref },
      }),
    );
  }
}

// --------------------------------------------------------------- singlepoint

class SinglepointPlots extends PlotSet {
  usesSecondary() {
    return false;
  }
  setResult(r) {
    this.result = r;
    this.redraw();
  }
  draw() {
    const r = this.result;
    if (!r || !Array.isArray(r.forces)) return;
    const L = baseLayout();
    const free = r.forces.filter((f) => f.free);
    const fixed = r.forces.filter((f) => !f.free);
    const bar = (fs, name, color) => ({
      type: "bar",
      x: fs.map((f) => f.index),
      y: fs.map((f) => f.f_norm),
      name,
      marker: { color },
      customdata: fs.map((f) => f.symbol),
      hovertemplate: "atom %{x} (%{customdata})<br>|F| = %{y:.4f} eV/Å<extra></extra>",
    });
    const traces = [bar(free, "free atoms", C.s1)];
    if (fixed.length) traces.push(bar(fixed, "fixed atoms", C.ghost));
    this.react(
      this.divs.main,
      traces,
      baseLayout({
        title: { text: "Force magnitude per atom", font: { size: 13 }, x: 0, xref: "paper", yref: "container", y: 0.985, yanchor: "top" },
        hovermode: "closest",
        showlegend: fixed.length > 0,
        bargap: 0.25,
        xaxis: { ...L.xaxis, title: { text: "Atom index" } },
        yaxis: { ...L.yaxis, title: { text: "|F| (eV/Å)" } },
      }),
    );
  }
}

const CLASSES = {
  optimize: OptimizePlots,
  md: MdPlots,
  neb: NebPlots,
  autoneb: NebPlots,
  freq: FreqPlots,
  singlepoint: SinglepointPlots,
};

/** Create the plot set for a calc type. divs = {main, secondary, extra}. */
export function createPlots(calcType, divs, ctx) {
  const Cls = CLASSES[calcType] || PlotSet;
  return new Cls(divs, { ...ctx, calcType, autoneb: calcType === "autoneb" });
}
