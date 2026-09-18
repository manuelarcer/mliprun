# MD Reference

Reference for the `md run` CLI command. Covers the three supported ensembles (NVE, NVT, NPT), every CLI flag, and the defaults used when nothing is passed.

For a top-level overview of the CLI, see the [main README](../README.md). For UMA-specific examples, see [UMA_USAGE_GUIDE.md](UMA_USAGE_GUIDE.md).

---

## Quick start

```bash
# NVT Langevin at 300 K, 1000 steps — the default
md run --structure structure.vasp

# NVE energy-conservation run
md run --structure structure.vasp --ensemble nve --steps 10000

# NVT Nose-Hoover, 50 fs time constant
md run --structure structure.vasp --ensemble nvt \
   --thermostat nose-hoover --temperature 300 --ttime 50 --steps 10000

# NPT (isotropic MTK) at zero pressure for lattice relaxation
md run --structure structure.vasp --ensemble npt \
   --temperature 300 --pressure 0.0 --steps 10000

# NPT Berendsen for fast pressure equilibration
md run --structure structure.vasp --ensemble npt --barostat berendsen \
   --temperature 300 --pressure 0.0 --taup 1000 --steps 10000
```

---

## Ensemble selection

| Use case | Ensemble | Default driver |
|----------|----------|----------------|
| Energy conservation test | `nve` | VelocityVerlet |
| Constant temperature, production MD | `nvt` | Langevin |
| Fast thermal equilibration only | `nvt` | Berendsen |
| Rigorous canonical sampling | `nvt` | Nose-Hoover |
| Lattice / volume relaxation | `npt` | NPT (isotropic MTK) |
| Fast pressure equilibration | `npt` | Berendsen |

NVT Langevin is the default when `--ensemble` is omitted.

---

## Thermostats (NVT)

| `--thermostat` | Backing class | Key parameter | Typical value | Best for |
|----------------|---------------|---------------|---------------|----------|
| `langevin` (default) | `ase.md.langevin.Langevin` | `--friction` (1/fs) | 0.01 | Production MD; good balance of accuracy and speed |
| `nose-hoover` | `ase.md.nose_hoover.NoseHoover` | `--ttime` (fs) | 25–100 | Rigorous canonical sampling |
| `berendsen` | `ase.md.nvtberendsen.NVTBerendsen` | `--taut` (fs) | 100–500 | Fast thermal equilibration only (not ergodic) |

Nose-Hoover requires a recent ASE; the platform raises `ImportError` if it is missing.

## Barostats (NPT)

| `--barostat` | Backing class | Key parameters | Best for |
|--------------|---------------|----------------|----------|
| `npt` (default) | `ase.md.npt.NPT` (Martyna-Tobias-Klein) | `--ttime` (fs); `pfactor` is auto-computed | Lattice constant optimization, isotropic expansion |
| `berendsen` | `ase.md.nptberendsen.NPTBerendsen` | `--taut`, `--taup` (fs) | Quick volume relaxation; not for production statistics |

### Restricting which axes may change (`--barostat-mask`)

By default the barostat couples to all three cell axes. `--barostat-mask`
takes three 0/1 values in Cartesian `(x, y, z)` order: `1` lets that axis
change, `0` holds its length fixed.

| Mask | Effect |
|------|--------|
| `"1,1,1"` (default) | Isotropic. Identical to the behaviour before this flag existed, and takes the same code path. |
| `"0,0,1"` | Only z may change. The intended case for a slab–liquid interface. |
| `"1,1,0"` | x and y may change **independently** of each other. This is *not* semi-isotropic coupling, which would tie them together; that is not implemented. |

The backing class changes with the mask:

- `--barostat berendsen` with a non-default mask uses
  `ase.md.nptberendsen.Inhomogeneous_NPTBerendsen`, which applies a separate
  scale factor per axis. Note it also multiplies by the periodic boundary
  flags, so an axis with `pbc=False` never scales regardless of the mask.
- `--barostat npt` passes the mask to `ase.md.npt.NPT`, which stores its
  **outer product**. `"0,0,1"` therefore frees the zz strain component alone:
  no in-plane strain and no xz/yz shear either.

The mask applies to NPT only. A non-default mask with `--ensemble nve` or
`nvt` is an error, not a silent no-op — neither ensemble scales the cell, so
accepting it would leave you believing an axis had been constrained.

**Worked case — oxide slab with liquid water, no vacuum gap.** The in-plane
lattice is fixed by the relaxed bulk oxide and must not be strained; the z
length must be free so the water reaches its own density at 1 bar
(`1e-4` GPa) instead of whatever the initial packing produced:

```bash
md run --structure interface.vasp --ensemble npt \
       --temperature 300 --pressure 0.0001 --steps 50000 \
       --barostat berendsen --barostat-mask "0,0,1"
```

Without this, the alternatives are a per-system packing calibration — which
does not transfer between facets, terminations, or MLIPs — or an unknown
density error in every interfacial energy computed from the run.

The resolved mask is echoed in the NPT setup block, written to
`md_params.txt`, and stored in the run record under
`parameters.barostat_mask`, so a masked run is distinguishable from an
isotropic one without opening the trajectory.

---

## CLI parameter reference

| Flag | Default | Units | Notes |
|------|---------|-------|-------|
| `--structure` | required | — | Path to input structure (any ASE-readable format) |
| `--ensemble` | `nvt` | — | One of `nve`, `nvt`, `npt` |
| `--steps` | `1000` | — | Number of MD steps |
| `--temperature` | `300` | K | Required for NVT and NPT |
| `--pressure` | `0.0` | GPa | NPT only; negative values = tension |
| `--timestep` | `1.0` | fs | Safe default for solids; lower (0.5) for high T or H-rich systems |
| `--thermostat` | `langevin` | — | NVT only: `langevin`, `nose-hoover`, `berendsen` |
| `--barostat` | `npt` | — | NPT only: `npt`, `berendsen` |
| `--barostat-mask` | `"1,1,1"` | — | NPT only: which axes the barostat may change, `(x,y,z)` as three 0/1 values. `"0,0,1"` relaxes z alone (slab–liquid interface). Rejected for NVE/NVT unless it is the default |
| `--friction` | `0.01` | 1/fs | Langevin friction coefficient |
| `--ttime` | `25.0` | fs | Time constant for Nose-Hoover and NPT (MTK) |
| `--taut` | `100.0` | fs | Berendsen temperature coupling time |
| `--taup` | `1000.0` | fs | Berendsen pressure coupling time (NPT Berendsen only) |
| `--mlip` | `auto` | — | MLIP model; auto-detect or explicit (`uma-s-1p2`, `mace`, `mace-mh-1`, `7net-mf-ompa`, `chgnet`, …) |
| `--uma-task` | `omat` | — | Task head for UMA models: `omat`, `oc20`, `omol`, `odac` |
| `--mace-head` | `omat_pbe` | — | Head for multi-head MACE foundation models (`mace-mh-*`): `omat_pbe`, `oc20_usemppbe`, `matpes_r2scan`, `mp_pbe_refit_add`, `omol`, `spice_wB97M`. Ignored for non-MH MACE |
| `--device` | `auto` | — | Compute device: `auto` (cuda if available, else cpu), `cuda`, or `cpu`. On multi-GPU nodes use `CUDA_VISIBLE_DEVICES` to pick the GPU |
| `--log-interval` | `10` | steps | Append a row to `md_energy.csv` (and drive the stdout MDLogger) every N steps. At dt=0.5 fs, 10 → one row every 5 fs |
| `--traj-interval` | `100` | steps | Write a frame to `md.traj` every N steps. Lower for finer dynamics; raise to shrink disk |
| `--csv-flush-every` | `100` | log calls | Flush buffered `md_energy.csv` rows to disk every N log calls. `0` disables incremental writes (flush only at end) |
| `--resume` | off | flag | Continue an existing run in the structure's directory. The last frame of `md.traj` is loaded as the starting state, momenta are preserved (no Maxwell-Boltzmann re-init), and `--steps` is interpreted as *additional* steps. New rows append to `md_energy.csv` and the trajectory; `md_params.txt` gets a `--- Resume invocation ---` block appended. Plots are regenerated over the full chain. |

**Not exposed on the CLI** but used internally with sensible defaults:

- `pfactor` for NPT MTK: auto-computed as `(ttime * 75 GPa)^2`.
- `compressibility` for NPT Berendsen: `4.57e-5 GPa⁻¹` (water value). For metals (~1e-6) or ceramics (~1e-7), the Berendsen barostat will still equilibrate but slower / faster than ideal. If you need to tune these, use the Python API (`mliprun.core.md.run_md`).

---

## Recommended defaults for solids

| Property | Default | Why |
|----------|---------|-----|
| Ensemble | NVT | Most common production setting |
| Thermostat (NVT) | Langevin | Best accuracy/speed balance |
| Barostat (NPT) | NPT (MTK, isotropic) | Rigorous; well suited to crystals |
| Temperature | 300 K | Room temperature |
| Pressure | 0.0 GPa | Ambient |
| Timestep | 1.0 fs | Safe for most solids; reduce to 0.5 fs for T > 1000 K or H-containing systems |
| Friction (Langevin) | 0.01 fs⁻¹ | Mild damping |
| `ttime` (NPT MTK) | 25 fs | Standard MTK value |

---

## Output files

Every `md run` invocation writes the following to the directory containing the input structure:

| File | Contents |
|------|----------|
| `md.traj` | Full ASE trajectory (every `--traj-interval` steps) |
| `md_energy.csv` | Step, time (fs), temperature (K), total / potential / kinetic energy (eV); plus `pressure(GPa)` and `volume(A^3)` columns for NPT |
| `md_params.txt` | Echo of every parameter the run was launched with |
| `md_energy.png` | Total / potential / kinetic energy vs time — **only with `--plot`** |
| `md_temperature.png` | Temperature vs time, with target line for NVT/NPT — **only with `--plot`** |
| `md_pressure.png` | NPT only: pressure vs time, with target line — **only with `--plot`** |
| `md_volume.png` | NPT only: volume vs time — **only with `--plot`** |

Plotting is opt-in: `md.traj`, `md_energy.csv`, and `md_params.txt` are always written; the PNGs require `--plot`. Output goes to `Path(--structure).parent`, not the current working directory.

---

## Worked examples

### Equilibration with Berendsen (NVT)

```bash
md run --structure structure.vasp \
   --ensemble nvt --thermostat berendsen \
   --temperature 300 --steps 5000 --timestep 1.0
```

Use only for the equilibration phase; switch to Langevin or Nose-Hoover for any data you intend to analyze.

### Production Langevin NVT

```bash
md run --structure structure.vasp \
   --ensemble nvt --thermostat langevin \
   --temperature 300 --friction 0.01 \
   --steps 50000 --timestep 1.0
```

### Lattice constant via NPT

```bash
md run --structure structure.vasp \
   --ensemble npt --temperature 300 --pressure 0.0 \
   --steps 10000 --timestep 1.0
```

After the run, average `volume(A^3)` from `md_energy.csv` over the equilibrated portion to extract the lattice constant.

### High-pressure simulation

```bash
md run --structure structure.vasp \
   --ensemble npt --temperature 1000 --pressure 50.0 \
   --steps 20000 --timestep 0.5
```

A shorter timestep (0.5 fs) is appropriate at high T and high P.

### Energy conservation test

```bash
md run --structure structure.vasp \
   --ensemble nve --steps 10000 --timestep 1.0
```

Inspect the total energy over time (plot `md_energy.csv`, or run with `--plot` to get `md_energy.png`): the total-energy curve should be flat. Drift is a sign that the timestep is too large or the calculator is stochastic.

---

## Notes

### Pressure units

- CLI: GPa
- ASE internal: eV/Å³
- Conversion: `1 GPa = 0.006241509 eV/Å³` (constant `GPA_TO_EV_PER_ANG3` in `core/utils.py`)

### MTK pfactor formula

```python
pfactor = (ttime * 75 * units.GPa) ** 2
```

For `ttime = 25 fs` this gives `pfactor ≈ 140 (eV/Å³)²·fs²`. The factor 75 GPa is a representative bulk modulus. For very stiff (>>200 GPa) or very soft (<10 GPa) systems, the time to reach pressure equilibrium may be unacceptably long; in that case, drive `pfactor` directly via the Python API.

### Initial velocities

Velocities are initialized from a Maxwell-Boltzmann distribution at `--temperature` whenever `--ensemble` is `nvt` or `npt` and `--temperature > 0`. NVE keeps whatever velocities are already on the input structure (or zero if none).
