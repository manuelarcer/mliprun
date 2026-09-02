# MLIP install recipes

Each supported MLIP needs **its own Python environment**. The packages pin mutually incompatible `torch` / `torch_geometric` / `e3nn` versions; installing two into one env will silently break at least one of them. See [ADR 0001](../adr/0001-per-mlip-envs.md).

Pick the MLIP that fits your problem, then follow its recipe to create a dedicated env.

| MLIP tag(s) | Best for | Recipe |
| --- | --- | --- |
| `uma-s-1p2`, any `uma-*` | Bulk inorganic, catalysis, molecules, ODAC — covered by per-task heads (`omat`, `oc20`, `omol`, `odac`). Strongest general-purpose foundation model in the set; requires HuggingFace auth. | [`uma.md`](./uma.md) |
| `mace`, `mace-mh-0`, `mace-mh-1` | MACE-MP-0 (medium) for fast inorganic baselines; multi-head foundation checkpoints (`mace-mh-*`) for cross-domain transfer (PBE bulk, r2SCAN, OC20, organics). | [`mace.md`](./mace.md) |
| `7net-omni` (recommended), `7net-omni-i8`, `7net-omni-i12`, `7net-mf-ompa`, `7net-mf-0`, `7net-omat`, `7net-l3i5`, `7net-0` | SevenNet. No auth. `7net-omni` is a 13-task multi-fidelity model and the only SevenNet checkpoint with surface heads (`oc20` RPBE, `oc22`), so it is the one to use for catalysis. Multi-task tags require `--sevennet-task`; there is no default. | [`sevenn.md`](./sevenn.md) |
| `chgnet` | Lightest install, fastest cold start, weights bundled. Good for quick scans on crystal structures (MP-trained). | [`chgnet.md`](./chgnet.md) |

## Recommended workflow

1. Pick one MLIP from the table.
2. Open its recipe and follow the conda or venv block (conda preferred on HPC; venv preferred for local dev).
3. Activate that env whenever you want to use that MLIP. Switch envs to switch MLIPs.

If you need to compare results across MLIPs, keep one env per MLIP and run each `mliprun` command after activating the relevant env.
