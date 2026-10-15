# PROGRESS (append-only)

One entry per subagent run.

- 2026-10-15 orchestrator: created branch feature/webapp and webapp/docs/.
- 2026-10-15 orchestrator: user delegated stack choice to orchestrator and approved new deps (keep minimal). Tentative: FastAPI+uvicorn+WebSocket, vanilla JS, 3Dmol.js + Plotly via CDN, optional extra [webapp].
- 2026-10-15 recon subagent: wrote RECON.md. No live hooks exist; plan optional observer kwarg in run_optimization, run_md, run_frequencies, CustomNEB.run_neb. EMT injectable via atoms.calc / setup_calculator override. Process-global side effects => one subprocess per job. No env had ase; creating .venv.
- 2026-10-15 orchestrator: installing .venv (mliprun[dev,neb], fastapi, uvicorn, python-multipart, httpx, playwright, torch-cpu, mace-torch). Launched design subagent for PLAN.md.
- 2026-10-15 orchestrator: .venv ready (py3.14.7, torch 2.14.1+cpu, mace 0.3.16, mlip doctor OK). Baseline suite (PATH must include .venv/bin): 1 pre-existing failure tests/test_md_mace.py::test_mace_md (stale kwarg temperature_K vs temperature), all else pass/skip. Not fixed (out of scope).
- 2026-10-15 design subagent: wrote PLAN.md (1246 lines). UI calc types: singlepoint, optimize, freq, md, neb, autoneb. Excluded: committee, batch, benchmark, md resume, neb restart, autoneb-results. Tier2 uses MACE-MP-0 medium (package hardcodes medium). Security: X-Mliprun header + Origin/Host checks.
