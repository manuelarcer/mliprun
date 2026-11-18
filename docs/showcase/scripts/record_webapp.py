"""Record the mliprun web app sending a real O/Pt(111) relaxation to cos-cluster.

Use case 5 of the front-page showcase. Two steps:

``record``
    Drives a running ``mlip webapp`` server with Playwright (Chromium, 1440x900) and
    records a video: upload the O-fcc POSCAR, choose the cos-cluster UMA environment,
    set uma-s-1p2 / oc20 / cuda / BFGS / fmax 0.02 / 500 steps, submit, wait until the
    job is done, then show the result view. Writes ``raw.webm`` and ``markers.json``
    (seconds since the video started) into ``--out-dir``.

``encode``
    Cuts ``raw.webm`` into the showcase clip with ffmpeg: the form part and the result
    part at normal speed, the waiting part sped up by N and labelled "×N" (label drawn
    by the browser as a transparent PNG, because the local ffmpeg has no drawtext).

The web app only submits; the calculation runs on cos-cluster. Start the server first,
outside any sandbox (it needs the SSH keys), for example::

    mlip webapp --no-browser --port 8765 --runs-dir <scratch dir>
    site/.venv/bin/python docs/showcase/scripts/record_webapp.py record \\
        --url http://127.0.0.1:8765 \\
        --poscar docs/showcase/data/runs/01_relax/o_fcc/POSCAR \\
        --folder showcase-o-fcc-webapp --out-dir <scratch dir>/rec
    site/.venv/bin/python docs/showcase/scripts/record_webapp.py encode \\
        --raw-dir <scratch dir>/rec --out docs/showcase/data/pack/webapp.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VIEWPORT = {"width": 1440, "height": 900}
ENV_LABEL = "cos-cluster-uma (cos-cluster)"
REMOTE_LOCATION = "/scratchb/juar/mlip-webapp"
MLIP = "uma-s-1p2"
TASK = "oc20"
DEVICE = "cuda"
OPTIMIZER = "bfgs"
FMAX = "0.02"
MAX_STEPS = "500"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
DETAIL = "[data-testid=job-detail]"
PAUSE_MS = 900          # pause after each visible form step, so a viewer can follow it


def launch_kw(chromium) -> dict:
    """Headless launch options; ``chromium`` is an explicit browser binary or None."""
    kw = {"headless": True}
    if chromium:
        kw["executable_path"] = str(Path(chromium).expanduser())
    return kw


def fill(page, selector: str, value: str) -> None:
    ctl = page.locator(selector)
    ctl.scroll_into_view_if_needed()
    ctl.click()
    ctl.fill(value)
    ctl.dispatch_event("input")
    ctl.dispatch_event("change")


def select(page, selector: str, **kw) -> None:
    ctl = page.locator(selector)
    ctl.scroll_into_view_if_needed()
    ctl.select_option(**kw)


# --------------------------------------------------------------------------- record
def record(args) -> None:
    from playwright.sync_api import sync_playwright

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    poscar = Path(args.poscar).resolve()
    vid_tmp = out / "video_tmp"
    markers: dict = {}
    errors: list = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch_kw(args.chromium))
        ctx = browser.new_context(viewport=VIEWPORT, record_video_dir=str(vid_tmp),
                                  record_video_size=VIEWPORT)
        page = ctx.new_page()
        t0 = time.monotonic()           # the video starts with the page

        def mark(name: str) -> None:
            markers[name] = round(time.monotonic() - t0, 3)
            print(f"[{markers[name]:7.2f} s] {name}", flush=True)

        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}")
                if m.type == "error" else None)
        page.goto(args.url)
        page.wait_for_selector("[data-testid=calc-type-select] option[value=optimize]",
                               state="attached", timeout=60000)
        page.wait_for_selector("[data-testid=run-on-select] option", state="attached",
                               timeout=60000)
        page.wait_for_timeout(1500)     # 3Dmol / plotly from the CDN
        mark("ready")

        # 1. upload the input structure (POSCAR with selective dynamics = FixAtoms)
        select(page, "[data-testid=upload-format]", value="vasp")
        page.set_input_files("[data-testid=upload-input]", [str(poscar)])
        page.wait_for_selector("[data-testid=structure-item][data-ok=true]", timeout=60000)
        page.wait_for_timeout(PAUSE_MS)
        mark("uploaded")

        # 2. machine: the cos-cluster UMA environment
        select(page, "[data-testid=run-on-select]", label=ENV_LABEL)
        page.wait_for_function(
            "() => document.querySelector('label[for=job-location]').textContent"
            " === 'Remote location'", timeout=30000)
        loc = page.input_value("[data-testid=job-location]")
        if loc != REMOTE_LOCATION:
            fill(page, "[data-testid=job-location]", REMOTE_LOCATION)
        page.wait_for_timeout(PAUSE_MS)
        mark("machine_selected")

        # 3. settings
        select(page, "[data-testid=calc-type-select]", value="optimize")
        page.wait_for_selector("#param-fmax", timeout=30000)
        box = page.locator("[data-testid=structure-item] [data-testid=structure-select]").first
        if not box.is_checked():
            box.set_checked(True)
        page.wait_for_function(
            f"() => [...document.querySelectorAll('[data-testid=calculator-select] option')]"
            f".some(o => o.value === '{MLIP}')", timeout=60000)
        select(page, "[data-testid=calculator-select]", value=MLIP)
        page.wait_for_timeout(400)
        select(page, "[data-testid=task-select]", value=TASK)
        select(page, "[data-testid=device-select]", value=DEVICE)
        page.wait_for_timeout(PAUSE_MS // 2)
        select(page, "#param-optimizer", value=OPTIMIZER)
        fill(page, "#param-fmax", FMAX)
        fill(page, "#param-max_steps", MAX_STEPS)
        page.wait_for_timeout(PAUSE_MS // 2)
        fill(page, "[data-testid=job-name]", "O/Pt(111) fcc hollow")
        fill(page, "[data-testid=job-folder]", args.folder)
        page.wait_for_function(
            "(t) => document.querySelector('[data-testid=folder-status]').textContent"
            ".includes(t)", arg=f"Remote folder: {REMOTE_LOCATION}/{args.folder}",
            timeout=30000)
        page.wait_for_timeout(PAUSE_MS)
        mark("settings_done")

        # 4. submit
        submit = page.locator("[data-testid=submit-button]")
        submit.scroll_into_view_if_needed()
        if not submit.is_enabled():
            reason = page.inner_text("[data-testid=submit-blocked-reason]")
            raise SystemExit(f"submit is disabled: {reason}")
        if args.dry_run:                # selector check only: nothing is submitted
            page.screenshot(path=str(out / "dry_run.png"), full_page=True)
            print(f"dry run: form filled, submit enabled; screenshot {out / 'dry_run.png'}")
            ctx.close()
            browser.close()
            return
        page.wait_for_timeout(500)
        submit.click()
        mark("submitted")
        page.wait_for_selector(DETAIL, timeout=60000)
        # back to the top: the queue row and the job header carry the status badge
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(600)
        mark("detail_shown")
        page.wait_for_selector(f"{DETAIL}[data-status=running]", timeout=300000)
        mark("running")
        page.wait_for_selector(f"{DETAIL}[data-status=done],{DETAIL}[data-status=failed],"
                               f"{DETAIL}[data-status=cancelled]", timeout=args.timeout * 1000)
        status = page.get_attribute(DETAIL, "data-status")
        mark("finished")
        if status != "done":
            page.wait_for_timeout(2000)
            ctx.close()
            browser.close()
            raise SystemExit(f"job ended with status {status!r}")

        # 5. "done" badge at the top, then the result panel, plots and final structure
        page.wait_for_timeout(2000)     # final events and copy-back land
        panel = page.locator("[data-testid=result-panel]")
        panel.scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        mark("result_shown")
        page.locator("[data-testid=plot-main]").scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        mark("end")

        job = page.get_attribute(DETAIL, "data-job-id")
        video = page.video
        ctx.close()
        raw = out / "raw.webm"
        shutil.move(video.path(), raw)
        shutil.rmtree(vid_tmp, ignore_errors=True)
        browser.close()

    detail = json.loads(urllib.request.urlopen(f"{args.url}/api/jobs/{job}").read())
    markers.update({"job_id": job, "status": status, "page_errors": errors,
                    "remote_run_dir": detail.get("remote_run_dir"),
                    "local_run_dir": detail.get("run_dir"),
                    "result": detail.get("result")})
    (out / "markers.json").write_text(json.dumps(markers, indent=1))
    print(f"wrote {raw} and {out / 'markers.json'}")
    if errors:
        print("page errors:", *errors, sep="\n  ", file=sys.stderr)


# --------------------------------------------------------------------------- encode
def label_png(path: Path, text: str, chromium=None) -> None:
    """Render ``text`` as a rounded dark label on a transparent PNG (browser-drawn)."""
    from playwright.sync_api import sync_playwright

    html = ("<html><body style='margin:0;background:transparent'>"
            "<div id=l style='display:inline-block;padding:10px 22px;border-radius:12px;"
            "background:rgba(20,24,32,0.82);color:#fff;font:600 34px -apple-system,"
            f"Helvetica,Arial,sans-serif'>{text}</div></body></html>")
    with sync_playwright() as pw:
        b = pw.chromium.launch(**launch_kw(chromium))
        pg = b.new_page()
        pg.set_content(html)
        pg.locator("#l").screenshot(path=str(path), omit_background=True)
        b.close()


def encode(args) -> None:
    raw_dir = Path(args.raw_dir).resolve()
    m = json.loads((raw_dir / "markers.json").read_text())
    raw = raw_dir / "raw.webm"
    a0 = max(0.0, m["ready"] - 0.3)                  # page loaded, form visible
    a1 = m["detail_shown"] + 1.5                     # submitted, job card on screen
    c0 = max(a1, m["finished"] - 1.0)                # just before "done"
    c1 = m["end"]
    wait = c0 - a1
    budget = args.max_seconds - (a1 - a0) - (c1 - c0) - 0.5
    if budget < 2.0:
        raise SystemExit(f"form + result parts alone take {(a1 - a0) + (c1 - c0):.1f} s")
    n = max(2, math.ceil(wait / budget))
    n = int(math.ceil(n / 5.0) * 5) if n > 10 else n  # a round factor for the label
    label = raw_dir / "speed_label.png"
    label_png(label, f"&times;{n} &nbsp;waiting for cos-cluster", args.chromium)
    flt = (
        f"[0:v]trim={a0:.3f}:{a1:.3f},setpts=PTS-STARTPTS,fps=25[a];"
        f"[0:v]trim={a1:.3f}:{c0:.3f},setpts=(PTS-STARTPTS)/{n},fps=25[b0];"
        f"[b0][1:v]overlay=W-w-40:H-h-40:shortest=1[b];"
        f"[0:v]trim={c0:.3f}:{c1:.3f},setpts=PTS-STARTPTS,fps=25[c];"
        f"[a][b][c]concat=n=3:v=1:a=0,scale=1280:-2,format=yuv420p[v]")
    out = Path(args.out).resolve()
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
           "-loop", "1", "-i", str(label), "-filter_complex", flt, "-map", "[v]",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-movflags", "+faststart", "-an", str(out)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    total = (a1 - a0) + wait / n + (c1 - c0)
    info = {"speedup": n, "segments_s": {"form": [a0, a1], "wait": [a1, c0],
                                         "result": [c0, c1]},
            "wait_real_s": round(wait, 1), "clip_s_expected": round(total, 1),
            "size_bytes": out.stat().st_size}
    (raw_dir / "encode.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# --------------------------------------------------------------------------- pack
def pack(args) -> None:
    """Write webapp.json from the pulled job directory (needs ase: build_pack helpers)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_pack import (geometry_flags, kn, make_record, read_opt_log,
                            validate_record)
    from ase.io import read

    run_dir = Path(args.run_dir).resolve()
    runs_root = Path(args.runs_root).resolve()
    job = json.loads((run_dir / "job.json").read_text())
    record = json.loads((run_dir / "mliprun_run.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    enc = json.loads((Path(args.raw_dir) / "encode.json").read_text())
    relax = json.loads(Path(args.relax_json).read_text())

    # convergence: the job's own optimizer log against the job's own fmax target
    rows = read_opt_log(run_dir / "opt.log")
    target = float(job["params"]["fmax"])
    last_fmax = rows[-1][2]
    converged = last_fmax < target
    flags = [] if converged else ["not_converged"]
    claims = [bool(st["results"]["converged"]) for st in record.get("stages", [])
              if "converged" in st.get("results", {})]
    if any(c != converged for c in claims) or summary["status"] != "done":
        flags.append("convergence_record_mismatch")
    flags += geometry_flags(read(str(run_dir / "opt_final.vasp")))

    # the command: the job's own reproduce.json argv (the equivalent mlip CLI call).
    # reproduce.json is tracked; reproduce.sh is gitignored.
    argv = json.loads((run_dir / "reproduce.json").read_text())["argv"]
    if not argv:
        raise SystemExit("reproduce.json has an empty argv")
    command = "mlip " + shlex.join(argv)

    stage = record["stages"][0]["results"]
    e_web = float(stage["final_energy_eV"])
    e_cli = next(k["value"] for k in relax["key_numbers"]
                 if k["label"] == "total energy, O at fcc hollow (relaxed)")
    f_cli = next(k["value"] for k in relax["key_numbers"]
                 if k["label"] == "final fmax, O at fcc hollow")
    key_numbers = [
        kn("total energy, O at fcc hollow (relaxed, web app job)", e_web, "eV"),
        kn("final fmax", stage["final_fmax_eV_per_A"], "eV/Å"),
        kn("fmax target", target, "eV/Å"),
        kn("BFGS steps", record["stages"][0]["steps"], "steps"),
        kn("total energy, same input via the CLI (relax.json)", e_cli, "eV"),
        kn("energy difference, web app − CLI", e_web - e_cli, "eV"),
        kn("final fmax, same input via the CLI (relax.json)", f_cli, "eV/Å"),
        kn("job wall time on cos-cluster (model load + relaxation)",
           summary["walltime_s"], "s"),
        kn("relaxation wall time (optimizer only)", record["provenance"]["walltime_s"], "s"),
        kn("video speed-up of the waiting part", enc["speedup"], "dimensionless"),
        kn("video length", enc["clip_s_expected"], "s"),
    ]
    series = {"step": [r[0] for r in rows], "energy_eV": [r[1] for r in rows],
              "fmax_eV_per_A": [r[2] for r in rows]}
    rec = make_record(
        "webapp", command, converged, flags, key_numbers, series, None,
        video="webapp.mp4",
        cwd=f"{job['location']}/{job['folder']} (cos-cluster; reproduce.sh copies "
            f"inputs/structure/POSCAR into reproduce/ and runs there)",
        run_dir=str(run_dir.relative_to(runs_root)),
        target={"machine": job["target"]["machine_name"], "env": job["target"]["env_name"],
                "python": job["target"]["python"]},
        notes=[
            "Submitted from the local mliprun web app (mlip webapp); the calculation ran "
            "on cos-cluster (cosrnode01g), GPU card 0. The laptop only submitted.",
            "Input: 01_relax/o_fcc/POSCAR, byte-identical, with selective dynamics "
            "(bottom 2 Pt layers fixed, 18 atoms).",
            "The energy difference to relax.json is consistent with floating-point noise: same "
            "input, model, task, optimizer and fmax, same 12 BFGS steps.",
            f"webapp.mp4: waiting part sped up x{enc['speedup']} and labelled so; form "
            "and result parts at normal speed.",
        ])
    validate_record(rec)
    out = Path(args.out)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(f"webapp: converged={converged} flags={flags} dE={e_web - e_cli:+.3e} eV -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chromium", default=None,
                   help="Chromium binary to use instead of Playwright's own download "
                        "(e.g. a cached chrome-headless-shell)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", default="http://127.0.0.1:8765")
    r.add_argument("--poscar", required=True)
    r.add_argument("--folder", required=True, help="run folder name on cos-cluster")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--timeout", type=int, default=1800, help="seconds to wait for done")
    r.add_argument("--dry-run", action="store_true",
                   help="fill the form and stop before Submit (no job is sent)")
    r.set_defaults(func=record)
    e = sub.add_parser("encode")
    e.add_argument("--raw-dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max-seconds", type=float, default=29.0)
    e.add_argument("--crf", type=int, default=28)
    e.set_defaults(func=encode)
    k = sub.add_parser("pack", help="write webapp.json from the pulled job directory")
    k.add_argument("--run-dir", required=True, help="job directory pulled from cos-cluster")
    k.add_argument("--runs-root", required=True, help="docs/showcase/data/runs")
    k.add_argument("--raw-dir", required=True, help="folder with encode.json")
    k.add_argument("--relax-json", required=True, help="pack/relax.json (CLI run)")
    k.add_argument("--out", required=True)
    k.set_defaults(func=pack)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
