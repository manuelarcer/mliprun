"""Copy the showcase data pack into site/alt-4-handbook/data/ and derive two small files.

Run from anywhere:  python site/alt-4-handbook/build_data.py

Copies (unchanged): relax.json, committee.json, neb.json, md.json, webapp.json,
relax.xyz, neb_band.xyz, ts_mode.xyz, webapp.mp4 from docs/showcase/data/pack/.

Derives:
- md_planned.json: the MD command that run_04_md.sh WILL run once the scientist
  decides how to proceed. It is parsed from the script (MODEL, MD_FLAGS,
  TARGET_STEPS), not typed. Nothing in it ran.
- committee.yaml: docs/showcase/committee.yaml, copied unchanged (the page
  replaces env paths with placeholders when it shows the recipe).
- meta.json: the mliprun commit that ran, parsed from PROVENANCE.md.
- webapp_result.png: a cropped still of the last seconds of webapp.mp4 (ffmpeg).

No calculation runs here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACK = REPO / "docs/showcase/data/pack"
OUT = HERE / "data"

COPY = [
    "relax.json", "committee.json", "neb.json", "md.json", "webapp.json",
    "relax.xyz", "neb_band.xyz", "ts_mode.xyz", "webapp.mp4",
]


def bash_array(text: str, name: str) -> list[str]:
    m = re.search(rf"^{name}=\((.*?)\)", text, re.S | re.M)
    if not m:
        raise SystemExit(f"{name}=(...) not found in run_04_md.sh")
    return m.group(1).split()


def planned_md() -> dict:
    script = REPO / "docs/showcase/scripts/run_04_md.sh"
    text = script.read_text()
    model = bash_array(text, "MODEL")
    flags = bash_array(text, "MD_FLAGS")
    m = re.search(r"^TARGET_STEPS=(\d+)", text, re.M)
    if not m:
        raise SystemExit("TARGET_STEPS not found in run_04_md.sh")
    steps = m.group(1)
    # The script calls: "$UMA_BIN/mlip" md run --structure "$MD/POSCAR" MODEL MD_FLAGS --steps N
    if '"$UMA_BIN/mlip" md run --structure "$MD/POSCAR" "${MODEL[@]}" "${MD_FLAGS[@]}"' not in text:
        raise SystemExit("md run call in run_04_md.sh changed; update build_data.py")
    cmd = ["mlip", "md", "run", "--structure", "$MD/POSCAR", *model, *flags, "--steps", steps]
    md = json.loads((PACK / "md.json").read_text())
    return {
        "status": md.get("status", "pending"),
        "ran": False,
        "command": " ".join(cmd),
        "source": "docs/showcase/scripts/run_04_md.sh",
        "structure_note": "$MD/POSCAR is the relaxed bilayer (opt_final.vasp of 04_md/bilayer_relax), copied by the script.",
    }


def meta() -> dict:
    prov = (PACK / "PROVENANCE.md").read_text()
    m = re.search(r"commit `([0-9a-f]{7})` \(`([0-9a-f]{40})`", prov)
    if not m:
        raise SystemExit("commit not found in PROVENANCE.md")
    return {"code_commit": m.group(1), "code_commit_full": m.group(2),
            "provenance": "docs/showcase/data/pack/PROVENANCE.md"}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name in COPY:
        shutil.copy2(PACK / name, OUT / name)
    shutil.copy2(REPO / "docs/showcase/committee.yaml", OUT / "committee.yaml")
    (OUT / "md_planned.json").write_text(json.dumps(planned_md(), indent=1) + "\n")
    (OUT / "meta.json").write_text(json.dumps(meta(), indent=1) + "\n")
    # Legible still of the web app result: a frame 1.5 s before the end of the real
    # recording, cropped (900x650 at x=380, y=30) to the result panel and live plots.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-1.5", "-i", str(OUT / "webapp.mp4"),
                    "-frames:v", "1", "-vf", "crop=900:650:380:30", str(OUT / "webapp_result.png")], check=True)
    print(f"wrote {len(COPY) + 3} files to {OUT}")


if __name__ == "__main__":
    main()
