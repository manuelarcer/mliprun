#!/usr/bin/env bash
# Stage 04: 6-H2O H-down bilayer on the relaxed Pt(111) slab -> relaxation ->
# water check -> Langevin NVT MD. Runs on cos-cluster only, on ONE GPU card,
# the two mlip runs in sequence (one mlip job on the card at a time).
#
# Usage (from anywhere on the cluster):
#   bash $REMOTE/scripts/run_04_md.sh <gpu_card>
#
# Safe to re-run: a converged relaxation is reused, and an unfinished MD is
# continued with --resume up to 10000 steps in total.
#
# Settings follow docs/showcase/CANON.md: uma-s-1p2 / oc20 (C1), fmax 0.02 (C2),
# BFGS, 6-H2O sqrt(3)-type H-down bilayer (A2), NVT Langevin 300 K, friction
# 0.01 fs^-1, 0.5 fs, 10000 steps (C5).
set -euo pipefail
source /scratchb/juar/mliprun-showcase/cluster.env

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <gpu_card>" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
# Shared, scheduler-less node: keep CPU threads modest (the work is on the GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

SCRIPTS="$REMOTE/scripts"
SLAB="$REMOTE/01_relax/slab/opt_final.vasp"
RELAX="$REMOTE/04_md/bilayer_relax"
MD="$REMOTE/04_md/md"
TARGET_STEPS=10000
MODEL=(--mlip uma-s-1p2 --uma-task oc20 --device cuda)
MD_FLAGS=(--ensemble nvt --thermostat langevin --friction 0.01 --temperature 300
          --timestep 0.5 --traj-interval 10)

mkdir -p "$RELAX" "$MD"

converged() {   # converged <dir>
    [[ -f "$1/opt_final.vasp" ]] && grep -Eq '^Converged:[[:space:]]+True' "$1/opt_params.txt" 2>/dev/null
}

# 1. Bilayer relaxation (A2, C1, C2).
if converged "$RELAX"; then
    echo "=== $(date -Is) bilayer relaxation already converged, reused"
else
    "$UMA_BIN/python" "$SCRIPTS/build_structures.py" bilayer "$SLAB" "$RELAX/POSCAR"
    echo "=== $(date -Is) relax $RELAX (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    "$UMA_BIN/mlip" optimize run --structure "$RELAX/POSCAR" "${MODEL[@]}" \
        --fmax 0.02 --optimizer bfgs --max-steps 500 \
        > "$RELAX/mlip_stdout.txt" 2>&1
    # optimize run exits 0 even when it does not converge: check explicitly.
    if ! converged "$RELAX"; then
        echo "STOP: $RELAX did not converge (see $RELAX/opt_params.txt)" >&2
        exit 1
    fi
    echo "=== $(date -Is) $RELAX converged"
fi

# 2. Water check on the relaxed bilayer: no O-H > 1.3 A, every O within 4.0 A
#    of the top Pt layer, and the 18 frozen Pt still frozen. Stop before MD
#    otherwise.
if ! "$UMA_BIN/python" "$SCRIPTS/check_water.py" "$RELAX/opt_final.vasp" \
        --max-oh 1.3 --max-height 4.0 > "$RELAX/water_check.txt"; then
    cat "$RELAX/water_check.txt"
    echo "STOP: relaxed bilayer failed the water check" >&2
    exit 1
fi
cat "$RELAX/water_check.txt"
"$UMA_BIN/python" -c '
import sys
from ase.io import read
a = read(sys.argv[1], format="vasp")
n = sum(len(c.index) for c in a.constraints)
print(f"fixed atoms in relaxed bilayer: {n}")
sys.exit(0 if n == 18 else 1)
' "$RELAX/opt_final.vasp" || { echo "STOP: expected 18 fixed Pt" >&2; exit 1; }

# 3. MD from the relaxed bilayer (C5). md run writes into the structure's dir.
if [[ ! -f "$MD/POSCAR" ]]; then
    cp "$RELAX/opt_final.vasp" "$MD/POSCAR"
fi
done_steps=0
if [[ -s "$MD/md.traj" && -f "$MD/md_energy.csv" ]] \
        && [[ $(wc -l < "$MD/md_energy.csv") -gt 1 ]]; then
    done_steps=$(tail -n 1 "$MD/md_energy.csv" | cut -d, -f1)
    done_steps=${done_steps%%.*}
fi
if (( done_steps >= TARGET_STEPS )); then
    echo "=== $(date -Is) MD already at step $done_steps >= $TARGET_STEPS"
elif (( done_steps > 0 )); then
    left=$(( TARGET_STEPS - done_steps ))
    echo "=== $(date -Is) MD resume from step $done_steps, $left steps left"
    "$UMA_BIN/mlip" md run --structure "$MD/POSCAR" "${MODEL[@]}" "${MD_FLAGS[@]}" \
        --steps "$left" --resume >> "$MD/mlip_stdout.txt" 2>&1
else
    echo "=== $(date -Is) MD start, $TARGET_STEPS steps (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    "$UMA_BIN/mlip" md run --structure "$MD/POSCAR" "${MODEL[@]}" "${MD_FLAGS[@]}" \
        --steps "$TARGET_STEPS" > "$MD/mlip_stdout.txt" 2>&1
fi

last=$(tail -n 1 "$MD/md_energy.csv" | cut -d, -f1)
echo "=== $(date -Is) MD last step in md_energy.csv: $last"
if (( ${last%%.*} < TARGET_STEPS )); then
    echo "STOP: MD ended before step $TARGET_STEPS" >&2
    exit 1
fi
echo "=== $(date -Is) stage 04 done"
