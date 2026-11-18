#!/usr/bin/env bash
# Stage 01: Pt bulk -> fcc check -> slab -> O at fcc and hcp hollows.
# Runs on cos-cluster only, on ONE GPU card, the four relaxations in sequence.
#
# Usage (from anywhere on the cluster):
#   bash $REMOTE/scripts/run_01_relax.sh <gpu_card>
#
# Settings follow docs/showcase/CANON.md: uma-s-1p2 / oc20 (C1), fmax 0.02 (C2),
# BFGS, cell relaxation for the bulk only (B2), slab from the relaxed a (S1).
set -euo pipefail
source /scratchb/juar/mliprun-showcase/cluster.env

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <gpu_card>" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
# Shared, scheduler-less node: keep CPU threads modest (the work is on the GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

BUILD="$REMOTE/scripts/build_structures.py"
OUT="$REMOTE/01_relax"
COMMON=(--mlip uma-s-1p2 --uma-task oc20 --device cuda --fmax 0.02
        --optimizer bfgs --max-steps 500)

relax() {   # relax <dir> [extra mlip args...]
    local dir="$1"; shift
    echo "=== $(date -Is) relax $dir (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    "$UMA_BIN/mlip" optimize run --structure "$dir/POSCAR" "${COMMON[@]}" "$@" \
        > "$dir/mlip_stdout.txt" 2>&1
    # optimize run exits 0 even when it does not converge: check explicitly.
    if ! grep -Eq '^Converged:[[:space:]]+True' "$dir/opt_params.txt"; then
        echo "STOP: $dir did not converge (see $dir/opt_params.txt)" >&2
        exit 1
    fi
    echo "=== $(date -Is) $dir converged"
}

mkdir -p "$OUT"/{bulk,slab,o_fcc,o_hcp}

# 1. Bulk: primitive fcc Pt, cell and atoms relaxed (B1, B2).
"$UMA_BIN/python" "$BUILD" bulk "$OUT/bulk/POSCAR"
relax "$OUT/bulk" --relax-cell all

# 2. The relaxed cell must still be fcc; stop otherwise (B2).
"$UMA_BIN/python" "$BUILD" check-fcc "$OUT/bulk/opt_final.vasp" \
    | tee "$OUT/bulk/fcc_check.txt"

# 3. Slab cut from the relaxed lattice constant, positions only (S1, S8).
"$UMA_BIN/python" "$BUILD" slab "$OUT/bulk/opt_final.vasp" "$OUT/slab/POSCAR"
relax "$OUT/slab"

# 4. O at the fcc and hcp hollows of the relaxed slab (A1).
for site in fcc hcp; do
    "$UMA_BIN/python" "$BUILD" adsorb "$OUT/slab/opt_final.vasp" "$site" \
        "$OUT/o_$site/POSCAR"
done
relax "$OUT/o_fcc"
relax "$OUT/o_hcp"

echo "=== $(date -Is) stage 01 done"
