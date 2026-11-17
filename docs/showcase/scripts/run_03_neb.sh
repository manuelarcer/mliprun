#!/usr/bin/env bash
# Stage 03: O hopping fcc -> hcp on Pt(111), CI-NEB with TS verification,
# then the band explosion check. Runs on cos-cluster only, on ONE GPU card.
#
# Usage (run.log lands in the NEB folder):
#   mkdir -p $REMOTE/03_neb && cd $REMOTE/03_neb
#   setsid nohup bash $REMOTE/scripts/run_03_neb.sh <gpu_card> > run.log 2>&1 < /dev/null &
#
# Settings follow docs/showcase/CANON.md: uma-s-1p2 / oc20 (C1), fmax 0.02 for
# band and endpoints (C2), BFGS endpoints (manager's call), 7 images, k 0.1,
# climbing image (C6), ts verify both (C8), explosion allowance 1.0 + 1.0 x
# travel (C7, band_check.py).
set -euo pipefail
source /scratchb/juar/mliprun-showcase/cluster.env

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <gpu_card>" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
# Shared, scheduler-less node: keep CPU threads modest (the work is on the GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RELAX="$REMOTE/01_relax"
OUT="$REMOTE/03_neb"

# Never overwrite an earlier NEB in place.
if [[ -e "$OUT/A2B.traj" || -e "$OUT/A2B_full.traj" ]]; then
    echo "STOP: $OUT already holds NEB output; move it aside first" >&2
    exit 1
fi

# Endpoints: the converged stage-01 relaxations, copied into the run folder.
for site in fcc hcp; do
    if ! grep -Eq '^Converged:[[:space:]]+True' "$RELAX/o_$site/opt_params.txt"; then
        echo "STOP: endpoint o_$site is not converged" >&2
        exit 1
    fi
    mkdir -p "$OUT/o_$site"
    cp "$RELAX/o_$site/opt_final.vasp" "$OUT/o_$site/opt_final.vasp"
done

cd "$OUT"
echo "=== $(date -Is) neb run (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
neb_status=0
"$UMA_BIN/mlip" neb run \
    --initial o_fcc/opt_final.vasp --final o_hcp/opt_final.vasp \
    --mlip uma-s-1p2 --uma-task oc20 --device cuda \
    --num-images 7 --k 0.1 --climb --fmax 0.02 \
    --optimize-endpoints --endpoint-fmax 0.02 --endpoint-optimizer bfgs \
    --verify-ts both --plot \
    > neb_stdout.txt 2>&1 || neb_status=$?
echo "=== $(date -Is) neb run exited with status $neb_status"

if [[ ! -e A2B.traj || ! -e A2B_full.traj ]]; then
    echo "STOP: no A2B.traj / A2B_full.traj; band check impossible" >&2
    exit 1
fi

echo "=== $(date -Is) band check"
"$UMA_BIN/python" "$REMOTE/scripts/band_check.py" "$OUT"

echo "=== $(date -Is) stage 03 done (neb status $neb_status)"
exit "$neb_status"
