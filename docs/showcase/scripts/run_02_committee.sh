#!/usr/bin/env bash
# Stage 02: committee relaxation of O at the Pt(111) fcc hollow.
# Three same-level (OC20/RPBE) MLIPs relax the structure together along their
# mean force; their disagreement is the uncertainty. Runs on cos-cluster only,
# on ONE GPU card shared by all three members (one job on that card).
#
# Usage (run.log lands in the committee folder):
#   mkdir -p $REMOTE/02_committee && cd $REMOTE/02_committee
#   setsid nohup bash $REMOTE/scripts/run_02_committee.sh <gpu_card> > run.log 2>&1 < /dev/null &
#
# Settings follow docs/showcase/CANON.md: the OC20-level trio of committee.yaml
# (C1), fmax 0.02 (C2), BFGS (manager's call). The start structure is the
# UNRELAXED O-fcc input of stage 01 (01_relax/o_fcc/POSCAR), so the trace covers
# the full relaxation. The committee owns model selection and device: no
# --mlip/--uma-task/--device here.
set -euo pipefail
source /scratchb/juar/mliprun-showcase/cluster.env

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <gpu_card>" >&2
    exit 2
fi
CARD="$1"
export CUDA_VISIBLE_DEVICES="$CARD"
# Shared, scheduler-less node: keep CPU threads modest (the work is on the GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

OUT="$REMOTE/02_committee"

# Never overwrite an earlier committee run in place.
if [[ -e "$OUT/opt_committee.csv" ]]; then
    echo "STOP: $OUT already holds committee output; move it aside first" >&2
    exit 1
fi
mkdir -p "$OUT"
cd "$OUT"

cp "$REMOTE/01_relax/o_fcc/POSCAR" POSCAR

# `gpu: N` in the yaml is the PHYSICAL card (mliprun overwrites each member's
# CUDA_VISIBLE_DEVICES), so the run-local copy pins all members to $CARD.
sed -E "s/^([[:space:]]*gpu:[[:space:]]*)[0-9]+/\1$CARD/" \
    "$REMOTE/committee.yaml" > committee.yaml
n_card=$(grep -Ec "^[[:space:]]*gpu:[[:space:]]*$CARD[[:space:]]*$" committee.yaml)
if [[ "$n_card" -ne 3 ]]; then
    echo "STOP: run-local committee.yaml has $n_card members on card $CARD, expected 3" >&2
    exit 1
fi

echo "=== $(date -Is) committee optimize run (card $CARD)"
"$UMA_BIN/mlip" optimize run --structure POSCAR --committee committee.yaml \
    --fmax 0.02 --optimizer bfgs --max-steps 500 --uncertainty-plot \
    > mlip_stdout.txt 2>&1

# optimize run exits 0 even when it does not converge: check explicitly.
if ! grep -Eq '^Converged:[[:space:]]+True' opt_params.txt; then
    echo "STOP: committee relaxation did not converge (see opt_params.txt)" >&2
    exit 1
fi
for f in opt_committee.csv opt_committee_peratom.csv opt_uncertainty.png; do
    [[ -s "$f" ]] || { echo "STOP: missing output $f" >&2; exit 1; }
done
echo "=== $(date -Is) stage 02 done (converged)"
