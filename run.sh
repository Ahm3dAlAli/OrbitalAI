#!/bin/sh
# OrbitSight container entrypoint (PRD FR-10, §6).
#
# Runs the full offline pipeline non-interactively:
#   1. Inference over every *.npy in the mounted dataset -> per-sequence
#      prediction files in the team output folder.
#   2. The frozen evaluator -> Evaluation_Metrics.xlsx in the same folder.
#
# Mounted paths (challenge spec):
#   /OrbitSight_dataset    (read-only)  event recordings + GT + dataloader
#   /work/teamName/DDMMYYYY (write)     predictions + scoring sheet
#
# Overridable via env vars for local runs.
set -e

export KMP_DUPLICATE_LIB_OK=TRUE      # guard against duplicate libomp on some hosts
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-0}"

DATASET="${ORBITSIGHT_DATASET:-/OrbitSight_dataset}"
TEAM="${ORBITSIGHT_TEAM:-OrbitSight}"
DATESTAMP="${ORBITSIGHT_DATE:-$(date +%d%m%Y)}"
OUT="${ORBITSIGHT_OUT:-/work/${TEAM}/${DATESTAMP}}"
MODEL="${ORBITSIGHT_MODEL:-models/coherence_lgbm.joblib}"

# Dataset may be split (Training_sets / Testing_sets) or flat.
TRAIN_DIR="${DATASET}/Training_sets"
TEST_DIR="${DATASET}/Testing_sets"

mkdir -p "${OUT}"
echo "[run.sh] dataset=${DATASET}  out=${OUT}  model=${MODEL}"

run_split () {
    src="$1"
    if [ -d "$src" ] && ls "$src"/*_labeled_events.npy >/dev/null 2>&1; then
        echo "[run.sh] inferring: $src"
        python3 scripts/infer.py --data-dir "$src" --model "${MODEL}" \
            --out-dir "${OUT}" --single-name
    fi
}

if [ -d "${TRAIN_DIR}" ] || [ -d "${TEST_DIR}" ]; then
    run_split "${TRAIN_DIR}"
    run_split "${TEST_DIR}"
else
    echo "[run.sh] inferring (flat layout): ${DATASET}"
    python3 scripts/infer.py --data-dir "${DATASET}" --model "${MODEL}" \
        --out-dir "${OUT}" --single-name
fi

# Scoring sheet (against provided GT, if present alongside the events).
echo "[run.sh] generating Evaluation_Metrics.xlsx"
python3 scripts/evaluate_wrapper.py --dataset "${DATASET}" \
    --pred-dir "${OUT}" --excel-out "${OUT}/Evaluation_Metrics.xlsx" || \
    echo "[run.sh] (evaluation skipped — GT not available)"

echo "[run.sh] done."
