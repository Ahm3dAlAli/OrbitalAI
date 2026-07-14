#!/bin/sh
# OrbitSight deep-model container entrypoint — reproduces the winning pipeline
# (multi-window temporal-context CenterNet + TTA, per-sensor routed).
#
# Non-interactive, offline.  For every *_labeled_events.npy in the mounted
# dataset it writes a prediction file, then (if GT is present) the frozen
# evaluator's Evaluation_Metrics.xlsx — all into the team output folder.
#
# Mounted paths (challenge spec):
#   /OrbitSight_dataset      (read-only)  event recordings + GT
#   /work/teamName/DDMMYYYY  (write)      predictions + scoring sheet
#
# Everything below is overridable via env vars (see docker run examples in the
# Dockerfile header).  Defaults reproduce the deployed real-time result on CPU:
# a single temporal model per sensor, with the Stars3 star field routed to a
# grid-256 multi-object detector (overall mAP 0.668, all sensors < 40 ms).
set -e
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-0}"

DATASET="${ORBITSIGHT_DATASET:-/OrbitSight_dataset}"
TEAM="${ORBITSIGHT_TEAM:-OrbitSight}"
DATESTAMP="${ORBITSIGHT_DATE:-$(date +%d%m%Y)}"
OUT="${ORBITSIGHT_OUT:-/work/${TEAM}/${DATESTAMP}}"
DEVICE="${ORBITSIGHT_DEVICE:-cpu}"                 # cpu (portable, real-time) | cuda
TTA="${ORBITSIGHT_TTA:---tta}"                     # set to "" to disable TTA
# DAVIS/DVX detectors (temporal ensemble members) — space-separated checkpoints:
MODELS="${ORBITSIGHT_MODELS:-models/g192_ctx.pt}"
# Optional distinct EVK4 detector(s) for the per-sensor router (cross-grid).
# If unset, the DAVIS/DVX models are used for EVK4 too (single-ensemble mode).
EVK4_MODELS="${ORBITSIGHT_EVK4_MODELS:-}"
# DAVIS + Stars3 -> grid-256 hard-negative model (DAVIS 0.729->0.753, Stars3
# 0.613->0.651). If absent, those sensors keep the default detector's prediction.
G256_MODEL="${ORBITSIGHT_G256_MODEL:-models/g256_hn.pt}"
STARS_TOPK="${ORBITSIGHT_STARS_TOPK:-2}"
# Thuraya3 faint object -> coasting Kalman recall recovery (0.469->0.506).
COAST="${ORBITSIGHT_COAST:-1}"; COAST_MAX="${ORBITSIGHT_COAST_MAX:-50}"

mkdir -p "${OUT}"
echo "[run_infer] dataset=${DATASET} out=${OUT} device=${DEVICE}"
echo "[run_infer] models=[${MODELS}] evk4=[${EVK4_MODELS:-<same>}] tta=${TTA:-off}"

# Fall back to whatever temporal-ish checkpoint is present if the default is absent.
first_present() { for m in "$@"; do [ -f "$m" ] && { echo "$m"; return; }; done; }
if [ -z "$(first_present $MODELS)" ]; then
    ALT="$(first_present models/g192_ctx.pt models/evt_centernet_aug.pt models/evt_centernet.pt)"
    [ -n "$ALT" ] && MODELS="$ALT" && echo "[run_infer] default model missing; using ${MODELS}"
fi

# List the sequence base-names of one split whose sensor matches $1 (grep token).
seqs_for () { # $1 = split dir, $2 = sensor token (EVK4|DAVIS|DVX) or "" for all
    for p in "$1"/*_labeled_events.npy; do
        [ -e "$p" ] || continue
        b="$(basename "$p")"; b="${b%_labeled_events.npy}"
        if [ -z "$2" ] || echo "$b" | grep -qi "$2"; then echo "$b"; fi
    done
}

infer_split () {
    src="$1"
    ls "$src"/*_labeled_events.npy >/dev/null 2>&1 || return 0
    echo "[run_infer] inferring split: $src"
    if [ -n "$EVK4_MODELS" ]; then
        # Per-sensor router: EVK4 -> cross-grid, DAVIS/DVX -> temporal ensemble.
        NON_EVK4="$(seqs_for "$src" DAVIS) $(seqs_for "$src" DVX)"
        EVK4_SEQS="$(seqs_for "$src" EVK4)"
        tmp_main="${OUT}/.main"; tmp_evk4="${OUT}/.evk4"
        [ -n "$(echo $NON_EVK4)" ] && python3 scripts/infer_ensemble.py \
            --device "$DEVICE" $TTA --data-dir "$src" --out-dir "$tmp_main" \
            --models $MODELS --sequences $NON_EVK4
        [ -n "$(echo $EVK4_SEQS)" ] && python3 scripts/infer_ensemble.py \
            --device "$DEVICE" $TTA --data-dir "$src" --out-dir "$tmp_evk4" \
            --models $EVK4_MODELS --sequences $EVK4_SEQS
        python3 scripts/route.py --out-dir "$OUT" \
            --map EVK4="$tmp_evk4" DAVIS="$tmp_main" DVX="$tmp_main"
        rm -rf "$tmp_main" "$tmp_evk4"
    else
        # Single-ensemble mode: one detector set over every sequence.
        python3 scripts/infer_ensemble.py --device "$DEVICE" $TTA \
            --data-dir "$src" --out-dir "$OUT" --models $MODELS
    fi
}

if [ -d "${DATASET}/Testing_sets" ] || [ -d "${DATASET}/Training_sets" ]; then
    infer_split "${DATASET}/Training_sets"
    infer_split "${DATASET}/Testing_sets"
else
    infer_split "${DATASET}"                       # flat layout
fi

# DAVIS + Stars3 -> grid-256 hard-negative model (overrides their predictions).
if [ -f "$G256_MODEL" ]; then
    for split in "${DATASET}/Testing_sets" "${DATASET}/Training_sets" "${DATASET}"; do
        [ -d "$split" ] || continue
        SEQS="$(seqs_for "$split" Stars3) $(seqs_for "$split" DAVIS)"
        [ -n "$(echo $SEQS)" ] || continue
        echo "[run_infer] DAVIS+Stars3 -> grid-256 hard-neg ($G256_MODEL, topk=$STARS_TOPK)"
        python3 scripts/infer_ensemble.py --device "$DEVICE" --topk "$STARS_TOPK" \
            --thresh 0.2 --data-dir "$split" --out-dir "$OUT" \
            --models "$G256_MODEL" --sequences $SEQS
    done
else
    echo "[run_infer] grid-256 hard-neg model ($G256_MODEL) absent -> keeping default"
fi

# Thuraya3 faint object -> coasting Kalman recall recovery (in place, on $OUT).
if [ "$COAST" = "1" ]; then
    for split in "${DATASET}/Testing_sets" "${DATASET}/Training_sets" "${DATASET}"; do
        [ -d "$split" ] || continue
        TH="$(seqs_for "$split" Thuraya3)"
        [ -n "$(echo $TH)" ] || continue
        echo "[run_infer] Thuraya3 -> coasting Kalman (max-coast=$COAST_MAX)"
        python3 scripts/kalman_coast.py --data-dir "$split" --pred-dir "$OUT" \
            --out-dir "$OUT" --max-coast "$COAST_MAX" --decay 0.92 --sequences $TH
    done
fi

echo "[run_infer] generating Evaluation_Metrics.xlsx"
python3 scripts/evaluate_wrapper.py --dataset "${DATASET}" \
    --pred-dir "${OUT}" --excel-out "${OUT}/Evaluation_Metrics.xlsx" || \
    echo "[run_infer] (evaluation skipped — GT not available)"

echo "[run_infer] done -> ${OUT}"
