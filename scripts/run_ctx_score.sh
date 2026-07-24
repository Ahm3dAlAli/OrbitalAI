#!/bin/bash
# Batch-2 scorer: score every g192_ctx lever variant on EVK4 + Thuraya3 with the frozen
# evaluator, applying the coasting Kalman to Thuraya3 (raw AND coasted reported). Prints
# one table vs the g192_ctx_v2 baseline (EVK4 0.881, Thuraya3 raw 0.524 / coasted 0.534).
# Skips checkpoints that don't exist yet, so it is safe to run while some still train.
#
# Run on rolf from ~/OrbitalAI:
#   bash scripts/run_ctx_score.sh
set -u
cd "$(dirname "$0")/.."

E=2025_12_23_20_53_46_EVK4_mag7.3
T=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
DATA=OrbitSight_Dataset/Testing_sets
DEV="${ORBITSIGHT_DEVICE:-cuda}"

declare -A CK=(
  [v2_baseline]=models/g192_ctx_v2.pt
  [iou_DIoU]=models/g192_ctx_iou.pt
  [r3_minradius]=models/g192_ctx_r3.pt
  [dw_dimweight]=models/g192_ctx_dw.pt
)
ORDER=(v2_baseline iou_DIoU r3_minradius dw_dimweight)

ap_of() {  # $1=eval.txt, $2=grep token -> AP (field NF-1; row ends with trailing pipe)
    grep -E "$2" "$1" 2>/dev/null | tail -1 | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}'
}

printf "\n%-15s | %8s | %11s | %13s\n" "variant" "EVK4" "Thur3 raw" "Thur3 coasted"
printf -- "----------------|----------|-------------|---------------\n"
for name in "${ORDER[@]}"; do
    ck="${CK[$name]}"
    if [ ! -f "$ck" ]; then
        printf "%-15s | %8s | %11s | %13s\n" "$name" "pending" "pending" "pending"; continue
    fi
    out="/tmp/ctxscore_$name"; mkdir -p "$out"
    # infer EVK4 + Thuraya3 (TTA on, matching the deployed g192_ctx path)
    python3 scripts/infer_ensemble.py --device "$DEV" --tta \
        --data-dir "$DATA" --out-dir "$out" --models "$ck" --sequences $E $T >/dev/null 2>&1
    # RAW score
    python3 scripts/evaluate_wrapper.py --dataset OrbitSight_Dataset \
        --pred-dir "$out" --excel-out "$out/m.xlsx" >"$out/eval_raw.txt" 2>&1
    evk=$(ap_of "$out/eval_raw.txt" "EVK4_mag7.3"); traw=$(ap_of "$out/eval_raw.txt" "Thuraya3")
    # COASTED Thuraya3 (in place, then re-score)
    cp -r "$out" "${out}_c"
    python3 scripts/kalman_coast.py --data-dir "$DATA" --pred-dir "${out}_c" \
        --out-dir "${out}_c" --max-coast 50 --decay 0.92 --sequences $T >/dev/null 2>&1
    python3 scripts/evaluate_wrapper.py --dataset OrbitSight_Dataset \
        --pred-dir "${out}_c" --excel-out "${out}_c/m.xlsx" >"${out}_c/eval.txt" 2>&1
    tco=$(ap_of "${out}_c/eval.txt" "Thuraya3")
    printf "%-15s | %8s | %11s | %13s\n" "$name" "${evk:-?}" "${traw:-?}" "${tco:-?}"
done
echo
echo "baseline (v2): EVK4 0.881, Thuraya3 raw 0.524, coasted 0.534."
echo "adopt per-sensor: iou if EVK4 up; r3/dw if Thuraya3 (coasted) up. DIoU expected"
echo "~flat on Thuraya3 (its floor is not a sizing problem); r3/dw are its real shot."
