#!/bin/bash
# GPU A/B: base (g192_ctx_v2) vs inject (g192_ctx_inject) on the faint DVX sequences.
# Runs on GPU with TTA, then coasts Thuraya3 (the deployed-style path), so the number is
# comparable to the deployed baseline (Thuraya3 coasted 0.5337). Fast: ~2 min vs the
# CPU A/B's ~30 min.
#
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_inject_ab.sh
set -u
cd "$(dirname "$0")/.."
DEV="${DEV:-cuda}"
T=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
S=DVX_Filtered_Stars3_2025-01-20-20-22-53
DATA=OrbitSight_Dataset/Testing_sets

ap_of() {  # $1=pred-dir  $2=grep token  -> AP (field NF-1; row ends with trailing pipe)
    python3 scripts/evaluate_wrapper.py --dataset OrbitSight_Dataset --pred-dir "$1" \
        --excel-out "$1/m.xlsx" 2>/dev/null | grep -E "$2" | tail -1 \
        | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}'
}

score_model() {  # $1=label  $2=checkpoint
    local name="$1" ck="$2" out="/tmp/inj_ab_$1"
    if [ ! -f "$ck" ]; then printf "%-8s | (missing %s)\n" "$name" "$ck"; return; fi
    python3 scripts/infer_ensemble.py --device "$DEV" --tta \
        --data-dir "$DATA" --out-dir "$out" --models "$ck" --sequences $T $S >/dev/null 2>&1
    local traw sap tco
    traw=$(ap_of "$out" "Thuraya3"); sap=$(ap_of "$out" "Stars3")
    rm -rf "${out}_c"; cp -r "$out" "${out}_c"
    python3 scripts/kalman_coast.py --data-dir "$DATA" --pred-dir "${out}_c" \
        --out-dir "${out}_c" --max-coast 50 --decay 0.92 --sequences $T >/dev/null 2>&1
    tco=$(ap_of "${out}_c" "Thuraya3")
    printf "%-8s | %-12s | %-16s | %-8s\n" "$name" "${traw:-?}" "${tco:-?}" "${sap:-?}"
}

echo "== base vs inject (GPU, TTA) =="
printf "%-8s | %-12s | %-16s | %-8s\n" "model" "Thur3 raw" "Thur3 coasted" "Stars3"
printf -- "---------|--------------|------------------|--------\n"
score_model base   models/g192_ctx_v2.pt
score_model inject models/g192_ctx_inject.pt
echo
echo "deployed baseline: Thuraya3 coasted 0.5337 (base). inject wins only if it beats that."
