#!/bin/bash
# Score every g256_hn DIoU-loss variant on DAVIS+Stars3 with the frozen evaluator and
# print one comparison table vs the adopted margin=0.15 baseline (g256_hn_iou.pt = 0.7299
# 2-seq mAP). Skips checkpoints that don't exist yet, so it is safe to run while some
# are still training. Uses the deployed g256 settings (--topk 2 --thresh 0.2, no TTA).
#
# Run on rolf from ~/OrbitalAI:
#   bash scripts/run_ablation_score.sh
set -u
cd "$(dirname "$0")/.."

S=DVX_Filtered_Stars3_2025-01-20-20-22-53
D=DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37
DATA=OrbitSight_Dataset/Testing_sets
DEV="${ORBITSIGHT_DEVICE:-cuda}"

# name | checkpoint (all built on the DIoU winner; each changes ONE thing)
declare -A CK=(
  [m15_baseline]=models/g256_hn_iou.pt
  [l0_pureDIoU]=models/g256_hn_iou_l0.pt
  [m10]=models/g256_hn_iou_m10.pt
  [m20]=models/g256_hn_iou_m20.pt
  [r3_minradius]=models/g256_hn_iou_r3.pt
  [dw_dimweight]=models/g256_hn_iou_dw.pt
)
ORDER=(m15_baseline l0_pureDIoU m10 m20 r3_minradius dw_dimweight)

ap_of() {  # $1=metrics dir, $2=grep token -> prints AP (last numeric col before the
           # trailing pipe; the row ends with "| ... | AP |", so AP is field NF-1)
    grep -E "$2" "$1/eval.txt" 2>/dev/null | tail -1 | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}'
}

printf "\n%-16s | %8s | %8s | %9s\n" "variant" "DAVIS" "Stars3" "2-seq mAP"
printf -- "-----------------|----------|----------|----------\n"
for name in "${ORDER[@]}"; do
    ck="${CK[$name]}"
    if [ ! -f "$ck" ]; then
        printf "%-16s | %8s | %8s | %9s\n" "$name" "pending" "pending" "pending"; continue
    fi
    out="/tmp/score_$name"; mkdir -p "$out"
    python3 scripts/infer_ensemble.py --device "$DEV" --topk 2 --thresh 0.2 \
        --data-dir "$DATA" --out-dir "$out" --models "$ck" --sequences $S $D \
        >/dev/null 2>&1
    python3 scripts/evaluate_wrapper.py --dataset OrbitSight_Dataset \
        --pred-dir "$out" --excel-out "$out/m.xlsx" >"$out/eval.txt" 2>&1
    dav=$(ap_of "$out" "SAOCOM"); star=$(ap_of "$out" "Stars3")
    map=$(grep -E "mAP @ IoU" "$out/eval.txt" | tail -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    printf "%-16s | %8s | %8s | %9s\n" "$name" "${dav:-?}" "${star:-?}" "${map:-?}"
done
echo
echo "baseline to beat: 2-seq mAP 0.7299 (DAVIS 0.783 / Stars3 0.6768)."
echo "reading: l0~m15 -> gain is scale-invariance (hinge optional); l0<m15 -> hinge adds value."
echo "         best of {m10,m15,m20} = margin to adopt; r3/dw beat m15 -> stack them."
