#!/bin/bash
# OFFLINE max-accuracy re-score with the CURRENT-era models (DIoU + min-radius).
# Not real-time (multi-model ensemble + TTA) -> refreshes the stale 0.715 offline number.
# Per-sensor ensemble of the variants we already have (no training):
#   EVK4       : {g192_ctx_r3, g192_ctx_v2, g192_ctx_iou}  + TTA   (grid-192 scale)
#   DAVIS+Stars3: {g256_hn_iou, g256_hn_v2}                + TTA   (grid-256 hard-neg)
#   Thuraya3   : {g192_ctx_v2, g192_ctx_iou}               + TTA -> coasting Kalman
# All members are averaged (heatmap) in infer_ensemble. Writes one pred dir, scores the
# 4 test sequences. Run on rolf; pin a FREE GPU to avoid OOM (offline != latency-bound,
# so --device cpu also works but is slow).
#
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_offline.sh          # gpu 2
#   DEV=cpu bash scripts/run_offline.sh                         # cpu (OOM-proof, slow)
set -u
cd "$(dirname "$0")/.."
DEV="${DEV:-cuda}"
T=OrbitSight_Dataset/Testing_sets
OUT="${OUT:-/tmp/off}"; mkdir -p "$OUT"
EVK4=2025_12_23_20_53_46_EVK4_mag7.3
DA=DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37
S=DVX_Filtered_Stars3_2025-01-20-20-22-53
TH=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43

present() { local o=""; for m in "$@"; do [ -f "$m" ] && o="$o $m"; done; echo "$o"; }
EVK4_M=$(present models/g192_ctx_r3.pt models/g192_ctx_v2.pt models/g192_ctx_iou.pt)
HN_M=$(present   models/g256_hn_iou.pt models/g256_hn_v2.pt)
TH_M=$(present   models/g192_ctx_v2.pt models/g192_ctx_iou.pt)
echo "[offline] EVK4 ensemble:   $EVK4_M"
echo "[offline] DAVIS+Stars3:    $HN_M"
echo "[offline] Thuraya3:        $TH_M (+coast)"

echo "== EVK4 (cross-member + TTA) =="
python3 scripts/infer_ensemble.py --device "$DEV" --tta \
    --data-dir "$T" --out-dir "$OUT" --models $EVK4_M --sequences $EVK4

echo "== DAVIS + Stars3 (grid-256 ensemble + TTA) =="
python3 scripts/infer_ensemble.py --device "$DEV" --tta --topk 2 --thresh 0.2 \
    --data-dir "$T" --out-dir "$OUT" --models $HN_M --sequences $DA $S

echo "== Thuraya3 (ensemble + TTA -> coast) =="
python3 scripts/infer_ensemble.py --device "$DEV" --tta \
    --data-dir "$T" --out-dir "$OUT" --models $TH_M --sequences $TH
python3 scripts/kalman_coast.py --data-dir "$T" --pred-dir "$OUT" --out-dir "$OUT" \
    --max-coast 50 --decay 0.92 --sequences $TH

echo "== SCORE (4 test sequences) =="
python3 scripts/evaluate_wrapper.py --dataset OrbitSight_Dataset --pred-dir "$OUT" \
    --excel-out "$OUT/m.xlsx" | grep -E "EVK4_mag7.3|SAOCOM|Stars3|Thuraya3|mAP @"
echo "baseline: real-time 0.721 (EVK4 0.891 DAVIS 0.783 Stars3 0.677 Thur 0.534); stale offline 0.715"
