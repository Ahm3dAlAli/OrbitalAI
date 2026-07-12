#!/bin/bash
# Improvement pipeline toward 0.8 — run AFTER `bash scripts/sync_from_rolf.sh`.
#
# Applies the two DVX-targeted levers on the synced TEMPORAL predictions + model,
# assembles the routed set, and scores against the test GT so you can compare to
# the 0.675 baseline.  Both levers are EXPERIMENTAL — the script prints the score
# so you can see whether each actually helped (they may not; see ROADMAP §2).
#
# Requires (from sync_from_rolf.sh):
#   models/g192_ctx.pt          temporal model (for Stars3 multi-peak re-inference)
#   predictions/test_ctx        temporal single-model predictions (all sequences)
#   predictions/test_xg         EVK4 cross-grid predictions
set -e
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1

T=${DATA_DIR:-OrbitSight_Dataset/Testing_sets}
CTX=${CTX_DIR:-predictions/test_ctx}
XG=${XG_DIR:-predictions/test_xg}
MODEL=${MODEL:-models/g192_ctx.pt}
DEV=${DEVICE:-cpu}
STARS=DVX_Filtered_Stars3_2025-01-20-20-22-53
THUR=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
EVK4=2025_12_23_20_53_46_EVK4_mag7.3
G=_bb_windows_40ms.txt

for need in "$T" "$CTX"; do
  [ -e "$need" ] || { echo "[ERR] missing $need — run scripts/sync_from_rolf.sh first"; exit 1; }
done

echo "== A. Stars3 (multi-object): re-infer with topk peaks =="
if [ -f "$MODEL" ]; then
  python3 scripts/infer_ensemble.py --device $DEV --tta --topk ${TOPK:-3} \
    --thresh ${THRESH:-0.25} --data-dir "$T" --out-dir predictions/imp_stars \
    --models "$MODEL" --sequences $STARS
else
  echo "  [skip] $MODEL absent — Stars3 stays as base"
fi

echo "== B. Thuraya3: trajectory-fill on temporal predictions =="
python3 scripts/trajectory_fill.py --data-dir "$T" --pred-dir "$CTX" \
  --out-dir predictions/imp_thur --sequences $THUR \
  --conf-min ${CONF_MIN:-0.4} --min-inlier-frac ${MIF:-0.45} \
  --max-resid-px ${MAXRES:-8} --max-expand ${MAXEXP:-2.0}

echo "== C. assemble routed set: base=temporal, EVK4->xg, Stars3->topk, Thuraya3->traj =="
OUT=predictions/router_imp; rm -rf "$OUT"; mkdir -p "$OUT"
cp "$CTX"/*$G "$OUT"/ 2>/dev/null || true
[ -d "$XG" ]                 && cp "$XG/${EVK4}$G"       "$OUT"/ 2>/dev/null || true
[ -f predictions/imp_stars/${STARS}$G ] && cp predictions/imp_stars/${STARS}$G "$OUT"/
[ -f predictions/imp_thur/${THUR}$G ]   && cp predictions/imp_thur/${THUR}$G   "$OUT"/

echo "== D. evaluate (compare overall mAP to 0.675) =="
python3 Dataloader/evaluate.py --gt-dir "$T" --pred-dir "$OUT" \
  --excel-out Evaluation_Metrics_improved.xlsx | grep -E "Stars3|Thuraya3|SAOCOM|EVK4|mAP"
echo
echo "Baseline per-seq (temporal+TTA): EVK4 0.896  DAVIS 0.729  Stars3 0.545  Thuraya3 0.469  -> mAP 0.675"
echo "If a lever DIDN'T help, revert that sequence to its baseline dir and re-route."
