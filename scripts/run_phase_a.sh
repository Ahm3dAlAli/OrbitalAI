#!/bin/bash
# Phase A: A/B the scale-free DIoU+hinge size loss (--iou-size) on the DAVIS+Stars3
# checkpoint. Same recipe as g256_hn_v2 (100 epochs, patience 15, grid-256 + hard-neg)
# but with the L1 size loss swapped for DIoU+hinge, written to a NEW checkpoint so the
# current g256_hn_v2.pt (L1) stays intact as the baseline for the comparison.
# Run on rolf from ~/OrbitalAI.
#
#   bash scripts/run_phase_a.sh        # auto-pick the least-used GPU
#   bash scripts/run_phase_a.sh 3      # force GPU 3
#
# Produces: models/g256_hn_iou.pt  (DAVIS+Stars3, grid-256 + hard-neg + DIoU size)
# Then score it vs models/g256_hn_v2.pt on DAVIS + Stars3 to decide adoption.
set -e
cd "$(dirname "$0")/.."

# --- pick GPU: from arg, or auto-detect the least-used --------------------- #
if [ "$#" -ge 1 ]; then
    G1="$1"
else
    echo "[gpu] auto-detecting the least-used GPU..."
    G1=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
fi
[ -n "$G1" ] || { echo "[err] could not determine a GPU"; exit 1; }
echo "[gpu] hn_iou -> GPU $G1"

E="${EPOCHS:-100}"; P="${PATIENCE:-15}"
COMMON="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1"

name=hn_iou; log=hn_iou.log
if screen -ls | grep -q "\.${name}\b"; then
    echo "[skip] screen '$name' already exists"; exit 0
fi
# g256_hn recipe + --iou-size (scale-free DIoU+hinge on the size head)
screen -dmS "$name" bash -c \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=$G1 nice -n 15 \
   python3 scripts/train_centernet.py $COMMON --grid 256 --batch 40 \
     --dvx-weight 3.0 --evk4-weight 0.7 --hard-neg 2.0 --iou-size \
     --out models/g256_hn_iou.pt 2>&1 | tee $log"
echo "[run ] $name on GPU $G1 -> $log"

sleep 3
echo
echo "== launched screen =="
screen -ls | grep -E "$name" || echo "(none — check errors above)"
echo
echo "watch:        tail -f $log"
echo "attach:       screen -r $name        (detach: Ctrl-a d)"
echo "verify GPU:   nvidia-smi | grep python"
echo "done when:    models/g256_hn_iou.pt exists"
echo
echo "then A/B (DAVIS+Stars3, frozen evaluator):"
echo "  # baseline (L1):  score models/g256_hn_v2.pt"
echo "  # candidate (IoU): score models/g256_hn_iou.pt"
echo "  adopt g256_hn_iou.pt into the router only if DAVIS and/or Stars3 AP goes up."
