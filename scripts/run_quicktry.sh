#!/bin/bash
# Two quick, cheap add-ons stacked ON TOP of the DIoU+hinge winner (g256_hn_iou),
# each a one-variable A/B on the DAVIS+Stars3 recipe:
#
#   r3  --min-radius 3   : floor the heatmap Gaussian splat at 3 cells. The
#                          0.3*max(w,h) formula collapses to ~1 cell for a ~10 px
#                          object (9 support cells); a floor of 3 gives 37 -> real
#                          positive signal for small objects. Aimed at Stars3/dim.
#   dw  --dim-weight 0.5 : density-weight windows by ~1/sqrt(n_events) (already in the
#                          sampler). Stops dense EVK4 windows dominating the gradient;
#                          aimed at the dim floor. (0.5 exponent == 1/sqrt(n).)
#
# Both keep --iou-size (margin 0.15, lambda 2) so they measure gain *on top of* DIoU.
# Score each vs g256_hn_iou.pt (0.7299 2-seq mAP); adopt only what adds.
#
# Run on rolf from ~/OrbitalAI:
#   bash scripts/run_quicktry.sh          # auto-pick 2 free GPUs
#   bash scripts/run_quicktry.sh 2 7      # force GPUs
set -e
cd "$(dirname "$0")/.."

E="${EPOCHS:-100}"; P="${PATIENCE:-15}"
COMMON="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1 \
  --grid 256 --batch 40 --dvx-weight 3.0 --evk4-weight 0.7 --hard-neg 2.0 --iou-size"

NAMES=(r3 dw)
FLAGS=("--min-radius 3"  "--dim-weight 0.5")
OUTS=(g256_hn_iou_r3 g256_hn_iou_dw)

N=${#NAMES[@]}
if [ "$#" -ge "$N" ]; then
    GPUS=("$@")
else
    echo "[gpu] auto-detecting the $N least-used GPUs..."
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                        | sort -t',' -k2 -n | head -"$N" | cut -d',' -f1 | tr -d ' ')
fi
[ "${#GPUS[@]}" -ge "$N" ] || { echo "[err] need $N GPUs, found ${#GPUS[@]}"; exit 1; }

for i in "${!NAMES[@]}"; do
    name="qt_${NAMES[$i]}"; gpu="${GPUS[$i]}"; log="${name}.log"
    out="models/${OUTS[$i]}.pt"; extra="${FLAGS[$i]}"
    if screen -ls | grep -q "\.${name}\b"; then
        echo "[skip] screen '$name' already exists"; continue
    fi
    screen -dmS "$name" bash -c \
      "cd '$PWD' && CUDA_VISIBLE_DEVICES=$gpu nice -n 15 \
       python3 scripts/train_centernet.py $COMMON $extra --out $out 2>&1 | tee $log"
    echo "[run ] $name on GPU $gpu ($extra) -> $out  ($log)"
done

sleep 3
echo
echo "== launched quick-try screens =="; screen -ls | grep -E "qt_r3|qt_dw" || echo "(none)"
echo "watch:     tail -f qt_r3.log qt_dw.log"
echo "done when: models/g256_hn_iou_r3.pt g256_hn_iou_dw.pt exist"
echo "then score each vs g256_hn_iou.pt (0.7299); keep only what beats it."
