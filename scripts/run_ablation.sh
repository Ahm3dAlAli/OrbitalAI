#!/bin/bash
# Loss ablation for the DIoU+hinge size head, run ONCE on the g256_hn (DAVIS+Stars3)
# checkpoint — the sensors whose boxes cluster near the IoU>=0.5 cliff, where the loss
# has measurable signal. NOT per-sensor: margin/lambda are properties of the loss, so
# we find the best config here and reuse it everywhere (incl. Phase B / EVK4).
#
# Same recipe as g256_hn_v2 (100 ep, patience 15, grid-256, hard-neg 2.0), varying only
# the DIoU knobs. The margin=0.15/lambda=2 point is ALREADY done (models/g256_hn_iou.pt).
# This launches the missing points, each in its own screen on a free GPU:
#
#   l0   pure DIoU (lambda=0)  -> is the win scale-invariance or the near-miss hinge?
#   m10  margin=0.10           -> margin sensitivity (tighter deadzone)
#   m20  margin=0.20           -> margin sensitivity (wider deadzone)
#
# Run on rolf from ~/OrbitalAI:
#   bash scripts/run_ablation.sh            # auto-pick free GPUs, one per run
#   bash scripts/run_ablation.sh 4 5 6      # force GPUs for l0 m10 m20
set -e
cd "$(dirname "$0")/.."

E="${EPOCHS:-100}"; P="${PATIENCE:-15}"
COMMON="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1 \
  --grid 256 --batch 40 --dvx-weight 3.0 --evk4-weight 0.7 --hard-neg 2.0 --iou-size"

# ablation variants: name | extra flags | output checkpoint
NAMES=(l0 m10 m20)
FLAGS=("--iou-lambda 0"      "--iou-margin 0.10"      "--iou-margin 0.20")
OUTS=(g256_hn_iou_l0 g256_hn_iou_m10 g256_hn_iou_m20)

# --- pick GPUs: from args, or auto-detect the N least-used ------------------ #
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
    name="ab_${NAMES[$i]}"; gpu="${GPUS[$i]}"; log="${name}.log"
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
echo "== launched ablation screens =="
screen -ls | grep -E "ab_l0|ab_m10|ab_m20" || echo "(none — check errors above)"
echo
echo "watch all:    tail -f ab_l0.log ab_m10.log ab_m20.log"
echo "done when:    models/g256_hn_iou_l0.pt g256_hn_iou_m10.pt g256_hn_iou_m20.pt exist"
echo
echo "then score each vs the margin=0.15 point (models/g256_hn_iou.pt = 0.7299 2-seq mAP):"
echo "  reuse the /tmp/run_ab.sh pattern, swapping --models to each checkpoint."
echo "reading: l0 ~ m15  -> the win is SCALE-INVARIANCE (DIoU), hinge is optional"
echo "         l0 <  m15  -> the near-miss HINGE adds real gain on top of DIoU"
echo "         pick the best margin from {m10, m15, m20} for Phase B / EVK4."
