#!/bin/bash
# Lever A/Bs on the g192_ctx checkpoint (EVK4 + Thuraya3) — the counterpart to
# run_quicktry.sh, which tested the g256_hn (DAVIS+Stars3) checkpoint. Thuraya3 is
# served by g192_ctx, NOT g256_hn, so these are the runs that tell us whether the
# dim/small-object levers help Thuraya3's recall floor.
#
#   ctx_r3   --min-radius 3   : floor the heatmap Gaussian at 3 cells -> real positive
#                               signal for tiny objects (Thuraya3 is the smallest/dimmest).
#   ctx_dw   --dim-weight 0.5 : density-weight windows by ~1/sqrt(n_events) -> upweights
#                               the sparse windows Thuraya3 is made of.
#   ctx_iou  --iou-size       : Phase B — scale-free DIoU+hinge size loss on EVK4/Thuraya3
#                               (low expectation for Thuraya3's intrinsic floor, but the
#                               EVK4 half of this checkpoint may still gain).
#
# Same recipe as g192_ctx_v2 (grid-192, 100ep/patience15, no hard-neg). Each writes a
# NEW checkpoint so g192_ctx_v2.pt stays the baseline. Score each on EVK4 + Thuraya3
# (raw AND coasted for Thuraya3); adopt only what beats g192_ctx_v2.
#
# Run on rolf from ~/OrbitalAI:
#   bash scripts/run_ctx_levers.sh            # auto-pick 3 free GPUs
#   bash scripts/run_ctx_levers.sh 0 3 7      # force GPUs for r3 dw iou
set -e
cd "$(dirname "$0")/.."

E="${EPOCHS:-100}"; P="${PATIENCE:-15}"
COMMON="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1 --grid 192 --batch 64"

NAMES=(r3 dw iou)
FLAGS=("--min-radius 3"  "--dim-weight 0.5"  "--iou-size")
OUTS=(g192_ctx_r3 g192_ctx_dw g192_ctx_iou)

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
    name="ctx_${NAMES[$i]}"; gpu="${GPUS[$i]}"; log="${name}.log"
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
echo "== launched ctx-lever screens =="; screen -ls | grep -E "ctx_r3|ctx_dw|ctx_iou" || echo "(none)"
echo "watch:     tail -f ctx_r3.log ctx_dw.log ctx_iou.log"
echo "done when: models/g192_ctx_r3.pt g192_ctx_dw.pt g192_ctx_iou.pt exist"
echo
echo "then score each on EVK4 + Thuraya3 vs g192_ctx_v2.pt (EVK4 0.874, Thuraya3 raw 0.524):"
echo "  infer with --sequences <EVK4> <Thuraya3>, then run kalman_coast on Thuraya3,"
echo "  and compare AP. Adopt per-sensor only what wins."
