#!/bin/bash
# TAR (Time-Surface fusion) ablation — NeuTAR-RSO's input-representation lever.
# Adds 2 exp-decay Time-Surface channels to the voxel (C = tbins*2 -> tbins*2+2),
# fusing the count "Time-Aggregated Frame" we already have with a per-pixel recency
# map (paper Fig. 6c). ONE variable changes vs. each deployed baseline: --time-surface.
#
# Because the input channel count changes, this REQUIRES a full retrain (a count-only
# checkpoint cannot be loaded with TAR on). Two members mirror the per-sensor router:
#
#   ctx_tar : g192_ctx_v2 recipe + --time-surface   -> score on EVK4 + Thuraya3
#   hn_tar  : g256_hn_iou recipe + --time-surface    -> score on DAVIS + Stars3
#
# Each writes a NEW checkpoint so the count-only baselines stay put. Adopt per-sensor
# only what beats its baseline (EVK4 0.8912, Thuraya3 0.5337, DAVIS 0.783, Stars3 0.6768).
# EITHER outcome is a result: a gain is a new representation lever; a wash shows the
# temporal-context input already carries the motion signal a Time Surface would add.
#
# Run on rolf from ~/OrbitalAI:
#   bash scripts/run_tar_ablation.sh          # auto-pick 2 free GPUs
#   bash scripts/run_tar_ablation.sh 0 3      # force GPUs for ctx_tar hn_tar
set -e
cd "$(dirname "$0")/.."

E="${EPOCHS:-100}"; P="${PATIENCE:-15}"
# g192_ctx_v2 recipe (EVK4 + Thuraya3): grid-192, ctx=3/tbins=7, no hard-neg.
CTX="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1 --grid 192 --batch 64"
# g256_hn_iou recipe (DAVIS + Stars3): grid-256, hard-neg 2.0, DIoU size loss.
HN="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1 --grid 256 --batch 40 \
  --dvx-weight 3.0 --evk4-weight 0.7 --hard-neg 2.0 --iou-size"

NAMES=(ctx_tar hn_tar)
COMMANDS=("$CTX" "$HN")
OUTS=(g192_ctx_tar g256_hn_tar)

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
    name="${NAMES[$i]}"; gpu="${GPUS[$i]}"; log="${name}.log"
    out="models/${OUTS[$i]}.pt"; recipe="${COMMANDS[$i]}"
    if screen -ls | grep -q "\.${name}\b"; then
        echo "[skip] screen '$name' already exists"; continue
    fi
    screen -dmS "$name" bash -c \
      "cd '$PWD' && CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 nice -n 15 \
       python3 scripts/train_centernet.py $recipe --time-surface --out $out 2>&1 | tee $log"
    echo "[run ] $name on GPU $gpu (--time-surface) -> $out  ($log)"
done

sleep 3
echo
echo "== launched TAR-ablation screens =="; screen -ls | grep -E "ctx_tar|hn_tar" || echo "(none)"
echo "watch:     tail -f ctx_tar.log hn_tar.log"
echo "done when: models/g192_ctx_tar.pt models/g256_hn_tar.pt exist"
echo
echo "then score (mirror the deployed re-score):"
echo "  # EVK4 + Thuraya3 from the ctx_tar checkpoint:"
echo "  python3 scripts/infer_centernet.py --data-dir OrbitSight_Dataset/Testing_sets \\"
echo "    --model models/g192_ctx_tar.pt --out-dir /tmp/tar_ctx"
echo "  # DAVIS + Stars3 from the hn_tar checkpoint:"
echo "  python3 scripts/infer_centernet.py --data-dir OrbitSight_Dataset/Testing_sets \\"
echo "    --model models/g256_hn_tar.pt --out-dir /tmp/tar_hn"
echo "  python3 scripts/evaluate_wrapper.py --dataset OrbitSight_Dataset \\"
echo "    --pred-dir /tmp/tar_ctx   # and /tmp/tar_hn ; compare per-seq AP vs baselines"
