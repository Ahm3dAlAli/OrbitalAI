#!/bin/bash
# Train the two DEPLOYED models to 100 epochs (patience 15), each in its own screen
# pinned to a free GPU. Run on rolf from ~/OrbitalAI.
#
#   bash scripts/run_100ep.sh          # auto-pick the 2 least-used GPUs
#   bash scripts/run_100ep.sh 1 2      # force GPUs 1 and 2
#
# Produces: models/g192_ctx_v2.pt (EVK4+Thuraya3, grid-192)
#           models/g256_hn_v2.pt  (DAVIS+Stars3, grid-256 + hard-neg)
set -e
cd "$(dirname "$0")/.."

# --- pick GPUs: from args, or auto-detect the two least-used ---------------- #
if [ "$#" -ge 2 ]; then
    G1="$1"; G2="$2"
else
    echo "[gpu] auto-detecting the 2 least-used GPUs..."
    FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t',' -k2 -n | head -2 | cut -d',' -f1 | tr -d ' ')
    G1=$(echo "$FREE" | sed -n 1p)
    G2=$(echo "$FREE" | sed -n 2p)
fi
[ -n "$G1" ] && [ -n "$G2" ] || { echo "[err] could not determine two GPUs"; exit 1; }
echo "[gpu] ctx_v2 -> GPU $G1   |   hn_v2 -> GPU $G2"

E="${EPOCHS:-100}"; P="${PATIENCE:-15}"
COMMON="--device cuda --workers 8 --patch 8 --dim 128 --tbins 7 --context 3 \
  --hm-div 2 --augment --epochs $E --patience $P --seed 1"

launch () {  # $1=screen  $2=gpu  $3=logfile  $4...=extra train flags
    local name="$1" gpu="$2" log="$3"; shift 3
    if screen -ls | grep -q "\.${name}\b"; then
        echo "[skip] screen '$name' already exists"; return
    fi
    screen -dmS "$name" bash -c \
      "cd '$PWD' && CUDA_VISIBLE_DEVICES=$gpu nice -n 15 \
       python3 scripts/train_centernet.py $COMMON $* 2>&1 | tee $log"
    echo "[run ] $name on GPU $gpu -> $log"
}

# g192_ctx_v2: EVK4 + Thuraya3 base (grid-192)
launch ctx_v2 "$G1" ctx_v2.log --grid 192 --batch 64 --out models/g192_ctx_v2.pt

# g256_hn_v2: DAVIS + Stars3 (grid-256 + hard-negative mining)
launch hn_v2  "$G2" hn_v2.log  --grid 256 --batch 40 \
       --dvx-weight 3.0 --evk4-weight 0.7 --hard-neg 2.0 --out models/g256_hn_v2.pt

sleep 3
echo
echo "== launched screens =="
screen -ls | grep -E "ctx_v2|hn_v2" || echo "(none — check errors above)"
echo
echo "watch both:   tail -f ctx_v2.log hn_v2.log"
echo "attach:       screen -r ctx_v2      (detach: Ctrl-a d)"
echo "verify GPUs:  nvidia-smi | grep python"
echo "done when:    models/g192_ctx_v2.pt and models/g256_hn_v2.pt exist"
