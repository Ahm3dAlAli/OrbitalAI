#!/bin/bash
# OrbitSight — ON-ROLF bootstrap: conda env + data placement + (optional) run.
#
# PREREQUISITE (do this FIRST, from your LAPTOP — rolf cannot reach your laptop):
#     bash scripts/sync_to_rolf.sh          # pushes code -> ~/OrbitalAI and data
#   OR clone the code and push only the data:
#     (on rolf)  git clone <your-repo-url> ~/OrbitalAI
#     (laptop)   rsync -av OrbitSight_Dataset/ alali@rolf.ifi.uzh.ch:/local/scratch/alali/OrbitSight_Dataset/
#
# THEN, on rolf:
#     cd ~/OrbitalAI
#     bash scripts/bootstrap_rolf.sh          # setup only
#     bash scripts/bootstrap_rolf.sh run      # setup, then run the full pipeline
set -e

USER=$(whoami)
SCRATCH=/local/scratch/$USER
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
echo "== project: $PROJ  |  user: $USER  |  scratch: $SCRATCH =="

# ---------------------------------------------------------------------------
# 1. conda (install miniconda to scratch if missing) + environment
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "== installing miniconda -> $SCRATCH/miniconda =="
    mkdir -p "$SCRATCH"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc_$USER.sh
    bash /tmp/mc_$USER.sh -b -p "$SCRATCH/miniconda"
    rm -f /tmp/mc_$USER.sh
    export PATH="$SCRATCH/miniconda/bin:$PATH"
fi
# make `conda activate` work inside this non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^orbitsight "; then
    echo "== creating conda env 'orbitsight' (this can take a few minutes) =="
    conda env create -f environment.yml
fi
conda activate orbitsight
echo "== env ready:  python=$(python3 --version 2>&1)"

# ---------------------------------------------------------------------------
# 2. place the dataset on fast local scratch and symlink it into the project
# ---------------------------------------------------------------------------
if [ -L "$PROJ/OrbitSight_Dataset" ] && [ -e "$PROJ/OrbitSight_Dataset" ]; then
    echo "== dataset symlink already present =="
elif [ -d "$SCRATCH/OrbitSight_Dataset" ]; then
    ln -sfn "$SCRATCH/OrbitSight_Dataset" "$PROJ/OrbitSight_Dataset"
    echo "== linked existing dataset from scratch =="
elif [ -d "$PROJ/OrbitSight_Dataset" ]; then
    echo "== moving dataset from home -> scratch (keeps home quota free) =="
    mkdir -p "$SCRATCH"
    mv "$PROJ/OrbitSight_Dataset" "$SCRATCH/OrbitSight_Dataset"
    ln -sfn "$SCRATCH/OrbitSight_Dataset" "$PROJ/OrbitSight_Dataset"
else
    echo "!! DATASET NOT FOUND."
    echo "   Push it from your LAPTOP first, e.g.:"
    echo "   rsync -av OrbitSight_Dataset/ $USER@rolf.ifi.uzh.ch:$SCRATCH/OrbitSight_Dataset/"
    exit 1
fi
echo "   train seqs: $(ls OrbitSight_Dataset/Training_sets/*_labeled_events.npy 2>/dev/null | wc -l),  test seqs: $(ls OrbitSight_Dataset/Testing_sets/*_labeled_events.npy 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
# 3. sanity: GPU + a 1-second forward pass
# ---------------------------------------------------------------------------
python3 - <<'PY'
import torch
print(f"== torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  (CPU!)"))
from orbitsight.evt_centernet import EventCenterNet
m = EventCenterNet(grid=128, patch=8, tbins=3, dim=128, hm_div=2)
dev = "cuda" if torch.cuda.is_available() else "cpu"; m = m.to(dev)
import torch as T
_ = m(T.rand(2, 6, 128, 128, device=dev))
print("== model forward OK on", dev)
PY

# ---------------------------------------------------------------------------
# 4. run the full pipeline if requested
# ---------------------------------------------------------------------------
if [ "$1" = "run" ]; then
    echo "== launching full pipeline (Stage 1: reproduce 0.398, Stage 2: grid-192) =="
    echo "   TIP: run this inside 'screen -S orbit' so it survives disconnects."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log
else
    echo
    echo "SETUP COMPLETE.  To run the pipeline (recommended inside screen):"
    echo "  screen -S orbit"
    echo "  conda activate orbitsight"
    echo "  CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log"
fi
