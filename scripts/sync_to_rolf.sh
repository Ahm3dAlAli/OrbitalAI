#!/bin/bash
# Sync OrbitSight to rolf.  RUN FROM YOUR LAPTOP, from the project root:
#     bash scripts/sync_to_rolf.sh
#
# Uses your ~/.ssh/config 'rolf' alias (User alali).  If passwordless SSH is
# set up (see below) it runs unattended; otherwise you'll enter your password
# a few times.
#
# ONE-TIME passwordless setup (recommended, do once in your terminal):
#     ssh-copy-id rolf        # enter your rolf password once
# after that this script (and rsync) run without prompting.
set -e

HOST=${1:-rolf}                                   # ssh alias or user@host
USER=${ROLF_USER:-alali}
SCRATCH=/local/scratch/$USER

echo "== 1/4  code -> ~/OrbitalAI (home, backed up) =="
rsync -av --progress \
    --exclude 'OrbitSight_Dataset' \
    --exclude 'Dataloader/output' \
    --exclude 'predictions' \
    --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.git' --exclude '.claude' \
    --exclude '*.tar' --exclude '*.tar.gz' \
    ./  "$HOST:~/OrbitalAI/"

echo "== 2/4  models (17 MB) -> ~/OrbitalAI/models (lets you skip Stage-1 training) =="
rsync -av --progress models/  "$HOST:~/OrbitalAI/models/"

echo "== 3/4  dataset (~6 GB) -> $SCRATCH (fast local scratch, NOT home) =="
ssh "$HOST" "mkdir -p $SCRATCH/OrbitSight_Dataset && chgrp aiml $SCRATCH 2>/dev/null || true"
rsync -av --progress OrbitSight_Dataset/  "$HOST:$SCRATCH/OrbitSight_Dataset/"

echo "== 4/4  symlink data into the project on rolf =="
ssh "$HOST" "cd ~/OrbitalAI && ln -sfn $SCRATCH/OrbitSight_Dataset OrbitSight_Dataset && ls -l OrbitSight_Dataset"

echo
echo "SYNC DONE.  Next on rolf:"
echo "  ssh $HOST"
echo "  cd ~/OrbitalAI && conda env create -f environment.yml && conda activate orbitsight"
echo "  screen -S orbit"
echo "  CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log"
