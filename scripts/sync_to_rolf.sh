#!/bin/bash
# Sync OrbitSight TO rolf.  RUN FROM YOUR LAPTOP, from the project root.
#
#   bash scripts/sync_to_rolf.sh            # full: code + models + dataset
#   CODE_ONLY=1 bash scripts/sync_to_rolf.sh   # code + models only (skip 6 GB dataset)
#
# Authenticates ONCE via a shared SSH master connection (multiplexing), so rolf's
# 2FA/OTP prompts a single time for the whole sync — not once per transfer.
set -e

HOST=${1:-rolf}
USER=${ROLF_USER:-alali}
SCRATCH=/local/scratch/$USER
REMOTE=OrbitalAI                                  # relative to remote home (no ~)

# ---- one shared, authenticated SSH connection -> single OTP prompt --------- #
CM="${TMPDIR:-/tmp}/rolf-cm-to-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"
cleanup() { ssh -o ControlPath="$CM" -O exit "$HOST" 2>/dev/null || true; }
trap cleanup EXIT

echo "== authenticate ONCE (password + OTP) =="
ssh $SSH_OPTS "$HOST" "mkdir -p $REMOTE && echo connected" \
    || { echo "[err] could not connect to $HOST"; exit 1; }
rs() { rsync -av --progress -e "ssh -o ControlPath=$CM" "$@"; }

echo "== 1  code -> ~/$REMOTE =="
rs --exclude 'OrbitSight_Dataset' --exclude 'Dataloader/output' --exclude 'predictions' \
   --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude '.claude' \
   --exclude 'logs' --exclude '*.tar' --exclude '*.tar.gz' \
   ./ "$HOST:$REMOTE/"

echo "== 2  models -> ~/$REMOTE/models =="
rs models/ "$HOST:$REMOTE/models/"

if [ -n "$CODE_ONLY" ]; then
    echo "== CODE_ONLY set -> skipping the ~6 GB dataset (already on rolf) =="
else
    echo "== 3  dataset (~6 GB) -> $SCRATCH (fast local scratch) =="
    ssh -o ControlPath="$CM" "$HOST" \
        "mkdir -p $SCRATCH/OrbitSight_Dataset && chgrp aiml $SCRATCH 2>/dev/null || true"
    rs OrbitSight_Dataset/ "$HOST:$SCRATCH/OrbitSight_Dataset/"
    echo "== 4  symlink dataset into the project =="
    ssh -o ControlPath="$CM" "$HOST" \
        "cd $REMOTE && ln -sfn $SCRATCH/OrbitSight_Dataset OrbitSight_Dataset && ls -l OrbitSight_Dataset"
fi

echo
echo "SYNC DONE (single authentication).  Next on rolf:"
echo "  cd ~/OrbitalAI && conda activate orbitsight"
echo "  bash scripts/run_100ep.sh          # launch the 100-epoch trainings"
