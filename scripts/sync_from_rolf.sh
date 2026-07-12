#!/bin/bash
# Pull ALL results BACK from rolf -> your laptop.  RUN FROM YOUR LAPTOP, project root:
#     bash scripts/sync_from_rolf.sh
#
# Brings home everything needed to reproduce the winning pipeline and every
# result we produced over time:
#   * models/*.pt              trained checkpoints (incl. the winning g192_ctx.pt)
#   * predictions/             ALL prediction dirs (the full progression + router_ctta)
#   * *.xlsx                    every Evaluation_Metrics scoring sheet
#   * logs/*.log               training logs (for training-curve figures)
#
# Then regenerate locally:
#   PRED_DIR=predictions/router_ctta bash scripts/build_all.sh
#
# Uses your ~/.ssh/config 'rolf' alias (User alali).  Passwordless setup:
#   ssh-copy-id rolf     # once, then this runs unattended
set -e

HOST=${1:-rolf}
REMOTE=${ROLF_DIR:-~/OrbitalAI}

echo "== 1/4  models/*.pt  (trained checkpoints, incl. winning temporal model) =="
mkdir -p models
rsync -av --progress --include '*/' --include '*.pt' --include '*.joblib' \
    --exclude '*' "$HOST:$REMOTE/models/" models/

echo "== 2/4  predictions/  (ALL dirs — the full over-time progression) =="
mkdir -p predictions
rsync -av --progress "$HOST:$REMOTE/predictions/" predictions/

echo "== 3/4  *.xlsx  (every scoring sheet) =="
rsync -av --progress --include '*.xlsx' --exclude '*' "$HOST:$REMOTE/" ./ || true

echo "== 4/4  logs  (training curves: *.log) =="
mkdir -p logs
rsync -av --progress --include '*.log' --exclude '*' "$HOST:$REMOTE/" logs/ || true

echo
echo "SYNC-BACK DONE.  What came home:"
echo "  models:      $(ls models/*.pt 2>/dev/null | wc -l | tr -d ' ') checkpoints"
echo "  predictions: $(ls -d predictions/*/ 2>/dev/null | wc -l | tr -d ' ') dirs"
echo "  xlsx:        $(ls *.xlsx 2>/dev/null | wc -l | tr -d ' ') sheets"
echo "  logs:        $(ls logs/*.log 2>/dev/null | wc -l | tr -d ' ') logs"
echo
echo "Now find the winning dir and regenerate every asset from it:"
echo "  python3 scripts/score_all_predictions.py        # ranks every pred dir by mAP"
echo "  PRED_DIR=predictions/<best-dir> bash scripts/build_all.sh"
