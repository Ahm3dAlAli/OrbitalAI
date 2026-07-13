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
# Authenticates ONCE via an SSH master connection (so rolf's 2FA/OTP prompts a
# single time, not once per transfer), then reuses it for every rsync.
set -e

HOST=${1:-rolf}
# Path is RELATIVE to the remote home (do NOT use ~ — the local shell expands it
# to your laptop's home before it ever reaches rolf).
REMOTE=${ROLF_DIR:-OrbitalAI}

# ---- one shared, authenticated SSH connection (multiplexing) --------------- #
CM="${TMPDIR:-/tmp}/rolf-cm-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"
cleanup() { ssh -o ControlPath="$CM" -O exit "$HOST" 2>/dev/null || true; }
trap cleanup EXIT

echo "== authenticate ONCE (password + OTP) to open a shared connection =="
ssh $SSH_OPTS "$HOST" "cd $REMOTE && echo connected; pwd" || {
    echo "[ERR] could not connect / cd into ~/$REMOTE on $HOST"; exit 1; }
rs() { rsync -av --progress -e "ssh -o ControlPath=$CM" "$@"; }

echo "== 1/4  models/*.pt  (trained checkpoints, incl. winning temporal model) =="
mkdir -p models
rs --include '*/' --include '*.pt' --include '*.joblib' --exclude '*' \
   "$HOST:$REMOTE/models/" models/ || echo "  (no models dir on rolf?)"

echo "== 2/4  predictions/  (ALL dirs — the full over-time progression) =="
mkdir -p predictions
rs "$HOST:$REMOTE/predictions/" predictions/ || echo "  (no predictions dir on rolf?)"

echo "== 3/4  *.xlsx  (every scoring sheet) =="
rs --include '*.xlsx' --exclude '*' "$HOST:$REMOTE/" ./ || true

echo "== 4/4  logs  (training curves: *.log) =="
mkdir -p logs
rs --include '*.log' --exclude '*' "$HOST:$REMOTE/" logs/ || true

echo
echo "SYNC-BACK DONE.  What came home:"
echo "  models:      $(ls models/*.pt 2>/dev/null | wc -l | tr -d ' ') checkpoints"
echo "  predictions: $(ls -d predictions/*/ 2>/dev/null | wc -l | tr -d ' ') dirs"
echo "  xlsx:        $(ls *.xlsx 2>/dev/null | wc -l | tr -d ' ') sheets"
echo "  logs:        $(ls logs/*.log 2>/dev/null | wc -l | tr -d ' ') logs"
echo
echo "Next:"
echo "  python3 scripts/score_all_predictions.py     # rank every run, find the winner"
echo "  bash scripts/run_improve.sh                  # apply the 0.8 levers + evaluate"
