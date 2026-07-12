#!/bin/bash
# DVX-lever ablation — isolate reweighting vs dim-augmentation vs both.
#
# Trains three grid-192 / context-3 variants (same architecture, only the DVX
# levers change), then infers + scores each on the two DVX sequences with a
# common decode (topk=2), and prints an ablation table against the baseline
# g192_ctx (Stars3 0.545 / Thuraya3 0.469).
#
# Run on rolf, in a screen, pinned to a free GPU:
#   cd ~/OrbitalAI && screen -S dvxabl
#   CUDA_VISIBLE_DEVICES=<free> nice -n 15 bash scripts/run_dvx_ablation.sh 2>&1 | tee dvxabl.log
#
# FASTER (parallel): launch the three train commands below on separate GPUs
# yourself, wait for the .pt files, then re-run this script — it skips trained
# models and jumps straight to scoring.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=${DATA_DIR:-OrbitSight_Dataset/Testing_sets}
DEV=${DEVICE:-cuda}; E=${EPOCHS:-80}; P=${PATIENCE:-12}; B=${BATCH:-64}
STARS=DVX_Filtered_Stars3_2025-01-20-20-22-53
THUR=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
COMMON="--device $DEV --workers 8 --grid 192 --patch 8 --dim 128 --tbins 7 \
  --context 3 --hm-div 2 --augment --epochs $E --patience $P --batch $B --seed 1"
RW="--dvx-weight 3.0 --evk4-weight 0.5 --dim-weight 0.5"

echo "== train ablation variants (skip if already present) =="
[ -f models/g192_rw.pt ]     || python3 scripts/train_centernet.py $COMMON $RW           --out models/g192_rw.pt
[ -f models/g192_dimaug.pt ] || python3 scripts/train_centernet.py $COMMON --dim-aug      --out models/g192_dimaug.pt
[ -f models/g192_dvx.pt ]    || python3 scripts/train_centernet.py $COMMON --dim-aug $RW  --out models/g192_dvx.pt

echo
echo "== DVX-lever ablation (AP@0.5, topk=2 decode) =="
printf "%-26s %-9s %-9s\n" "variant (grid192,ctx3)" "Stars3" "Thuraya3"
printf "%-26s %-9s %-9s\n" "--------------------------" "------" "--------"
score () { # $1 = model name (without models/ and .pt)
  local m="models/$1.pt"; [ -f "$m" ] || { printf "%-26s %-9s\n" "$1" "(missing)"; return; }
  python3 scripts/infer_ensemble.py --device $DEV --topk 2 --thresh 0.2 --data-dir "$T" \
    --out-dir "predictions/abl_$1" --models "$m" --sequences $STARS $THUR >/dev/null 2>&1
  local out=$(python3 Dataloader/evaluate.py --gt-dir "$T" --pred-dir "predictions/abl_$1" 2>/dev/null)
  local s=$(echo "$out"  | grep Stars3   | grep -oE "0\.[0-9]+" | tail -1)
  local th=$(echo "$out" | grep Thuraya3 | grep -oE "0\.[0-9]+" | tail -1)
  printf "%-26s %-9s %-9s\n" "$1" "$s" "$th"
}
score g192_ctx        # baseline: no reweight, no dim-aug
score g192_rw         # + DVX reweight only
score g192_dimaug     # + dim-aug only
score g192_dvx        # + both (full DVX specialist)
echo
echo "baseline reference (topk=1): Stars3 0.545  Thuraya3 0.469"
echo "if a variant wins, route it into DVX and re-score the full 4-sequence mAP."
