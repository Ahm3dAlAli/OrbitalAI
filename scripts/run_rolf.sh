#!/bin/bash
# OrbitSight — full GPU pipeline for rolf.
#
#   Stage 1: reproduce the augmented router  (target mAP ~= 0.398)
#   Stage 2: grid-192 follow-up              (finer localization for Stars3/DAVIS)
#
# Each CenterNet is trained with a val split + EARLY STOPPING (--epochs 40
# --patience 6), so the old 16-epoch CPU cap is gone.  On an RTX 2080 Ti an
# epoch is seconds.
#
# Run:
#   screen -S orbit ; conda activate orbitsight ; cd ~/OrbitalAI
#   CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log
set -e
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

DATA=OrbitSight_Dataset
TRAIN=$DATA/Training_sets
TEST=$DATA/Testing_sets
DEV=${DEV:-cuda}
WORKERS=${WORKERS:-8}
BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-40}
PATIENCE=${PATIENCE:-6}
EVAL="python3 Dataloader/evaluate.py --gt-dir $TEST"

mkdir -p models predictions

banner () { echo; echo "=================================================="; echo "  $1"; echo "=================================================="; }

# =====================================================================
# STAGE 1 — reproduce the augmented router (mAP ~ 0.398)
# =====================================================================
banner "STAGE 1  train CenterNet: aug + no-aug (grid 128, early stop)"

python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
    --grid 128 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
    --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
    --out models/evt_centernet_aug.pt

python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
    --grid 128 --patch 8 --dim 128 --tbins 3 --hm-div 2 \
    --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
    --out models/evt_centernet.pt

banner "STAGE 1  inference (GPU)"
python3 scripts/infer_centernet.py --device $DEV --data-dir $TEST \
    --model models/evt_centernet_aug.pt --out-dir predictions/test_aug128 --thresh 0.3
python3 scripts/infer_centernet.py --device $DEV --data-dir $TEST \
    --model models/evt_centernet.pt    --out-dir predictions/test_noaug128 --thresh 0.3

banner "STAGE 1  router  {EVK4,DVX -> aug ; DAVIS -> no-aug}  => target mAP ~0.398"
python3 scripts/route.py --out-dir predictions/router_398 \
    --map EVK4=predictions/test_aug128 DVX=predictions/test_aug128 \
          DAVIS=predictions/test_noaug128
$EVAL --pred-dir predictions/router_398 --excel-out Evaluation_Metrics_stage1.xlsx

# =====================================================================
# STAGE 2 — grid-192 follow-up (finer localization)
# =====================================================================
banner "STAGE 2  train CenterNet grid-192 (aug, early stop)"
python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
    --grid 192 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
    --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
    --out models/evt_centernet_g192.pt

banner "STAGE 2  inference (GPU)"
python3 scripts/infer_centernet.py --device $DEV --data-dir $TEST \
    --model models/evt_centernet_g192.pt --out-dir predictions/test_g192 --thresh 0.3

banner "STAGE 2  router  {EVK4,DVX -> g192 ; DAVIS -> no-aug g128}"
python3 scripts/route.py --out-dir predictions/router_g192 \
    --map EVK4=predictions/test_g192 DVX=predictions/test_g192 \
          DAVIS=predictions/test_noaug128
$EVAL --pred-dir predictions/router_g192 --excel-out Evaluation_Metrics.xlsx

banner "DONE  — compare Stage-1 (0.398 repro) vs Stage-2 (grid-192) mAP above"
echo "Stage 1 sheet: Evaluation_Metrics_stage1.xlsx"
echo "Stage 2 sheet: Evaluation_Metrics.xlsx  (grid-192 router = final)"
