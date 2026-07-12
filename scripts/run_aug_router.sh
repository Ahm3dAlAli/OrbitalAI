#!/bin/bash
# Augmented per-sensor router — retrain longer + early stopping, then score.
# Trains BOTH CenterNet checkpoints the router needs (aug for EVK4/DVX, no-aug
# for DAVIS), runs inference, builds the router, and evaluates.
#
# Defaults: 50 epochs, patience 10 (override with EPOCHS=/PATIENCE= env vars).
#
# On rolf, in its own screen:
#   screen -S augrouter
#   conda activate orbitsight && cd ~/OrbitalAI
#   CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_aug_router.sh 2>&1 | tee augrouter.log
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; WORKERS=${WORKERS:-8}; BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-50}; PATIENCE=${PATIENCE:-10}
mkdir -p models predictions
echo "== config: grid-128  epochs=$EPOCHS patience=$PATIENCE batch=$BATCH dev=$DEV =="

echo "== 1/5  train AUGMENTED grid-128 (EVK4/DVX) =="
python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
    --grid 128 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
    --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
    --out models/evt_centernet_aug.pt

echo "== 2/5  train NON-AUG grid-128 (DAVIS) =="
python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
    --grid 128 --patch 8 --dim 128 --tbins 3 --hm-div 2 \
    --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
    --out models/evt_centernet.pt

echo "== 3/5  inference: augmented model =="
python3 scripts/infer_centernet.py --device $DEV --data-dir $T \
    --model models/evt_centernet_aug.pt --out-dir predictions/test_aug --thresh 0.3
echo "== 4/5  inference: non-aug model =="
python3 scripts/infer_centernet.py --device $DEV --data-dir $T \
    --model models/evt_centernet.pt --out-dir predictions/test_noaug --thresh 0.3

echo "== 5/5  route {EVK4,DVX -> aug ; DAVIS -> non-aug} + score =="
python3 scripts/route.py --out-dir predictions/router_aug \
    --map EVK4=predictions/test_aug DVX=predictions/test_aug DAVIS=predictions/test_noaug
python3 Dataloader/evaluate.py --gt-dir $T \
    --pred-dir predictions/router_aug --excel-out Evaluation_Metrics_aug.xlsx

echo "== DONE.  mAP above; sheet: Evaluation_Metrics_aug.xlsx.  (current best: 0.398) =="
