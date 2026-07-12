#!/bin/bash
# OrbitSight — grid-192 experiment only (train + infer + route + score).
# Run on rolf inside an activated 'orbitsight' env, from ~/OrbitalAI:
#   conda activate orbitsight
#   CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_g192.sh 2>&1 | tee g192.log
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

TEST=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; WORKERS=${WORKERS:-8}; BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-40}; PATIENCE=${PATIENCE:-6}
mkdir -p models predictions

echo "== 1/4  train CenterNet grid-192 (aug, early stop) =="
python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
    --grid 192 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
    --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
    --out models/evt_centernet_g192.pt

echo "== 2/4  inference: grid-192 (all test sequences) =="
python3 scripts/infer_centernet.py --device $DEV --data-dir $TEST \
    --model models/evt_centernet_g192.pt --out-dir predictions/test_g192 --thresh 0.3

echo "== 3/4  DAVIS needs the NON-aug model (aug hurt DAVIS) =="
if [ ! -f models/evt_centernet.pt ]; then
    echo "   training non-aug grid-128 for DAVIS..."
    python3 scripts/train_centernet.py --device $DEV --workers $WORKERS \
        --grid 128 --patch 8 --dim 128 --tbins 3 --hm-div 2 \
        --epochs $EPOCHS --patience $PATIENCE --batch $BATCH \
        --out models/evt_centernet.pt
fi
python3 scripts/infer_centernet.py --device $DEV --data-dir $TEST \
    --model models/evt_centernet.pt --out-dir predictions/test_noaug128 --thresh 0.3

echo "== 4/4  route {EVK4,DVX -> g192 ; DAVIS -> non-aug} + score =="
python3 scripts/route.py --out-dir predictions/router_g192 \
    --map EVK4=predictions/test_g192 DVX=predictions/test_g192 \
          DAVIS=predictions/test_noaug128
python3 Dataloader/evaluate.py --gt-dir $TEST \
    --pred-dir predictions/router_g192 --excel-out Evaluation_Metrics_g192.xlsx

echo "== DONE.  grid-192 router mAP printed above; sheet: Evaluation_Metrics_g192.xlsx =="
echo "   Compare to current best (augmented grid-128 router): mAP 0.398"
