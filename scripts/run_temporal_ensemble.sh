#!/bin/bash
# Temporal ENSEMBLE — the push from 0.66 toward 0.7.
# Trains extra temporal-context members (more seeds + a wider context=5), then
# ensembles them with TTA and routes (EVK4 -> cross-grid, DAVIS/DVX -> temporal).
#
# Prereqs: models/g192_ctx.pt (context=3, from run_temporal.sh) and
#          predictions/test_xg (cross-grid, best EVK4).
#
# On rolf, pinned to a FREE gpu, in a screen:
#   screen -S tens
#   CUDA_VISIBLE_DEVICES=<free> nice -n 15 bash scripts/run_temporal_ensemble.sh 2>&1 | tee tens.log
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; W=${WORKERS:-8}; E=${EPOCHS:-80}; P=${PATIENCE:-12}
STARS=DVX_Filtered_Stars3_2025-01-20-20-22-53
THUR=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
mkdir -p models predictions

echo "== train temporal members: 2 more context=3 seeds + 1 context=5 =="
[ -f models/g192_ctx_s2.pt ] || python3 scripts/train_centernet.py --device $DEV --workers $W \
    --grid 192 --dim 128 --tbins 7  --context 3 --hm-div 2 --augment \
    --epochs $E --patience $P --batch 96 --seed 2 --out models/g192_ctx_s2.pt
[ -f models/g192_ctx_s3.pt ] || python3 scripts/train_centernet.py --device $DEV --workers $W \
    --grid 192 --dim 128 --tbins 7  --context 3 --hm-div 2 --augment \
    --epochs $E --patience $P --batch 96 --seed 3 --out models/g192_ctx_s3.pt
[ -f models/g192_ctx5.pt ] || python3 scripts/train_centernet.py --device $DEV --workers $W \
    --grid 192 --dim 128 --tbins 11 --context 5 --hm-div 2 --augment \
    --epochs $E --patience $P --batch 64 --seed 1 --out models/g192_ctx5.pt

# all share grid=192 (context differs -> handled per-model in the voxelizer)
CTX_MODELS="models/g192_ctx.pt models/g192_ctx_s2.pt models/g192_ctx_s3.pt models/g192_ctx5.pt"

echo "== temporal ensemble + TTA on DAVIS + DVX =="
python3 scripts/infer_ensemble.py --device $DEV --tta --data-dir $T \
    --out-dir predictions/test_tens --models $CTX_MODELS \
    --sequences DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37 $STARS $THUR

echo "== route: EVK4 -> cross-grid, DAVIS/DVX -> temporal ensemble =="
python3 scripts/route.py --out-dir predictions/router_tens \
    --map EVK4=predictions/test_xg DAVIS=predictions/test_tens DVX=predictions/test_tens
python3 Dataloader/evaluate.py --gt-dir $T \
    --pred-dir predictions/router_tens --excel-out Evaluation_Metrics_tens.xlsx
echo "== DONE. compare to current best 0.660 =="
