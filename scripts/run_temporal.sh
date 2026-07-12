#!/bin/bash
# Multi-window TEMPORAL-CONTEXT CenterNet — the structural lever for dim recall.
# Feeds the model +/- CONTEXT windows (~280 ms) as extra time-bins, so it sees
# the object's TRACK (many windows of evidence) instead of one 40 ms slice.
# This is the change with a real path to 0.65-0.7 on Thuraya3/Stars3.
#
# On rolf, pinned to a FREE gpu, in a screen:
#   screen -S ctx
#   CUDA_VISIBLE_DEVICES=<free> nice -n 15 bash scripts/run_temporal.sh 2>&1 | tee ctx.log
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; W=${WORKERS:-8}; B=${BATCH:-96}
E=${EPOCHS:-80}; P=${PATIENCE:-12}; CTX=${CONTEXT:-3}; TB=${TBINS:-7}
mkdir -p models predictions

echo "== train temporal model: context=+/-$CTX windows, tbins=$TB, grid 192 =="
[ -f models/g192_ctx.pt ] || python3 scripts/train_centernet.py --device $DEV --workers $W \
    --grid 192 --patch 8 --dim 128 --tbins $TB --context $CTX --hm-div 2 --augment \
    --epochs $E --patience $P --batch $B --seed 1 --out models/g192_ctx.pt

echo "== inference (context applied automatically from the checkpoint cfg) =="
python3 scripts/infer_centernet.py --device $DEV --data-dir $T \
    --model models/g192_ctx.pt --out-dir predictions/test_ctx --thresh 0.3

echo "== STANDALONE score — does temporal context lift DVX vs the g192 baseline? =="
python3 Dataloader/evaluate.py --gt-dir $T --pred-dir predictions/test_ctx \
    --excel-out Evaluation_Metrics_ctx.xlsx

echo "== stack-merge DVX + route {EVK4->cross-grid, DAVIS->g192-ens, DVX->ctx+stack} =="
python3 scripts/stack_merge.py --data-dir $T --base-dir predictions/test_ctx \
    --out-dir predictions/test_ctx_stack \
    --sequences DVX_Filtered_Stars3_2025-01-20-20-22-53 \
                DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
python3 scripts/route.py --out-dir predictions/router_ctx \
    --map EVK4=predictions/test_xg DAVIS=predictions/test_ens_stack \
          DVX=predictions/test_ctx_stack
python3 Dataloader/evaluate.py --gt-dir $T \
    --pred-dir predictions/router_ctx --excel-out Evaluation_Metrics_ctx_router.xlsx
echo "== DONE. compare the DVX rows + overall mAP to the current best 0.554 =="
