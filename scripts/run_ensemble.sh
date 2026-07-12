#!/bin/bash
# OrbitSight — ensemble + TTA campaign (the big push).
# Trains N diverse grid-192 models (different seeds), then does ensemble + TTA
# inference over ALL sensors, stack-merges DVX, and scores.
#
# Ensembling + TTA are the "free" accuracy levers we hadn't used; a 2-model
# ensemble+TTA already lifted DAVIS 0.411 -> 0.581 in a local test.
#
# Run on rolf (inside screen).  N models train sequentially here (~15 min each
# on a 2080 Ti); to parallelize, launch each train_centernet call in its own
# screen on a different GPU, then run the inference block.
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; W=${WORKERS:-8}; B=${BATCH:-128}
E=${EPOCHS:-60}; P=${PATIENCE:-10}
NSEED=${NSEED:-3}
mkdir -p models predictions

echo "== train $NSEED diverse grid-192 models (seeds 1..$NSEED) =="
MODELS=""
for S in $(seq 1 $NSEED); do
    OUT=models/g192_s$S.pt
    if [ ! -f "$OUT" ]; then
        python3 scripts/train_centernet.py --device $DEV --workers $W \
            --grid 192 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
            --epochs $E --patience $P --batch $B --seed $S --out $OUT
    fi
    MODELS="$MODELS $OUT"
done

echo "== ensemble + TTA inference over all sensors =="
python3 scripts/infer_ensemble.py --device $DEV --tta --data-dir $T \
    --out-dir predictions/test_ens --models $MODELS

echo "== stack-merge on DVX (extra dim recall) =="
python3 scripts/stack_merge.py --data-dir $T --base-dir predictions/test_ens \
    --out-dir predictions/test_ens_stack \
    --sequences DVX_Filtered_Stars3_2025-01-20-20-22-53 \
                DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43

echo "== score (single ensemble serves all 4 sensors) =="
python3 Dataloader/evaluate.py --gt-dir $T \
    --pred-dir predictions/test_ens_stack --excel-out Evaluation_Metrics_ensemble.xlsx

echo "== DONE. compare mAP above to current best 0.454 =="
