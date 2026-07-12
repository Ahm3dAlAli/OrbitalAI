#!/bin/bash
# Cross-grid ensemble campaign (scale diversity: grid-128 + 192 + 256).
# Different grids have different receptive fields, catching different object
# sizes (EVK4's 47px objects vs DVX's 4-10px).  Ensembling across scales +
# TTA + stack is the push from ~0.55 toward 0.62+.
#
# Prereqs: models/g192_s{1,2,3}.pt (from run_ensemble.sh) and
#          models/evt_centernet_aug.pt (the grid-128 aug model) must exist.
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; W=${WORKERS:-8}; E=${EPOCHS:-60}; P=${PATIENCE:-10}
mkdir -p models predictions

echo "== train grid-256 member (finer localization for small DVX objects) =="
[ -f models/g256_s1.pt ] || python3 scripts/train_centernet.py --device $DEV --workers $W \
    --grid 256 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
    --epochs $E --patience $P --batch 48 --seed 1 --out models/g256_s1.pt

# scale-diverse members (all tbins=3 so they share the voxel time-binning)
MODELS="models/evt_centernet_aug.pt \
        models/g192_s1.pt models/g192_s2.pt models/g192_s3.pt \
        models/g256_s1.pt"
# include the bigger g192 members too, if present
for M in models/g192_big_s4.pt models/g192_big_s5.pt; do
    [ -f "$M" ] && MODELS="$MODELS $M"
done
echo "== cross-grid ensemble + TTA over: $MODELS =="
python3 scripts/infer_ensemble.py --device $DEV --tta --data-dir $T \
    --out-dir predictions/test_xg --models $MODELS

echo "== stack-merge on DVX =="
python3 scripts/stack_merge.py --data-dir $T --base-dir predictions/test_xg \
    --out-dir predictions/test_xg_stack \
    --sequences DVX_Filtered_Stars3_2025-01-20-20-22-53 \
                DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43

echo "== score =="
python3 Dataloader/evaluate.py --gt-dir $T \
    --pred-dir predictions/test_xg_stack --excel-out Evaluation_Metrics_xg.xlsx
echo "== DONE.  compare to 0.547 (5-model same-grid was the prior step) =="
