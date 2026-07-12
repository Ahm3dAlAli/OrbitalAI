#!/bin/bash
# DVX-specialist ensemble: grid-256 models (finer cells for 4-10px DVX objects)
# ensembled for DVX ONLY, then routed with the existing EVK4/DAVIS winners.
#
# Prereqs: predictions/test_xg (cross-grid, best EVK4) and
#          predictions/test_ens_stack (g192 ensemble, best DAVIS) must exist.
set -e
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}; W=${WORKERS:-8}; E=${EPOCHS:-60}; P=${PATIENCE:-10}
STARS=DVX_Filtered_Stars3_2025-01-20-20-22-53
THUR=DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
mkdir -p models predictions

echo "== train grid-256 members (s1 exists; add s2, s3) =="
for S in 2 3; do
    [ -f models/g256_s$S.pt ] || python3 scripts/train_centernet.py --device $DEV --workers $W \
        --grid 256 --patch 8 --dim 128 --tbins 3 --hm-div 2 --augment \
        --epochs $E --patience $P --batch 48 --seed $S --out models/g256_s$S.pt
done

echo "== grid-256 ensemble + TTA on DVX only =="
python3 scripts/infer_ensemble.py --device $DEV --tta --data-dir $T \
    --out-dir predictions/test_dvx256 --sequences $STARS $THUR \
    --models models/g256_s1.pt models/g256_s2.pt models/g256_s3.pt

echo "== stack-merge on DVX =="
python3 scripts/stack_merge.py --data-dir $T --base-dir predictions/test_dvx256 \
    --out-dir predictions/test_dvx256_stack --sequences $STARS $THUR

echo "== route: EVK4->cross-grid, DAVIS->g192-ens, DVX->g256-specialist+stack =="
python3 scripts/route.py --out-dir predictions/router_v2 \
    --map EVK4=predictions/test_xg DAVIS=predictions/test_ens_stack \
          DVX=predictions/test_dvx256_stack
python3 Dataloader/evaluate.py --gt-dir $T \
    --pred-dir predictions/router_v2 --excel-out Evaluation_Metrics_v2.xlsx
echo "== DONE. compare to 0.554 (per-sensor router of ens/cross-grid) =="
