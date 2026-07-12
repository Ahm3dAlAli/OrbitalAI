#!/bin/bash
# Run AFTER the training screens finish.  Infers every checkpoint that exists,
# builds the aug-128 and grid-192 routers, and scores both through the frozen
# evaluator.  DAVIS is served by the non-aug model (models/evt_centernet.pt).
#
#   conda activate orbitsight && cd ~/OrbitalAI
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_router.sh 2>&1 | tee eval.log
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
T=OrbitSight_Dataset/Testing_sets
DEV=${DEV:-cuda}

echo "== inference =="
for M in evt_centernet_aug evt_centernet_g192 evt_centernet; do
    if [ -f models/$M.pt ]; then
        python3 scripts/infer_centernet.py --device $DEV --data-dir $T \
            --model models/$M.pt --out-dir predictions/test_$M
    else
        echo "   (skip: models/$M.pt not found)"
    fi
done

score () {  # $1=aug|g192  $2=cnet-dir-for-EVK4/DVX
    python3 scripts/route.py --out-dir predictions/router_$1 \
        --map EVK4=$2 DVX=$2 DAVIS=predictions/test_evt_centernet >/dev/null
    echo; echo "############## ROUTER: $1 ##############"
    python3 Dataloader/evaluate.py --gt-dir $T \
        --pred-dir predictions/router_$1 --excel-out Evaluation_Metrics_$1.xlsx
}

[ -d predictions/test_evt_centernet_aug ]  && score aug  predictions/test_evt_centernet_aug
[ -d predictions/test_evt_centernet_g192 ] && score g192 predictions/test_evt_centernet_g192

echo; echo "== done. sheets: Evaluation_Metrics_aug.xlsx / _g192.xlsx =="
