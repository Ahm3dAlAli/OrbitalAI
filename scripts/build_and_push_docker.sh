#!/bin/bash
# Build the OrbitSight submission image with the checkpoints currently in ./models/
# and push it to Docker Hub. Verifies the deployed checkpoints are present first so
# you never ship an image that silently fell back to a weaker model.
#
#   bash scripts/build_and_push_docker.sh              # build + push :0.721 and :latest
#   TAG=0.72 bash scripts/build_and_push_docker.sh     # custom version tag
#   NO_PUSH=1 bash scripts/build_and_push_docker.sh    # build + verify only, no push
set -eu
cd "$(dirname "$0")/.."

REPO="${REPO:-ahm3dalali/orbitsight}"
TAG="${TAG:-0.721}"

# The checkpoints the deployed 0.721 router actually uses.
REQUIRED=(models/g192_ctx_r3.pt models/g256_hn_iou.pt models/g192_ctx_v2.pt)

echo "== verifying deployed checkpoints in ./models/ =="
missing=0
for m in "${REQUIRED[@]}"; do
    if [ -f "$m" ]; then
        printf "  [ok ] %-28s %s\n" "$(basename "$m")" "$(du -h "$m" | cut -f1)"
    else
        printf "  [MISSING] %s\n" "$m"; missing=1
    fi
done
if [ "$missing" = 1 ]; then
    echo "[err] a deployed checkpoint is missing — sync it from rolf first:"
    echo "      bash scripts/sync_from_rolf.sh"
    echo "      (or targeted: rsync the specific .pt into ./models/)"
    exit 1
fi

echo ""
echo "== docker build -t $REPO:$TAG -t $REPO:latest . =="
docker build -t "$REPO:$TAG" -t "$REPO:latest" .

if [ "${NO_PUSH:-}" = 1 ]; then
    echo ""; echo "== NO_PUSH set -> built but not pushed =="
    echo "   test:  docker run --rm -e ORBITSIGHT_DATASET=/OrbitSight_dataset/Testing_sets \\"
    echo "            -v \$PWD/OrbitSight_Dataset:/OrbitSight_dataset:ro -v \$PWD/docker_out:/work $REPO:latest"
    exit 0
fi

echo ""
echo "== pushing to Docker Hub (must be 'docker login'ed as the repo owner) =="
docker push "$REPO:$TAG"
docker push "$REPO:latest"
echo ""
echo "== done -> https://hub.docker.com/r/${REPO} (tags: $TAG, latest) =="
