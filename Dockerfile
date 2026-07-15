# OrbitSight — reproducible inference container for the deployed real-time pipeline.
#
# Deployed per-sensor pipeline (mAP 0.704 real-time, all sensors < 40 ms CPU):
#   EVK4          -> temporal-context CenterNet (g192_ctx_v2, 100-epoch)
#   DAVIS, Stars3 -> grid-256 CenterNet w/ hard-negative mining (g256_hn)
#   Thuraya3      -> g192_ctx_v2 + coasting Kalman recall recovery
# The 100-epoch g192_ctx_v2 checkpoint drives the 0.704 result (EVK4 0.859->0.874,
# Thuraya3 raw 0.469->0.524, coasted->0.538); the entrypoint falls back to g192_ctx
# if v2 is not baked in. CPU-only by design: measured 15-38 ms/window end-to-end, so
# the image runs anywhere with no CUDA/nvidia-docker dependency.  (Offline max-accuracy
# 0.715 uses cross-grid ensembling + TTA at ~211 ms — not the real-time path.)
#
# Build:
#   docker build -t orbitsight .
#
# Run (challenge mounts):
#   docker run --rm \
#     -v /path/to/OrbitSight_dataset:/OrbitSight_dataset:ro \
#     -v /path/to/output:/work \
#     orbitsight
#
# The Stars3/DAVIS grid-256 model and Thuraya3 coasting are on by default; override
# via ORBITSIGHT_G256_MODEL / ORBITSIGHT_COAST if needed. Classical LightGBM path:
#   docker run --rm ... orbitsight sh run.sh
FROM python:3.11-slim

# System libs: libgomp1 for lightgbm/torch OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Pinned base deps (numpy/scipy/lightgbm/pillow/openpyxl/...) — layer-cached.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) CPU build of PyTorch (training env requires >=2.1; no CUDA).
#    Pin to a specific version once the exact training wheel is confirmed;
#    the environment.yml floor is >=2.1 so any current stable build works.
RUN pip install --no-cache-dir "torch>=2.1" \
        --index-url https://download.pytorch.org/whl/cpu

# 3) Application: package, scripts, trained checkpoints, entrypoints.
COPY orbitsight/ ./orbitsight/
COPY scripts/    ./scripts/
COPY models/     ./models/
COPY run.sh run_infer.sh ./
RUN chmod +x run.sh run_infer.sh

# Guard: the deployed checkpoints must be baked in. A temporal checkpoint is required
# (EVK4 + Thuraya3 base) — g192_ctx_v2 (100-epoch, the 0.704 model) preferred, g192_ctx
# accepted as fallback. g256_hn drives DAVIS/Stars3; the pipeline degrades to the
# temporal model for those sensors if it is absent.
RUN test -f models/g192_ctx_v2.pt -o -f models/g192_ctx.pt || \
    { echo "ERROR: no temporal checkpoint (g192_ctx_v2.pt or g192_ctx.pt) in models/."; exit 1; }
RUN test -f models/g192_ctx_v2.pt || \
    echo "WARNING: models/g192_ctx_v2.pt absent — using g192_ctx (mAP ~0.692, not 0.704). Sync the 100-epoch checkpoint for the deployed result."
RUN test -f models/g256_hn.pt || \
    echo "WARNING: models/g256_hn.pt absent — DAVIS/Stars3 fall back to the temporal model (mAP ~0.66, not 0.704)."

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1 \
    ORBITSIGHT_DATASET=/OrbitSight_dataset \
    ORBITSIGHT_DEVICE=cpu

# Default: the winning deep pipeline. Finishes on its own (offline).
CMD ["sh", "run_infer.sh"]

# ---------------------------------------------------------------------------
# GPU variant (faster eval hardware): swap the base image and the torch install:
#   FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#   ... install python3.11 ...
#   RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
#   ENV ORBITSIGHT_DEVICE=cuda
#   run with:  docker run --gpus all ...
# ---------------------------------------------------------------------------
