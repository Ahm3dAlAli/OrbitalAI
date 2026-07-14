# OrbitSight — reproducible inference container for the deployed real-time pipeline.
#
# Deployed per-sensor pipeline (mAP 0.692 real-time, all sensors < 40 ms CPU):
#   EVK4          -> temporal-context CenterNet (g192_ctx)
#   DAVIS, Stars3 -> grid-256 CenterNet w/ hard-negative mining (g256_hn)
#   Thuraya3      -> temporal model + coasting Kalman recall recovery
# CPU-only by design: measured 15-38 ms/window end-to-end, so the image runs
# anywhere with no CUDA/nvidia-docker dependency.  (Offline max-accuracy 0.709 uses
# cross-grid ensembling + TTA at ~211 ms — not the real-time path.)
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

# Guard: the deployed checkpoints must be baked in. g192_ctx is required (EVK4 +
# Thuraya3 base); g256_hn drives DAVIS/Stars3 (0.692) — the pipeline degrades to the
# temporal model for those sensors if it is absent, so only g192_ctx is hard-required.
RUN test -f models/g192_ctx.pt || \
    { echo "ERROR: models/g192_ctx.pt missing — bake the temporal checkpoint into models/."; exit 1; }
RUN test -f models/g256_hn.pt || \
    echo "WARNING: models/g256_hn.pt absent — DAVIS/Stars3 fall back to g192_ctx (mAP ~0.66, not 0.692)."

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
