# OrbitSight — reproducible inference container for the winning deep pipeline
# (multi-window temporal-context CenterNet + TTA, per-sensor routed).
#
# CPU-only by design: the pipeline is measured real-time on CPU (~15-17 ms/window
# end-to-end, well under the 40 ms budget — see ROADMAP.md §4), so the image runs
# anywhere with no CUDA/nvidia-docker dependency.  A GPU variant is one line away
# (see the note at the bottom).
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
# Reproduce the exact 0.675 per-sensor router (needs the cross-grid EVK4 models
# baked into models/):
#   docker run --rm -v .../dataset:/OrbitSight_dataset:ro -v .../out:/work \
#     -e ORBITSIGHT_MODELS="models/g192_ctx.pt models/g192_ctx_s2.pt" \
#     -e ORBITSIGHT_EVK4_MODELS="models/g128_xg.pt models/g192_xg.pt" \
#     orbitsight
#
# Run the classical LightGBM pipeline instead (no torch path):
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

# Guard: the 0.675 temporal-context checkpoint must be baked in.
# Copy models/g192_ctx.pt from rolf before building.
RUN test -f models/g192_ctx.pt || \
    { echo "ERROR: models/g192_ctx.pt missing — copy the 0.675 checkpoint into models/ first."; exit 1; }

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
