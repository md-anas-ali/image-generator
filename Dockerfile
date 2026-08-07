# Tiny, CPU-only image-generation server using a distilled Stable Diffusion
# UNet (bk-sdm-tiny) + stock CLIP text encoder + stock VAE, all int8
# quantized. No CUDA, no torch, no transformers/optimum at runtime.
# Realistic memory floor is ~570MB (model weights + runtime import
# overhead) — see app/server.py module docstring and the README's
# "Realistic memory budget" section for the honest numbers and why a
# strict 430-460MB ceiling isn't achievable with a real diffusion model.
FROM python:3.11-slim AS base

# Keep the image itself small too — fewer cached layers/files means less
# chance of page-cache pressure competing with the app's RSS budget.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ORT_NUM_THREADS=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app/ ./app/

# Model weights are NOT baked into the image by default — set MODEL_REPO
# to pull a pre-quantized int8 ONNX pipeline at first boot, or mount a
# volume at /srv/model-cache with one already exported (see
# scripts/export_model.py). Baking a model in is fine too, but only if
# you've confirmed on-disk size stays well under the container's memory
# budget once loaded (loaded size is usually larger than on-disk size).
ENV MODEL_DIR=/srv/model-cache/bk-sdm-tiny-onnx-int8 \
    MAX_SIDE=192 \
    DEFAULT_SIDE=96 \
    MAX_STEPS=6 \
    INTRA_OP_THREADS=1 \
    SOFT_RSS_LIMIT_MB=620 \
    PORT=8000

EXPOSE 8000

# Single process, single worker — see server.py CONCURRENCY_LIMIT for why.
CMD ["python", "-m", "app.server"]
