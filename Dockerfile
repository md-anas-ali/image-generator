# Tiny, CPU-only, zero-model-weight procedural image generator.
# No torch, no onnxruntime, no transformers — every image is generated
# with numpy math at request time. See app/server.py module docstring for
# the full explanation of why this replaced the earlier diffusion-based
# version (that pipeline's own realistic floor was ~570MB before serving
# a single request — impossible to fit in Render free's 512MB no matter
# how it's tuned). Measured with this version: idle RSS ~60-90MB, peak
# RSS well under 250MB even with 2 concurrent 768x768 generations.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app/ ./app/

# Tunables — see app/server.py for what each one does. Generous relative
# to Render free's 512MB limit; the real bottleneck on that plan is the
# shared 0.1 vCPU (generation time), not RAM.
ENV MAX_SIDE=768 \
    DEFAULT_SIDE=512 \
    CONCURRENCY_LIMIT=2 \
    SOFT_RSS_LIMIT_MB=400

EXPOSE 8000

# Render sets $PORT itself at runtime; app/server.py reads it via
# os.getenv("PORT", "8000") so this also works unchanged for local
# `docker run -p 8000:8000` testing.
CMD ["python", "-m", "app.server"]
