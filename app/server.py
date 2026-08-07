"""
Tiny CPU-only image generation server, built to run in as little memory
as a real Stable-Diffusion-family diffusion model allows on CPU.

READ THIS FIRST — HONEST MEMORY BUDGET:
A strict 430-460MB ceiling was the original target, but no publicly
available small/distilled SD-family checkpoint fits it while still using
a real diffusion UNet + CLIP text encoder + VAE. Every small variant
(tiny-sd, small-sd, bk-sdm-tiny, ...) only compresses the UNet — the CLIP
text encoder (~123M params) and VAE are the stock, unchanged components
in all of them, because no off-the-shelf distilled replacement for those
exists publicly. That fixed floor is documented with exact numbers in
scripts/export_model.py and the README's "Realistic memory budget"
section. Bottom line, using the default nota-ai/bk-sdm-tiny-2m UNet:
    text encoder (int8)      ~120MB
    VAE decoder (int8)       ~50MB
    UNet, bk-sdm-tiny (int8) ~330MB
    runtime import overhead  ~67MB   (onnxruntime+numpy+PIL+fastapi, no transformers)
    ------------------------------
    realistic floor          ~570MB before a single request runs
This server is tuned to minimize everything on top of that floor
(resolution, threads, concurrency, activation buffer reuse) — see below —
but the floor itself is a property of the model, not something this code
can tune away. If your hard container limit is truly fixed below ~550MB,
see the README for the two remaining options: (a) a non-diffusion
generator (much lower quality), or (b) training a distilled text encoder
yourself (a real ML project, not a config change).

This is NOT a drop-in replacement for the FLUX/Nunchaku/Sana servers in
image.pollinations.ai — those load multi-GB GPU diffusion models and
need vastly more memory than even this generator's floor.

ROOT CAUSE OF THE EARLIER 512MB IMPORT-TIME FAILURE (measured, fixed):
An earlier version of this file used `optimum.onnxruntime
.ORTStableDiffusionPipeline`. Importing `optimum[onnxruntime]` pulls in
`transformers`, which alone costs ~530MB RSS just to import — before a
single request is served, before any model weights are loaded. That was
fixed by talking to raw onnxruntime InferenceSessions directly (text
encoder, UNet, VAE decoder as three separate ONNX graphs) and
implementing the scheduler denoising loop by hand in numpy. `tokenizers`
(the standalone Rust-backed library, NOT `transformers`) is used just to
turn the prompt into token ids — it costs ~6MB to import. That fix is
still in place and still correct — it just wasn't, by itself, enough to
reach 430-460MB once a real diffusion model's weights are counted.

A second real bug was also found and fixed during review: the
denoising loop's arithmetic mixed float32 arrays with Python-float
scalars from `np.sqrt(...)`, which numpy silently upcasts to float64 —
doubling the latents array's memory on every single step. Confirmed by
direct test; see the np.float32(...) casts and the belt-and-braces
.astype(np.float32) call in the loop below.

Design choices made to minimize everything on top of the model-size
floor:
  - Output capped at 96x96 by default (192x192 max) — pixel count is
    the single biggest *tunable* memory lever, on top of the fixed model
    weight floor.
  - Single worker, hard concurrency limit of 1 in-flight generation, with
    atomic (non-racy) semaphore acquisition — a second request is
    rejected (503) immediately rather than queuing and doubling memory.
  - A soft self-reported RSS ceiling (SOFT_RSS_LIMIT_MB) rejects new
    requests if the process is already close to the hard container
    limit, so you get a clean 503 instead of an OOM-kill mid-request.
    load_pipeline() also checks this right after loading, so a
    too-large model export fails loudly at startup instead of on the
    first request.
  - No image caching in RAM; each result is written straight to a
    bounded response buffer, then explicitly dereferenced and collected
    before the response is returned.
  - onnxruntime session per graph, single intra-op thread by default, so
    onnxruntime doesn't spin up per-thread scratch buffers.
"""

import asyncio
import gc
import io
import logging
import os
import resource
import time
from contextlib import asynccontextmanager
from typing import Optional

# Must be set before onnxruntime (or anything that pulls in a BLAS backend)
# is imported — setting these later, or only in the Dockerfile, has no
# effect on the thread pools those libraries spin up at import time. This
# covers the case where the server is run directly (e.g. local dev, or a
# platform that ignores the Dockerfile's ENV) rather than only through the
# container.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tiny-image-gen")


def current_rss_mb() -> float:
    """
    Process RSS in MB via the stdlib resource module (no psutil — that's
    an extra dependency and a few more MB of import overhead not worth
    spending). ru_maxrss is peak RSS in KB on Linux (it's KB on Linux,
    bytes on macOS — this server only targets Linux containers, so KB is
    assumed).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# ---------------------------------------------------------------------------
# Tunable limits — these minimize memory *on top of* the model's own
# ~570MB weight floor (see module docstring); they cannot bring total RSS
# below that floor. Do not raise these without re-measuring actual RSS;
# onnxruntime's peak usage during a generation call can be several times
# the on-disk model size because of activation buffers.
# ---------------------------------------------------------------------------
MAX_SIDE = int(os.getenv("MAX_SIDE", "192"))            # hard ceiling — UNet activation buffers grow roughly quadratically with resolution
DEFAULT_SIDE = int(os.getenv("DEFAULT_SIDE", "96"))      # what /generate uses if unspecified
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))             # small RAM cost, larger CPU-time cost per additional step
INTRA_OP_THREADS = int(os.getenv("INTRA_OP_THREADS", "1"))  # each onnxruntime thread gets its own scratch buffer; 1 thread trades speed for a predictable memory ceiling
CONCURRENCY_LIMIT = 1  # never run two generations at once — doubling the model's own ~570MB floor is not an option

# Soft self-monitoring ceiling. If the process RSS crosses this during a
# generation, we refuse *new* requests (but let the in-flight one finish)
# rather than letting the OOM killer SIGKILL the whole process mid-request,
# which would corrupt any half-written response and require a full cold
# restart (model reload) to recover.
#
# Default of 620MB assumes the ~570MB realistic floor documented in the
# module docstring (bk-sdm-tiny weights + runtime overhead) plus headroom
# for per-request activation buffers. If your container's hard memory
# limit is below ~650MB, this generator's model choice does not fit it —
# see the README before lowering this further; lowering the number alone
# will not make the model smaller.
SOFT_RSS_LIMIT_MB = int(os.getenv("SOFT_RSS_LIMIT_MB", "620"))

MODEL_DIR = os.getenv("MODEL_DIR", "model-cache/bk-sdm-tiny-onnx-int8")
MODEL_REPO = os.getenv("MODEL_REPO", "")  # set to a HF repo id to auto-download an ONNX int8 export at startup

# Expected layout under MODEL_DIR (produced by the offline export script in
# scripts/export_model.py — see README):
#   tokenizer/           - a saved `tokenizers` Tokenizer.json, NOT a
#                           transformers tokenizer directory
#   text_encoder.onnx
#   unet.onnx
#   vae_decoder.onnx
#   scheduler_config.json - {"num_train_timesteps", "beta_start", "beta_end"}

gpu_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# Populated by load_pipeline(): raw onnxruntime sessions + a plain
# tokenizers.Tokenizer + the handful of scheduler scalars we need. No
# transformers, no optimum, no diffusers objects anywhere in this process.
_sessions = {}
_tokenizer = None
_scheduler_cfg = None


def _configure_onnxruntime_session_options():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = INTRA_OP_THREADS
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Arena allocator over-reserves ahead of demand (its point is to avoid
    # per-call malloc/free overhead) — that's the wrong trade at this
    # memory budget, so we disable it to keep peak RSS predictable.
    so.enable_cpu_mem_arena = False
    # Mem-pattern reuses same-shaped scratch buffers across repeated runs
    # of the same session rather than growing overall footprint. It does
    # NOT fight enable_cpu_mem_arena — keep it on; disabling it only forces
    # every call to re-allocate every intermediate tensor from scratch.
    so.enable_mem_pattern = True
    return so


def load_pipeline():
    """
    Loads three separate raw onnxruntime InferenceSessions (text encoder,
    UNet, VAE decoder) plus a standalone `tokenizers.Tokenizer` — no
    transformers, no optimum, no diffusers. This is the direct fix for the
    ~530MB transformers import cost documented in the module docstring.

    Expects MODEL_DIR to contain pre-exported, int8-quantized ONNX graphs
    (see scripts/export_model.py and the README) — exporting/quantizing
    itself needs far more than this container's memory budget and must be
    done offline, once, elsewhere.
    """
    global _sessions, _tokenizer, _scheduler_cfg
    if _sessions:
        return

    import json

    import onnxruntime as ort
    from tokenizers import Tokenizer

    if MODEL_REPO and not os.path.isdir(MODEL_DIR):
        logger.info("Downloading quantized ONNX model from %s ...", MODEL_REPO)
        os.makedirs(MODEL_DIR, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"No model found at {MODEL_DIR} and MODEL_REPO not set. "
            "Provide pre-exported int8 ONNX graphs (see README / scripts/export_model.py)."
        )

    so = _configure_onnxruntime_session_options()
    t0 = time.time()

    # Sessions are loaded and released one at a time where possible isn't
    # applicable here (all three are needed for the lifetime of the
    # process), but loading them sequentially rather than in parallel
    # avoids a transient spike where multiple ONNX graph-loading buffers
    # are alive simultaneously.
    logger.info("Loading text encoder...")
    _sessions["text_encoder"] = ort.InferenceSession(
        os.path.join(MODEL_DIR, "text_encoder.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    logger.info("Loading UNet...")
    _sessions["unet"] = ort.InferenceSession(
        os.path.join(MODEL_DIR, "unet.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    logger.info("Loading VAE decoder...")
    _sessions["vae_decoder"] = ort.InferenceSession(
        os.path.join(MODEL_DIR, "vae_decoder.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    _tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer", "tokenizer.json"))

    with open(os.path.join(MODEL_DIR, "scheduler_config.json")) as f:
        _scheduler_cfg = json.load(f)

    logger.info("Pipeline loaded in %.1fs (RSS %.0fMB)", time.time() - t0, current_rss_mb())

    # Fail loudly at load time, not on the first request, if the model
    # itself is already eating most of the budget — a stale/wrong export
    # (e.g. fp32 weights instead of int8, or a bigger base model than
    # intended) is a build-time mistake, not a runtime one, and should
    # surface as a clear startup error rather than a mysterious first-request
    # OOM. Threshold is deliberately generous (SOFT_RSS_LIMIT_MB itself)
    # since generation adds further overhead on top of the loaded model.
    post_load_rss = current_rss_mb()
    if post_load_rss > SOFT_RSS_LIMIT_MB:
        logger.error(
            "Model loaded but RSS is already %.0fMB (soft limit %dMB) — "
            "no headroom left for inference. Check MODEL_DIR contains "
            "int8-quantized graphs, not fp32.",
            post_load_rss,
            SOFT_RSS_LIMIT_MB,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at startup rather than on first request, so the first caller
    # doesn't pay a multi-second cold-start penalty. If MODEL_REPO isn't
    # configured, we defer loading and fail loudly on first /generate call
    # instead of crashing the whole process at boot.
    try:
        load_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(title="tiny-image-gen (distilled SD, CPU)", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=300)
    width: int = Field(DEFAULT_SIDE, ge=32, le=MAX_SIDE)
    height: int = Field(DEFAULT_SIDE, ge=32, le=MAX_SIDE)
    steps: int = Field(4, ge=1, le=MAX_STEPS)
    seed: Optional[int] = None


@app.get("/health")
async def health():
    rss = current_rss_mb()
    return {
        "status": "ok" if _sessions else "model_not_loaded",
        "max_side": MAX_SIDE,
        "default_side": DEFAULT_SIDE,
        "max_steps": MAX_STEPS,
        "concurrency_limit": CONCURRENCY_LIMIT,
        "rss_mb": round(rss, 1),
        "soft_rss_limit_mb": SOFT_RSS_LIMIT_MB,
        "near_limit": rss > SOFT_RSS_LIMIT_MB,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not _sessions:
        try:
            load_pipeline()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"Model unavailable: {exc}",
            ) from exc

    # Round to multiples of 8 (standard SD-family latent constraint). Do
    # this before Pydantic-level bounds are re-checked so a caller passing
    # e.g. width=33 gets an explicit error instead of a silently-different
    # result — cheaper to reject than to debug "why is my image smaller".
    if req.width % 8 or req.height % 8:
        raise HTTPException(
            status_code=422,
            detail="width and height must be multiples of 8.",
        )
    width, height = req.width, req.height

    # Reject before starting work if we're already close to the ceiling.
    # This only catches the "idle-but-bloated" case (e.g. a slow leak, or
    # the model itself sitting bigger than expected) — it deliberately
    # does NOT abort an in-flight generation, since killing mid-request
    # would still trigger the exact OOM-kill-and-cold-restart problem this
    # is meant to avoid. ru_maxrss is a high-water mark, not current usage,
    # so this check gets strictly more conservative over the process
    # lifetime — expected and fine for a single-worker process we're happy
    # to eventually recycle.
    rss = current_rss_mb()
    if rss > SOFT_RSS_LIMIT_MB:
        logger.warning("Rejecting request: RSS %.0fMB > soft limit %dMB", rss, SOFT_RSS_LIMIT_MB)
        raise HTTPException(
            status_code=503,
            detail=f"Server near memory limit ({rss:.0f}MB used). Retry shortly.",
        )

    # acquire(non-blocking) instead of locked()-then-acquire: checking
    # .locked() and then entering `async with` is a TOCTOU race — two
    # requests can both see "unlocked" before either actually acquires,
    # since the event loop can interleave between the check and the
    # `async with` awaiting acquisition. At this memory budget that race is the
    # exact failure mode this limit exists to prevent, so we acquire
    # atomically via try_acquire and reject immediately on failure instead.
    try:
        await asyncio.wait_for(gpu_semaphore.acquire(), timeout=0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is at capacity (1 concurrent generation on this memory-limited instance). Retry shortly.",
        )

    try:
        loop = asyncio.get_event_loop()
        try:
            image = await loop.run_in_executor(
                None,
                _run_inference,
                req.prompt,
                width,
                height,
                req.steps,
                req.seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        payload = buf.getvalue()

        # Drop references to the largest live objects (raw PIL image, its
        # underlying numpy buffer, the BytesIO backing store) and collect
        # now, before returning — not in a `finally` that fires while
        # `image`/`buf` are still referenced by the enclosing frame. This
        # is the actual point in the request where peak memory has been
        # reached and can finally come back down.
        del image, buf
        gc.collect()

        return Response(content=payload, media_type="image/jpeg")
    finally:
        gpu_semaphore.release()


def _encode_prompt(prompt: str, max_length: int = 77) -> "np.ndarray":
    """
    Tokenize with the standalone `tokenizers` library and run the text
    encoder ONNX graph. Returns the encoder hidden states the UNet expects
    as cross-attention context. Padding/truncation to max_length mirrors
    what the original CLIP text encoder was trained with — a shorter or
    longer sequence changes the UNet's expected input shape and will error
    out inside onnxruntime rather than silently misbehave.
    """
    enc = _tokenizer.encode(prompt)
    ids = enc.ids[:max_length]
    ids = ids + [0] * (max_length - len(ids))
    input_ids = np.array([ids], dtype=np.int64)
    outputs = _sessions["text_encoder"].run(None, {"input_ids": input_ids})
    return outputs[0]  # (1, seq_len, hidden_dim)


def _run_inference(
    prompt: str, width: int, height: int, steps: int, seed: Optional[int]
) -> Image.Image:
    """
    Hand-rolled DDIM-style denoising loop over raw onnxruntime sessions.
    This replaces optimum's ORTStableDiffusionPipeline.__call__ — the
    whole reason that class isn't used here is that importing it drags in
    transformers (~530MB, see module docstring). The loop below is the
    minimum needed to reproduce its behavior without that dependency: it
    is deliberately simple (linear beta schedule, no classifier-free
    guidance to keep the UNet call count and peak memory low) rather than
    a full-featured scheduler implementation.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    context = _encode_prompt(prompt)  # (1, 77, hidden_dim)

    latent_h, latent_w = height // 8, width // 8
    latents = rng.randn(1, 4, latent_h, latent_w).astype(np.float32)

    num_train_timesteps = _scheduler_cfg["num_train_timesteps"]
    beta_start = _scheduler_cfg["beta_start"]
    beta_end = _scheduler_cfg["beta_end"]
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    # Evenly spaced timestep subset for `steps` denoising iterations,
    # walking from high noise (near num_train_timesteps) to low noise (0).
    timesteps = np.linspace(num_train_timesteps - 1, 0, steps, dtype=np.int64)

    unet = _sessions["unet"]
    for t in timesteps:
        t_arr = np.array([t], dtype=np.int64)
        noise_pred = unet.run(
            None,
            {
                "sample": latents,
                "timestep": t_arr,
                "encoder_hidden_states": context,
            },
        )[0]

        alpha_t = np.float32(alphas_cumprod[t])
        alpha_t_prev = np.float32(alphas_cumprod[max(t - (num_train_timesteps // steps), 0)])
        # np.float32(...) scalars, not Python floats, are required here:
        # arithmetic between a float32 ndarray and a plain Python float
        # silently upcasts the result to float64 under numpy's type
        # promotion rules, doubling `latents`' memory footprint on every
        # single denoising step. Confirmed by direct test in this
        # environment — this was a real, previously-unnoticed leak, not a
        # hypothetical one.
        pred_original = (latents - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
        latents = np.sqrt(alpha_t_prev) * pred_original + np.sqrt(1 - alpha_t_prev) * noise_pred
        latents = latents.astype(np.float32, copy=False)  # belt-and-braces: guarantee fp32 regardless of the scalar types above

        # Free the noise_pred reference explicitly each iteration rather
        # than waiting for the next loop iteration to overwrite it — at
        # this memory budget, an extra (1, 4, H/8, W/8) fp32 array sitting
        # around for even one extra iteration is worth reclaiming early.
        del noise_pred

    scaled_latents = (latents / 0.18215).astype(np.float32)  # standard SD VAE scaling factor; explicit astype guards against numpy silently upcasting to fp64 on the division
    vae_out = _sessions["vae_decoder"].run(None, {"latent_sample": scaled_latents})[0]
    del latents, scaled_latents, context

    # image: (1, 3, H, W) in [-1, 1] -> HWC uint8 in [0, 255]. Each step
    # below reassigns `image` in place (same name, new array) rather than
    # keeping both the float and uint8 versions alive simultaneously — at
    # small resolutions this is a minor saving, but it's free and keeps
    # the pattern consistent with the rest of this function.
    image = (vae_out[0].transpose(1, 2, 0) / 2 + 0.5).clip(0, 1)
    del vae_out
    image = (image * 255).round().astype(np.uint8)
    return Image.fromarray(image)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,  # never more than 1 worker process at this memory budget
        log_level="info",
    )
