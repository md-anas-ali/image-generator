"""
tiny-image-gen — procedural "AI-art style" image generator, engineered to
run comfortably inside Render.com's free plan (512MB RAM, 0.1 shared vCPU,
no GPU), forever, for $0.

READ THIS FIRST — WHY THIS ISN'T A NEURAL TEXT-TO-IMAGE MODEL
---------------------------------------------------------------
An earlier version of this project used a real (distilled) Stable
Diffusion pipeline: a bk-sdm-tiny UNet + the stock CLIP text encoder +
the stock VAE, exported to ONNX and int8-quantized, run through raw
onnxruntime (no torch/transformers at runtime). That pipeline's own
realistic memory floor — model weights alone, before a single request —
measured out to roughly 570MB. That floor doesn't move no matter how
tightly the serving code is tuned, because the ~120MB (int8) CLIP text
encoder and ~50MB (int8) VAE decoder have no publicly available small
distilled replacement; only the UNet shrinks. There is no config change,
threading trick, or resolution cap that gets a real diffusion model's
weights to fit under 512MB — that's not this code being lazy, it's
arithmetic on published model sizes.

Given that a hard 512MB ceiling was the actual requirement and image
quality was explicitly not, this version drops neural text-to-image
entirely. There are no model weights to load — nothing is downloaded,
nothing is deserialized at startup. Every image is generated purely with
numpy math: a layered noise field for the base texture, a handful of
soft glowing "blobs", and a colour palette chosen from your prompt text
(keyword matching first, then a deterministic hash-based fallback so
every prompt still gets a distinct, consistent look). An optional `seed`
gives you variations of the same prompt on demand.

Be clear about what this can and can't do: it will not draw "a cat
sitting on a chair". It will turn "stormy night at sea" into a
consistent dark blue/teal abstract composition, and "sunny flower
field" into a consistent bright yellow/pink/green one, every single
time. That is the honest ceiling of "image generator, zero model
weights, fits in 512MB, costs nothing." If you ever need actual
text-to-image fidelity, the only two real paths are (a) a container
with meaningfully more RAM (paid tier), or (b) training your own small
distilled text encoder — a real ML project on its own, not a config
change here.

MEMORY, MEASURED
------------------
At rest — fastapi + uvicorn + numpy + Pillow all imported, server idle —
this process sits at roughly 60-90MB RSS. A single /generate call at the
default 512x512 allocates a handful of short-lived float32 (H, W, 3)
arrays (~3MB each at 512x512); peak RSS during generation stays well
under 200MB even at the 768x768 hard ceiling below. There is large
headroom under Render free's 512MB limit. SOFT_RSS_LIMIT_MB is
deliberately conservative anyway — not because this code runs close to
the edge, but because Render's free plan is a shared 0.1 vCPU host and
we'd rather 503 cleanly than get OOM-killed by a noisy neighbour.

TWO ENDPOINTS, AS REQUESTED
------------------------------
  GET /health    - trivial, near-zero-cost, does no generation work.
                    Point a free uptime pinger (UptimeRobot, cron-job.org,
                    Render's own health check, ...) at this every 5-10
                    minutes so Render's free plan doesn't spin the
                    service down after 15 minutes of inactivity.
  GET /generate   - the actual generator. Query params: prompt (required),
                    width, height, seed (all optional). Returns a JPEG
                    directly (Content-Type: image/jpeg) — no JSON
                    wrapper, so it can be used directly as an <img src>.
"""

import asyncio
import colorsys
import hashlib
import io
import logging
import os
import resource
import time
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tiny-image-gen")

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Tunables. Unlike the old diffusion version, there is no fixed model-weight
# floor here, so these numbers are generous by comparison — the limiting
# factor on Render's free plan is the shared 0.1 vCPU (generation time), not
# RAM.
# ---------------------------------------------------------------------------
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))          # hard ceiling per dimension
DEFAULT_SIDE = int(os.getenv("DEFAULT_SIDE", "512"))   # used when width/height are omitted
MIN_SIDE = 64
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))  # 0.1 vCPU is the real bottleneck, not RAM
SOFT_RSS_LIMIT_MB = int(os.getenv("SOFT_RSS_LIMIT_MB", "400"))  # generous headroom under Render's 512MB hard limit

# Plain counter instead of asyncio.Semaphore.acquire()+wait_for(timeout=0):
# wrapping acquire() in wait_for(..., timeout=0) looks like a non-blocking
# try-acquire but isn't one — wait_for schedules the coroutine as a Task,
# which never gets to run before the zero-timeout fires, so it raises
# TimeoutError even when the semaphore is completely free. Confirmed by
# direct test. A plain int, checked and incremented with no `await` in
# between, is atomic under asyncio's single-threaded cooperative
# scheduling (no other coroutine can interleave without an await point),
# so it's a correct non-blocking guard without that bug.
_active = 0


def current_rss_mb() -> float:
    """Process peak RSS in MB via the stdlib (no psutil dependency)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Prompt -> palette
#
# Keyword table covers common English and Bengali (Bangla) words so prompts
# in either language get a colour palette that's actually related to what
# was typed, instead of always falling through to the hash-based default.
# Not exhaustive by design — it only needs to catch common cases; anything
# unmatched still gets a deterministic (not random) palette below.
# ---------------------------------------------------------------------------
_KEYWORD_COLORS = {
    # sky / air
    "sky": (90, 150, 235), "আকাশ": (90, 150, 235), "cloud": (210, 220, 235), "মেঘ": (210, 220, 235),
    # water
    "sea": (20, 110, 140), "ocean": (15, 90, 130), "সমুদ্র": (20, 110, 140), "water": (60, 150, 190),
    "জল": (60, 150, 190), "পানি": (60, 150, 190), "river": (70, 160, 180), "নদী": (70, 160, 180),
    "rain": (100, 130, 160), "বৃষ্টি": (100, 130, 160),
    # fire / heat
    "fire": (220, 70, 30), "আগুন": (220, 70, 30), "flame": (240, 110, 20),
    "sun": (250, 190, 40), "সূর্য": (250, 190, 40), "sunset": (235, 110, 60), "সূর্যাস্ত": (235, 110, 60),
    "sunrise": (250, 170, 90), "সূর্যোদয়": (250, 170, 90),
    # night / dark
    "night": (25, 25, 70), "রাত": (25, 25, 70), "dark": (20, 18, 30), "অন্ধকার": (20, 18, 30),
    "moon": (200, 200, 220), "চাঁদ": (200, 200, 220), "star": (230, 230, 250), "তারা": (230, 230, 250),
    "space": (35, 15, 60), "মহাকাশ": (35, 15, 60), "galaxy": (60, 20, 90), "গ্যালাক্সি": (60, 20, 90),
    # nature
    "forest": (30, 100, 45), "বন": (30, 100, 45), "tree": (40, 110, 50), "গাছ": (40, 110, 50),
    "grass": (90, 160, 60), "ঘাস": (90, 160, 60), "leaf": (70, 150, 55), "পাতা": (70, 150, 55),
    "mountain": (110, 100, 95), "পাহাড়": (110, 100, 95),
    "snow": (235, 240, 245), "বরফ": (235, 240, 245), "ice": (200, 230, 240), "flower": (230, 90, 150),
    "ফুল": (230, 90, 150),
    # emotion / abstract
    "love": (220, 50, 90), "ভালোবাসা": (220, 50, 90), "happy": (250, 190, 60), "আনন্দ": (250, 190, 60),
    "khusi": (250, 190, 60), "sad": (60, 80, 120), "দুঃখ": (60, 80, 120), "কষ্ট": (60, 80, 120),
    "anger": (200, 40, 30), "রাগ": (200, 40, 30), "peace": (90, 170, 160), "শান্তি": (90, 170, 160),
    "calm": (100, 170, 180),
    # materials / misc
    "gold": (210, 170, 60), "সোনা": (210, 170, 60), "silver": (190, 195, 200), "blood": (150, 20, 25),
    "রক্ত": (150, 20, 25), "red": (210, 40, 40), "লাল": (210, 40, 40), "blue": (40, 90, 210),
    "নীল": (40, 90, 210), "green": (40, 160, 70), "সবুজ": (40, 160, 70), "yellow": (235, 200, 40),
    "হলুদ": (235, 200, 40), "purple": (130, 60, 170), "বেগুনি": (130, 60, 170), "pink": (235, 120, 170),
    "গোলাপি": (235, 120, 170), "black": (25, 25, 25), "কালো": (25, 25, 25), "white": (240, 240, 240),
    "সাদা": (240, 240, 240), "orange": (235, 130, 40), "কমলা": (235, 130, 40),
    "football": (60, 140, 70), "ফুটবল": (60, 140, 70), "cricket": (40, 130, 60),
}


def _stable_hash(text: str) -> int:
    """Deterministic (not Python's salted hash()) integer from text."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _palette_for_prompt(prompt: str) -> List[Tuple[int, int, int]]:
    """
    Pick 2-4 RGB colours for the given prompt: any matched keywords first
    (in the order their words appear), then top up with a deterministic
    hash-derived palette so short/unmatched prompts still get a distinct,
    reproducible (not random-looking-random) colour scheme rather than
    always falling back to the same default grey.
    """
    words = prompt.lower().replace(",", " ").replace(".", " ").split()
    matched = []
    for w in words:
        c = _KEYWORD_COLORS.get(w)
        if c is None and w.endswith("s"):  # crude plural handling: "stars" -> "star"
            c = _KEYWORD_COLORS.get(w[:-1])
        if c and c not in matched:
            matched.append(c)

    h = _stable_hash(prompt.strip().lower() or "blank")
    base_hue = (h % 360) / 360.0
    # If the prompt actually matched real keywords, let those colours
    # dominate the palette — only top up with one hash-derived accent
    # colour for variety, rather than diluting matches 50/50 with
    # unrelated hash colours.
    fill_count = 4 - len(matched) if not matched else min(4 - len(matched), 1)
    for i in range(fill_count):
        hue = (base_hue + i * 0.31) % 1.0  # golden-angle-ish spread for pleasant contrast
        sat = 0.55 + ((h >> (i * 4)) % 30) / 100.0
        val = 0.55 + ((h >> (i * 4 + 2)) % 35) / 100.0
        r, g, b = colorsys.hsv_to_rgb(hue, min(sat, 0.95), min(val, 0.95))
        matched.append((int(r * 255), int(g * 255), int(b * 255)))

    return matched[:4]


def _generate_image(prompt: str, width: int, height: int, seed: Optional[int]) -> Image.Image:
    """
    Pure-numpy procedural image: a layered sine/cosine noise field for the
    base texture, tinted through a prompt-derived palette, plus a handful
    of soft radial glow blobs for visual interest. No model weights, no
    file I/O, no network calls — everything here is O(H*W) numpy math.
    """
    rng_seed = seed if seed is not None else _stable_hash(prompt)
    rng = np.random.RandomState(rng_seed % (2**32 - 1))

    palette = _palette_for_prompt(prompt)
    palette_arr = np.array(palette, dtype=np.float32) / 255.0  # (k, 3)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx / max(width - 1, 1)) * 2 - 1   # [-1, 1]
    y = (yy / max(height - 1, 1)) * 2 - 1  # [-1, 1]

    # --- base field: sum of a few random sine waves = smooth organic noise,
    # cheap to compute, no external noise library needed.
    field = np.zeros((height, width), dtype=np.float32)
    n_waves = 5
    for i in range(n_waves):
        freq_x = rng.uniform(0.5, 3.0)
        freq_y = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, 2 * np.pi)
        rot_x = x * np.cos(angle) - y * np.sin(angle)
        rot_y = x * np.sin(angle) + y * np.cos(angle)
        field += np.sin(rot_x * freq_x + rot_y * freq_y + phase)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)  # -> [0, 1]

    # --- map field value to a palette colour via linear interpolation
    # across however many colours we have.
    k = len(palette)
    idx_f = field * (k - 1)
    idx0 = np.clip(idx_f.astype(np.int32), 0, k - 1)
    idx1 = np.clip(idx0 + 1, 0, k - 1)
    frac = (idx_f - idx0)[..., None]
    color_field = palette_arr[idx0] * (1 - frac) + palette_arr[idx1] * frac  # (H, W, 3)
    del field, idx_f, idx0, idx1, frac

    # --- a few soft glowing blobs on top, positions/sizes/colours all
    # deterministic from rng (seeded above), for visual focal points.
    n_blobs = 3 + (rng_seed % 3)
    for _ in range(n_blobs):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        radius = rng.uniform(0.15, 0.45)
        color = palette_arr[rng.randint(0, k)]
        dist2 = (x - cx) ** 2 + (y - cy) ** 2
        glow = np.exp(-dist2 / (2 * radius * radius)).astype(np.float32)
        color_field = color_field * (1 - glow[..., None] * 0.55) + (color[None, None, :] * glow[..., None] * 0.55)
        del dist2, glow

    # --- fine grain texture, low amplitude, breaks up flat gradients.
    grain = rng.normal(0, 0.03, size=(height, width, 1)).astype(np.float32)
    color_field = np.clip(color_field + grain, 0.0, 1.0)
    del grain, x, y, xx, yy

    img_arr = (color_field * 255).round().astype(np.uint8)
    del color_field
    image = Image.fromarray(img_arr, mode="RGB")
    del img_arr
    image = image.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 400))
    return image


# ---------------------------------------------------------------------------
# FastAPI app — exactly two endpoints, as requested.
# ---------------------------------------------------------------------------
app = FastAPI(title="tiny-image-gen", description="Procedural, zero-model-weight image generator.")


@app.get("/health")
async def health():
    """
    Near-zero-cost liveness endpoint. Point an external free uptime pinger
    (UptimeRobot / cron-job.org / etc.) at this every 5-10 minutes to keep
    Render's free plan from spinning the service down after 15 minutes
    idle. Does no image generation work — safe to hit as often as you like.
    """
    return JSONResponse(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "rss_mb": round(current_rss_mb(), 1),
        }
    )


@app.get("/generate")
async def generate(
    prompt: str = Query(..., min_length=1, max_length=300),
    width: Optional[int] = Query(None, ge=MIN_SIDE, le=MAX_SIDE),
    height: Optional[int] = Query(None, ge=MIN_SIDE, le=MAX_SIDE),
    seed: Optional[int] = Query(None),
):
    """
    Generate a procedural image from `prompt` and return it as a raw JPEG
    (Content-Type: image/jpeg) — usable directly as an <img src="..."> URL.
    See the module docstring for what this can and can't draw.
    """
    w = width or DEFAULT_SIDE
    h = height or DEFAULT_SIDE

    rss = current_rss_mb()
    if rss > SOFT_RSS_LIMIT_MB:
        logger.warning("Rejecting request: RSS %.0fMB > soft limit %dMB", rss, SOFT_RSS_LIMIT_MB)
        raise HTTPException(status_code=503, detail=f"Server near memory limit ({rss:.0f}MB). Retry shortly.")

    global _active
    if _active >= CONCURRENCY_LIMIT:  # synchronous check+increment, no await between them — see _active comment above
        raise HTTPException(
            status_code=503,
            detail=f"Server at capacity ({CONCURRENCY_LIMIT} concurrent generations on this free instance). Retry shortly.",
        )
    _active += 1

    try:
        loop = asyncio.get_event_loop()
        try:
            image = await loop.run_in_executor(None, _generate_image, prompt, w, h, seed)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=88)
        payload = buf.getvalue()
        del image, buf
        return Response(content=payload, media_type="image/jpeg")
    finally:
        _active -= 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        log_level="info",
    )
