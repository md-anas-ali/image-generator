"""
ONE-TIME OFFLINE export script. Run this on a normal development machine
with plenty of RAM (NOT inside the 432-460MB target container) to produce
the artifacts app/server.py expects under MODEL_DIR:

    text_encoder.onnx
    unet.onnx
    vae_decoder.onnx
    tokenizer/tokenizer.json
    scheduler_config.json

This script needs torch/diffusers/transformers/optimum — the exact
packages the runtime server deliberately avoids. That's expected: export
and quantization are a build-time step, not a runtime one. Nothing this
script imports ends up in the container that actually serves requests.

MODEL CHOICE — read this before picking --model:
The default here is nota-ai/bk-sdm-tiny-2m, a block-removed
knowledge-distilled UNet (0.33B params vs 0.86B in the SD-v1.4 base) that
its authors distribute as a UNet-only checkpoint — its text encoder and
VAE are the STOCK CompVis/stable-diffusion-v1-4 components, unchanged.
That matters for the memory budget: no publicly available SD-family
distillation shrinks the CLIP text encoder or VAE, only the UNet. So the
realistic floor for ANY of these small variants (tiny-sd, small-sd,
bk-sdm-tiny, ...) is roughly:
    text encoder (CLIP, ~123M params, unchanged everywhere)  ~120MB int8
    VAE decoder (unchanged everywhere)                        ~50MB int8
    UNet (this is the part that actually varies by model)     ~330MB int8 for bk-sdm-tiny
    ---------------------------------------------------------------
    total on-disk (int8)                                     ~500MB
Combined with the ~67MB runtime import footprint and per-request
activation buffers, this does NOT fit a strict 430-460MB ceiling — see
the README's "Realistic memory budget" section for the honest numbers
and what your actual options are if you need to go lower than that.

Usage:
    pip install torch diffusers transformers optimum[onnxruntime] onnx onnxruntime tokenizers
    python scripts/export_model.py \
        --model CompVis/stable-diffusion-v1-4 \
        --unet-repo nota-ai/bk-sdm-tiny-2m \
        --out ./model-cache/bk-sdm-tiny-onnx-int8
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="CompVis/stable-diffusion-v1-4",
        help="HF hub id of the base pipeline to source the text encoder, "
        "VAE, tokenizer, and scheduler config from. Almost always the "
        "original SD-v1.4/v1.5 base, even when using a distilled UNet — "
        "distilled UNet checkpoints are typically published UNet-only and "
        "expect the stock text encoder/VAE alongside them.",
    )
    parser.add_argument(
        "--unet-repo",
        default="nota-ai/bk-sdm-tiny-2m",
        help="HF hub id to load the UNet from. Set equal to --model (or "
        "omit by passing the same id) if you want the *un-distilled* "
        "UNet instead — useful for comparing quality/size trade-offs.",
    )
    parser.add_argument("--out", default="./model-cache/bk-sdm-tiny-onnx-int8")
    args = parser.parse_args()

    # Imported lazily, inside main(), so `python scripts/export_model.py --help`
    # doesn't require torch to be installed just to print usage.
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from optimum.onnxruntime import ORTStableDiffusionPipeline
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    os.makedirs(args.out, exist_ok=True)
    fp32_dir = os.path.join(args.out, "_fp32_tmp")

    print(f"Loading base pipeline {args.model} (text encoder, VAE, tokenizer, scheduler)...")
    src_pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32)

    if args.unet_repo != args.model:
        print(f"Swapping in distilled UNet from {args.unet_repo}...")
        src_pipe.unet = UNet2DConditionModel.from_pretrained(
            args.unet_repo, subfolder="unet", torch_dtype=torch.float32
        )

    print("Exporting the (possibly UNet-swapped) pipeline to ONNX (fp32, intermediate)...")
    # ORTStableDiffusionPipeline.from_pretrained(..., export=True) re-downloads
    # from the hub rather than exporting an in-memory pipeline, which would
    # silently ignore the UNet swap above. Save the swapped pipeline to disk
    # first, then export from that local path.
    swapped_dir = os.path.join(args.out, "_swapped_tmp")
    src_pipe.save_pretrained(swapped_dir)
    ort_pipe = ORTStableDiffusionPipeline.from_pretrained(swapped_dir, export=True)
    ort_pipe.save_pretrained(fp32_dir)
    shutil.rmtree(swapped_dir, ignore_errors=True)

    print("Quantizing text encoder, UNet, and VAE decoder to int8 (AVX2)...")
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    for component in ("text_encoder", "unet", "vae_decoder"):
        component_dir = os.path.join(fp32_dir, component)
        quantizer = ORTQuantizer.from_pretrained(component_dir, file_name="model.onnx")
        quantizer.quantize(save_dir=os.path.join(args.out), quantization_config=qconfig)
        # optimum names the quantized file "model_quantized.onnx" by default;
        # normalize to the flat filenames app/server.py expects.
        produced = os.path.join(args.out, "model_quantized.onnx")
        target = os.path.join(args.out, f"{component}.onnx")
        shutil.move(produced, target)
        print(f"  {component} -> {target}")

    print("Exporting tokenizer as a standalone `tokenizers` file (no transformers at runtime)...")
    # StableDiffusionPipeline's tokenizer is a transformers CLIPTokenizer,
    # which has a `backend_tokenizer` (the underlying Rust `tokenizers`
    # object) we can save directly — that's the file app/server.py loads
    # with `tokenizers.Tokenizer.from_file`, with no transformers import
    # required at serve time.
    tok_dir = os.path.join(args.out, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    src_pipe.tokenizer.backend_tokenizer.save(os.path.join(tok_dir, "tokenizer.json"))

    print("Writing scheduler_config.json (only the 3 scalars the hand-rolled loop needs)...")
    sched = src_pipe.scheduler.config
    with open(os.path.join(args.out, "scheduler_config.json"), "w") as f:
        json.dump(
            {
                "num_train_timesteps": sched.num_train_timesteps,
                "beta_start": sched.beta_start,
                "beta_end": sched.beta_end,
            },
            f,
            indent=2,
        )

    shutil.rmtree(fp32_dir, ignore_errors=True)

    total_mb = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
        if os.path.isfile(os.path.join(args.out, f))
    ) / (1024 * 1024)
    print(f"\nDone. On-disk size: {total_mb:.0f}MB in {args.out}")
    print("Remember: on-disk size is NOT peak RSS during inference — test with")
    print("`docker run --memory=460m` (or your real limit) before assuming this fits.")
    print("See the README's 'Realistic memory budget' section for expected numbers.")


if __name__ == "__main__":
    main()
