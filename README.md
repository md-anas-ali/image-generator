# tiny-image-gen — CPU-only diffusion image generator (distilled, low-res)

এটা `image.pollinations.ai`-এর FLUX/Nunchaku/Sana/z-image সার্ভারগুলোর
প্রতিস্থাপন **না** — ওগুলো GPU-নির্ভর মাল্টি-GB diffusion মডেল। এটা একটা
সম্পূর্ণ আলাদা, CPU-only, distilled UNet ব্যবহারকারী generator।

## ⚠️ Realistic memory budget — আগে এটা পড়ুন

আপনি চেয়েছিলেন এটা কড়াকড়ি **430-460MB**-এ ফিট করুক। সৎভাবে বলছি:
**এটা কোনো real diffusion মডেল দিয়ে সম্ভব না।** কেন, তার হিসাব:

| Component | সব SD-family distilled মডেলেই কি ছোট হয়? | int8 আনুমানিক সাইজ |
|---|---|---|
| CLIP Text Encoder (~123M params) | **না** — কোনো পাবলিক distillation এটা ছোট করে না | ~120MB |
| VAE decoder | **না** — সবসময় stock/unchanged | ~50MB |
| UNet | **হ্যাঁ** — এটাই একমাত্র অংশ যেটা distillation-এ ছোট হয় | bk-sdm-tiny: ~330MB |
| Runtime import overhead (onnxruntime+numpy+PIL+fastapi, no transformers) | — | ~67MB |
| **মোট বাস্তবিক floor** | | **~570MB, কোনো request চালানোর আগেই** |

`tiny-sd`, `small-sd`, `bk-sdm-tiny` — যেটাই বাছুন, Text Encoder ও VAE-এর
~170MB সবসময় থেকেই যাবে, কারণ **এই দুটোর কোনো পাবলিক ছোট (distilled)
ভার্সন নেই**। এটা ২০২৫-এর গবেষণাতেও (T5-XXL distillation নিয়ে) একটা
খোলা সমস্যা হিসেবে আলোচিত — SD1.4/1.5-এর CLIP text encoder-এর জন্য
এমন কিছু এখনো নেই।

**তাহলে ৪৩০-৪৬০MB-এ পাক্কা যেতে চাইলে আপনার আসল অপশন:**
1. নিজে একটা ছোট text encoder distill/train করা (একটা পূর্ণাঙ্গ ML
   প্রজেক্ট, এই কোডবেসের স্কোপের বাইরে)
2. Diffusion ছেড়ে সম্পূর্ণ ভিন্ন architecture (GAN-স্টাইল ছোট generator)
   ব্যবহার করা — কোয়ালিটি অনেক নিচু হবে
3. **এই README-এর সাজেশন:** limit-টা বাস্তবিক জায়গায় (~৬৫০MB+) রাখা,
   diffusion-ভিত্তিক ভালো কোয়ালিটি ধরে রাখতে

এই সার্ভারটা option ৩ ধরে বানানো — floor-এর উপরে যা কিছু tune করা
সম্ভব (resolution, threads, concurrency, activation buffer reuse)
সবকিছু tight করা হয়েছে, কিন্তু model floor নিজেই কোনো কনফিগ দিয়ে
সরানো যায় না।

## ব্যবহৃত মডেল

`nota-ai/bk-sdm-tiny-2m` — একটা block-removed knowledge-distilled UNet
(0.33B params, SD-v1.4-এর 0.86B UNet থেকে অনেক কম), stock
`CompVis/stable-diffusion-v1-4` text encoder + VAE-এর সাথে। এটা একটা
পুরোনো (২০২৩), পরীক্ষিত, ভালো কোয়ালিটির distillation — `-2m` ভার্সন
বেশি ডেটায় ট্রেইন্ড, তাই মূল `bk-sdm-tiny`-এর চেয়ে ভালো আউটপুট দেয়।

## ডিবাগ হিস্টরি (প্রাসঙ্গিক প্রেক্ষাপট)

আগের একটা ভার্সন `optimum.onnxruntime.ORTStableDiffusionPipeline`
ব্যবহার করত, যেটা `transformers` টেনে আনত — **শুধু import করতেই ~530MB**
(মাপা হয়েছে)। সেটা এখন সম্পূর্ণ বাদ; সার্ভার এখন raw
`onnxruntime.InferenceSession` দিয়ে সরাসরি ৩টা ONNX গ্রাফ চালায় এবং
denoising loop হাতে numpy দিয়ে লেখা। এই ফিক্সটা সঠিক ছিল এবং এখনো আছে
— কিন্তু একা এটা দিয়ে real diffusion model-সহ ৪৩০-৪৬০MB-এ যাওয়া যায় না,
কারণ model weights-ই তার চেয়ে বড়।

এছাড়া রিভিউয়ের সময় আরেকটা আসল bug পাওয়া গেছে ও ঠিক করা হয়েছে:
denoising loop-এ `np.sqrt()` থেকে পাওয়া Python float দিয়ে fp32 array-কে
গুণ/ভাগ করলে numpy silently সেটাকে **float64-এ upcast** করত (মেমরি
দ্বিগুণ)। টেস্ট করে নিশ্চিত করা হয়েছে, এখন `np.float32()` cast +
`.astype(np.float32)` guard দিয়ে ঠিক করা।

## মডেল বানানো

```bash
pip install torch diffusers transformers "optimum[onnxruntime]" onnx onnxruntime tokenizers
python scripts/export_model.py \
  --model CompVis/stable-diffusion-v1-4 \
  --unet-repo nota-ai/bk-sdm-tiny-2m \
  --out ./model-cache/bk-sdm-tiny-onnx-int8
```

**এটা একটা বড় RAM মেশিনে একবারই চালাতে হবে** — torch/diffusers/optimum
লাগে, তাই এটা কখনোই memory-limited container-এর ভেতরে চালাবেন না।
স্ক্রিপ্টটা:
1. Base pipeline (text encoder, VAE, tokenizer, scheduler) লোড করে
2. `--unet-repo` থেকে distilled UNet swap করে বসায়
3. পুরো (swap করা) pipeline ONNX-এ export করে, তারপর int8 (AVX2) quantize করে
4. tokenizer-কে standalone `tokenizers` ফরম্যাটে সেভ করে (transformers ছাড়া লোড করা যায়)
5. scheduler-এর ৩টা scalar `scheduler_config.json`-এ লেখে

শেষে টার্মিনালে on-disk সাইজ প্রিন্ট হবে — এটা লক্ষ্য করুন, কিন্তু
মনে রাখবেন **on-disk সাইজ ≠ inference-এর সময় peak RSS**, তাই সবসময়
`--memory` লিমিট দিয়ে টেস্ট করুন।

## চালানো

```bash
docker build -t tiny-image-gen .
docker run -p 8000:8000 \
  -e MODEL_REPO=your-username/bk-sdm-tiny-onnx-int8 \
  --memory=700m \
  tiny-image-gen
```

`700m` একটা নিরাপদ শুরুর পয়েন্ট এই মডেলের জন্য (~570MB floor + headroom)।
এটা কমাতে চাইলে ধাপে ধাপে টেস্ট করুন — `/health`-এ `rss_mb` দেখে বুঝবেন
কতটা কমানো যাবে।

## API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a small house", "width": 96, "height": 96, "steps": 4}' \
  --output out.jpg

curl http://localhost:8000/health
# {"status": "ok", "rss_mb": 590.2, "soft_rss_limit_mb": 620, "near_limit": false, ...}
```

## টিউনিং লিভার (model floor-এর উপরে যা tune করা যায়)

- `MAX_SIDE` / `DEFAULT_SIDE` (ডিফল্ট 192/96) — resolution বাড়লে UNet
  activation buffer roughly quadratic বাড়ে।
- `MAX_STEPS` (ডিফল্ট 6) — মেমরিতে ছোট প্রভাব, CPU সময়ে বড় প্রভাব।
- `INTRA_OP_THREADS` (ডিফল্ট 1) — প্রতি থ্রেড নিজের scratch buffer নেয়।
- `SOFT_RSS_LIMIT_MB` (ডিফল্ট 620) — hard container limit-এর নিচে
  headroom রাখতে; আপনার আসল limit অনুযায়ী বদলান (limit থেকে ~৫০-৮০MB
  কম রাখুন)।
- মডেল choice-ই সবচেয়ে বড় লিভার — `--unet-repo` বদলে আরও agressive
  compression (bk-sdm-tiny non-2m, বা নিজে distill করা) চেষ্টা করা যায়,
  কিন্তু text encoder+VAE floor (~170MB) কখনো কমবে না এই পদ্ধতিতে।

## জানা সীমাবদ্ধতা

- **৪৩০-৪৬০MB-এ ফিট করে না** — উপরের বাস্তবিক floor ব্যাখ্যা দেখুন।
- একসাথে একটার বেশি request সার্ভ হবে না (503, atomic semaphore acquire দিয়ে race-free)।
- `SOFT_RSS_LIMIT_MB` পার হলে নতুন request 503 পাবে, চলমান request নিরাপদে শেষ হয়।
- Denoising loop-এ classifier-free guidance নেই (মেমরি/স্পিড বাঁচাতে বাদ)।
- CPU inference দশ সেকেন্ড থেকে মিনিট-স্কেলে সময় নিতে পারে।
- এটা প্রোডাকশন-গ্রেড না — experimental/hobby ব্যবহারের জন্য।
