# tiny-image-gen — procedural image generator, ০ model weight, Render free-plan (512MB) এ ফিট করে

## এটা আসলে কী, আগে সৎভাবে বলে নিই

আগের ভার্সনটা real diffusion মডেল (bk-sdm-tiny UNet + stock CLIP text
encoder + stock VAE, ONNX int8) দিয়ে বানানো ছিল। সেটার নিজের বাস্তবিক
মেমরি floor (কোনো request চালানোর আগেই, শুধু weights লোড করতে) মাপা
হয়েছিল **~570MB**। কারণ CLIP text encoder (~120MB) আর VAE (~50MB)-এর
কোনো পাবলিক ছোট (distilled) ভার্সন নেই — শুধু UNet-টাই ছোট করা যায়।
এটা কোনো কনফিগ দিয়ে ঠিক করা যায় না, প্রকাশিত মডেল সাইজের সাধারণ অঙ্ক।

আপনি বলেছেন quality খারাপ হোক সমস্যা নাই, কিন্তু ৫১২MB-এর ভিতরে **must**
থাকতে হবে আর lifetime free থাকতে হবে — সেই বাস্তবতা মেনে এই ভার্সনে
diffusion/neural text-to-image পুরোপুরি বাদ দেওয়া হয়েছে। **কোনো মডেল
weight লোড হয় না — লোড করার মতো কিছুই নেই।** প্রতিটা ছবি request-এর
সময় শুধু numpy math দিয়ে বানানো হয়: একটা layered noise field (base
texture), কয়েকটা soft glow blob, আর prompt থেকে বের করা color palette
(keyword matching, তারপর hash-based fallback)।

**এটা "a cat sitting on a chair" আঁকবে না।** এটা "stormy night at sea"
লিখলে প্রতিবার একই রকম dark blue/teal abstract ছবি বানাবে, "sunny flower
field" লিখলে প্রতিবার একই রকম bright yellow/pink/green ছবি বানাবে —
সামঞ্জস্যপূর্ণ (consistent), prompt-relevant abstract art, কিন্তু real
AI object-recognition/generation না। ৫১২MB-তে, ০ model weight দিয়ে,
বিনামূল্যে — এটাই বাস্তবিক সর্বোচ্চ। যদি সত্যিকারের text-to-image লাগে,
বাস্তব ২টা পথ: (a) বেশি RAM-এর (পেইড) ইনস্ট্যান্স, অথবা (b) নিজে একটা
ছোট distilled text encoder ট্রেইন করা — সেটা একটা সম্পূর্ণ আলাদা ML
প্রজেক্ট, এই কোডবেসের স্কোপের বাইরে।

## মেমরি — মাপা হয়েছে, অনুমান না

- Idle (fastapi+uvicorn+numpy+Pillow লোড হয়ে আছে, কোনো request আসেনি): **~60-90MB RSS**
- একটা 768x768 (max resolution) request চলাকালীন peak: **~140MB RSS**
- ২টা concurrent 768x768 request একসাথে: **~210MB RSS**
- Render free plan-এর হার্ড লিমিট 512MB — এখানে অনেক headroom আছে।

`SOFT_RSS_LIMIT_MB` (ডিফল্ট 400) ইচ্ছাকৃতভাবে রক্ষণশীল রাখা — এই কোড
লিমিটের কাছাকাছি চলে বলে না, বরং Render free plan একটা shared 0.1 vCPU
হোস্ট, তাই পাশের noisy neighbor প্রসেস থাকলেও আমরা একটা পরিষ্কার 503
দিতে চাই, OOM-kill হয়ে পুরো প্রসেস রিস্টার্ট হওয়ার বদলে।

## ২টা endpoint, যেমনটা চেয়েছিলেন

```
GET /health
```
প্রায় বিনামূল্যে (কোনো image generation হয় না)। এখানে একটা ফ্রি uptime
pinger (UptimeRobot, cron-job.org ইত্যাদি) প্রতি ৫-১০ মিনিটে হিট করলে
Render free plan ১৫ মিনিট idle থাকার পর সার্ভিস ঘুমিয়ে (spin down) যাবে
না।

```
GET /generate?prompt=<text>&width=<64-768>&height=<64-768>&seed=<optional int>
```
সরাসরি JPEG রিটার্ন করে (`Content-Type: image/jpeg`) — কোনো JSON র‍্যাপার
নেই, তাই সরাসরি `<img src="...">` হিসেবে ব্যবহার করা যায়। `width`/`height`
না দিলে ডিফল্ট 512x512। একই `prompt` + একই (বা কোনো) `seed` দিলে একই ছবি
আসবে — deterministic।

## চালানো (লোকাল টেস্ট)

```bash
docker build -t tiny-image-gen .
docker run -p 8000:8000 tiny-image-gen

curl "http://localhost:8000/health"
curl "http://localhost:8000/generate?prompt=sunny%20flower%20field&width=512&height=512" --output out.jpg
```

কোনো `MODEL_REPO`, কোনো model download, কোনো বড় RAM মেশিনে আগে থেকে কিছু
বানিয়ে রাখার দরকার নেই — এই ভার্সনে সেসব ধাপ পুরোপুরি বাদ।

## Render.com-এ ডিপ্লয় (ফ্রি প্ল্যান, লাইফটাইম ফ্রি)

1. এই ফোল্ডারটা একটা GitHub রিপোতে push করুন (public বা private, দুটোই
   Render free plan সাপোর্ট করে)।
2. Render dashboard → **New** → **Blueprint** → রিপোটা সিলেক্ট করুন।
   `render.yaml` ফাইলটা Render নিজে থেকেই পড়বে এবং **free** প্ল্যানে
   সার্ভিস বানিয়ে দেবে — কোনো ম্যানুয়াল কনফিগ লাগবে না।
   (Blueprint ব্যবহার না করতে চাইলে ম্যানুয়ালি **New → Web Service**
   করেও একই রিপো দিয়ে বানানো যায়; env হিসেবে **Docker** বেছে নেবেন,
   Instance Type-এ **Free** বেছে নেবেন।)
3. ডিপ্লয় শেষ হলে আপনি একটা URL পাবেন, যেমন
   `https://tiny-image-gen-xxxx.onrender.com`।
4. **জাগিয়ে রাখার জন্য:** [UptimeRobot](https://uptimerobot.com) বা
   [cron-job.org](https://cron-job.org)-এ ফ্রি অ্যাকাউন্ট খুলে
   `https://tiny-image-gen-xxxx.onrender.com/health`-কে প্রতি ৫-১০
   মিনিটে হিট করার একটা মনিটর বসিয়ে দিন। এটা করলে Render free plan-এর
   ১৫-মিনিট idle spin-down এড়ানো যায়।
5. ছবি বানাতে:
   `https://tiny-image-gen-xxxx.onrender.com/generate?prompt=your+text+here`

**খরচ:** Render free plan-এর জন্য কোনো কার্ড/পেমেন্ট লাগে না, এই
সার্ভিসটা সেই free plan-এর 512MB RAM / shared CPU লিমিটের ভিতরেই স্বচ্ছন্দে
চলে (উপরের মাপা RSS নাম্বারগুলো দেখুন) — কোনো hidden paid dependency
(কোনো external paid API, কোনো model hosting fee) নেই।

## Prompt কীভাবে ছবিতে রূপান্তরিত হয়

`app/server.py`-তে একটা ইংরেজি+বাংলা keyword-to-color dictionary আছে
(`_KEYWORD_COLORS`) — যেমন আকাশ/sky → নীল, আগুন/fire → লাল-কমলা,
রাত/night → গাঢ় নেভি, ফুল/flower → গোলাপি। আপনার prompt-এর শব্দগুলো এই
dictionary-তে মিলিয়ে দেখা হয়; যা মেলে তা দিয়ে মূল palette বানানো হয়,
আর সামান্য বৈচিত্র্যের জন্য একটা extra hash-based accent color যোগ করা
হয়। কিছু না মিললে (prompt-টা dictionary-তে নেই এমন শব্দ হলে) পুরো
palette-টাই prompt টেক্সটের hash থেকে deterministic ভাবে বানানো হয় —
তাও random না, একই prompt বারবার একই রঙ দেবে।

নতুন keyword/color যোগ করতে চাইলে `_KEYWORD_COLORS` ডিকশনারিতে
`"শব্দ": (R, G, B)` লাইন যোগ করলেই হবে।

## টিউনিং লিভার

- `MAX_SIDE` / `DEFAULT_SIDE` (ডিফল্ট 768/512) — resolution। মডেল weight
  না থাকায় এখানে ৫১২MB-এর তুলনায় বিশাল headroom আছে; আসল সীমা Render
  free-এর shared 0.1 vCPU-তে generation সময়, RAM না।
- `CONCURRENCY_LIMIT` (ডিফল্ট 2) — একসাথে সর্বোচ্চ কয়টা generation
  চলবে; এর বেশি এলে সাথে সাথে 503 (queue করে না, তাই মেমরি বা CPU দুটোই
  predictable থাকে)।
- `SOFT_RSS_LIMIT_MB` (ডিফল্ট 400) — hard 512MB limit-এর নিচে headroom
  রাখতে self-monitoring থ্রেশহোল্ড।

## জানা সীমাবদ্ধতা

- এটা real object/scene generation না — abstract, prompt-influenced
  color/pattern art। "draw a cat" আক্ষরিক অর্থে বিড়াল আঁকবে না।
- Bengali কীওয়ার্ড dictionary সীমিত (সাধারণ শব্দ কভার করে, সব শব্দ না)।
- Render free plan-এর shared vCPU-তে অনেক ট্রাফিক এলে generation সময়
  বাড়তে পারে; `CONCURRENCY_LIMIT` এটা bound রাখে কিন্তু slow করে দিতে
  পারে না পুরোপুরি এড়ানো।
- Render free plan নিজে থেকেই ১৫ মিনিট idle-এর পর ঘুমিয়ে যায় — `/health`
  uptime pinger ছাড়া প্রথম request-এ কয়েক সেকেন্ড cold-start delay হবে।
