# Diffusion Jev — Visual decisions with DiffusionGemma

**English** | [简体中文](README.zh-CN.md)

**A diffusion model that guesses your doodles, recognizes flowers, and picks an emoji for your message. Built with DiffusionGemma and SGLang.**

[Try the playground](https://diffusion-jev-sglang.vercel.app/#doodle) · [Model](https://huggingface.co/google/diffusiongemma-26B-A4B-it) · [Results](reports/three-model-comparison/README.md) · [Setup guide](docs/diffusiongemma.md)

**One model · Three demos · No fine-tuning**

## See it in action

### Draw a cat → get “cat”

[![Draw a scruffy cat and get a real DiffusionGemma prediction. Click to try the playground.](reports/doodle-demo/sketch-demo.gif)](https://diffusion-jev-sglang.vercel.app/#doodle)

A wonky sketch goes in. **Cat** comes out, with scores for all 16 choices. Draw with a mouse or a finger, upload your own image, or try one of **384 Google Quick, Draw! sketches**. Click the GIF to open the drawing pad.

The 16 choices are **airplane, apple, bicycle, cat, clock, fish, pizza, umbrella, dog, car, house, tree, sun, star, cup, and sailboat**. The model must pick from this list.

### Choose a flower → get its class

[![A daisy photo on the left and the model's daisy prediction on the right. Click to try flower classification.](reports/readme-demo/flower.png)](https://diffusion-jev-sglang.vercel.app/#flowers)

Pick a photo and click **Classify image**. Here, the model picks **daisy** from daisy, dandelion, rose, sunflower, and tulip. It receives the image pixels. The dataset label is shown only after the prediction, so you can compare.

### Write a message → get an emoji, a yes/no answer, and a score

[![The message “I love you so much. You make every day better!” produces a heart emoji, a positive answer, and intensity 3 out of 3. Click to try text decisions.](reports/readme-demo/emoji.png)](https://diffusion-jev-sglang.vercel.app/#text)

**“I love you so much. You make every day better!”** returns **❤️**, **positive**, and intensity **3/3**. One message shows all three answer types. Change the text or edit the questions to try your own decision task.

All three examples show real responses from the local deployment. They are individual examples; the benchmark results are below. [Sketch recording](reports/doodle-demo/README.md) · [Photo and text capture](reports/readme-demo/README.md).

**Trying the public app:** the GPU sleeps between visits to keep costs low. The first prediction may take a few minutes. Drawing and browsing do not start the GPU.

## What it does

- **Text and images:** DiffusionGemma reads the image pixels along with your question.
- **Three answer types:** `choice` picks a label, `noul` answers yes/no, and `score` rates ordered levels.
- **Scores for every option:** inspect the distribution and raw result in the app.
- **Your own server:** run the web app and decision API with SGLang on one GPU.

This is an independent project with a Jev-like API. It uses Google's published DiffusionGemma weights. It does not use TypeSafe's Jev model or its calibration. The image demos use DiffusionGemma; the compared Jev 1.13.0 and LLaDA2.1-mini interfaces are text-only.

## How it works

```text
  Picture or text + Question + Choices
                   |
                   v
         DiffusionGemma / SGLang
                   |
                   v
          Answer-letter scores
                   |
                   v
   Choice / Yes-no / Score + Distribution
```

We map your options to letters: A for cat, B for clock, and so on. DiffusionGemma runs its denoising steps. The adapter reads the answer-letter scores from the last active step and maps them back to your labels.

Each question gets its own model request. SGLang can batch these requests. This uses the existing model without training a new decision head.

**The scores compare the supplied options. They are not calibrated probabilities of being correct.** A high score can still be wrong. [Readout details](docs/diffusiongemma.md#how-the-decision-engine-works) · [Local API](docs/api.md).

## Results so far

The same **231 public JevBench tasks**. Local runs are from September 21, 2026:

| Model | Correct | Accuracy | Median / p95 latency |
|---|---:|---:|---:|
| Local DiffusionGemma 26B A4B | 196/231 | 84.8% | 183 / 392 ms |
| Local LLaDA2.1-mini | 152/231 | 65.8% | 172–190 / 2,324–2,346 ms |
| Official Jev 1.13.0, historical published run | 200/231 | 86.6% | 665 / 803 ms |

Local runs used an A100. Official Jev numbers come from an older published API run. Different hardware and network paths mean these timings do not establish which model is faster. We have not completed a fresh hosted Jev comparison. The public playground also adds queueing, startup, and network time.

Separate DiffusionGemma tests: **98/100 flowers**, **42/50 Emojify**, and **311/1,000 TweetEval**. TweetEval includes 105 output-format failures, counted as wrong. Doodle Detective has no held-out accuracy result.

[Full comparison](reports/three-model-comparison/README.md) · [Image results](reports/diffusiongemma/README.md) · [Emoji results](reports/diffusiongemma/emoji/README.md)

## Run locally

The tested setup uses **one NVIDIA A100 80GB**, Linux, and Python 3.12. Allow about **50 GB of disk space for weights**. The engine reserves about **72 GiB of GPU memory**.

```bash
git clone https://github.com/Hangzhi/diffusion-jev-sglang.git
cd diffusion-jev-sglang
uv sync --locked
```

Follow the [setup guide](docs/diffusiongemma.md) to download the model and prepare the pinned SGLang environment. Then replace the paths below with your own:

```bash
uv run python scripts/launch_diffusiongemma.py \
  --model-path /path/to/diffusiongemma-model \
  --sglang-source /path/to/sglang-diffusiongemma \
  --engine-python /path/to/gemma-engine/bin/python
```

Open **http://localhost:8000/#doodle**. The web app is included. The local decision endpoint is **`POST /v1/systemone`**. Node.js is only needed to change the frontend.

For the optional galleries, run:

```bash
uv run python scripts/prepare_quickdraw.py
uv run python scripts/prepare_flowers.py
```

The drawing pad works without these downloads. If your GPU is on another machine, use the [SSH forwarding guide](docs/remote.md).

## Public hosting

Vercel serves the page, drawing pad, and galleries. Modal starts an A100 only when you request a prediction. We keep at most one GPU worker and set its idle window to 60 seconds.

The public allowance is **100 prediction jobs per day and 1,000 per month**, shared by all visitors. A separate monthly compute budget can pause predictions earlier. [Deploy your own](docs/cloud-hosting.md).

## Development

| Part | Tested environment |
|---|---|
| GPU and OS | A100 80GB · Ubuntu 24.04.3 |
| Model runtime | Python 3.12.3 · PyTorch 2.13.0 · Transformers 5.12.1 |
| SGLang | Pinned experimental revision with this repo's readout patch |
| Frontend | Node.js 22.14.0 · React 19.1 · TypeScript 5.8.3 · Vite 6.4.3 |

[Exact versions and source revisions](docs/development.md) · [Tested Modal container](docs/cloud-hosting.md)

```bash
uv sync --locked
uv run pytest
uv run ruff check src tests scripts deployment
cd web
npm ci
npm run build
```

The build updates the bundled web app in `src/diffusion_jev/static`.

[Dataset preparation and credits](docs/image-demos.md) · [Benchmark method](docs/experiments.md) · [LLaDA backend](docs/engine.md) · [AI Gateway examples](docs/gateway.md)

Project code: [MIT](LICENSE). Model weights and datasets have [separate licenses](THIRD_PARTY.md).
