# Diffusion Jev / SGLang

Draw a wonky cat. Click **Guess doodle**. See what the model thinks.

**[Open the live demo](https://diffusion-jev-sglang.vercel.app/#doodle).** No setup needed. The model sleeps between visits, so the first prediction may take a few minutes.

![Live sketch demo: drawing a scruffy cat and getting a prediction from local DiffusionGemma](reports/doodle-demo/sketch-demo.gif)

This GIF records the running app and a real model response. It is one example, not an accuracy test. [Recording and checks](reports/doodle-demo/README.md).

We use **DiffusionGemma** with **SGLang** to make a local decision API. Give it text or an image. Ask it to pick a label, answer yes/no, or give a score. No fine-tuning is needed.

This is an independent project with a Jev-like API. It does not use TypeSafe's Jev model or its calibration.

## Try the sketch demo

With DiffusionGemma running, open **http://localhost:8000/#doodle**. The drawing pad opens first.

1. Draw an airplane, apple, bicycle, cat, clock, fish, pizza, or umbrella.
2. Click **Guess doodle** at the top.
3. See the answer and the scores for all eight choices. Use **Clear drawing** to try again.

You can draw with a mouse or a finger. You can also upload an image. **Try a sketch** opens 192 drawings from Google Quick, Draw! The gallery download is optional; the drawing pad works without it.

The model must choose from the listed objects. It can be wrong. The scores compare these choices; they are not the chance that the answer is correct.

**Flowers** and **Text decisions** are separate tabs. The text demo includes emoji prediction. Images go to your own inference server.

## Run it

The tested setup uses **one NVIDIA A100 80GB**, Linux, and Python 3.12. It needs about 50 GB for model weights on disk. The running engine reserves about 72 GiB of GPU memory.

First follow the [DiffusionGemma setup guide](docs/diffusiongemma.md). It covers the model download, the pinned SGLang source, and the GPU environment. This is still an experimental setup. We tested it in an existing GPU environment; a full install on a clean machine has not been validated.

Then, from this repository:

```bash
uv sync --locked
uv run python scripts/launch_diffusiongemma.py \
  --model-path /path/to/diffusiongemma-model \
  --sglang-source /path/to/sglang-diffusiongemma \
  --engine-python /path/to/gemma-engine/bin/python
```

Open **http://localhost:8000/#doodle**. The web app is included. You only need Node.js if you change the frontend.

If the GPU is on another machine, run this on your laptop:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18000:127.0.0.1:8000 YOUR_GPU_SSH_ALIAS
```

Keep the terminal open. Visit **http://localhost:18000/#doodle**. [More about remote access](docs/remote.md).

For the optional galleries, run these on the API host:

```bash
uv run python scripts/prepare_quickdraw.py
uv run python scripts/prepare_flowers.py
```

## How it becomes a decision engine

```text
  [Your sketch or text] + [Question + choices]
                       |
                       v
           [DiffusionGemma / SGLang]
                       |
                       v
            [Answer-letter scores]
                       |
                       v
      [Choice / yes-no / score + distribution]
```

We turn your question into letter choices, such as A for cat and B for clock. SGLang runs DiffusionGemma on the text and image. Our adapter reads the model's answer-letter scores and maps them back to your labels.

The API returns a choice, a yes/no score, or a rubric score. It also returns the distribution over your options. Each question gets its own model request. SGLang can batch these requests.

The scores come from the denoiser after self-conditioning. They are not calibrated probabilities. See the [implementation guide](docs/diffusiongemma.md#how-the-decision-engine-works) for the readout, SGLang patches, and limits. See the [API reference](docs/api.md) for request examples.

## Results so far

All three rows use the same 231 public JevBench tasks.

| Model | Correct | Accuracy | Median / p95 latency |
|---|---:|---:|---:|
| Local DiffusionGemma 26B A4B | 196/231 | 84.8% | 183 / 392 ms |
| Local LLaDA2.1-mini | 152/231 | 65.8% | 172–190 / 2,324–2,346 ms |
| Official Jev 1.13.0, historical published run | 200/231 | 86.6% | 665 / 803 ms |

The local models ran on an A100. The published Jev run used a remote API. These timings do **not** prove that one engine is faster. We have not completed a fresh hosted Jev run. [Full comparison and raw results](reports/three-model-comparison/README.md).

DiffusionGemma also scored **98/100 on flowers**, **42/50 on Emojify**, and **311/1,000 on TweetEval**. The TweetEval total includes 105 output-format errors. [Image results](reports/diffusiongemma/README.md) · [Emoji results](reports/diffusiongemma/emoji/README.md).

Doodle Detective is a demo, not a held-out benchmark. The image demos use DiffusionGemma. Official Jev 1.13.0 and LLaDA2.1-mini are text-only.

## Development

We developed this on Ubuntu 24.04.3 with Python 3.12.3, Node.js 22.14.0, and one A100 80GB. The engine uses PyTorch 2.13.0 and a pinned SGLang checkout. [Full environment and recording steps](docs/development.md).

```bash
uv sync --locked
uv run pytest
uv run ruff check src tests scripts
cd web
npm ci
npm run build
```

The build updates the bundled web app in `src/diffusion_jev/static`. Reload the browser after a build.

More: [image demos and dataset credits](docs/image-demos.md) · [earlier LLaDA backend](docs/engine.md) · [benchmark method](docs/experiments.md) · [AI Gateway examples](docs/gateway.md).

The project code uses the [MIT license](LICENSE). Model weights and datasets have their own [licenses and credits](THIRD_PARTY.md).
