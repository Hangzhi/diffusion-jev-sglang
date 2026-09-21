# Diffusion Jev / SGLang

A local decision service and green React playground, backed by **real LLaDA2.1-mini masked-position logits in SGLang**. Classify text, answer yes/no questions, or score a rubric through `POST /v1/systemone`.

This is an independent experimental implementation, not TypeSafe's Jev model. No generated confidence numbers, heuristic inference fallback, hosted API, or fine-tuning is used.

Validated with real BF16 inference on an A100 80GB: all three playground presets, browser flow, and 1/8/32-question requests. A ten-request warm single-question run measured 147 ms p50 and 153 ms p95. See [measurements and limitations](reports/README.md). Emoji is an introductory classification demo; support triage better demonstrates multiple typed decisions over shared context.

## Run

On **Linux x86-64, Python 3.11/3.12, an NVIDIA A100 80GB or compatible GPU, a CUDA-compatible C++ build toolchain, and a CUDA 13-compatible NVIDIA driver**, install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and run:

```bash
uv run diffusion-jev serve
```

Open **http://localhost:8000**. API documentation: **http://localhost:8000/docs**.

First launch installs the pinned engine dependencies and downloads approximately 32 GB of weights into the Hugging Face cache. Allow additional storage for CUDA/PyTorch and several minutes for first startup. No Node installation is needed to use the bundled webapp. The CLI waits for engine readiness and stops its child process group on Ctrl+C. Both servers bind to localhost by default. No Docker is required.

```bash
# Explicit provisioning, also useful before working offline:
uv sync --extra engine
uv run diffusion-jev doctor

# Attach to an already running, patched JevScoring engine:
uv run diffusion-jev serve --engine-url http://127.0.0.1:30000

# Show the UI while an externally managed model is still loading:
uv run diffusion-jev serve --engine-url http://127.0.0.1:30000 --no-wait-for-engine

# Use another Python environment with the pinned SGLang installed:
uv run diffusion-jev serve --engine-python /path/to/engine/bin/python

# Compare 1, 2, or 4 denoising passes (restart between runs):
uv run diffusion-jev serve --passes 2
```

The engine installer applies a small version-checked patch to the selected SGLang installation. Use a project-specific environment. See [the engine design](docs/engine.md) for the exact source patch and local-branch workflow.

## API

```bash
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusion-jev",
    "state": "I love you so much!",
    "questions": {
      "positive": {"type": "noul", "instructions": "Is the sentiment positive?"},
      "emotion": {
        "type": "choice", "instructions": "Choose the main emotion.",
        "criteria": {"love": null, "sadness": null, "anger": null}
      },
      "intensity": {
        "type": "score", "instructions": "Rate the emotional intensity.",
        "criteria": ["Neutral", "Mild", "Strong"]
      }
    }
  }'
```

- `noul`: a probability of yes, between 0 and 1.
- `choice`: winning option, complete candidate distribution, and confidence.
- `score`: expected **zero-based** rubric level, legend, distribution, and confidence.

The model alias is `diffusion-jev`; the full checkpoint name is also accepted. State and instructions can be strings, objects, or arrays. Question IDs are returned unchanged and are not sent to the model. See [API compatibility](docs/api.md) for explicit differences from [TypeSafe](https://docs.typesafe.ai/api).

## Evaluate and calibrate

Original examples are bundled; they are **smoke tests, not a representative benchmark**. Development and test files are separate.

```bash
uv run diffusion-jev benchmark src/diffusion_jev/data/emoji-dev.jsonl --output reports/dev.json
uv run diffusion-jev calibrate reports/dev.json --output calibration.json
# Restart with the fitted development temperature:
uv run diffusion-jev serve --calibration calibration.json
uv run diffusion-jev benchmark src/diffusion_jev/data/emoji-test.jsonl --output reports/test.json
```

Reports contain raw logits, probabilities, accuracy, negative log-likelihood, multiclass Brier score, 10-bin ECE, p50/p95 client-observed latency, and a dataset hash. Warmup requests are excluded. The calibration command refuses any report containing test records. Never select temperatures or prompts on the test set.

Optional [Kaggle Emojify](https://www.kaggle.com/datasets/alvinrindra/emojify): download it yourself under its applicable terms, inspect the CSV column names, and convert it:

```bash
uv run python scripts/import_emojify.py /path/to/data.csv \
  --text-column text --label-column label
```

The importer deduplicates text and uses a deterministic hash split so duplicate text cannot cross development/test boundaries. Third-party examples and Kaggle credentials are not bundled. Use meaningful emoji labels or edit the criteria descriptions if the source labels are numeric.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check src tests scripts
# Optional frontend work:
cd web
npm ci
npm run build
```

The TypeScript frontend is built into `src/diffusion_jev/static` and included in the wheel. CI checks the API/scoring/calibration tests, Python lint, TypeScript build, and committed frontend reproducibility. GPU validation is recorded separately under `reports/`.

## Scope

v0.1 scores independent question branches using one masked answer position per branch. It supports up to 32 questions, 26 choice candidates, 10 score levels, and 7,900 prompt tokens per question. SGLang batches branches; it does **not yet** pack several question slots into one shared diffusion block. Prefix reuse is disabled in the conservative initial recipe. Candidate probabilities are conditional on the options and are not calibrated until a separate development temperature is fitted.

No claim of Jev-level accuracy or a speed advantage over direct Qwen scoring is made. A controlled comparison requires the same hardware, prompts, task sets, precision, warmup, and concurrency. See [the research protocol](docs/experiments.md).

Optional container packaging is supplied as `Dockerfile` and `compose.yaml` (`docker compose up --build`). It requires NVIDIA Container Toolkit. The uv workflow is primary; container builds are not part of the CPU CI checks.

For a remote A100 with a Mac browser, see [SSH forwarding instructions](docs/remote.md). The browser and API share one forwarded port; inference remains on the remote GPU.
