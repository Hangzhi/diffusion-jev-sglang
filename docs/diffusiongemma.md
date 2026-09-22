# DiffusionGemma as a local Jev service

The image-capable deployment uses `google/diffusiongemma-26B-A4B-it` with native BF16 SGLang inference on one A100 80GB. The HTTP interface retains `noul`, `choice`, and `score` decisions and adds image inputs. This is an independent Jev-compatible service; it is not TypeSafe's hosted model.

Completed measurements and raw artifacts are in the [benchmark report](../reports/diffusiongemma/README.md).

The subsequent [emoji prediction report](../reports/diffusiongemma/emoji/README.md) evaluates five-category Emojify and 20-category TweetEval, including deployment failures and development-only temperature calibration. Run `python scripts/benchmark_emoji.py` in the project environment after preparing the pinned data with `python scripts/prepare_emoji_benchmark.py`; existing outputs are protected from overwrite.

## Pinned runtime

- Model revision: `f7f5b7f5fa82ffc52addd066915886d497f5517b`.
- SGLang source: `ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f`, from [open PR 34061](https://github.com/sgl-project/sglang/pull/34061). This is experimental support outside the released SGLang version used by the older LLaDA deployment.
- Engine: Python 3.12, PyTorch 2.13.0, Transformers 5.12.1, FlashInfer 0.6.18, sglang-kernel 0.4.7, Triton 3.7.1.
- Native `Gemma4Renoise`, 256-position canvas, at most 48 adaptive denoising steps, seed 42. BF16, Triton attention, synchronous scheduling, four running requests maximum, 8,192-token context, no CUDA graphs or prefix cache.

The model weights occupy about 49 GiB of GPU memory. The current service reserves about 72 GiB including KV pools. Startup imports and first-use kernels can take several minutes on network storage.

## Prepare the API, source, and checkpoint

From the repository root, install the lightweight API and test environment:

```bash
uv sync --locked
```

Clone an isolated SGLang checkout. Use the exact commit, rather than the moving PR head; the installer verifies source hashes before applying its patch.

```bash
git clone https://github.com/sgl-project/sglang.git /path/to/sglang-diffusiongemma
git -C /path/to/sglang-diffusiongemma fetch origin pull/34061/head
git -C /path/to/sglang-diffusiongemma checkout ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f
uv run python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    'google/diffusiongemma-26B-A4B-it',
    revision='f7f5b7f5fa82ffc52addd066915886d497f5517b',
    local_dir='/path/to/diffusiongemma-model',
)
PYTHON
```

The weights require approximately 50 GB of disk space, plus space for the engine dependencies and caches. Follow any access requirements on [the model's page](https://huggingface.co/google/diffusiongemma-26B-A4B-it).

## Prepare the GPU engine environment

Use a dedicated Python 3.12 environment on a compatible Linux/CUDA host. The validated core versions are in the pinned-runtime list above and [environment.json](../reports/diffusiongemma/environment.json). The launcher imports SGLang directly from the chosen checkout via `PYTHONPATH`; it does not upgrade your engine environment.

The source's `python/pyproject.toml` lists engine dependencies. The original measurements reused an already compatible environment. For a tested clean container build, see [Modal hosting](cloud-hosting.md). This upstream revision lists `torch==2.13.0` alongside `torchaudio==2.11.0`; resolving its entire dependency list can conflict. This text/image service does not use audio. The Modal build excludes those unused audio dependencies. For the original local setup, start from a compatible SGLang/CUDA environment, verify the versions below, and use its interpreter as `--engine-python`.

```bash
/path/to/gemma-engine/bin/python - <<'PYTHON'
from importlib.metadata import version
expected = {
    'torch': '2.13.0',
    'transformers': '5.12.1',
    'flashinfer-python': '0.6.18',
    'sglang-kernel': '0.4.7',
    'triton': '3.7.1',
}
for name, wanted in expected.items():
    actual = version(name)
    print(name, actual)
    assert actual.split('+')[0] == wanted, (name, actual, wanted)
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)
PYTHON
```

These checks verify core packages and GPU visibility, not the full dependency closure. The project's `engine` extra and Dockerfile target the older LLaDA deployment; do not use them as a DiffusionGemma installer.

## Launch the service

```bash
uv run python scripts/launch_diffusiongemma.py \
  --model-path /path/to/diffusiongemma-model \
  --sglang-source /path/to/sglang-diffusiongemma \
  --engine-python /path/to/gemma-engine/bin/python
```

The launcher binds the web/API to `127.0.0.1:8000` and SGLang to `127.0.0.1:30000`. It checks ports, applies the integration patch, verifies that A–Z are single-token candidates, and supervises both child process groups. Stop both with Ctrl+C. `GET /health` reports model identity and readiness. A real decision request is the end-to-end inference check; readiness polling does not generate GPU work.

Open **http://localhost:8000**. From a laptop with an SSH connection to the GPU host:

```bash
ssh -N -L 18000:127.0.0.1:8000 YOUR_GPU_SSH_ALIAS
```

Then open **http://localhost:18000**. See [image demos](image-demos.md) for gallery preparation and drawing. Keep the engine bound to localhost; exposing the application publicly requires your own authentication and access controls.

To update only the API while keeping a separately managed, patched engine running:

```bash
uv run diffusion-jev serve-gemma \
  --model-path /path/to/diffusiongemma-model \
  --engine-url http://127.0.0.1:30000 --port 8000 --denoising-steps 48
```

Do not use the older `diffusion-jev serve` command to launch Gemma: that command starts LLaDA.

## How the decision engine works

1. **Render the typed question.** [`scoring.py`](../src/diffusion_jev/scoring.py) turns each `choice`, `noul`, or `score` rubric into up to 26 options labeled A–Z. The state is treated as data; question IDs are not part of the prompt.
2. **Encode the input.** [`gemma_backend.py`](../src/diffusion_jev/gemma_backend.py) applies the checkpoint's chat template with thinking disabled. For images, the adapter supplies pixels before the question text; SGLang runs the model's native vision encoder.
3. **Run native diffusion.** SGLang's `Gemma4Renoise` encodes the prompt context, then denoises a 256-position canvas using native self-conditioning and adaptive stopping. The configured maximum is 48 steps. Requesting five output tokens does not reduce the canvas to five positions.
4. **Capture the answer logits.** Our [readout extension](../src/diffusion_jev/sglang_extension/gemma_readout.py) records A–Z logits at position 4 from each batch row's last active step. It retains a finished row's values while other rows continue. It does not replace the sampler or generate synthetic probabilities.
5. **Validate and normalize.** The adapter checks the expected four-token empty thought prefix, finite logits, and step metadata. A format mismatch returns HTTP 503. It applies `softmax(logits / T)` over the options offered by that question; default `T=1`.
6. **Return typed answers.** `choice` returns argmax and the option distribution. `noul` returns the probability of its “yes” option. `score` returns the expected zero-based rubric index, `sum(i * p[i])`. Local `confidence` is normalized entropy, not TypeSafe's proprietary confidence calculation.

Each question is a separate model request. The API submits question branches concurrently; SGLang can batch them. We do not pack several answer slots into one shared canvas or claim that the prefix is encoded only once.

### Why the SGLang patch exists

[`install_gemma_readout.py`](../scripts/install_gemma_readout.py) makes three version-checked changes in the isolated checkout: append the readout wrapper to the sampler; propagate its metadata through the scheduler; and correct context admission for the encoder/canvas boundary. SHA256 checks require the exact upstream files before patching. Re-running the installer on its own patched files is supported.

The pinned upstream PR's normal admission path advertised 256 tokens for a 90-token text prompt, causing an out-of-bounds attention read. It also split image context at a canvas boundary. The correction admits the actual context length within the available budget and reserves a complete canvas for denoising. Sampling math, stopping, and weights are unchanged. Regression tests cover short/image contexts, canvas allocation, and retaining an early-finished row's logits.

## What the probabilities mean

The model performs its native diffusion generation with thinking disabled. It still generates the checkpoint's documented empty thought block (`<|channel>thought\n<channel|>`): four protocol tokens precede the answer. The adapter verifies those tokens and reads A–Z logits at canvas position 4 from that request's **last active denoising step**, then selects the supplied answer letters and applies a T=1 softmax. A mismatched protocol is an error, not a fabricated distribution. The native request returns five tokens including the protocol; the full 256-position canvas is denoised. Each question is a separate request.

These logits are self-conditioned by earlier denoising steps. They are **not independent masked-token likelihoods or calibrated probabilities of correctness**. This differs from the previous LLaDA one-pass readout. `meta.probability_source` is `self_conditioned_denoiser_logits`; `passes=48` is the configured maximum. Per-question `denoising_steps` reports actual work, and `answer_position` records the offset. `candidate_mass` reports the full-vocabulary mass in the offered options; `unrestricted_first_token` is the first **answer** token after the protocol. Candidate normalization can hide substantial mass outside the options, so retain these diagnostics when evaluating.

## Flower dataset and image API

Prepared from [Kaggle Flowers, version 1](https://www.kaggle.com/datasets/abdelrahmanatef01/flowers-dataset-for-image-classification), whose uploader declares Apache 2.0. The source metadata and archive hash are saved locally. Images are not committed to Git.

```bash
uv run python scripts/prepare_flowers.py
```

Preparation removes three repeated image entries and one unique image with conflicting labels, yielding **2,742 images**. Exact decoded pixels are deduplicated before the deterministic split: 1,929 train, 288 dev, 525 test. Near-duplicates are not detected; model-pretraining overlap is unknown. The frozen benchmark contains 20 test images from each of daisy, dandelion, rose, sunflower, and tulip. No fitting is performed. For the separate sketch gallery and drawing pad, see [Doodle Detective](image-demos.md).

Images are EXIF-oriented, resized to at most 768 pixels per side, re-encoded as JPEG, and named with opaque SHA256 IDs. The model receives pixels without filenames or gold labels. Gallery labels are shown in the interface for checking the returned prediction.

- `GET /api/datasets/flowers`: counts and provenance.
- `GET /api/datasets/flowers/images?split=test&label=rose&offset=0&limit=20`: gallery records.
- `GET /api/datasets/flowers/image/{id}`: prepared JPEG.
- `POST /v1/systemone`: add `"images": ["<gallery ID>"]`, or a JPEG/PNG/WebP data URL. Up to four images are accepted, uploads up to 6 MB each with a combined encoded-input bound. Arbitrary paths and remote URLs are rejected.

## Reproduce the benchmarks

Use an unused output directory for each run. Run serially while no browser evaluations or other GPU jobs are active:

```bash
uv run python scripts/compare_jevbench.py \
  --backend local --upstream /path/to/jevbench \
  --expected-passes 48 --expected-model google/diffusiongemma-26B-A4B-it \
  --output reports/diffusiongemma/jevbench
uv run python scripts/compare_jevbench.py \
  --summarize --upstream /path/to/jevbench \
  --output reports/diffusiongemma/jevbench
uv run python scripts/compare_published_jev.py \
  --upstream /path/to/jevbench --output reports/diffusiongemma/jevbench
uv run python scripts/benchmark_flowers.py
```

JevBench uses the exact 231 public tasks and upstream scoring. Hosted Jev comparison uses matched published per-item results when a fresh Gateway credential is unavailable; that column must be labeled historical. Published artifacts lack matching probability vectors and image tests, so no hosted calibration or flower-accuracy comparison is implied. See [Gateway setup](gateway.md) for an authenticated live run.

## Provenance and scope

- [Google model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card): architecture, native image inputs, diffusion sampling, and the empty thought protocol.
- [Pinned SGLang integration](https://github.com/sgl-project/sglang/tree/ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f): runtime implementation.
- [TypeSafe model documentation](https://docs.typesafe.ai/models): official Jev 1.13.0 is text-only. The image field is our extension, not a claim of native Jev image compatibility.

The local implementation reproduces a useful decision API shape, not the official model's training or calibration. Accuracy depends on prompts, options, model, and deployment; test on your own data and calibrate only on a separate development split.
