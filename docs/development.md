# Development environment

This is the environment used for the sketch demo and the local GPU checks. It is a record of the working setup, not a lockfile for every system package.

| Part | Version or setting |
|---|---|
| GPU host | Ubuntu 24.04.3 LTS, Linux x86_64 |
| GPU | One NVIDIA A100-SXM4-80GB |
| NVIDIA driver | 580.159.04 |
| Python | 3.12.3; separate API and GPU engine environments |
| uv | 0.9.0 |
| Node.js / npm | 22.14.0 / 10.9.2 |
| Frontend | React 19.1.0, TypeScript 5.8.3, Vite 6.4.3 |
| Model | `google/diffusiongemma-26B-A4B-it` |
| Model revision | `f7f5b7f5fa82ffc52addd066915886d497f5517b` |
| SGLang revision | `ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f`, with this repo's integration patches |
| PyTorch / Transformers | 2.13.0 / 5.12.1 |
| FlashInfer / sglang-kernel / Triton | 0.6.18 / 0.4.7 / 3.7.1 |
| Inference | BF16, Triton attention, 256-position canvas, up to 48 denoising steps |
| Web app and API | `127.0.0.1:8000` |
| Model engine | `127.0.0.1:30000` |
| Laptop browser | Mac, SSH forwarding to `localhost:18000` |
| Demo capture | Playwright Chromium and FFmpeg 6.1.1 |

The engine reuses a working GPU environment. A complete clean-machine install has not been validated. The pinned upstream dependency list has a PyTorch/torchaudio version conflict. See the [setup guide](diffusiongemma.md#prepare-the-gpu-engine-environment) before installing it.

The runtime configuration and model source hashes are in [environment.json](../reports/diffusiongemma/environment.json). Frontend dependencies are pinned in `web/package-lock.json`; API and test dependencies are in `uv.lock`.

## Change the app

```bash
uv sync --locked
uv run pytest
uv run ruff check src tests scripts
cd web
npm ci
npm run build
```

Vite writes the production app into `src/diffusion_jev/static`. Commit these files with the frontend source. The running API serves the new files without restarting the model. Reload the browser to use the new build. If an old tab still shows the previous page, do a hard refresh.

Doodle Detective opens by default on an image-capable backend. Direct links use `/#doodle`, `/#flowers`, and `/#text`. Text-only backends open the text demo.

## Record the live sketch demo

Start the real DiffusionGemma service first. For the full browser check, also prepare the flower and Quick, Draw! galleries using the commands in [image demos](image-demos.md).

Install Playwright's browser and FFmpeg on the capture host. Then run:

```bash
uvx --with playwright playwright install chromium
uvx --with playwright python scripts/browser_sketch_demo.py \
  --url http://127.0.0.1:8000/ \
  --record --output .cache/sketch-demo
```

The script draws a scruffy cat with browser mouse events. Its uneven outline, ears, and whiskers use hand-authored points with a little seeded wobble. It clicks **Guess doodle**, waits for the real GPU response, and records the answer. FFmpeg converts the browser video into a looping GIF at 12 fps. Playback stays at normal speed. The script does not insert an answer or change the API response.

The output includes `sketch-demo.gif`, desktop and mobile screenshots, and `sketch-validation.json` with the actual responses. The browser checks also cover touch drawing, clear/reset, uploads, gallery navigation, text decisions, and flowers. Missing gallery data, slow example loading, and text-only capabilities are simulated in separate interface checks; inference responses are never mocked.

To update the README recording, review the GIF and copy it into `reports/doodle-demo/sketch-demo.gif`. Update the validation report with it. The few live requests in this script are integration checks, not an accuracy or latency benchmark.
