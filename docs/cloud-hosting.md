# A small public demo

The website runs on Vercel Hobby. DiffusionGemma runs on Modal when someone asks
for a prediction. Opening the website or browsing examples does not start a GPU.
The first prediction after a quiet period can take several minutes.

```text
Vercel: page + drawing pad + gallery
                  |
             submit a job
                  |
Modal CPU: validate + reserve allowance + return job ID
                  |
Modal A100 80GB: load model + predict + sleep
                  |
Browser polls CPU for the result
```

## Cost settings

- One A100 80GB at most. Zero GPU workers are kept warm.
- The worker shuts down after up to 60 idle seconds.
- Startup is limited to 10 minutes. Each prediction job has a two-minute timeout.
- No automatic inference retries.
- The public allowance is 100 prediction jobs per UTC day and 1,000 per UTC month,
  shared by all visitors. Up to 10 jobs can be submitted per minute.
- Each job accepts one image and at most four questions. Each question runs one
  model evaluation. Failed jobs still use allowance because startup may cost money.
- Duplicate submissions with the same request ID do not start another job.
- Quota reservations use atomic Modal Dict entries. They survive worker restarts.

The job allowance is not a dollar budget. Repeated cold starts can use much more
compute than several predictions in one visit. Set a separate monthly usage limit
in Modal's **Settings → Usage & Billing**. The initial workspace budget chosen by
the owner is $50. Credits are money too; keep the usage limit low even when the
credit balance is high. See [Modal budgets](https://modal.com/docs/guide/budgets).

The static page and galleries remain available if Modal pauses inference. When
the application allowance is exhausted, the app explains the limit. A provider
outage or workspace budget stop may instead return a service error.

## Deploy

Use Python 3.12, Node.js 22, Modal CLI 1.5.5, and Vercel CLI 59.25.0. Connect the
CLI tools with `modal token new` and `vercel login`. Keep credentials outside Git.
The Vercel project must use Hobby only if it is a personal, noncommercial demo.

Prepare the datasets described in [image demos](image-demos.md), then build:

```bash
python scripts/export_cloud_gallery.py
cd web
npm ci
npm run build
cd ..
```

Prepare model weights with an explicit CPU-only job. It downloads the pinned
checkpoint to a Modal Volume and writes a ready marker after checking its shards.
The GPU worker refuses to download missing weights.

```bash
modal run deployment/modal_app.py::prepare_model
modal deploy deployment/modal_app.py
```

The deployment prints the Modal web endpoint. Use that URL to build Vercel's output:

```bash
python scripts/build_vercel.py --backend https://YOUR-ENDPOINT.modal.run
vercel link --project diffusion-jev-sglang
vercel deploy --prebuilt --prod
```

Vercel serves the complete drawing pad, examples and 717 gallery images as static
files. Only `/api/jobs` requests go to Modal. This deployment does not need a Vercel
GPU, an AI Gateway subscription, a paid database, or a custom domain.

The Modal web endpoint is also usable directly. The GPU method itself has no
public HTTP endpoint. The public CPU API is intentionally anonymous and applies
the same shared quota through either URL.

`deployment/gemma-requirements.txt` pins the working text/image engine dependencies.
The image builds the exact SGLang source revision and applies this repo's verified
readout patch. Audio dependencies are excluded because this app does not use them
and the upstream audio pin conflicts with PyTorch. This does not change the model,
precision, denoiser, or candidate-logit readout.

## Verify and stop

```bash
uvx --with playwright python scripts/browser_cloud_demo.py --url https://YOUR-SITE.vercel.app
modal container list
modal billing summary --json
```

The browser check draws a cat, submits one real GPU prediction, and checks gallery
loading and mobile layout. It never inserts a model answer. Wait beyond the idle
window and check that the GPU container is gone. Opening the page again must not
start another GPU.

To stop the Modal deployment, use `modal app stop APP_ID`. Model files remain in
the `diffusion-jev-model` Volume until that volume is separately removed. The old
development GPU rental is a separate bill; migrating this app does not stop it.

Published benchmark numbers still describe the original local deployment. Public
request times also include queueing, cold starts and Internet latency.
