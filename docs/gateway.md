# Vercel AI Gateway and JevBench

The Node examples are server-side scripts under `web`, using the existing npm project. The React browser bundle never imports the Gateway modules or credentials. AI SDK 7 exposes Jev through `experimental_evaluate`; text models use `generateText`.

## Credential setup

Run this yourself in a remote terminal; input is hidden and does not enter shell history:

```bash
python /workspace/diffusion-jev-sglang/scripts/set_gateway_key.py
```

This writes `AI_GATEWAY_API_KEY` to `web/.env.local` with mode 0600. The file is ignored by Git. Do not paste a key into chat. Do not give it a `VITE_` prefix. The runtime loads the file without displaying it; error reporting deliberately omits SDK errors, headers and response bodies.

## Run the examples

With Node 22 and npm:

```bash
cd /workspace/diffusion-jev-sglang/web
npm ci
npm run gateway:check
npm run gateway:example  # openai/gpt-5.5: invent a holiday and its traditions
npm run gateway:jev      # typesafe-ai/jev: boolean, choice and score
```

Successful requests save allowlisted output under `reports/gateway`. Installing or compiling alone does not verify access to either model. Jev uses its evaluation API, with `noul` mapped to SDK `boolean` and back; no prompt asks it to invent probabilities.

## Reproduce the public comparison

```bash
git clone https://github.com/fstandhartinger/jevbench.git /path/to/jevbench
git -C /path/to/jevbench checkout c6004e008ffba24aec091261ca1a5c02f7324702
cd /workspace/diffusion-jev-sglang
uv run python scripts/compare_jevbench.py --upstream /path/to/jevbench --backend local
uv run python scripts/compare_jevbench.py --upstream /path/to/jevbench --backend gateway
uv run python scripts/compare_jevbench.py --upstream /path/to/jevbench --summarize
```

Use a fresh `--output` directory for a new run: the runner refuses to overwrite measurements or run metadata. Put Node on PATH for the Gateway worker. Gateway benchmarking sends these public benchmark inputs to Vercel and TypeSafe. Each run has 231 decisions, one question per request, no retries, and two excluded warmups. Original rubrics and option order are preserved; labels, rationales and gold probabilities are used only by the scorer. Model settings remain frozen at one pass and T=1 for the local run; the runner verifies these on successful responses. Transport failures and invalid distributions count wrong, and no calibration values are synthesized for them. Summarization verifies task coverage and dataset hashes and re-scores native probabilities, rather than trusting saved correctness flags.

## Local throughput and stability

Measure the full public workload with one and four in-flight requests. Run these sequentially on an otherwise idle server:

```bash
uv run python scripts/compare_jevbench.py --upstream /path/to/jevbench --backend local --concurrency 1 --output reports/my-load-run/c1
uv run python scripts/compare_jevbench.py --upstream /path/to/jevbench --backend local --concurrency 4 --output reports/my-load-run/c4
uv run python scripts/analyze_jev_load.py --upstream /path/to/jevbench --runs reports/my-load-run/c1 reports/my-load-run/c4 --reference reports/jevbench --output reports/my-load-run
```

Concurrency is supported for the local backend; Gateway runs currently remain serial. Both local runs use the same item submission order. Each records latency per request, valid decisions per measured wall-clock second, and complete task-level outcomes. The report checks label and probability stability by exact task ID and groups latency by actual prompt token count. This is a closed-loop load comparison, not a peak-capacity estimate. The older repeated emoji matrix measures a different workload and should not be reported as JevBench throughput.

[Measured September 21 local load comparison](../reports/jevbench-load-20260921/README.md).

Upstream `score_task` determines argmax accuracy for all types, including ordinal scores, and validates distributions. Ordinal MAE uses the probability-weighted level. Brier, top-label ECE and exact-gold TVD on the ten probability items measure different aspects of calibration. Pooled public accuracy is not the official JevBench composite score; private cohorts and the imported judge tier are absent. Cloud network latency and local loopback latency are reported as measured, with no hypothetical production multiplier. Local GPU cost is not assumed to be zero.

If a new Gateway run cannot be authenticated, a separate command compares local measurements to **historical published Jev 1.13.0** outcomes for the exact public IDs:

```bash
uv run python scripts/compare_published_jev.py --upstream /path/to/jevbench
```

That artifact is labeled `published-comparison.json`; it is never written as a new Gateway run. Published public-item results contain outcomes and timings but no distributions, so matched calibration cannot be inferred from them.

Sources: [Vercel evaluation API](https://vercel.com/docs/ai-gateway/getting-started/evaluation), [TypeSafe-compatible Gateway API](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe), [JevBench](https://github.com/fstandhartinger/jevbench).
