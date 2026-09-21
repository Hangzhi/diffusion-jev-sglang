# Research protocol

The first milestone is real model inference through the API and browser, not a claim of beating Jev. The bundled 20 original emoji sentences only test the plumbing.

For a meaningful comparison:

1. Freeze a licensed task corpus and deterministic train/development/test split. Use emoji, spam, support routing, and review scoring; preserve provenance. Group duplicates before splitting.
2. Freeze the prompt and option-label mapping. Test sensitivity to option order on development data. Do not tune on held-out examples.
3. Run LLaDA with 1, 2, and 4 passes, BF16, on the same A100, after warmup. Test 1, 8, and 32 question branches per request.
4. Run a Qwen direct-first-token-logit baseline with identical labeled options, state, precision, hardware, and concurrency. Do not compare against long-form Qwen generation.
5. Fit a temperature separately for each model/pass setting using only development logits. Keep an uncalibrated report too.
6. Report accuracy, NLL, Brier, ECE, p50/p95 complete HTTP latency, requests/sec and decisions/sec. Include batch size, input-token distributions, model revision, SGLang version/patch, GPU, cache policy, and failure rate.
7. Use enough held-out examples for confidence intervals and a range of request sizes. Preserve raw report records.

The supplied benchmark CLI measures sequential end-to-end latency and classification quality. Run `uv run python scripts/benchmark_matrix.py --output reports/matrix.json` for the separate multi-request throughput/multi-question matrix; sequential p50 is not throughput. No fine-tuning or packed multi-slot shared-context experiment is included in v0.1.

Calibration reports containing any non-development record are rejected. A calibration file is bound to model and denoising passes. Refit after changing prompts, candidate mappings, model weights, or datasets.
