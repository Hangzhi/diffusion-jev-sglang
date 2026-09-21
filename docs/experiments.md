# Research protocol

The first milestone is real model inference through the API and browser, not a claim of beating Jev. The bundled 20 original emoji sentences only test the plumbing.

The [completed emoji evaluation](../reports/diffusiongemma/emoji/README.md) adds two frozen tasks: all 50 distinct retained Kaggle Emojify test sentences and 1,000 TweetEval test tweets across 20 emoji labels. Its development splits are separate, test failures remain in the accuracy/F1 denominators, and published TweetEval predictions are compared on matched source indices. These samples are distinct from the bundled smoke sentences and from the 231-task JevBench comparison. Reproduction is provided by `scripts/prepare_emoji_benchmark.py` and `scripts/benchmark_emoji.py`.

For a meaningful comparison:

1. Freeze a licensed task corpus and deterministic train/development/test split. Use emoji, spam, support routing, and review scoring; preserve provenance. Group duplicates before splitting.
2. Freeze the prompt and option-label mapping. Test sensitivity to option order on development data. Do not tune on held-out examples.
3. Run LLaDA with 1, 2, and 4 passes, BF16, on the same A100, after warmup. Test 1, 8, and 32 question branches per request.
4. Run a Qwen direct-first-token-logit baseline with identical labeled options, state, precision, hardware, and concurrency. Do not compare against long-form Qwen generation.
5. Fit a temperature separately for each model/pass setting using only development logits. Keep an uncalibrated report too.
6. Report accuracy, NLL, Brier, ECE, p50/p95 complete HTTP latency, requests/sec and decisions/sec. Include batch size, input-token distributions, model revision, SGLang version/patch, GPU, cache policy, and failure rate.
7. Use enough held-out examples for confidence intervals and a range of request sizes. Preserve raw report records.

The supplied benchmark CLI measures sequential end-to-end latency and classification quality. Run `uv run python scripts/benchmark_matrix.py --output reports/matrix.json` for the separate multi-request throughput/multi-question matrix; sequential p50 is not throughput. No fine-tuning or packed multi-slot shared-context experiment is included in v0.1.

The pinned public JevBench harness also supports local request concurrency via `scripts/compare_jevbench.py --concurrency 4`. Its [load report](../reports/jevbench-load-20260921/README.md) measures throughput on the full 231-item corpus and checks whether native probabilities and predicted labels change under batching. Keep concurrency fixed when comparing decision quality or calibration; changing batch composition can change model outputs even with the same prompts and temperature.

Calibration reports containing any non-development record are rejected. A calibration file is bound to model and denoising passes. Refit after changing prompts, candidate mappings, model weights, or datasets.
