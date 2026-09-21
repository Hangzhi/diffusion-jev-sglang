# JevBench: local load and reproducibility measurements

Fresh measurements on the same 231 public JevBench tasks, with the original state, rubrics and option order. Model: BF16 LLaDA2.1-mini on the A100 80GB; one pass, temperature 1. Each request contains one decision.

| Run | Concurrent requests | Correct | Valid | Median | p95 | Valid decisions/s |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 1 | 152/231 (65.8%) | 231/231 | 172 ms | 2346 ms | 1.66 |
| c4 | 4 | 151/231 (65.4%) | 231/231 | 216 ms | 2749 ms | 5.34 |
| c1-after | 1 | 152/231 (65.8%) | 231/231 | 190 ms | 2324 ms | 1.66 |
| c4-repeat | 4 | 151/231 (65.4%) | 231/231 | 217 ms | 2778 ms | 5.50 |

## Same-item stability

- c1 → c4: 10/231 predicted labels changed; probability TVD (mean/max): 0.039772 / 0.318518 across 231 pairs with valid distributions.
- c1 → c1-after: 0/231 predicted labels changed; probability TVD (mean/max): 0.000000 / 0.000000 across 231 pairs with valid distributions.
- c1 → c4-repeat: 10/231 predicted labels changed; probability TVD (mean/max): 0.038580 / 0.271936 across 231 pairs with valid distributions.
- c4 → c1-after: 10/231 predicted labels changed; probability TVD (mean/max): 0.039772 / 0.318518 across 231 pairs with valid distributions.
- c4 → c4-repeat: 0/231 predicted labels changed; probability TVD (mean/max): 0.007822 / 0.269479 across 231 pairs with valid distributions.
- c1-after → c4-repeat: 10/231 predicted labels changed; probability TVD (mean/max): 0.038580 / 0.271936 across 231 pairs with valid distributions.
- previous run → c1: 0/231 predicted labels changed; probability TVD (mean/max): 0.000000 / 0.000000 across 231 pairs with valid distributions.

TVD is half the sum of absolute probability differences. A zero label-change count does not imply identical probabilities. Changed item IDs and correctness are recorded in `load-comparison.json`.

Treat concurrency and batch composition as part of the measured inference configuration. These observations do not isolate the cause of any output differences in the model or serving implementation.

## Prompt length

| Run | Input tokens | Items | Accuracy | Median | p95 |
|---|---|---:|---:|---:|---:|
| c1 | 1–256 | 127 | 84.3% | 138 ms | 202 ms |
| c1 | 257–1,024 | 60 | 53.3% | 354 ms | 770 ms |
| c1 | 1,025–4,096 | 44 | 29.5% | 2208 ms | 3090 ms |
| c4 | 1–256 | 127 | 83.5% | 182 ms | 363 ms |
| c4 | 257–1,024 | 60 | 53.3% | 515 ms | 2165 ms |
| c4 | 1,025–4,096 | 44 | 29.5% | 2216 ms | 3314 ms |
| c1-after | 1–256 | 127 | 84.3% | 152 ms | 208 ms |
| c1-after | 257–1,024 | 60 | 53.3% | 357 ms | 788 ms |
| c1-after | 1,025–4,096 | 44 | 29.5% | 2152 ms | 3014 ms |
| c4-repeat | 1–256 | 127 | 83.5% | 185 ms | 270 ms |
| c4-repeat | 257–1,024 | 60 | 53.3% | 521 ms | 2088 ms |
| c4-repeat | 1,025–4,096 | 44 | 29.5% | 2112 ms | 3202 ms |

Token count includes the local model's prompt template. Task content and difficulty differ across length groups; this is not a controlled causal test of length.

## Method and limits

Runs execute sequentially on the same engine. Two serial warmups precede each run and are excluded. Tasks are submitted in the frozen easy/original/hard order; concurrent runs save completion order and compare by exact task ID. Concurrency is a closed-loop cap on in-flight requests. Request latency includes server queueing; throughput divides valid decisions by measured wall time after warmup. Failures remain in accuracy denominators. There are no retries.

Each row is one sweep; repeated settings are separate measurements, not confidence intervals or an open-loop saturation test. These runs do not establish peak capacity or cloud latency. No prompts, thresholds, temperatures or labels were tuned using these results.

Historical hosted Jev scored 200/231 (86.6%) on these items; that remains a published reference, not a new Gateway measurement. A valid local Gateway credential is still needed for a fresh hosted comparison. [Earlier comparison](../jevbench/README.md).

## Raw artifacts

- [c1 measurements](c1/local.jsonl) · [run metadata](c1/local-run.json)
- [c1 versus historical Jev](c1/published-comparison.json)
- [c4 measurements](c4/local.jsonl) · [run metadata](c4/local-run.json)
- [c1-after measurements](c1-after/local.jsonl) · [run metadata](c1-after/local-run.json)
- [c4-repeat measurements](c4-repeat/local.jsonl) · [run metadata](c4-repeat/local-run.json)
- [Aggregate and paired results](load-comparison.json)
- [Engine settings and source hashes](environment.json)
