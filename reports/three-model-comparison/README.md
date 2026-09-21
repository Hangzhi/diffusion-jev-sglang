# Local DiffusionGemma, local LLaDA, and official Jev

Comparison compiled from completed September 21, 2026 runs. All three columns cover the same 231 public JevBench task IDs and upstream scoring. The local engines used the same A100 80GB host, BF16, T=1, two excluded warmups, serial requests, and no retries. No new inference was performed for this consolidation.

| Metric | Local DiffusionGemma | Local LLaDA2.1-mini | Official Jev 1.13.0, published |
|---|---:|---:|---:|
| Overall accuracy | **196/231 — 84.8%** | **152/231 — 65.8%** | **200/231 — 86.6%** |
| Easy accuracy | 48/48 — 100.0% | 48/48 — 100.0% | 48/48 — 100.0% |
| Original accuracy | 70/72 — 97.2% | 55/72 — 76.4% | 71/72 — 98.6% |
| Hard accuracy | 78/111 — 70.3% | 49/111 — 44.1% | 81/111 — 73.0% |
| Median latency | 183 ms | 172–190 ms | 665.0 ms |
| p95 latency | 392 ms | 2324–2346 ms | 802.5 ms |
| Serial valid decisions/second | 4.42 | 1.66 | Not available |
| Valid local distributions | 231/231 | 231/231 | Not established from published vectors |

Latency is client-observed. LLaDA ranges span the two completed serial load runs, not confidence intervals. Its initial earlier sweep had median 183 ms and p95 2,594 ms; the repeat measurements above provide both throughput and repeatability. Four-request LLaDA runs reached 5.34–5.50 decisions/s, but those are a different concurrency setting and are not substituted into this serial comparison. Gemma has no matching four-request sweep.

Official Jev figures come from the [pinned published per-task artifact](https://github.com/fstandhartinger/jevbench/blob/c6004e008ffba24aec091261ca1a5c02f7324702/results/v1.2/jevbench-v1.2-per-task.json). They are historical, not a fresh Gateway run. Jev latency includes the production API network path from Germany; local measurements use loopback. Different networks, hardware, versions, and measurement times prevent attributing the latency difference to model inference alone. A fresh hosted comparison still needs an available Gateway credential.

## What the measured results show

- DiffusionGemma gains **44 net correct answers / 19.05 percentage points** over LLaDA. Its gap to published Jev is **four net answers / 1.73 points**; the small public sample does not establish equivalence.
- The largest local improvement is on hard items: **70.3% versus 44.1%**. Published Jev scores 73.0% on that subset.
- Median local latency is similar. DiffusionGemma reduces p95 from about **2.3 seconds to 0.39 seconds** and delivers about **2.66× the serial decisions/second** on this workload.
- These are deployment comparisons: LLaDA uses a one-pass masked-token readout; Gemma uses native adaptive self-conditioned denoising (2–6 actual steps on these items; maximum 48). Neither uses a calibrated correctness guarantee.

## Probability quality on the matched public tasks

Lower is better for every metric below. Official matched probability vectors were not published, so their values cannot be filled in.

| Metric | DiffusionGemma | LLaDA | Official Jev |
|---|---:|---:|---:|
| Negative log-likelihood | 0.834 | 1.020 | Not available |
| Multiclass Brier | 0.259 | 0.491 | Not available |
| 10-bin calibration error | 0.122 | 0.204 | Not available |
| Ordinal MAE | 0.123 | 0.317 | Not available |
| Gold-distribution TVD, ten tasks | 0.367 | 0.346 | Not available |

Gemma improves average classification/calibration metrics here, but slightly worsens exact-distribution TVD. These results do not establish generally reliable probabilities; the emoji results below expose further limitations.

## Emoji and image coverage

| Separate test | DiffusionGemma | LLaDA | Official Jev |
|---|---:|---:|---:|
| Emojify, same 50 retained test sentences | 42/50 — 84.0% | Not measured | Not measured |
| TweetEval, same 1,000 tweets / 20 classes | 311/1,000 — 31.1% | Not measured | Not measured |
| Flowers, same 100 images | 98/100 — 98.0% | Not measured | Not measured |

The TweetEval result includes 105 adapter output-format failures. Its 49.1% published baseline is task-trained RoBERTa, not Jev. Earlier LLaDA emoji smoke results used different, much smaller examples and cannot fill these cells. A three-model emoji comparison has not been run.

## Sources and validation

Raw task IDs and dataset hashes were checked across Gemma and both serial LLaDA runs; the published Jev IDs match them. LLaDA repeat probability vectors were identical. [comparison.json](comparison.json) records the aggregates, checks, and input-artifact hashes. This public subset is not the complete 534-item official ranking. Pretraining overlap is unknown and paraphrase pairs limit independence.

- [DiffusionGemma text and image report](../diffusiongemma/README.md)
- [LLaDA serial and concurrent repeats](../jevbench-load-20260921/README.md)
- [Original LLaDA / published Jev comparison](../jevbench/README.md)
- [Emoji benchmark, calibration, and failure analysis](../diffusiongemma/emoji/README.md)
