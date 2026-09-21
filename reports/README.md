# Live A100 validation

Measured on September 21, 2026 with the environment in `environment.json`. BF16, one denoising pass, prefix caching and CUDA graphs disabled. These are small functional smoke tests, not representative quality or production load benchmarks.

The browser exercised the actual API and model, including choice, noul, score, returned logits, JSON view, preset switching and mobile layout. All three presets also ran concurrently with different prompt lengths. See `browser-smoke.json`, `presets-live.json`, and screenshots.

## Warm client-observed latency

Ten requests per configuration; two warmup requests excluded. Repeated original emoji question. Latency is per whole API request; throughput counts question decisions.

| Questions | Concurrent requests | p50 ms | p95 ms | Decisions/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 147.1 | 153.3 | 6.8 |
| 1 | 4 | 207.5 | 241.3 | 15.9 |
| 8 | 1 | 312.7 | 383.0 | 24.6 |
| 8 | 4 | 1360.6 | 1584.0 | 22.0 |
| 32 | 1 | 1257.1 | 1499.4 | 24.7 |
| 32 | 4 | 4757.2 | 5000.2 | 26.3 |

## Quality and calibration smoke checks

Original development set: 9/10 correct. Original held-out test set: 10/10 correct. These sets contain only ten examples each. Development-fitted temperature was 2.692. It reduced development NLL from 0.776 to 0.376, but increased held-out NLL from 0.00030 to 0.08683. This illustrates why fitting on a tiny development set does not establish calibration. The running demo retains T=1. Test calibration was computed offline from saved logits; it was never fitted on test labels.

No comparison against Qwen, autoregressive serving, or TypeSafe Jev was measured. Two/four-pass behavior has tensor tests but no GPU benchmark here. Docker and a fresh engine installation from the lockfile have not been exercised; the actual GPU run used an existing pinned SGLang environment with the included patch.
