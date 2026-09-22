# README examples

Captured on September 22, 2026 from the running local DiffusionGemma app.
The screenshots show the actual browser interface and model responses.
The capture script does not insert answers or change the page layout.

| Example | Input | Observed answer |
|---|---|---|
| [Flower](flower.png) | First daisy in the filtered test gallery | `daisy` |
| [Text](emoji.png) | “I love you so much. You make every day better!” | `❤️`, positive, intensity approximately 3/3 |

[examples.json](examples.json) contains the requests, gallery image ID, full
responses, and browser error check. Dataset labels are not included in the model
request. Scores compare supplied candidates and are not calibrated correctness
probabilities. These two examples are not a benchmark.

The original [sketch GIF and recording](../doodle-demo/README.md) show a separate
real drawing session. All README images link to the matching public playground
tab. Public request times include additional queueing, cold starts, and network
overhead; screenshot latency badges describe the local API measurement.

To capture the photo and text examples again, start the local DiffusionGemma
service and prepare the flower gallery, then run from the repo root:

```bash
uvx --with playwright playwright install chromium
uvx --with playwright python scripts/capture_readme_examples.py \
  --output .cache/readme-demo
```

Use the [development environment](../../docs/development.md) and retain the
[dataset credits](../../THIRD_PARTY.md). Detailed technical docs are in English;
the repo overview is available in [English](../../README.md) and
[简体中文](../../README.zh-CN.md).
