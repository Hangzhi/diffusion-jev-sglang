# API compatibility

The HTTP shape follows the [TypeSafe reference](https://docs.typesafe.ai/api), retrieved September 21, 2026. This project is independent and makes no claim of model equivalence.

| Area | Local implementation |
| --- | --- |
| Endpoint | `POST /v1/systemone` |
| Model | `jev`, `diffusion-jev`, or the active checkpoint name; aliases select the local model |
| Input | `model`, `state`, a named `questions` map, and optional `images` for DiffusionGemma |
| Types | `noul`, `choice`, `score` |
| Criteria | Noul true/false descriptions; choice description map; ordered score array |
| Choice limit | 2–26 options (single-token letter mapping), rather than the hosted API's larger limit |
| Score | 2–10 levels, zero-based expectation |
| Questions | 1–32 per request |
| Size | 100,000 serialized text characters; 7,000 prompt text tokens for Gemma, 7,900 for LLaDA; Gemma additionally accepts bounded image inputs |
| Confidence | 1 − normalized Shannon entropy; not guaranteed to match TypeSafe's formula |
| Authentication | None; intended for localhost only |
| Extra response fields | `meta`: latency, passes, temperature, probability source, raw candidate logits |
| Errors | 422 validation/input failures; 503 engine/metadata failures |

There is no silent conversion of generated prose to probabilities. Missing extension metadata is an error. Noul maps A to false and B to true. A score's probabilities and legend use stringified level indices. Candidate probabilities sum to one within the supplied options, not across every token in the vocabulary.

For DiffusionGemma, scores come from the last active self-conditioned denoising step, with actual steps and candidate mass in `meta`. They are not the old LLaDA masked-position scores. See [Gemma image inputs and scoring semantics](diffusiongemma.md).

`GET /health` reports engine reachability; it does not prove a particular prompt will succeed. `GET /api/examples` supplies the UI presets. `/docs` and `/openapi.json` expose the local OpenAPI schema.

Image inputs are a local extension. Use `images: ["flowers:<id>"]`, `images: ["quickdraw:<id>"]`, or a JPEG/PNG/WebP data URL. Bare flower IDs remain supported. Gallery metadata, paging, and image routes are documented in [Image demos](image-demos.md#gallery-api). Official Jev 1.13.0 accepts text only.
