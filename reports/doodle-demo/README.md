# Doodle Detective: live demo and checks

Recorded on September 22, 2026 with the running DiffusionGemma service on one A100 80GB.

![Draw a scruffy cat and get a real model answer](sketch-demo.gif)

The GIF shows browser mouse input, a click on **Guess doodle**, and the actual GPU response. The app sends the canvas pixels to the model. No answer is inserted into the recording. Playback is at normal speed, with 12 frames per second. The image is 1000 × 750 pixels and about 3.7 MB.

This is an example of the app working. It is **not an accuracy or latency benchmark**.

## What changed for the user

The drawing pad opens first on DiffusionGemma. Doodle Detective, Flowers, and Text decisions each have a top-level tab. The image demos keep the question settings in a closed section so the canvas and answer are easier to find. Direct links use `/#doodle`, `/#flowers`, and `/#text`.

## Live browser checks

[sketch-validation.json](sketch-validation.json) contains the actual model responses and the check results.

- Draw a scruffy cat with the mouse and get **cat** from the real model.
- Keep the Guess doodle button at the same brightness between strokes and while waiting for the model. Submission is blocked until the stroke ends, so it cannot send the previous image.
- Clear the drawing, or reopen the demo. Pixels reset and guessing is disabled.
- Page through the gallery, filter cats, and classify a sketch. Its dataset label appears after the answer.
- Edit the question and check that the old dataset-label result is hidden.
- Classify a rose and evaluate the text/emoji example through the same service.
- Reload the Flowers and Text links and keep the selected demo.
- Upload a drawing while optional gallery endpoints return 404. The real model still classifies it.
- Delay the examples endpoint and confirm it does not replace a demo the user already selected.
- Simulate a text-only backend and keep the text interface available.
- Draw on a 390-pixel touch viewport and get **cat**, with no horizontal overflow.

Inference responses are never mocked. Only the missing-gallery, delayed-startup, and text-only capability checks simulate interface conditions. No JavaScript exceptions were recorded.

Screenshots: [desktop](sketch-desktop.png), [mobile](sketch-mobile.png), and [the drawn cat](drawn-cat.png).

The Python suite passed **52 tests**, with three skipped in the lightweight environment. Ruff and the TypeScript/Vite production build passed. See [development and recording steps](../../docs/development.md) for the environment and repeatable capture command.

## Gallery source

The optional gallery has 192 sketches across eight categories. [dataset.json](dataset.json) records source URLs, immutable GCS object generations, selection hashes, and opaque image IDs. Preparation takes the first 24 recognized, pixel-unique sketches per category without looking at this model's predictions. Gallery images and raw source vectors stay outside Git.

Sketch source: [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), Google, Inc., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Simplified strokes were rendered as black-on-white JPEGs. The scruffy cat in this recording was drawn with browser pointer events from hand-authored, deliberately uneven points. It is not a traced gallery image.

The earlier gallery checks remain in [browser-smoke.json](browser-smoke.json). The older [gallery preview](preview.png) and `gallery-*` / `drawing-*` screenshots show the previous navigation layout.
