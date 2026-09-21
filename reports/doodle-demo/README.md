# Doodle Detective demo validation

Validated September 21, 2026 against the local A100 / native SGLang DiffusionGemma service. This is an interface and inference smoke check, **not an accuracy benchmark**.

The gallery contains 192 sketches across eight categories. [dataset.json](dataset.json) records source URLs, immutable GCS object generations, selection hashes, and opaque image IDs. The preparation method selects the first 24 recognized, pixel-unique sketches per category without looking at this model's predictions. Gallery images and raw source vectors are excluded from Git.

Browser checks passed:

- Load the gallery, move between pages, and filter a category.
- Classify a gallery cat through the real GPU API; predicted **cat**.
- Draw a clock with browser pointer input and submit its PNG pixels; predicted **clock**.
- Clear the drawing and verify that guessing is disabled until another image is supplied.
- Switch back to Flowers and classify a namespaced flower ID through the real API.
- Switch to text decisions; retain the single top action button and no Copy cURL control.
- Check a 390-pixel mobile viewport for horizontal overflow; no JavaScript errors.

[browser-smoke.json](browser-smoke.json) contains actual model responses and check results. These few examples were integration checks; their timings include cold-start effects and are not comparable to the measured benchmark runs.

The Python suite passed 52 tests with three skipped in the lightweight environment; eight tensor/readout checks passed in the engine environment. Ruff and the TypeScript/Vite production build passed. The additional gallery test verifies that dataset namespaces cannot access another gallery's files and that source IDs/labels do not reach inference.

![Doodle Detective with a real model response](preview.png)

Screenshots: [desktop gallery](gallery-desktop.png), [mobile gallery](gallery-mobile.png), [drawn clock](drawing-desktop.png), [mobile drawing](drawing-mobile.png). The drawing screenshots were captured during the initial smoke check, before category hints were moved above the canvas; the gallery preview shows the final UI.

Sketch attribution: [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), Google, Inc., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Simplified vectors were rendered into black-on-white JPEGs. The clock in the drawing test was drawn with browser pointer events for this project.
