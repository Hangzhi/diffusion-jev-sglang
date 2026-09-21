# Image demos

Both demos use **DiffusionGemma's native image input** through the same typed decision API. Images are sent as pixels; gallery filenames, source IDs, and gold labels are removed before inference. Official Jev 1.13.0 and the LLaDA2.1-mini checkpoint do not have native image input.

## Doodle Detective

Open **Doodle Detective** at the top of the app, or go directly to `/#doodle`. It also opens by default when the backend supports images. The drawing pad is ready right away. Draw with a mouse or finger, or use **Try a sketch** to open the gallery. Click **Guess doodle** at the top to see the model's prediction and distribution. Clear the drawing to start over. Uploading a PNG/JPEG/WebP works too.

The eight categories are **airplane, apple, bicycle, cat, clock, fish, pizza, and umbrella**. The model must pick among those eight options; an unrelated drawing will still get a guess. The scores are relative to this candidate set, not a guarantee that the drawing depicts one of them. Edit questions to try your own categories.

Prepare the gallery on the API host:

```bash
uv run python scripts/prepare_quickdraw.py
```

This downloads a small selection from [Google's Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), made available by Google, Inc. under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We render the simplified vector strokes into black-on-white JPEGs. Source records, immutable GCS object generations, and selection hashes stay in `data/quickdraw/`, which is excluded from Git. The app links to the original source and credits Google.

The default collection has **192 sketches: 24 per category**. Selection uses the first recognized, pixel-unique sketches in each source stream, without querying DiffusionGemma. “Recognized” is the original game's metadata, not our model's outcome. This deliberately approachable demo collection is **not a random or held-out test set**. Do not report its performance as general sketch-recognition accuracy. Dataset labels describe what players were asked to draw and can be ambiguous.

Preparation refuses to overwrite an existing manifest. Use a new `--output` directory to build another selection; set `DIFFUSION_JEV_DATA_DIR` to its parent when starting the API if you move the dataset root. Drawing and uploads work without the gallery.

## Flowers

Open **Flowers** at the top of the app (or `/#flowers`) to identify daisy, dandelion, rose, sunflower, or tulip from a gallery image or upload. Prepare the gallery with:

```bash
uv run python scripts/prepare_flowers.py
```

The source is [Kaggle Flowers, version 1](https://www.kaggle.com/datasets/abdelrahmanatef01/flowers-dataset-for-image-classification), whose uploader lists Apache 2.0. After exact deduplication, 2,742 images remain. The app browses 525 test images; the separate frozen benchmark uses 100. See [the measured flower results](../reports/diffusiongemma/README.md#image-classification).

## Gallery API

The dataset IDs are `flowers` and `quickdraw`:

- `GET /api/datasets/{dataset}` — title, categories, counts, source, license, preparation metadata.
- `GET /api/datasets/{dataset}/images?split=demo&offset=0&limit=12` — paginated sketches; use `split=test` for flowers. Optional `label` filters the gallery.
- `GET /api/datasets/{dataset}/image/{id}` — JPEG bytes.
- `POST /v1/systemone` — `images: ["quickdraw:<opaque SHA256 ID>"]` or `images: ["data:image/png;base64,..."]` alongside state and typed questions.

Namespaced `flowers:<id>` and legacy bare flower IDs both work. Arbitrary filesystem paths, remote image URLs, and unknown dataset names are rejected. Inputs are restricted to four images, 6 MB each and 20 megapixels per upload, then normalized to RGB JPEG with a maximum side of 768 pixels. The API sends neither gallery labels nor dataset identifiers to the model. An edited question may change which answer is meaningful, so expected-label badges disappear when results are stale.

User drawings stay in browser memory until evaluation; inference is sent to your own API host. The application does not save uploads or drawings to its dataset.
