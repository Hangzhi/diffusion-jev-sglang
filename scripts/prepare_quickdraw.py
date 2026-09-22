"""Prepare a small Google Quick, Draw! gallery for the Doodle Detective demo.

Streams the first 24 recognized, pixel-unique sketches per category. This is a
curated demo selection, not a random test split or a benchmark. Source generations,
selected source records, and hashes are retained in ignored data/quickdraw/.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
LABELS = [
    "airplane", "apple", "bicycle", "cat", "clock", "fish", "pizza", "umbrella",
    "dog", "car", "house", "tree", "sun", "star", "cup", "sailboat",
]
SOURCE = "https://github.com/googlecreativelab/quickdraw-dataset"


def render(drawing):
    image = Image.new("RGB", (320, 320), "white")
    pen = ImageDraw.Draw(image)
    for stroke in drawing:
        points = [(int(x) + 32, int(y) + 32) for x, y in zip(stroke[0], stroke[1], strict=True)]
        if len(points) > 1:
            pen.line(points, fill="black", width=4, joint="curve")
        for x, y in points:
            pen.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")
    return image


def main(args):
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("Prepared dataset exists; choose another --output")
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows, sources, seen = [], {}, set()
    for label in LABELS:
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{label}.ndjson"
        source_file = args.output / f"{label}.ndjson"
        metadata_file = args.output / f"{label}.source.json"
        if source_file.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            records = [json.loads(line) for line in source_file.read_text().splitlines()]
        else:
            print(f"Downloading {label} sketches", flush=True)
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=30
            ) as r:
                generation = r.headers["x-goog-generation"]
            if not generation:
                raise ValueError("Missing immutable source generation")
            metadata = {"url": url, "generation": generation}
            records = []
            with urllib.request.urlopen(f"{url}?generation={generation}", timeout=60) as stream:
                # Deliberately bounded demo selection, independent of our model's predictions.
                for line in stream:
                    record = json.loads(line)
                    if not record.get("recognized"):
                        continue
                    image = render(record["drawing"])
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    records.append({k: record[k] for k in ("key_id", "drawing")})
                    if len(records) == args.per_class:
                        break
            source_file.write_text("".join(json.dumps(r) + "\n" for r in records))
            metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
        if len(records) != args.per_class:
            raise ValueError("Cached selection size differs; choose a new output directory")
        metadata["selected_records_sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()
        sources[label] = metadata
        for record in records:
            image = render(record["drawing"])
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            image.save(image_dir / f"{digest}.jpg", "JPEG", quality=95)
            rows.append(
                {
                    "id": digest,
                    "label": label,
                    "split": "demo",
                    "benchmark": False,
                    "width": image.width,
                    "height": image.height,
                    "source_key": record["key_id"],
                }
            )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate rendered sketches in cached selections")
    manifest = {
        "dataset": "quickdraw",
        "title": "Google Quick, Draw!",
        "labels": LABELS,
        "source": SOURCE,
        "license": "CC BY 4.0 · Google, Inc.",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "count": len(rows),
        "benchmark_count": 0,
        "sources": sources,
        "method": "First recognized, pixel-unique sketches per category; rendered from simplified "
        "vectors as black strokes on white JPEGs. Curated demo only, not a held-out benchmark.",
        "images": sorted(rows, key=lambda r: r["id"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(rows)} sketches across {len(LABELS)} classes in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/quickdraw")
    parser.add_argument("--per-class", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.per_class <= 1000:
        parser.error("--per-class must be between 1 and 1000")
    main(args)
