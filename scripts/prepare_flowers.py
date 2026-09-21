"""Download a pinned Kaggle flower dataset and prepare a local, labeled gallery.

Images and manifests stay in ignored data/flowers. Labels never appear in image IDs.
"""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET = "abdelrahmanatef01/flowers-dataset-for-image-classification"
VERSION = 1
LABELS = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("Prepared dataset already exists; choose another --output")
    archive = args.output / "source.zip"
    if not archive.exists():
        print("Downloading Kaggle Flowers, version 1", flush=True)
        urllib.request.urlretrieve(
            f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber={VERSION}",
            archive,
        )
    with urllib.request.urlopen(
        f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"
    ) as response:
        upstream = json.load(response)
    (args.output / "source-metadata.json").write_text(json.dumps(upstream, indent=2) + "\n")
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    rows = {}
    conflicts = set()
    rejected = 0
    duplicates = 0
    with zipfile.ZipFile(archive) as source:
        for member in sorted(source.namelist()):
            path = Path(member)
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            label = next((part.lower() for part in path.parts[:-1] if part.lower() in LABELS), None)
            if label is None:
                continue
            try:
                # Decode directly from the archive; never extract archive-provided paths.
                with Image.open(io.BytesIO(source.read(member))) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    image.load()
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                if digest in rows:
                    duplicates += 1
                    if rows[digest]["label"] != label:
                        conflicts.add(digest)
                    continue
                width, height = image.size
                image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                image.save(image_dir / f"{digest}.jpg", "JPEG", quality=92)
                split_hash = hashlib.sha256(("flower-split-v1:" + digest).encode()).hexdigest()
                bucket = int(split_hash[:8], 16) % 100
                split = "test" if bucket < 20 else "dev" if bucket < 30 else "train"
                rows[digest] = {
                    "id": digest,
                    "label": label,
                    "split": split,
                    "width": image.width,
                    "height": image.height,
                    "original_width": width,
                    "original_height": height,
                    "original_member": member,
                }
            except (OSError, ValueError, Image.DecompressionBombError):
                rejected += 1
    for digest in conflicts:
        rows.pop(digest, None)
        (image_dir / f"{digest}.jpg").unlink(missing_ok=True)
    by_class = defaultdict(list)
    for row in rows.values():
        if row["split"] == "test":
            by_class[row["label"]].append(row)
    selected = set()
    for label in LABELS:
        candidates = sorted(by_class[label], key=lambda row: row["id"])
        if len(candidates) < args.per_class:
            raise ValueError(f"Not enough test images for {label}")
        selected.update(row["id"] for row in candidates[: args.per_class])
    for row in rows.values():
        row["benchmark"] = row["id"] in selected
    images = sorted(rows.values(), key=lambda row: row["id"])
    manifest = {
        "dataset": "flowers",
        "title": "Kaggle Flowers",
        "labels": LABELS,
        "source": f"https://www.kaggle.com/datasets/{DATASET}",
        "version": VERSION,
        "license": upstream.get("licenseName", "Unknown"),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "count": len(images),
        "duplicates_removed": duplicates,
        "conflicting_images_removed": len(conflicts),
        "invalid_images_removed": rejected,
        "counts": dict(Counter(f"{r['split']}:{r['label']}" for r in images)),
        "benchmark_count": len(selected),
        "benchmark_per_class": args.per_class,
        "method": "Exact decoded-pixel deduplication; SHA256 split 70/10/20 train/dev/test; "
        "first sorted IDs per test class for the frozen benchmark. Near-duplicates are not detected.",
        "images": images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ("count", "counts", "benchmark_count", "license")}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/flowers")
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    main(args)
