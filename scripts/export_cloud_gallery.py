"""Prepare only the public demo images for the CPU web container."""

import argparse
import json
import shutil
from pathlib import Path


def export(source, target):
    for name, split in (("quickdraw", "demo"), ("flowers", "test")):
        root = source / name
        manifest = json.loads((root / "manifest.json").read_text())
        rows = [row for row in manifest["images"] if row["split"] == split]
        destination = target / name
        (destination / "images").mkdir(parents=True, exist_ok=True)
        for row in rows:
            filename = row["id"] + ".jpg"
            shutil.copy2(root / "images" / filename, destination / "images" / filename)
        manifest["images"] = rows
        manifest["count"] = len(rows)
        manifest["benchmark_count"] = sum(bool(row.get("benchmark")) for row in rows)
        (destination / "manifest.json").write_text(json.dumps(manifest))
        print(f"{name}: {len(rows)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(".cache/cloud-gallery"))
    args = parser.parse_args()
    export(args.source, args.output)
