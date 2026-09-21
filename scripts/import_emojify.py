"""Convert a user-downloaded Emojify CSV to reproducible dev/test JSONL.

Usage: uv run python scripts/import_emojify.py input.csv --text-column text --label-column label
No Kaggle credentials or third-party data are bundled. Review the source license first.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("--text-column", required=True)
parser.add_argument("--label-column", required=True)
parser.add_argument("--output", type=Path, default=Path("data/emojify"))
args = parser.parse_args()
rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig")))
labels = sorted({r[args.label_column] for r in rows})
if not 2 <= len(labels) <= 26:
    raise ValueError("Expected 2–26 label values")
args.output.mkdir(parents=True, exist_ok=True)
outputs = {split: [] for split in ("dev", "test")}
seen = set()
for row in rows:
    text = row[args.text_column].strip()
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    if not text or fingerprint in seen:
        continue
    seen.add(fingerprint)
    split = "dev" if int(fingerprint[:8], 16) % 5 == 0 else "test"
    outputs[split].append(
        {
            "id": fingerprint[:16],
            "split": split,
            "label": row[args.label_column],
            "request": {
                "model": "diffusion-jev",
                "state": text,
                "questions": {
                    "emoji": {
                        "type": "choice",
                        "instructions": "Choose the emoji category that best matches the text.",
                        "criteria": dict.fromkeys(labels),
                    }
                },
            },
        }
    )
for split, values in outputs.items():
    (args.output / f"{split}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values)
    )
print({split: len(values) for split, values in outputs.items()})
