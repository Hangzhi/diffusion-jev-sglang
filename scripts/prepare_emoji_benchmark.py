"""Prepare pinned Emojify and TweetEval sources without mixing their original splits."""

import argparse
import csv
import hashlib
import io
import json
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWEETEVAL_REVISION = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
EMOJIFY_VERSION = 2
EMOJIFY_LABELS = {
    "❤️": "Love and affection",
    "⚾": "Sports and ball games",
    "😄": "Happiness and joy",
    "😞": "Sadness, disappointment, or distress",
    "🍴": "Food, eating, or hunger",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(url, target)


def clean_splits(splits):
    """Remove conflicting text groups, then keep first split occurrence (train before dev/test)."""
    labels = defaultdict(set)
    for rows in splits.values():
        for row in rows:
            labels[fingerprint(row["text"])].add(row["label"])
    seen = set()
    cleaned, audit = {}, {}
    for split, rows in splits.items():
        cleaned[split] = []
        counts = Counter()
        for row in rows:
            key = fingerprint(row["text"])
            if not row["text"].strip():
                counts["empty"] += 1
            elif len(labels[key]) != 1:
                counts["conflicting_label"] += 1
            elif key in seen:
                counts["duplicate_or_earlier_split_overlap"] += 1
            else:
                seen.add(key)
                cleaned[split].append({**row, "id": key, "split": split})
        audit[split] = {"original": len(rows), "retained": len(cleaned[split]), **counts}
    return cleaned, audit


def sample(rows, count, salt):
    if not 1 <= count <= len(rows):
        raise ValueError(f"Requested {count} from {len(rows)} available rows")
    return sorted(rows, key=lambda row: hashlib.sha256((salt + row["id"]).encode()).hexdigest())[
        :count
    ]


def read_emojify_csv(content, labels):
    # These CSVs have NO header. Extra worksheet columns are not model inputs.
    rows = []
    for index, row in enumerate(csv.reader(io.StringIO(content.decode("utf-8-sig")))):
        if len(row) < 2 or row[1].strip() not in {str(i) for i in range(len(labels))}:
            raise ValueError(f"Malformed Emojify source row {index}")
        rows.append({"source_index": index, "text": row[0].strip(), "label": labels[int(row[1])]})
    return rows


def write_dataset(root, splits, manifest):
    labels = list(manifest["question"]["criteria"])
    if {row["label"] for row in splits["test"]} != set(labels):
        raise ValueError("Frozen test must represent every emoji class")
    files = {}
    for split, rows in splits.items():
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[split] = {
            "file": path.name,
            "sha256": digest(path),
            "count": len(rows),
            "class_counts": dict(Counter(row["label"] for row in rows)),
        }
    manifest["files"] = files
    manifest["labels"] = labels
    manifest["preparation_sha256"] = digest(Path(__file__))
    manifest["deduplication"] = (
        "NFKC, casefold, collapsed whitespace for matching only; original stripped text for "
        "inference. Remove all conflicting-label groups and duplicates; earlier source split "
        "takes precedence. No near-duplicate or model-pretraining overlap detection."
    )
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Directly reusable API bodies, with no gold labels or source identifiers in the request.
    (root / "test-requests.jsonl").write_text(
        "".join(
            json.dumps(
                {"model": "jev", "state": r["text"], "questions": {"emoji": manifest["question"]}},
                ensure_ascii=False,
            )
            + "\n"
            for r in splits["test"]
        )
    )
    print(manifest["dataset"], json.dumps(files, ensure_ascii=False), flush=True)


def prepare_emojify(root):
    source = root / "source"
    url = "https://www.kaggle.com/datasets/alvinrindra/emojify"
    download(
        "https://www.kaggle.com/api/v1/datasets/download/alvinrindra/emojify"
        f"?datasetVersionNumber={EMOJIFY_VERSION}",
        source / "source.zip",
    )
    download(
        "https://www.kaggle.com/api/v1/datasets/view/alvinrindra/emojify",
        source / "source-metadata.json",
    )
    with zipfile.ZipFile(source / "source.zip") as archive:
        splits = {
            split: read_emojify_csv(archive.read(f"{split}_emoji.csv"), list(EMOJIFY_LABELS))
            for split in ("train", "test")
        }
    splits, audit = clean_splits(splits)
    dev = []
    for label in EMOJIFY_LABELS:
        dev.extend(
            sample([r for r in splits["train"] if r["label"] == label], 4, "emojify-dev-v1:")
        )
    ids = {r["id"] for r in dev}
    splits["train"] = [r for r in splits["train"] if r["id"] not in ids]
    splits["dev"] = [{**r, "split": "dev"} for r in dev]
    write_dataset(
        root,
        splits,
        {
            "dataset": "emojify",
            "source": url,
            "source_version": EMOJIFY_VERSION,
            "license": "Unknown in uploader metadata; source data retained locally, not bundled",
            "source_sha256": {p.name: digest(p) for p in source.iterdir() if p.is_file()},
            "source_split_audit": audit,
            "selection": "All retained original test rows; four development rows per class from train",
            "combined_csv_used": False,
            "numeric_class_mapping": dict(enumerate(EMOJIFY_LABELS)),
            "question": {
                "type": "choice",
                "instructions": "Which emoji best matches the emotion or activity in this message?",
                "criteria": EMOJIFY_LABELS,
            },
        },
    )


def prepare_tweeteval(root, test_count, dev_count):
    source = root / "source"
    paths = [
        f"datasets/emoji/{split}_{kind}.txt"
        for split in ("train", "val", "test")
        for kind in ("text", "labels")
    ]
    paths += [
        "datasets/emoji/mapping.txt",
        "predictions/emoji.txt",
        "README.md",
        "evaluation_script.py",
    ]
    for path in paths:
        download(
            f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{TWEETEVAL_REVISION}/{path}",
            source / path,
        )
    mapping = [
        line.split("\t")
        for line in (source / "datasets/emoji/mapping.txt").read_text().splitlines()
        if line.strip()
    ]
    if [int(row[0]) for row in mapping] != list(range(20)):
        raise ValueError("Unexpected TweetEval emoji mapping")
    criteria = {row[1]: row[2].strip("_").replace("_", " ") for row in mapping}
    labels = list(criteria)
    published = [int(line) for line in (source / "predictions/emoji.txt").read_text().splitlines()]
    splits = {}
    for split, upstream in (("train", "train"), ("dev", "val"), ("test", "test")):
        texts = (source / f"datasets/emoji/{upstream}_text.txt").read_text().splitlines()
        targets = (source / f"datasets/emoji/{upstream}_labels.txt").read_text().splitlines()
        if split == "test" and len(published) != len(texts):
            raise ValueError("Published predictions do not align with source test rows")
        splits[split] = [
            {
                "source_index": i,
                "text": text.strip(),
                "label": labels[int(target)],
                **({"published_prediction": labels[published[i]]} if split == "test" else {}),
            }
            for i, (text, target) in enumerate(zip(texts, targets, strict=True))
        ]
    splits, audit = clean_splits(splits)
    splits["test"] = sample(splits["test"], test_count, "tweeteval-emoji-test-v1:")
    splits["dev"] = sample(splits["dev"], dev_count, "tweeteval-emoji-dev-v1:")
    write_dataset(
        root,
        splits,
        {
            "dataset": "tweeteval",
            "source": "https://github.com/cardiffnlp/tweeteval",
            "source_revision": TWEETEVAL_REVISION,
            "license": "No extra TweetEval restrictions; emoji subset license unspecified; data kept locally",
            "source_sha256": {path: digest(source / path) for path in paths},
            "source_split_audit": audit,
            "selection": "First salted SHA256 IDs from cleaned official test/validation, without class balancing",
            "selection_test_count": test_count,
            "selection_dev_count": dev_count,
            "published_baseline": "TweetEval RoBERTa re-trained on Twitter; published predictions on these exact source indices",
            "question": {
                "type": "choice",
                "instructions": "Predict the single emoji the author most likely used with this tweet. Choose the most likely emoji from the provided options.",
                "criteria": criteria,
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/emoji-benchmark")
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--dev-count", type=int, default=200)
    args = parser.parse_args()
    for name in ("emojify", "tweeteval"):
        if (args.output / name / "manifest.json").exists():
            raise ValueError("Prepared dataset already exists; choose another --output")
    prepare_emojify(args.output / "emojify")
    prepare_tweeteval(args.output / "tweeteval", args.test_count, args.dev_count)
