"""Measure a frozen, balanced image-only flower classification test through the web API."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from diffusion_jev.evaluation import metrics
from diffusion_jev.gemma_backend import MODEL, REVISION, SGLANG_REVISION

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {
    "type": "choice",
    "instructions": "Which type of flower is most prominent in the attached image?",
    "criteria": {
        "daisy": "A daisy flower",
        "dandelion": "A dandelion flower or seed head",
        "rose": "A rose flower",
        "sunflower": "A sunflower",
        "tulip": "A tulip flower",
    },
}


def request(client, row):
    # Gold label and source filename are used only for scoring after the response.
    body = {
        "model": "jev",
        "state": "Classify the flower shown in the attached image.",
        "images": [row["id"]],
        "questions": {"flower": QUESTION},
    }
    start = time.perf_counter()
    try:
        response = client.post("/v1/systemone", json=body)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        result = response.json()
        if result["model"] != MODEL or result["meta"]["passes"] != 48:
            raise ValueError("Active model or denoising configuration changed")
        if result["meta"]["temperature"] != 1.0:
            raise ValueError("Readout temperature changed")
        probabilities = result["answers"]["flower"]["probabilities"]
        values = list(probabilities.values())
        if (
            set(probabilities) != set(QUESTION["criteria"])
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values)
            or not math.isclose(sum(values), 1, abs_tol=1e-6)
        ):
            raise ValueError("Invalid candidate distribution")
        prediction = max(probabilities, key=probabilities.get)
        return {
            "id": row["id"],
            "label": row["label"],
            "valid": True,
            "correct": prediction == row["label"],
            "prediction": prediction,
            "latency_s": elapsed,
            "response": result,
        }
    except httpx.HTTPError:
        return {
            "id": row["id"],
            "label": row["label"],
            "valid": False,
            "correct": False,
            "latency_s": time.perf_counter() - start,
            "error": "HTTP request failed",
        }


def main(args):
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["images"] if row["benchmark"]]
    labels = list(QUESTION["criteria"])
    if (
        len(rows) != 100
        or Counter(row["label"] for row in rows) != dict.fromkeys(labels, 20)
        or any(row["split"] != "test" for row in rows)
        or len({row["id"] for row in rows}) != len(rows)
    ):
        raise ValueError("Expected the frozen 100 unique test images, 20 per class")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {
        "completed": False,
        "started": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "model_revision": REVISION,
        "sglang_revision": SGLANG_REVISION,
        "dataset_source": manifest["source"],
        "dataset_version": manifest["version"],
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "items": len(rows),
        "labels": labels,
        "warmup_requests": 2,
        "concurrency": 1,
        "retries": 0,
        "temperature": 1.0,
        "max_denoising_steps": 48,
        "latency_scope": "client HTTP loopback, including image loading and vision encoding",
        "question": QUESTION,
        "limitations": [
            "Public dataset: pretraining overlap is unknown.",
            "Only exact decoded-pixel duplicates were removed; near duplicates may remain.",
            "Self-conditioned denoiser scores are not calibrated correctness probabilities.",
            "No official hosted Jev image result was measured.",
        ],
    }
    meta_path = args.output / "run.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    records = []
    started = None
    try:
        with httpx.Client(base_url=args.url, timeout=180) as client:
            # Warm with a development image, not a test label or prompt adjustment.
            warm = next(row for row in manifest["images"] if row["split"] == "dev")
            for _ in range(2):
                if not request(client, warm)["valid"]:
                    raise RuntimeError("Image warmup failed")
            started = time.perf_counter()
            with (args.output / "predictions.jsonl").open("x") as output:
                for i, row in enumerate(rows, 1):
                    record = request(client, row)
                    records.append(record)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    if i % 10 == 0:
                        print(
                            f"{i}/{len(rows)}; correct={sum(r['correct'] for r in records)}",
                            flush=True,
                        )
                    if len(records) >= 3 and not any(r["valid"] for r in records[-3:]):
                        raise RuntimeError("Three consecutive image inference failures")
        valid = [row for row in records if row["valid"]]
        confusion = {label: dict.fromkeys(labels + ["invalid"], 0) for label in labels}
        for row in records:
            confusion[row["label"]][row.get("prediction", "invalid")] += 1
        summary = {
            "count": len(rows),
            "valid": len(valid),
            "correct": sum(row["correct"] for row in records),
            "accuracy_including_failures": sum(row["correct"] for row in records) / len(rows),
            "chance_accuracy": 0.2,
            "by_class": {
                label: {
                    "count": 20,
                    "correct": sum(r["correct"] for r in records if r["label"] == label),
                }
                for label in labels
            },
            "confusion_matrix": confusion,
            "latency_s": dict(
                zip(
                    ["p50", "p95"],
                    np.percentile([r["latency_s"] for r in records], [50, 95]).tolist(),
                    strict=True,
                )
            ),
            "metrics_valid_only": metrics(
                [
                    [r["response"]["answers"]["flower"]["probabilities"][label] for label in labels]
                    for r in valid
                ],
                [labels.index(r["label"]) for r in valid],
            )
            if valid
            else None,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        meta["completed"] = True
    finally:
        meta["finished"] = datetime.now(UTC).isoformat()
        meta["measured_items"] = len(records)
        meta["measured_elapsed_s"] = time.perf_counter() - started if started else None
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/flowers/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/diffusiongemma/flowers")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    main(parser.parse_args())
