"""Run frozen emoji tasks through the current Jev API; save raw and dev-calibrated results."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from diffusion_jev.emoji_benchmark import baselines, summarize, validate_response
from diffusion_jev.evaluation import fit_calibration
from diffusion_jev.gemma_backend import MODEL, REVISION, SGLANG_REVISION

ROOT = Path(__file__).resolve().parents[1]


def save(path, body):
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")


def request(client, row, question, labels):
    # Labels and source IDs are retained only in output records, never in model inputs.
    start = time.perf_counter()
    record = {k: row[k] for k in ("id", "source_index", "split", "label")}
    try:
        response = client.post(
            "/v1/systemone",
            json={"model": "jev", "state": row["text"], "questions": {"emoji": question}},
        )
        response.raise_for_status()
        body = response.json()
        prediction = validate_response(body, labels)
        record.update(
            valid=True, prediction=prediction, correct=prediction == row["label"], response=body
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        record.update(valid=False, correct=False, error_type=type(error).__name__)
    record["latency_ms"] = 1000 * (time.perf_counter() - start)
    return record


def run(args, name):
    data, output = args.data / name, args.output / name
    manifest = json.loads((data / "manifest.json").read_text())
    splits = {}
    for split, spec in manifest["files"].items():
        path = data / spec["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Frozen {name} {split} file changed")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != spec["count"] or any(row["split"] != split for row in rows):
            raise ValueError("Dataset row count or split mismatch")
        splits[split] = rows
    ids = [r["id"] for rows in splits.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate text across prepared splits")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "dataset.json", manifest)
    save(
        output / "selection.json",
        {
            split: [{k: v for k, v in r.items() if k != "text"} for r in splits[split]]
            for split in ("dev", "test")
        },
    )
    labels = manifest["labels"]
    meta = {
        "completed": False,
        "started": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "model_revision": REVISION,
        "sglang_revision": SGLANG_REVISION,
        "dataset": name,
        "dataset_manifest_sha256": hashlib.sha256(
            (data / "manifest.json").read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(
            (ROOT / "src/diffusion_jev/emoji_benchmark.py").read_bytes()
        ).hexdigest(),
        "concurrency": 1,
        "retries": 0,
        "warmup_requests": 2,
        "temperature": 1.0,
        "max_denoising_steps": 48,
        "latency_scope": "client HTTP loopback including adapter and response validation",
        "question": manifest["question"],
        "split_runs": {},
        "live_official_jev_measured": False,
    }
    save(output / "run.json", meta)
    all_records = {}
    try:
        with httpx.Client(base_url=args.url, timeout=180) as client:
            for _ in range(2):
                if not request(client, splits["train"][0], manifest["question"], labels)["valid"]:
                    raise RuntimeError("Emoji benchmark warmup failed")
            for split in ("dev", "test"):
                records = []
                all_records[split] = records
                started = time.perf_counter()
                with (output / f"{split}-predictions.jsonl").open("x") as stream:
                    for i, row in enumerate(splits[split], 1):
                        record = request(client, row, manifest["question"], labels)
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        if i % 50 == 0 or i == len(splits[split]):
                            print(
                                f"{name} {split}: {i}/{len(splits[split])}; correct={sum(r['correct'] for r in records)}; valid={sum(r['valid'] for r in records)}",
                                flush=True,
                            )
                        if len(records) >= 3 and not any(r["valid"] for r in records[-3:]):
                            raise RuntimeError(
                                "Three consecutive inference failures; abort benchmark"
                            )
                elapsed = time.perf_counter() - started
                meta["split_runs"][split] = {
                    "items": len(records),
                    "elapsed_s": elapsed,
                    "valid_decisions_per_second": sum(r["valid"] for r in records) / elapsed,
                }
                save(output / f"{split}-summary.json", summarize(records, labels))
                if split == "dev":
                    dev_report = {
                        "model": MODEL,
                        "records": [
                            {
                                "id": r["id"],
                                "split": "dev",
                                "label": r["label"],
                                "passes": 48,
                                "logits": r["response"]["meta"]["candidate_logits"]["emoji"],
                                "probabilities": {
                                    label: r["response"]["answers"]["emoji"]["probabilities"][label]
                                    for label in labels
                                },
                            }
                            for r in records
                            if r["valid"]
                        ],
                    }
                    save(output / "calibration-dev-input.json", dev_report)
                    fit_calibration(
                        output / "calibration-dev-input.json", output / "calibration.json"
                    )
                    # Freeze the development fit before any test request is evaluated.
                    meta["calibration_frozen_before_test"] = datetime.now(UTC).isoformat()
                save(output / "run.json", meta)
        temperature = json.loads((output / "calibration.json").read_text())["temperature"]
        save(
            output / "test-calibrated-summary.json",
            summarize(all_records["test"], labels, temperature),
        )
        save(output / "baselines.json", baselines(splits["train"], splits["test"], labels))
        meta["completed"] = True
    finally:
        meta["finished"] = datetime.now(UTC).isoformat()
        meta["measured_items"] = {split: len(rows) for split, rows in all_records.items()}
        save(output / "run.json", meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/emoji-benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/diffusiongemma/emoji")
    parser.add_argument("--dataset", choices=["both", "emojify", "tweeteval"], default="both")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    names = ("emojify", "tweeteval") if args.dataset == "both" else (args.dataset,)
    for name in names:
        if (args.output / name).exists():
            raise ValueError("Benchmark output exists; choose a new --output")
    for name in names:
        run(args, name)
