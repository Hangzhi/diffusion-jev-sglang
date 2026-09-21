"""Reproducible classification metrics; development-only temperature fitting."""

import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from .backend import MODEL
from .scoring import probabilities


def metrics(distributions, targets):
    p = np.asarray(distributions, dtype=float)
    y = np.asarray(targets, dtype=int)
    prediction = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = prediction == y
    ece = 0.0
    for i in range(10):
        selected = (confidence >= i / 10) & (
            confidence < (i + 1) / 10 if i < 9 else confidence <= 1
        )
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
        "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean()),
        "ece_10_bins": float(ece),
    }


def run_benchmark(data: Path, output: Path, url: str, warmup: int):
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if not rows or warmup < 0:
        raise ValueError("Need nonempty JSONL and nonnegative warmup")
    from .schemas import EvaluationRequest
    from .scoring import options

    for row in rows:
        req = EvaluationRequest.model_validate(row["request"])
        if len(req.questions) != 1:
            raise ValueError("Classification benchmark requires one question per row")
        keys = [k for k, _ in options(next(iter(req.questions.values())))]
        if str(row["label"]) not in keys:
            raise ValueError("Unknown label")
    records = []
    with httpx.Client(base_url=url, timeout=180) as client:
        for _ in range(warmup):
            client.post("/v1/systemone", json=rows[0]["request"]).raise_for_status()
        for row in rows:
            start = time.perf_counter()
            response = client.post("/v1/systemone", json=row["request"])
            response.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            body = response.json()
            key, ans = next(iter(body["answers"].items()))
            dist = ans.get("probabilities") or {"false": 1 - ans["noul"], "true": ans["noul"]}
            records.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "label": str(row["label"]),
                    "probabilities": dist,
                    "logits": body["meta"]["candidate_logits"][key],
                    "passes": body["meta"]["passes"],
                    "latency_ms": elapsed,
                }
            )
    # Group by class vocabulary to avoid mixing unrelated labels or dimensions.
    groups = {}
    for row in records:
        group = row["split"] + ":" + ",".join(row["probabilities"])
        groups.setdefault(group, []).append(row)
    results = {}
    for key, group in groups.items():
        p = [list(row["probabilities"].values()) for row in group]
        y = [list(row["probabilities"]).index(row["label"]) for row in group]
        results[key] = {"count": len(group), **metrics(p, y)}
    latencies = [row["latency_ms"] for row in records]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model": MODEL,
                "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "warmup_requests": warmup,
                "metrics": results,
                "latency_ms": {
                    "p50": float(np.percentile(latencies, 50)),
                    "p95": float(np.percentile(latencies, 95)),
                },
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


def fit_calibration(report: Path, output: Path):
    data = json.loads(report.read_text())
    rows = data["records"]
    if not rows or any(row["split"] != "dev" for row in rows):
        raise ValueError(
            "Calibration requires exclusively development-split records; never fit test data"
        )
    passes = {row["passes"] for row in rows}
    if len(passes) != 1:
        raise ValueError("Do not mix inference pass settings")

    def loss(t):
        return float(
            np.mean(
                [
                    -np.log(
                        max(
                            probabilities(row["logits"], t)[
                                list(row["probabilities"]).index(row["label"])
                            ],
                            1e-12,
                        )
                    )
                    for row in rows
                ]
            )
        )

    temperatures = np.geomspace(0.1, 10, 401)
    best = float(min(temperatures, key=loss))
    output.write_text(
        json.dumps(
            {
                "model": data["model"],
                "passes": passes.pop(),
                "temperature": best,
                "source_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "dev_count": len(rows),
                "nll_before": loss(1),
                "nll_after": loss(best),
            },
            indent=2,
        )
        + "\n"
    )
