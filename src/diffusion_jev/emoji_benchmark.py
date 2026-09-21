"""Failure-aware metrics and response validation for frozen emoji prediction tasks."""

import math
from collections import Counter

import numpy as np

from .evaluation import metrics
from .gemma_backend import MODEL
from .schemas import EvaluationResponse
from .scoring import probabilities


def validate_response(body, labels):
    response = EvaluationResponse.model_validate(body)
    if (
        response.model != MODEL
        or response.meta.passes != 48
        or response.meta.temperature != 1.0
        or response.meta.probability_source != "self_conditioned_denoiser_logits"
    ):
        raise RuntimeError("Active model or inference settings changed; abort benchmark")
    if set(response.answers) != {"emoji"}:
        raise ValueError("Expected exactly one emoji answer")
    answer = response.answers["emoji"]
    if answer.type != "choice" or set(answer.probabilities) != set(labels):
        raise ValueError("Emoji class vocabulary changed")
    values = [answer.probabilities[label] for label in labels]
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Emoji distribution is not normalized")
    expected = labels[int(np.argmax(values))]
    if answer.choice != expected:
        raise ValueError("Returned choice disagrees with candidate probabilities")
    logits = response.meta.candidate_logits.get("emoji", [])
    if len(logits) != len(labels) or any(not math.isfinite(v) for v in logits):
        raise ValueError("Candidate logits are invalid")
    if not np.allclose(probabilities(logits, 1), values, atol=1e-6, rtol=1e-5):
        raise ValueError("Candidate probabilities disagree with the reported logits")
    if (
        not response.meta.denoising_steps
        or not 1 <= response.meta.denoising_steps.get("emoji", 0) <= 48
        or not response.meta.answer_position
        or response.meta.answer_position.get("emoji") != 4
        or not response.meta.candidate_mass
        or "emoji" not in response.meta.candidate_mass
    ):
        raise ValueError("Denoiser readout diagnostics are incomplete")
    return expected


def classification_metrics(targets, predictions, labels):
    if not targets or len(targets) != len(predictions):
        raise ValueError("Need paired, nonempty targets and predictions")
    confusion = {a: dict.fromkeys([*labels, "invalid"], 0) for a in labels}
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target][prediction if prediction in labels else "invalid"] += 1
    by_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[target][label] for target in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        by_class[label] = {
            "count": support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    count = len(targets)
    correct = sum(a == b for a, b in zip(targets, predictions, strict=True))
    p, z = correct / count, 1.959963984540054
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return {
        "count": count,
        "correct": correct,
        "accuracy": p,
        "accuracy_wilson_95": [center - radius, center + radius],
        "macro_f1": float(np.mean([r["f1"] for r in by_class.values()])),
        "balanced_accuracy": float(np.mean([r["recall"] for r in by_class.values()])),
        "by_class": by_class,
        "confusion_matrix": confusion,
    }


def summarize(records, labels, temperature=1.0):
    predictions = [r.get("prediction") for r in records]
    result = classification_metrics([r["label"] for r in records], predictions, labels)
    valid = [r for r in records if r["valid"]]
    p = [
        probabilities(r["response"]["meta"]["candidate_logits"]["emoji"], temperature)
        for r in valid
    ]
    y = [labels.index(r["label"]) for r in valid]
    result.update(
        {
            "valid": len(valid),
            "failures": len(records) - len(valid),
            "top3_accuracy_including_failures": sum(
                target in sorted(range(len(labels)), key=lambda j: (-dist[j], j))[:3]
                for dist, target in zip(p, y, strict=True)
            )
            / len(records),
            "probability_metrics_valid_only": metrics(p, y) if valid else None,
            "readout_temperature": temperature,
            "latency_ms": dict(
                zip(
                    ["p50", "p95"],
                    np.percentile([r["latency_ms"] for r in records], [50, 95]).tolist(),
                    strict=True,
                )
            ),
            "incorrect_with_top_probability_at_least_90_percent": sum(
                labels[int(np.argmax(dist))] != r["label"] and max(dist) >= 0.9
                for r, dist in zip(valid, p, strict=True)
            ),
            "candidate_mass_below_half_count": sum(
                r["response"]["meta"]["candidate_mass"]["emoji"] < 0.5 for r in valid
            ),
            "denoising_steps": dict(
                Counter(r["response"]["meta"]["denoising_steps"]["emoji"] for r in valid)
            ),
        }
    )
    return result


def baselines(train, test, labels):
    counts = Counter(r["label"] for r in train)
    majority = max(labels, key=lambda label: counts[label])
    targets = [r["label"] for r in test]
    result = {
        "uniform_random_expected_accuracy": 1 / len(labels),
        "train_majority_label": majority,
        "train_majority": classification_metrics(targets, [majority] * len(test), labels),
    }
    if all("published_prediction" in r for r in test):
        result["published_tweeteval_roberta"] = classification_metrics(
            targets, [r["published_prediction"] for r in test], labels
        )
    return result
