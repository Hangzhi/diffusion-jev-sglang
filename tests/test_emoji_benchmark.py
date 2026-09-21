import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from diffusion_jev.emoji_benchmark import classification_metrics, validate_response
from diffusion_jev.gemma_backend import MODEL

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emojify_headerless_csv_preserves_first_example_and_ignores_worksheet_columns():
    prepare = script("prepare_emoji_benchmark")
    rows = prepare.read_emojify_csv(b'"food, please",1,, [0]\n"hi\t",0\n', ["happy", "food"])
    assert [(r["text"], r["label"]) for r in rows] == [("food, please", "food"), ("hi", "happy")]
    with pytest.raises(ValueError, match="Malformed"):
        prepare.read_emojify_csv(b"text,label\n", ["happy", "food"])


def test_duplicate_groups_cannot_cross_splits_or_keep_conflicting_gold():
    prepare = script("prepare_emoji_benchmark")
    cleaned, audit = prepare.clean_splits(
        {
            "train": [{"text": "HELLO   world", "label": "a"}, {"text": "ambiguous", "label": "a"}],
            "dev": [{"text": "hello world", "label": "a"}, {"text": "different", "label": "b"}],
            "test": [
                {"text": "ｈｅｌｌｏ world", "label": "a"},
                {"text": "ambiguous", "label": "b"},
                {"text": "unique", "label": "a"},
            ],
        }
    )
    assert [r["text"] for r in cleaned["train"]] == ["HELLO   world"]
    assert [r["text"] for r in cleaned["dev"]] == ["different"]
    assert [r["text"] for r in cleaned["test"]] == ["unique"]
    assert audit["train"]["conflicting_label"] == audit["test"]["conflicting_label"] == 1


def test_freezing_is_independent_of_source_order():
    prepare = script("prepare_emoji_benchmark")
    rows = [{"id": str(i)} for i in range(100)]
    assert prepare.sample(rows, 20, "fixed:") == prepare.sample(rows[::-1], 20, "fixed:")
    with pytest.raises(ValueError):
        prepare.sample(rows, 101, "fixed:")


def test_failures_count_as_wrong_and_reduce_class_recall():
    result = classification_metrics(["a", "a", "b"], ["a", None, "a"], ["a", "b"])
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == 0.25
    assert result["balanced_accuracy"] == 0.25
    assert result["by_class"]["a"]["recall"] == 0.5
    assert result["confusion_matrix"]["a"]["invalid"] == 1
    assert result["accuracy_wilson_95"][0] < 1 / 3 < result["accuracy_wilson_95"][1]


def response():
    return {
        "model": MODEL,
        "answers": {
            "emoji": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.75, "b": 0.25},
                "confidence": 0.2,
            }
        },
        "usage": {"input_tokens": 20, "output_tokens": 5},
        "meta": {
            "latency_ms": 1,
            "passes": 48,
            "temperature": 1,
            "probability_source": "self_conditioned_denoiser_logits",
            "candidate_logits": {"emoji": [math.log(3), 0]},
            "denoising_steps": {"emoji": 2},
            "answer_position": {"emoji": 4},
            "candidate_mass": {"emoji": 0.99},
            "unrestricted_first_token": {"emoji": "A"},
        },
    }


def test_report_checks_model_and_actual_probability_values():
    body = response()
    assert validate_response(body, ["a", "b"]) == "a"
    body["model"] = "wrong-model"
    with pytest.raises(RuntimeError, match="settings changed"):
        validate_response(body, ["a", "b"])
    body = response()
    body["answers"]["emoji"]["probabilities"]["a"] = 0.5
    with pytest.raises(ValueError, match="normalized"):
        validate_response(body, ["a", "b"])
    body = response()
    body["meta"]["candidate_logits"]["emoji"] = [0, 0]
    with pytest.raises(ValueError, match="disagree"):
        validate_response(body, ["a", "b"])


def test_api_request_withholds_gold_and_counts_http_failures():
    benchmark = script("benchmark_emoji")
    row = {
        "id": "secret-source-id",
        "source_index": 4,
        "split": "test",
        "label": "a",
        "text": "hello",
    }
    question = {"type": "choice", "instructions": "Choose", "criteria": {"a": None, "b": None}}

    def handle(request):
        assert json.loads(request.content) == {
            "model": "jev",
            "state": "hello",
            "questions": {"emoji": question},
        }
        return httpx.Response(503)

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(handle)) as client:
        record = benchmark.request(client, row, question, ["a", "b"])
    assert not record["valid"] and not record["correct"]
    assert record["error_type"] == "HTTPStatusError"
    assert "prediction" not in record


def test_complete_run_retains_three_protocol_failures_when_server_is_healthy(tmp_path, monkeypatch):
    benchmark = script("benchmark_emoji")
    data = tmp_path / "data" / "fixture"
    data.mkdir(parents=True)
    files = {}
    for split, count in (("train", 2), ("dev", 4), ("test", 2)):
        rows = [
            {
                "id": f"{split}-{i}",
                "source_index": i,
                "split": split,
                "label": "a",
                "text": f"{split}-{i}",
            }
            for i in range(count)
        ]
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[split] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": count,
        }
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "labels": ["a", "b"],
                "files": files,
                "question": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": None, "b": None},
                },
            }
        )
    )
    health_checks = []

    def handle(req):
        if req.url.path == "/health":
            health_checks.append(True)
            return httpx.Response(200, json={"ready": True})
        text = json.loads(req.content)["state"]
        if text in {"dev-0", "dev-1", "dev-2"}:
            return httpx.Response(
                503,
                json={"detail": "DiffusionGemma did not produce the expected empty thought prefix"},
            )
        return httpx.Response(200, json=response())

    original_client = httpx.Client
    monkeypatch.setattr(
        benchmark.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    args = SimpleNamespace(data=tmp_path / "data", output=tmp_path / "report", url="http://local")
    benchmark.run(args, "fixture")
    report = args.output / "fixture"
    assert json.loads((report / "run.json").read_text())["completed"]
    summary = json.loads((report / "dev-summary.json").read_text())
    assert summary["count"] == 4 and summary["valid"] == 1 and summary["accuracy"] == 0.25
    assert len(health_checks) == 1
