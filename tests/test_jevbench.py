"""Regression checks for benchmark integrity and bounded request concurrency."""

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "compare_jevbench", Path(__file__).parents[1] / "scripts/compare_jevbench.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_concurrent_requests_are_bounded_and_keep_task_identity():
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=5)
    active = peak = calls = 0

    def request(task):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            invocation = calls
            peak = max(active, peak)
        if invocation <= 4:
            barrier.wait()
        time.sleep(0.002 * (task.id % 3))
        with lock:
            active -= 1
        return {"task_id": task.id}

    tasks = [("test", SimpleNamespace(id=i)) for i in range(13)]
    runner = SimpleNamespace(backend="local", request=request)
    results = list(bench.measured_requests(runner, tasks, 4))
    assert peak == 4
    assert sorted(task.id for _, task, _ in results) == list(range(13))
    assert all(result["task_id"] == task.id for _, task, result in results)


@pytest.mark.parametrize(
    "settings", [None, {"temperature": 0.5, "passes": 1}, {"temperature": 1.0, "passes": 4}]
)
def test_frozen_settings_reject_different_experiment(settings):
    with pytest.raises(ValueError, match="Frozen benchmark"):
        bench.check_settings({"ok": True, "settings": settings}, "local")


@pytest.mark.parametrize(
    "ids,completed,error",
    [
        (["a", "a"], True, "Duplicate"),
        (["a"], True, "missing"),
        (["a", "foreign"], False, "Unknown"),
    ],
)
def test_saved_run_cannot_claim_false_coverage(tmp_path, ids, completed, error):
    (tmp_path / "local.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ids))
    (tmp_path / "local-run.json").write_text(
        json.dumps(
            {
                "upstream_commit": bench.UPSTREAM_COMMIT,
                "dataset_sha256": {"test": "abc"},
                "completed": completed,
            }
        )
    )
    tasks = [("test", SimpleNamespace(id=i)) for i in ("a", "b")]
    with pytest.raises(ValueError, match=error):
        bench.load_run(tmp_path, "local", tasks, {"test": "abc"})


def test_malformed_http_success_counts_as_invalid_response():
    import httpx

    # No scorer is needed: an HTTP 200 with malformed JSON is a failed response.
    runner = bench.Runner("local", "http://localhost", Path("."))
    runner.client.close()
    runner.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON")),
        base_url="http://localhost",
    )
    # request() needs upstream build_question; keep this check independent of its install.
    import sys
    from unittest.mock import patch

    base = SimpleNamespace(build_question=lambda task: {"type": "noul", "instructions": "?"})
    with patch.dict(sys.modules, {"jevbench.adapters.base": base}):
        result = runner.request(SimpleNamespace(state="public state"))
    runner.close()
    assert result["ok"] is False
    assert result["category"] == "invalid_response"
    assert result["latency_s"] >= 0
