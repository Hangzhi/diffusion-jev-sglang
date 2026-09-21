"""Run the frozen PUBLIC JevBench tasks against local Jev or the AI SDK worker.

uv run python scripts/compare_jevbench.py --backend local --upstream /path/to/jevbench
uv run python scripts/compare_jevbench.py --backend gateway --upstream /path/to/jevbench
uv run python scripts/compare_jevbench.py --summarize --upstream /path/to/jevbench
"""

import argparse
import hashlib
import json
import math
import select
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import httpx

UPSTREAM_COMMIT = "c6004e008ffba24aec091261ca1a5c02f7324702"
ROOT = Path(__file__).resolve().parents[1]


def load_tasks(upstream):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Expected JevBench commit {UPSTREAM_COMMIT}; got {revision}")
    sys.path.insert(0, str(upstream))
    from jevbench.tasks import load_jsonl

    tasks, hashes = [], {}
    for name in ("easy", "original", "hard"):
        path = upstream / "datasets/public" / f"{name}.jsonl"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tasks.extend((name, t) for t in load_jsonl(str(path)))
    assert len({t.id for _, t in tasks}) == len(tasks)
    assert all(t.split == "public" for _, t in tasks)
    return tasks, hashes


class Runner:
    def __init__(self, backend, url, web):
        self.backend = backend
        self.client = httpx.Client(base_url=url, timeout=120)
        self.worker = None
        if backend == "gateway":
            self.worker = subprocess.Popen(
                ["node", "--import", "tsx", "jev-worker.ts"],
                cwd=web,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

    def request(self, task):
        from jevbench.adapters.base import build_question

        # Gold labels, rationale and provenance NEVER enter either model request.
        body = {"state": task.state, "questions": {"decision": build_question(task)}}
        start = time.perf_counter()
        if self.worker:
            self.worker.stdin.write(json.dumps(body, ensure_ascii=False) + "\n")
            self.worker.stdin.flush()
            if not select.select([self.worker.stdout], [], [], 140)[0]:
                raise RuntimeError("Gateway worker timed out")
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError("Gateway worker stopped; check local credentials")
            return json.loads(line)
        try:
            response = self.client.post("/v1/systemone", json={"model": "diffusion-jev", **body})
            elapsed = time.perf_counter() - start
            if response.status_code != 200:
                return {"ok": False, "latency_s": elapsed, "error": f"HTTP {response.status_code}"}
            result = response.json()
            return {
                "ok": True,
                "latency_s": elapsed,
                "model": result["model"],
                "answers": result["answers"],
                "usage": result.get("usage", {}),
                "settings": {k: result.get("meta", {}).get(k) for k in ("temperature", "passes")},
                "readout": {
                    k: result.get("meta", {}).get(k)
                    for k in (
                        "probability_source",
                        "denoising_steps",
                        "candidate_mass",
                        "unrestricted_first_token",
                        "answer_position",
                    )
                },
            }
        except httpx.HTTPError:
            return {
                "ok": False,
                "latency_s": time.perf_counter() - start,
                "error": "Local HTTP request failed",
            }
        except (KeyError, TypeError, ValueError, AttributeError):
            return {
                "ok": False,
                "latency_s": time.perf_counter() - start,
                "error": "Local response has an invalid schema",
                "category": "invalid_response",
            }

    def close(self):
        self.client.close()
        if self.worker:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()


def outcome(result, task):
    from jevbench.scoring import score_task

    if not result.get("ok"):
        return {"valid": False, "correct": False, "predicted": None}
    answers = result.get("answers")
    answer = answers.get("decision") if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        return {"valid": False, "correct": False, "predicted": None}
    if answer.get("type") != task.question["type"]:
        return {"valid": False, "correct": False, "predicted": None}
    probs = answer.get("probabilities")
    if task.question["type"] == "choice" and answer.get("choice") not in task.labels:
        return {"valid": False, "correct": False, "predicted": None}
    if task.question["type"] == "noul":
        value = answer.get("noul")
        if type(value) not in (int, float):
            return {"valid": False, "correct": False, "predicted": None}
        probs = {"yes": value, "no": 1 - value}
    return score_task(probs, task)


def measured_requests(runner, tasks, concurrency):
    """Bound in-flight HTTP requests; never queue the entire dataset on failure."""
    if concurrency == 1:
        for dataset, task in tasks:
            yield dataset, task, runner.request(task)
        return
    if runner.backend != "local":
        raise ValueError("Concurrent runs currently support only the local HTTP backend")
    pending = {}
    remaining = iter(tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:

        def submit_next():
            item = next(remaining, None)
            if item is not None:
                dataset, task = item
                pending[pool.submit(runner.request, task)] = (dataset, task)

        for _ in range(concurrency):
            submit_next()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    dataset, task = pending.pop(future)
                    yield dataset, task, future.result()
                    submit_next()
        finally:
            for future in pending:
                future.cancel()


def check_settings(result, backend, expected=None, expected_model=None):
    expected = expected or {"temperature": 1.0, "passes": 1}
    if backend == "local" and result.get("ok") and result.get("settings") != expected:
        raise ValueError(f"Frozen benchmark settings differ: expected {expected}")
    if expected_model and result.get("ok") and result.get("model") != expected_model:
        raise ValueError("Active model differs from the requested benchmark model")


def run(args, tasks, hashes):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.backend}.jsonl"
    meta_path = args.output / f"{args.backend}-run.json"
    if path.exists() or meta_path.exists():
        raise ValueError(f"Refusing to overwrite measured results: {path}; choose another --output")
    if args.concurrency < 1 or (args.backend == "gateway" and args.concurrency != 1):
        raise ValueError("Concurrency must be positive; Gateway currently requires concurrency=1")
    runner = Runner(args.backend, args.url, ROOT / "web")
    meta = {
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_sha256": hashes,
        "public_items": len(tasks),
        "backend": args.backend,
        "concurrency": args.concurrency,
        "warmup_requests": 2,
        "retries": 0,
        "completed": False,
        "started": datetime.now(UTC).isoformat(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_model": args.expected_model,
        "expected_settings": {"temperature": 1.0, "passes": args.expected_passes}
        if args.backend == "local"
        else None,
        "latency_scope": "client HTTP loopback"
        if args.backend == "local"
        else "AI SDK through Vercel gateway; excludes Node startup",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    measured_start = None
    measured_count = 0
    measured_valid = 0
    requests = None
    try:
        for _ in range(2):
            warmup = runner.request(tasks[0][1])
            if not warmup.get("ok"):
                raise RuntimeError("Warmup failed: " + warmup.get("error", "invalid response"))
            check_settings(warmup, args.backend, meta["expected_settings"], args.expected_model)
            if not outcome(warmup, tasks[0][1])["valid"]:
                raise RuntimeError("Warmup returned an invalid distribution")
        failures = 0
        with path.open("x") as output:
            measured_start = time.perf_counter()
            requests = measured_requests(runner, tasks, args.concurrency)
            for i, (dataset, task, result) in enumerate(requests):
                check_settings(result, args.backend, meta["expected_settings"], args.expected_model)
                scored = outcome(result, task)
                row = {
                    "id": task.id,
                    "dataset": dataset,
                    "family": task.family,
                    "question_type": task.question["type"],
                    "expected": task.expected,
                    "excluded": bool(task.provenance.get("exclude_reason")),
                    "result": result,
                    "score": scored,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                measured_count += 1
                measured_valid += int(scored["valid"])
                if result.get("status_code") in (401, 403, 429):
                    raise RuntimeError(f"Gateway HTTP {result['status_code']}; run stopped")
                failures = failures + 1 if not result.get("ok") else 0
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    print(f"{args.backend}: {i + 1}/{len(tasks)} evaluated", flush=True)
                if failures >= 3:
                    raise RuntimeError(
                        "Three consecutive transport failures; stopped instead of charging for a failing run"
                    )
        meta["completed"] = True
    finally:
        if measured_start is not None:
            seconds = time.perf_counter() - measured_start
            meta.update(
                measured_seconds=seconds,
                measured_items=measured_count,
                valid_items=measured_valid,
                attempts_per_second=measured_count / seconds,
                valid_decisions_per_second=measured_valid / seconds,
            )
        if requests is not None:
            requests.close()
        meta["finished"] = datetime.now(UTC).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        runner.close()


def aggregate(rows, tasks_by_id):
    from jevbench.composite_v12 import tvd
    from jevbench.metrics import brier_score, ece_top_label, latency_summary, ordinal_mae

    scored = [r for r in rows if r["expected"] is not None and not r["excluded"]]
    valid = [r for r in scored if r["score"]["valid"]]
    pairs = [(max(r["score"]["probs"].values()), r["score"]["correct"]) for r in valid]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    gold_rows = [r for r in valid if tasks_by_id[r["id"]].provenance.get("gold_probs")]
    return {
        "count": len(rows),
        "scorable": len(scored),
        "valid_distributions": len(valid),
        "strict_valid_distributions": sum(r["score"].get("strict_valid", False) for r in valid),
        "renormalized_distributions": sum(r["score"].get("renormalized", False) for r in valid),
        "correct": sum(bool(r["score"]["correct"]) for r in scored),
        "accuracy": mean([int(bool(r["score"]["correct"])) for r in scored]),
        "schema_failures": sum(not r["score"]["valid"] for r in rows),
        "transport_failures": sum(
            not r["result"]["ok"] and r["result"].get("category") != "invalid_response"
            for r in rows
        ),
        "nll": mean(
            [-math.log(max(r["score"]["probs"][str(r["expected"])], 1e-12)) for r in valid]
        ),
        "brier": mean(
            [
                brier_score(r["score"]["probs"], str(r["expected"]), tasks_by_id[r["id"]].labels)
                for r in valid
            ]
        ),
        "ece": ece_top_label(pairs)["ece"] if pairs else None,
        "ordinal_mae": ordinal_mae(
            [
                (int(r["expected"]), r["score"]["ordinal_ev"])
                for r in valid
                if r["question_type"] == "score"
            ]
        ),
        "gold_distribution_count": len(gold_rows),
        "gold_tvd": mean(
            [
                tvd(
                    r["score"]["probs"],
                    tasks_by_id[r["id"]].provenance["gold_probs"],
                    tasks_by_id[r["id"]].labels,
                )
                for r in gold_rows
            ]
        ),
        "latency": latency_summary([r["result"]["latency_s"] for r in rows]),
    }


def load_run(directory, backend, tasks, hashes):
    """Verify coverage and rescore saved native probabilities before aggregation."""
    rows = [json.loads(line) for line in (directory / f"{backend}.jsonl").read_text().splitlines()]
    meta = json.loads((directory / f"{backend}-run.json").read_text())
    if meta["upstream_commit"] != UPSTREAM_COMMIT or meta["dataset_sha256"] != hashes:
        raise ValueError("Dataset revision or hashes differ from the run")
    task_map = {t.id: (dataset, t) for dataset, t in tasks}
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs in measured run")
    if not set(ids) <= task_map.keys():
        raise ValueError("Unknown task IDs in measured run")
    if meta["completed"] and set(ids) != task_map.keys():
        raise ValueError("Completed run is missing task IDs")
    for row in rows:
        dataset, task = task_map[row["id"]]
        expected = {
            "dataset": dataset,
            "family": task.family,
            "question_type": task.question["type"],
            "expected": task.expected,
            "excluded": bool(task.provenance.get("exclude_reason")),
        }
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("Saved task metadata differs from the pinned dataset")
        latency = row["result"].get("latency_s")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid recorded latency")
        check_settings(
            row["result"], backend, meta.get("expected_settings"), meta.get("expected_model")
        )
        row["score"] = outcome(row["result"], task)
    return rows, meta


def summarize(args, tasks, hashes):
    task_map = {t.id: t for _, t in tasks}
    summary = {
        "scope": "231 public JevBench items, not the official 534-item ranking",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_sha256": hashes,
        "models": {},
    }
    by_model = {}
    for backend in ("local", "gateway"):
        path = args.output / f"{backend}.jsonl"
        if not path.exists():
            continue
        rows, meta = load_run(args.output, backend, tasks, hashes)
        by_model[backend] = {r["id"]: r for r in rows}
        groups = defaultdict(list)
        for row in rows:
            groups[f"dataset:{row['dataset']}"].append(row)
            groups[f"family:{row['family']}"].append(row)
            groups[f"type:{row['question_type']}"].append(row)
        summary["models"][backend] = {
            "completed": meta["completed"],
            "coverage": len(rows) / len(tasks),
            "concurrency": meta["concurrency"],
            "measured_seconds": meta.get("measured_seconds"),
            "valid_decisions_per_second": meta.get("valid_decisions_per_second"),
            "overall": aggregate(rows, task_map),
            "groups": {k: aggregate(v, task_map) for k, v in groups.items()},
        }
    if set(by_model) == {"local", "gateway"}:
        ids = by_model["local"].keys() & by_model["gateway"].keys()
        summary["paired"] = {
            "count": len(ids),
            "local_only_correct": [],
            "gateway_only_correct": [],
            "both_wrong": [],
        }
        for tid in sorted(ids):
            a, b = (bool(by_model[m][tid]["score"]["correct"]) for m in ("local", "gateway"))
            key = (
                "local_only_correct"
                if a and not b
                else "gateway_only_correct"
                if b and not a
                else "both_wrong"
                if not a and not b
                else None
            )
            if key:
                summary["paired"][key].append(tid)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v["overall"] for k, v in summary["models"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "gateway"], default="local")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--expected-passes", type=int, default=1)
    parser.add_argument("--expected-model")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/jevbench")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    tasks, hashes = load_tasks(args.upstream)
    if args.summarize:
        summarize(args, tasks, hashes)
    else:
        run(args, tasks, hashes)
