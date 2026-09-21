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
            }
        except httpx.HTTPError:
            return {
                "ok": False,
                "latency_s": time.perf_counter() - start,
                "error": "Local HTTP request failed",
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
    answer = result.get("answers", {}).get("decision", {})
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


def run(args, tasks, hashes):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.backend}.jsonl"
    meta_path = args.output / f"{args.backend}-run.json"
    if path.exists():
        raise ValueError(f"Refusing to overwrite measured results: {path}; choose another --output")
    runner = Runner(args.backend, args.url, ROOT / "web")
    meta = {
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_sha256": hashes,
        "public_items": len(tasks),
        "backend": args.backend,
        "concurrency": 1,
        "warmup_requests": 2,
        "retries": 0,
        "completed": False,
        "started": datetime.now(UTC).isoformat(),
        "latency_scope": "client HTTP loopback"
        if args.backend == "local"
        else "AI SDK through Vercel gateway; excludes Node startup",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    try:
        for _ in range(2):
            warmup = runner.request(tasks[0][1])
            if not warmup.get("ok"):
                raise RuntimeError("Warmup failed: " + warmup.get("error", "invalid response"))
        failures = 0
        with path.open("x") as output:
            for i, (dataset, task) in enumerate(tasks):
                result = runner.request(task)
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
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        meta = json.loads((args.output / f"{backend}-run.json").read_text())
        if meta["dataset_sha256"] != hashes:
            raise ValueError("Dataset hashes differ from the run")
        by_model[backend] = {r["id"]: r for r in rows}
        groups = defaultdict(list)
        for row in rows:
            groups[f"dataset:{row['dataset']}"].append(row)
            groups[f"family:{row['family']}"].append(row)
            groups[f"type:{row['question_type']}"].append(row)
        summary["models"][backend] = {
            "completed": meta["completed"],
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
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/jevbench")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    tasks, hashes = load_tasks(args.upstream)
    if args.summarize:
        summarize(args, tasks, hashes)
    else:
        run(args, tasks, hashes)
