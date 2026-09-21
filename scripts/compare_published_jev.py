"""Match a live local run to published Jev 1.13.0 outcomes by exact task ID.

This is explicitly historical; it never impersonates a new Gateway run.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compare_jevbench import ROOT, UPSTREAM_COMMIT, load_run, load_tasks


def main(args):
    tasks, hashes = load_tasks(args.upstream)
    task_map = {t.id: (dataset, t) for dataset, t in tasks}
    from jevbench.metrics import latency_summary

    source = args.upstream / "results/v1.2/jevbench-v1.2-per-task.json"
    published = json.loads(source.read_text())["systems"]["jev-1.13.0"]["public_tasks"]
    rows, meta = load_run(args.output, "local", tasks, hashes)
    local = {r["id"]: r for r in rows}
    if not meta["completed"] or set(local) != set(task_map) or not set(task_map) <= set(published):
        raise ValueError("Need a complete local run and published outcomes for all matching IDs")
    matched = [
        {
            "id": tid,
            "dataset": task_map[tid][0],
            "family": task_map[tid][1].family,
            "local_correct": bool(local[tid]["score"]["correct"]),
            "published_jev_correct": published[tid][0] == "c",
            "published_jev_outcome": published[tid][0],
            "local_latency_s": local[tid]["result"]["latency_s"],
            "published_jev_latency_s": published[tid][1],
        }
        for tid in task_map
    ]

    def summarize(rows):
        return {
            "count": len(rows),
            **{
                name: {
                    "correct": sum(r[f"{name}_correct"] for r in rows),
                    "accuracy": sum(r[f"{name}_correct"] for r in rows) / len(rows),
                    "latency": latency_summary([r[f"{name}_latency_s"] for r in rows]),
                }
                for name in ("local", "published_jev")
            },
        }

    report = {
        "comparison_type": "new local measurements versus historical Jev 1.13.0 measurements",
        "gateway_rerun": False,
        "source_url": f"https://github.com/fstandhartinger/jevbench/blob/{UPSTREAM_COMMIT}/results/v1.2/jevbench-v1.2-per-task.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Exact same 231 public task IDs, no private or imported judge items",
        "limitations": [
            "Local latency uses loopback on the A100 server; published Jev latency was measured through TypeSafe's production API from Germany.",
            "Different measurement time, model version and network path; these are not same-environment Vercel Gateway measurements.",
            "Published public-item artifact has outcomes and rounded timings, not distributions; no matched calibration comparison can be computed.",
            "One attempt per item; original items include paraphrase pairs, so observations are not independent.",
            "No estimated production latency multiplier, no official composite rank, and no invented local hosting cost.",
        ],
        "overall": summarize(matched),
        "datasets": {
            d: summarize([r for r in matched if r["dataset"] == d])
            for d in ("easy", "original", "hard")
        },
        "families": {
            f: summarize([r for r in matched if r["family"] == f])
            for f in sorted({r["family"] for r in matched})
        },
        "paired": {
            "local_only_correct": [
                r["id"] for r in matched if r["local_correct"] and not r["published_jev_correct"]
            ],
            "published_jev_only_correct": [
                r["id"] for r in matched if r["published_jev_correct"] and not r["local_correct"]
            ],
            "both_wrong": [
                r["id"]
                for r in matched
                if not r["local_correct"] and not r["published_jev_correct"]
            ],
        },
        "matched_items": matched,
    }
    (args.output / "published-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "datasets")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "reports/jevbench")
    main(p.parse_args())
