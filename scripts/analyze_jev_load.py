"""Compare complete local JevBench runs at different request concurrency levels.

Example:
  uv run python scripts/analyze_jev_load.py --upstream /path/to/jevbench \
    --runs reports/jevbench-load/c1 reports/jevbench-load/c4 \
    --reference reports/jevbench --output reports/jevbench-load
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

from compare_jevbench import aggregate, load_run, load_tasks


def paired(reference, candidate):
    before = {r["id"]: r for r in reference}
    after = {r["id"]: r for r in candidate}
    if before.keys() != after.keys():
        raise ValueError("Paired comparison requires identical task IDs")
    changed = []
    distances = []
    for tid, a in before.items():
        b = after[tid]
        if a["score"].get("predicted") != b["score"].get("predicted"):
            changed.append(
                {
                    "id": tid,
                    "before": a["score"].get("predicted"),
                    "after": b["score"].get("predicted"),
                    "before_correct": a["score"]["correct"],
                    "after_correct": b["score"]["correct"],
                }
            )
        if a["score"]["valid"] and b["score"]["valid"]:
            pa, pb = a["score"]["probs"], b["score"]["probs"]
            distances.append(sum(abs(pa[k] - pb[k]) for k in pa) / 2)
    return {
        "items": len(before),
        "changed_predictions": changed,
        "both_valid": len(distances),
        "mean_probability_tvd": sum(distances) / len(distances) if distances else None,
        "max_probability_tvd": max(distances) if distances else None,
    }


def main(args):
    tasks, hashes = load_tasks(args.upstream)
    task_map = {t.id: t for _, t in tasks}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "created": datetime.now(UTC).isoformat(),
        "description": "Measured local request concurrency; no fresh hosted Jev run",
        "dataset_sha256": hashes,
        "runs": {},
        "paired": {},
    }
    records = {}
    for directory in args.runs:
        name = directory.name
        if name in records:
            raise ValueError("Run directory names must be unique")
        rows, meta = load_run(directory, "local", tasks, hashes)
        if not meta["completed"]:
            raise ValueError("Load comparison requires completed runs")
        records[name] = rows
        by_length, by_set = defaultdict(list), defaultdict(list)
        for row in rows:
            tokens = row["result"].get("usage", {}).get("input_tokens")
            band = "unavailable"
            if isinstance(tokens, int):
                band = (
                    "1–256"
                    if tokens <= 256
                    else "257–1,024"
                    if tokens <= 1024
                    else "1,025–4,096"
                    if tokens <= 4096
                    else "4,097+"
                )
            by_length[band].append(row)
            by_set[row["dataset"]].append(row)
        report["runs"][name] = {
            "path": os.path.relpath(directory, args.output),
            "metadata": meta,
            "overall": aggregate(rows, task_map),
            "by_dataset": {k: aggregate(v, task_map) for k, v in by_set.items()},
            "by_prompt_tokens": {
                k: aggregate(by_length[k], task_map)
                for k in ("1–256", "257–1,024", "1,025–4,096", "4,097+", "unavailable")
                if k in by_length
            },
        }
    first_name, first_rows = next(iter(records.items()))
    for (a, a_rows), (b, b_rows) in combinations(records.items(), 2):
        report["paired"][f"{a} → {b}"] = paired(a_rows, b_rows)
    if args.reference:
        rows, meta = load_run(args.reference, "local", tasks, hashes)
        if not meta["completed"]:
            raise ValueError("Reference must be complete")
        report["paired"][f"previous run → {first_name}"] = paired(rows, first_rows)

    lines = [
        "# JevBench: local load and reproducibility measurements",
        "",
        (
            "Fresh measurements on the same 231 public JevBench tasks, with the original "
            "state, rubrics and option order. Model: BF16 LLaDA2.1-mini on the A100 80GB; "
            "one pass, temperature 1. Each request contains one decision."
        ),
        "",
        "| Run | Concurrent requests | Correct | Valid | Median | p95 | Valid decisions/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in report["runs"].items():
        m, a = run["metadata"], run["overall"]
        rate = m.get("valid_decisions_per_second")
        rate_text = f"{rate:.2f}" if rate is not None else "unavailable"
        lines.append(
            f"| {name} | {m['concurrency']} | {a['correct']}/{a['scorable']} "
            f"({a['accuracy']:.1%}) | {a['valid_distributions']}/{a['count']} | "
            f"{a['latency']['p50_s'] * 1000:.0f} ms | "
            f"{a['latency']['p95_s'] * 1000:.0f} ms | {rate_text} |"
        )
    lines += ["", "## Same-item stability", ""]
    for name, comparison in report["paired"].items():
        mean_tvd = comparison["mean_probability_tvd"]
        max_tvd = comparison["max_probability_tvd"]
        drift = f"{mean_tvd:.6f} / {max_tvd:.6f}" if mean_tvd is not None else "unavailable"
        lines.append(
            f"- {name}: {len(comparison['changed_predictions'])}/{comparison['items']} "
            f"predicted labels changed; probability TVD (mean/max): "
            f"{drift} across "
            f"{comparison['both_valid']} pairs with valid distributions."
        )
    lines += [
        "",
        (
            "TVD is half the sum of absolute probability differences. A zero label-change "
            "count does not imply identical probabilities. Changed item IDs and correctness "
            "are recorded in `load-comparison.json`."
        ),
        "",
        (
            "Treat concurrency and batch composition as part of the measured inference "
            "configuration. These observations do not isolate the cause of any output "
            "differences in the model or serving implementation."
        ),
        "",
        "## Prompt length",
        "",
        "| Run | Input tokens | Items | Accuracy | Median | p95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, run in report["runs"].items():
        for band, a in run["by_prompt_tokens"].items():
            lines.append(
                f"| {name} | {band} | {a['count']} | {a['accuracy']:.1%} | "
                f"{a['latency']['p50_s'] * 1000:.0f} ms | "
                f"{a['latency']['p95_s'] * 1000:.0f} ms |"
            )
    lines += [
        "",
        (
            "Token count includes the local model's prompt template. Task content and "
            "difficulty differ across length groups; this is not a controlled causal test of length."
        ),
        "",
        "## Method and limits",
        "",
        (
            "Runs execute sequentially on the same engine. Two serial warmups precede each run "
            "and are excluded. Tasks are submitted in the frozen easy/original/hard order; "
            "concurrent runs save completion order and compare by exact task ID. Concurrency "
            "is a closed-loop cap on in-flight requests. Request latency includes server "
            "queueing; throughput divides valid decisions by measured wall time after warmup. "
            "Failures remain in accuracy denominators. There are no retries."
        ),
        "",
        (
            "Each row is one sweep; repeated settings are separate measurements, not confidence "
            "intervals or an open-loop saturation test. These runs do not establish peak capacity "
            "or cloud latency. No prompts, thresholds, "
            "temperatures or labels were tuned using these results."
        ),
        "",
        (
            "Historical hosted Jev scored 200/231 (86.6%) on these items; that remains a "
            "published reference, not a new Gateway measurement. A valid local Gateway "
            "credential is still needed for a fresh hosted comparison. "
            "[Earlier comparison](../jevbench/README.md)."
        ),
        "",
        "## Raw artifacts",
        "",
    ]
    for name, run in report["runs"].items():
        lines.append(
            f"- [{name} measurements]({run['path']}/local.jsonl) · "
            f"[run metadata]({run['path']}/local-run.json)"
        )
        if (args.output / run["path"] / "published-comparison.json").exists():
            lines.append(
                f"- [{name} versus historical Jev]({run['path']}/published-comparison.json)"
            )
    lines += ["- [Aggregate and paired results](load-comparison.json)", ""]
    if (args.output / "environment.json").exists():
        lines.insert(-1, "- [Engine settings and source hashes](environment.json)")
    (args.output / "load-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "README.md").write_text("\n".join(lines))
    print(args.output / "README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
