"""Measure client-observed latency/throughput for 1, 8, and 32 questions.

uv run python scripts/benchmark_matrix.py --output reports/matrix.json
Runs are sequential across configurations; each configuration uses the chosen concurrency.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np


async def run(args):
    preset = json.loads(
        (Path(__file__).parents[1] / "src/diffusion_jev/data/presets.json").read_text()
    )[0]["request"]
    rows = []
    async with httpx.AsyncClient(base_url=args.url, timeout=180) as client:
        for count in (1, 8, 32):
            request = {
                **preset,
                "questions": {f"q{i}": preset["questions"]["emoji"] for i in range(count)},
            }
            for concurrency in (1, 4):
                for _ in range(2):
                    (await client.post("/v1/systemone", json=request)).raise_for_status()
                semaphore = asyncio.Semaphore(concurrency)

                async def evaluate(semaphore=semaphore, request=request, count=count):
                    async with semaphore:
                        start = time.perf_counter()
                        response = await client.post("/v1/systemone", json=request)
                        response.raise_for_status()
                        body = response.json()
                        assert len(body["answers"]) == count
                        assert all(
                            a["choice"] == body["answers"]["q0"]["choice"]
                            for a in body["answers"].values()
                        )
                        return (time.perf_counter() - start) * 1000, body["meta"]["passes"]

                start = time.perf_counter()
                results = await asyncio.gather(*(evaluate() for _ in range(args.requests)))
                elapsed = time.perf_counter() - start
                latencies = [r[0] for r in results]
                rows.append(
                    {
                        "questions": count,
                        "concurrency": concurrency,
                        "requests": args.requests,
                        "passes": results[0][1],
                        "p50_ms": float(np.percentile(latencies, 50)),
                        "p95_ms": float(np.percentile(latencies, 95)),
                        "requests_per_second": args.requests / elapsed,
                        "decisions_per_second": args.requests * count / elapsed,
                        "latencies_ms": latencies,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "description": "Original emoji preset; warmup excluded; client-observed latency",
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/matrix.json"))
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    asyncio.run(run(args))
