"""Render the completed local-versus-published comparison using matplotlib.

uvx --with matplotlib python scripts/render_jev_comparison.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "reports/jevbench"
report = json.loads((folder / "published-comparison.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
fig.set_facecolor("#f5f8f3")
colors = ["#266448", "#9f78b5"]
labels = ["Easy", "Original", "Hard", "Overall"]
groups = [*report["datasets"].values(), report["overall"]]
x = np.arange(len(groups))
for i, name in enumerate(("local", "published_jev")):
    vals = [100 * g[name]["accuracy"] for g in groups]
    bars = axes[0].bar(x + (i - 0.5) * 0.36, vals, width=0.36, color=colors[i])
    axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in vals], fontsize=9, padding=3)
axes[0].set_xticks(x, labels)
axes[0].set_ylim(0, 112)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Identical 231 public task IDs", loc="left", fontsize=12)
for i, name in enumerate(("local", "published_jev")):
    vals = [report["overall"][name]["latency"][f"{q}_s"] for q in ("p50", "p95")]
    bars = axes[1].bar(np.arange(2) + (i - 0.5) * 0.36, vals, width=0.36, color=colors[i])
    axes[1].bar_label(bars, labels=[f"{v:.3f}s" for v in vals], fontsize=10, padding=3)
axes[1].set_xticks(np.arange(2), ["Median", "p95"])
axes[1].set_ylim(0, 3.1)
axes[1].set_ylabel("Client-observed seconds")
axes[1].set_title("Different deployment/network paths", loc="left", fontsize=12)
for ax in axes:
    ax.set_facecolor("#f5f8f3")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
fig.suptitle("Local diffusion Jev vs published TypeSafe Jev 1.13.0", fontsize=16, x=0.07, ha="left")
fig.legend(
    [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
    ["Local A100 · measured in this run", "TypeSafe Jev · historical published run"],
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.04),
)
fig.text(
    0.07,
    0.015,
    "Local: BF16 LLaDA2.1-mini, 1 pass, T=1, loopback. Jev: production API from Germany. No new Gateway run.",
    fontsize=9,
    color="#4e5b53",
)
fig.tight_layout(rect=(0, 0.12, 1, 0.93))
fig.savefig(folder / "comparison.png", dpi=180, facecolor=fig.get_facecolor())
