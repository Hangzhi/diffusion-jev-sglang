"""Render completed Gemma benchmarks: uvx --with matplotlib python scripts/render_gemma_report.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "reports/diffusiongemma"
report = json.loads((ROOT / "jevbench/published-comparison.json").read_text())
flowers = json.loads((ROOT / "flowers/summary.json").read_text())
colors = ["#266448", "#9f78b5"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
groups = [report["datasets"][key] for key in ("easy", "original", "hard")] + [report["overall"]]
for i, key in enumerate(("local", "published_jev")):
    accuracy = [100 * group[key]["accuracy"] for group in groups]
    bars = axes[0].bar(np.arange(4) + (i - 0.5) * 0.36, accuracy, width=0.36, color=colors[i])
    axes[0].bar_label(
        bars,
        labels=["100%" if v == 100 else f"{v:.1f}%" for v in accuracy],
        fontsize=9,
        padding=3,
    )
    latency = [report["overall"][key]["latency"][f"{q}_s"] for q in ("p50", "p95")]
    bars = axes[1].bar(np.arange(2) + (i - 0.5) * 0.36, latency, width=0.36, color=colors[i])
    axes[1].bar_label(bars, labels=[f"{v:.3f}s" for v in latency], padding=3)
axes[0].set(
    xticks=np.arange(4),
    xticklabels=["Easy", "Original", "Hard", "Overall"],
    ylim=(0, 112),
    ylabel="Accuracy (%)",
    title="Same 231 public JevBench tasks",
)
axes[1].set(
    xticks=np.arange(2),
    xticklabels=["Median", "p95"],
    ylabel="Client-observed seconds",
    title="Different deployment and network paths",
)
axes[1].set_ylim(
    0, max(report["overall"][key]["latency"]["p95_s"] for key in ("local", "published_jev")) * 1.2
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
fig.suptitle("DiffusionGemma vs published TypeSafe Jev 1.13.0", fontsize=16)
fig.legend(
    [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
    ["DiffusionGemma · measured locally", "Jev · historical published results"],
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.055),
)
fig.text(
    0.05,
    0.02,
    "Gemma: A100 80GB, BF16, native adaptive denoising ≤48 steps, loopback. No fresh hosted Jev run.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.14, 1, 0.94))
fig.savefig(ROOT / "comparison.png", dpi=180)
plt.close(fig)

labels = list(flowers["by_class"])
columns = labels + (["invalid"] if flowers["valid"] < flowers["count"] else [])
matrix = np.array([[flowers["confusion_matrix"][a][b] for b in columns] for a in labels])
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.imshow(matrix, cmap="Greens", vmin=0, vmax=20)
for i in range(len(labels)):
    for j in range(len(columns)):
        ax.text(
            j,
            i,
            str(matrix[i, j]),
            ha="center",
            va="center",
            color="white" if matrix[i, j] > 10 else "#18261e",
        )
ax.set(
    xticks=np.arange(len(columns)),
    xticklabels=columns,
    yticks=np.arange(len(labels)),
    yticklabels=labels,
    xlabel="Predicted class",
    ylabel="Dataset label",
)
ax.set_title(
    f"DiffusionGemma · {flowers['correct']}/{flowers['count']} flower images correct", pad=18
)
fig.text(
    0.5,
    0.025,
    "Frozen test: 20 images per class. Pixels only; labels withheld from inference.\nExact duplicates removed; pretraining overlap unknown.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(ROOT / "flowers-confusion.png", dpi=180)
plt.close(fig)
