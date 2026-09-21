"""Render completed emoji benchmark artifacts: uvx --with matplotlib python scripts/render_emoji_report.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/diffusiongemma/emoji"


def read(name, file):
    return json.loads((REPORT / name / file).read_text())


def percent(value):
    return f"{100 * value:.1f}%"


def main():
    names = ("emojify", "tweeteval")
    for name in names:
        if not read(name, "run.json")["completed"]:
            raise ValueError(f"{name} benchmark is incomplete")
    summaries = {name: read(name, "test-summary.json") for name in names}
    calibrated = {name: read(name, "test-calibrated-summary.json") for name in names}
    baseline = {name: read(name, "baselines.json") for name in names}
    manifests = {name: read(name, "dataset.json") for name in names}
    e, t = (summaries[name] for name in names)
    rb = baseline["tweeteval"]["published_tweeteval_roberta"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = ["#266448", "#9f78b5", "#aeb9b1"]
    groups = [t, rb, baseline["tweeteval"]["train_majority"]]
    labels = ["DiffusionGemma", "Published RoBERTa", "Train-majority baseline"]
    for ax, metric, title in zip(
        axes, ["accuracy", "macro_f1"], ["Top-1 accuracy", "Macro F1"], strict=True
    ):
        values = [100 * group[metric] for group in groups]
        bars = ax.bar(range(3), values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4)
        ax.set(
            xticks=range(3),
            xticklabels=["Gemma", "RoBERTa", "Majority"],
            ylim=(0, max(values) * 1.25),
            title=title,
            ylabel="Percent",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"Emoji prediction · same {t['count']:,} TweetEval test items", fontsize=16)
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.text(
        0.5,
        0.025,
        f"Gemma: zero-shot local API; {t['failures']} invalid responses count as wrong. RoBERTa: historical task-trained predictions.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    fig.savefig(REPORT / "comparison.png", dpi=180)
    plt.close(fig)

    criteria = manifests["tweeteval"]["question"]["criteria"]
    ordered = sorted(criteria, key=lambda label: -t["by_class"][label]["recall"])
    fig, ax = plt.subplots(figsize=(10, 8))
    values = [100 * t["by_class"][label]["recall"] for label in ordered]
    bars = ax.barh(np.arange(len(ordered)), values, color=colors[0])
    ax.bar_label(
        bars,
        labels=[
            f"{t['by_class'][label]['correct']}/{t['by_class'][label]['count']}"
            for label in ordered
        ],
        padding=4,
        fontsize=8,
    )
    ax.set(
        yticks=np.arange(len(ordered)),
        yticklabels=[criteria[label].replace("hearteyes", "heart eyes") for label in ordered],
        xlabel="Recall (%) · failures included",
        xlim=(0, 110),
        title="DiffusionGemma · performance across 20 emoji classes",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(REPORT / "per-class.png", dpi=180)
    plt.close(fig)

    def overview_row(title, result):
        return f"| {title} | {result['correct']}/{result['count']} — {percent(result['accuracy'])} | {percent(result['macro_f1'])} | {percent(result['top3_accuracy_including_failures'])} | {result['valid']}/{result['count']} | {result['latency_ms']['p50']:.1f} / {result['latency_ms']['p95']:.1f} ms |"

    text = [
        "# Emoji prediction benchmark",
        "",
        "Measured September 21, 2026 through the running DiffusionGemma Jev-compatible API on one A100 80GB. Two tasks measure different uses: mapping short sentences into five broad emoji categories, and predicting an author's choice among 20 overlapping emoji classes in real tweets.",
        "",
        "| Test | Top-1 accuracy | Macro F1 | Top-3 accuracy | Valid responses | Median / p95 |",
        "|---|---:|---:|---:|---:|---:|",
        overview_row("Emojify, five classes", e),
        overview_row("TweetEval, 20 classes", t),
        "",
        "Invalid responses count as wrong in accuracy, top-3 accuracy, per-class recall, and macro F1. Probability metrics use valid responses only. Macro F1 averages the classes equally, which matters because TweetEval contains many more heart examples than some other emoji.",
        "",
        "## Comparisons and uncertainty",
        "",
        f"On the exact same {t['count']:,} TweetEval source indices, the [published TweetEval RoBERTa predictions](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/predictions) score **{rb['correct']}/{rb['count']} ({percent(rb['accuracy'])}) accuracy and {percent(rb['macro_f1'])} macro F1**. This is a historical task-trained baseline; DiffusionGemma uses no task training or few-shot examples. The local sample is not the complete official test set and does not establish a leaderboard rank. No RoBERTa latency or probability comparison was measured.",
        "",
        f"A constant label selected from training frequencies gets {percent(baseline['emojify']['train_majority']['accuracy'])} accuracy on Emojify and {percent(baseline['tweeteval']['train_majority']['accuracy'])} on TweetEval. Uniform random expected accuracy is 20% and 5%, respectively.",
        "",
        f"DiffusionGemma is {100 * (rb['accuracy'] - t['accuracy']):.1f} percentage points below the trained RoBERTa baseline on this sample. Output-format failures account for at most {100 * t['failures'] / t['count']:.1f} points if the existing valid predictions stay fixed, so improving the response protocol alone would not close the entire gap.",
        "",
        f"Wilson 95% intervals for top-1 accuracy are {percent(e['accuracy_wilson_95'][0])}–{percent(e['accuracy_wilson_95'][1])} for Emojify and {percent(t['accuracy_wilson_95'][0])}–{percent(t['accuracy_wilson_95'][1])} for TweetEval. These are descriptive item-level intervals; they do not account for near-duplicates, author clusters, subjective labels, or unknown pretraining overlap.",
        "",
        "**No official hosted Jev emoji result was measured.** The previously reported 231-task JevBench figures cover different tasks and cannot be reused as an emoji baseline. Gateway credentials were unavailable for a hosted comparison in the preceding deployment work.",
        "",
        "![Matched TweetEval accuracy and macro F1](comparison.png)",
        "",
        "## Frozen datasets and task definitions",
        "",
        "**Emojify:** [Kaggle `alvinrindra/emojify`, version 2](https://www.kaggle.com/datasets/alvinrindra/emojify). Its headerless CSVs contain 132 training and 56 test rows. One training duplicate and six test duplicates/overlaps are removed. The result is 111 training rows, 20 development rows (four per class), and all 50 retained original test rows. The combined CSV is never merged into either split. Classes are love ❤️, sports ⚾, happiness 😄, sadness/distress 😞, and food 🍴. These represent the source numeric classes; the small web demo uses different icons for some categories.",
        "",
        "**TweetEval:** [Cardiff NLP source](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/emoji), commit `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`. After duplicate/conflicting-label filtering, 44,389 training, 4,807 validation, and 49,937 test rows remain. Salted SHA256 order selects 200 development and 1,000 test rows without class balancing. All 20 original labels are retained. Gold labels represent the emoji originally used by the author, so an alternative sensible emoji is still scored wrong. Upstream text formatting is retained for inference.",
        "",
        "Deduplication matches NFKC-normalized, case-folded text with collapsed whitespace. Conflicting-label groups are removed before sampling; repeated text keeps its earliest split. Near-duplicate detection is not performed. The samples, option order, descriptions, and prompts were frozen before test inference. No weights or prompts were fitted on test labels.",
        "",
        "Raw source files, prepared JSONL, and directly reusable API request bodies are saved under `/workspace/diffusion-jev-sglang/data/emoji-benchmark/{emojify,tweeteval}/`. Each `manifest.json` records source hashes, selection rules, source row counts, and prepared-file hashes. Raw corpora remain outside Git. Kaggle reports an unknown license, and TweetEval's emoji subset does not specify a separate license; provenance is retained without asserting an unrestricted redistribution license.",
        "",
        "## Reliability and confidence",
        "",
        f"The completed TweetEval test has **{t['failures']} invalid responses**, all rejected with HTTP 503 because the native generation did not match the adapter's required empty-thought prefix. Accuracy among the {t['valid']} valid responses is {percent(t['probability_metrics_valid_only']['accuracy'])}; the headline result includes all {t['count']} requests. The first development-only attempt stopped after three consecutive rejections; it is preserved under `incomplete-dev/`. The harness was corrected to check server readiness and continue counting individual failures. The development run was repeated before touching the frozen test set. Model weights, prompts, and inference behavior were not changed.",
        "",
        "Candidate scores come from the last active self-conditioned denoising step. They are not calibrated correctness probabilities. The following temperature fits use exclusively valid development records and were frozen before test inference. The calibrated columns are an offline transformation of saved logits; the running web app remains at T=1.",
        "",
        "| Test | Dev-fitted T | Raw NLL → calibrated | Raw Brier → calibrated | Raw ECE → calibrated | Wrong answers with raw top probability ≥90% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        s, c = summaries[name], calibrated[name]
        raw, cal = s["probability_metrics_valid_only"], c["probability_metrics_valid_only"]
        temp = read(name, "calibration.json")["temperature"]
        text.append(
            f"| {name} | {temp:.3f} | {raw['nll']:.3f} → {cal['nll']:.3f} | {raw['brier']:.3f} → {cal['brier']:.3f} | {raw['ece_10_bins']:.3f} → {cal['ece_10_bins']:.3f} | {s['incorrect_with_top_probability_at_least_90_percent']} |"
        )
    text += [
        "",
        "Temperature search uses the existing 401-point geometric grid from 0.1 to 10. A positive temperature preserves the class ranking and does not resolve invalid responses. The small development samples limit calibration conclusions.",
        "",
        f"Full-vocabulary candidate mass fell below 50% on {e['candidate_mass_below_half_count']} valid Emojify responses and {t['candidate_mass_below_half_count']} valid TweetEval responses. Candidate normalization can hide out-of-option answers; retain the raw diagnostics when deciding whether to abstain.",
        "",
        "![Per-class TweetEval recall](per-class.png)",
        "",
        "## Reproduce and inspect",
        "",
        "The runtime is unchanged from the [DiffusionGemma deployment](../README.md): pinned model/source, BF16, native 256-position diffusion canvas, at most 48 steps, no prefix caching or CUDA graphs. Each benchmark uses serial requests, two excluded training-example warmups, zero retries, and no competing evaluation job. `run.json` records timings, actual model identity, file hashes, and calibration timing. The app and model remain running.",
        "",
        "```bash",
        "cd /workspace/diffusion-jev-sglang",
        "# For a new data copy, choose an unused output directory:",
        ".venv/bin/python scripts/prepare_emoji_benchmark.py --output data/emoji-repeat",
        "# Existing frozen data; always write each new run to an unused report directory:",
        ".venv/bin/python scripts/benchmark_emoji.py --output reports/emoji-repeat",
        "# Render the original completed report:",
        "uvx --with matplotlib python scripts/render_emoji_report.py",
        "```",
        "",
        "For each task, `dataset.json` and `selection.json` capture the frozen setup and source indices; `dev-predictions.jsonl` and `test-predictions.jsonl` retain individual failures, predictions, distributions, logits, and latency. `test-summary.json`, `test-calibrated-summary.json`, `calibration.json`, and `baselines.json` provide the aggregate results. The archived original harness accompanies the initial development attempt and the unchanged Emojify measurements.",
        "",
        "Validation covers headerless CSV parsing, duplicate/conflict exclusion, stable sampling, failure-inclusive metrics, model/configuration drift, distribution/logit consistency, gold-label exclusion from requests, and completing a run despite individual protocol failures while the server is healthy.",
        "",
        "The full Python suite passed **52 tests**, with two tensor-dependent skips in the API environment; seven tests specifically cover this benchmark. Lint passed. Macro F1 was independently checked with scikit-learn, and published gold/prediction alignment was verified against all 1,000 selected source indices. [validation.json](validation.json) records metric agreement, package version, artifact hashes, and final service health.",
    ]
    (REPORT / "README.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
