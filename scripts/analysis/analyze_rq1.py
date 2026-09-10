#!/usr/bin/env python3
"""Analyze RQ1: Does pretraining evidence determine corrective retrieval?

The analysis uses existing generations:
  - scripts/inference/closed_book/results/run_*_{model}_simple.json
  - scripts/inference/correct_context/results/run_*_{model}_simple.json

It filters to closed-book failures and measures whether adding the correct
passage corrected the answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "dataset.json"
DEFAULT_CLOSED_BOOK_DIR = REPO_ROOT / "scripts" / "inference" / "closed_book" / "results"
DEFAULT_CORRECT_CONTEXT_DIR = REPO_ROOT / "scripts" / "inference" / "correct_context" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq1"
DEFAULT_MODELS = ("amber", "redpajama", "olmo")
QUESTION_TYPE = "simple"
RUN_RE = re.compile(r"run_(\d+)_(.+)_(.+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RQ1 corrective retrieval rates against RAS."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--closed-book-dir", type=Path, default=DEFAULT_CLOSED_BOOK_DIR)
    parser.add_argument(
        "--correct-context-dir", type=Path, default=DEFAULT_CORRECT_CONTEXT_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to include. Default: amber redpajama olmo",
    )
    parser.add_argument("--q-type", default=QUESTION_TYPE)
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Number of within-model RAS quantile bins for Figure 2.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_key_for_model(model: str) -> str:
    return "olmo" if model == "olmo32" else model


def result_files(results_dir: Path, model: str, q_type: str) -> dict[int, Path]:
    files = {}
    for path in sorted(results_dir.glob(f"run_*_{model}_{q_type}.json")):
        match = RUN_RE.fullmatch(path.name)
        if not match:
            continue
        run_id, parsed_model, parsed_q_type = match.groups()
        if parsed_model == model and parsed_q_type == q_type:
            files[int(run_id)] = path
    return files


def is_true(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def collect_examples(args: argparse.Namespace) -> list[dict]:
    dataset = load_json(args.dataset)
    rows = []

    for model in args.models:
        ds_model = dataset_key_for_model(model)
        if ds_model not in dataset:
            raise KeyError(f"Missing dataset section for model {model!r}: {ds_model!r}")

        cb_files = result_files(args.closed_book_dir, model, args.q_type)
        cc_files = result_files(args.correct_context_dir, model, args.q_type)
        shared_runs = sorted(set(cb_files) & set(cc_files))

        if not shared_runs:
            print(f"[warn] no paired runs for {model}")
            continue

        for run_id in shared_runs:
            closed_book = load_json(cb_files[run_id])
            correct_context = load_json(cc_files[run_id])
            shared_qids = sorted(set(closed_book) & set(correct_context))

            for qid in shared_qids:
                cb_item = closed_book[qid]
                cc_wrapper = correct_context[qid]
                if model not in cc_wrapper:
                    raise KeyError(f"Missing {model!r} in {cc_files[run_id]}:{qid}")
                cc_item = cc_wrapper[model]

                cb_correct = is_true(cb_item.get("is_correct"))
                if cb_correct:
                    continue

                fact = dataset[ds_model][qid]
                cc_correct = is_true(cc_item.get("is_correct"))
                confidence = cb_item.get("confidence", {})

                rows.append(
                    {
                        "model": model,
                        "run_id": run_id,
                        "qid": qid,
                        "subject": fact.get("sub", ""),
                        "relation": fact.get("rel", ""),
                        "object": fact.get("obj", ""),
                        "ras": int(fact.get("relation_support", 0)),
                        "lexical_so": int(fact.get("lexical_so", 0)),
                        "lexical_sro": int(fact.get("lexical_sro", 0)),
                        "closed_book_correct": int(cb_correct),
                        "correct_context_correct": int(cc_correct),
                        "rcr": int(cc_correct),
                        "token_logprob": confidence.get("token_logprob", ""),
                        "verbalized_confidence": confidence.get(
                            "verbalized_confidence", ""
                        ),
                        "p_true": confidence.get("p_true", ""),
                        "self_consistency": confidence.get("self_consistency", ""),
                    }
                )

    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x_bar = mean(xs)
    y_bar = mean(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_bar) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_bar) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return float("nan")
    return num / (x_den * y_den)


def build_binned_rows(rows: list[dict], n_bins: int) -> list[dict]:
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out = []
    for model, model_rows in sorted(by_model.items()):
        sorted_rows = sorted(model_rows, key=lambda r: (r["ras"], r["qid"], r["run_id"]))
        total_rows = len(sorted_rows)
        for bin_id in range(1, min(n_bins, total_rows) + 1):
            start = math.floor((bin_id - 1) * total_rows / n_bins)
            end = math.floor(bin_id * total_rows / n_bins)
            group = sorted_rows[start:end]
            if not group:
                continue
            successes = sum(r["rcr"] for r in group)
            total = len(group)
            lo, hi = wilson_interval(successes, total)
            ras_values = [r["ras"] for r in group]
            out.append(
                {
                    "model": model,
                    "bin": bin_id,
                    "ras_min": min(ras_values),
                    "ras_max": max(ras_values),
                    "ras_mean": round(mean(ras_values), 4),
                    "n": total,
                    "corrected": successes,
                    "correction_rate": round(successes / total, 6),
                    "ci95_low": round(lo, 6),
                    "ci95_high": round(hi, 6),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        successes = sum(r["rcr"] for r in group)
        support = [math.log10(r["ras"] + 1) for r in group]
        outcome = [r["rcr"] for r in group]
        summaries.append(
            {
                "model": model,
                "n_closed_book_wrong": len(group),
                "corrected": successes,
                "rcr": successes / len(group) if group else float("nan"),
                "mean_ras": mean(r["ras"] for r in group) if group else float("nan"),
                "pearson_log_ras_rcr": pearson(support, outcome),
            }
        )
    return summaries


def write_summary(path: Path, rows: list[dict], binned_rows: list[dict]) -> None:
    lines = [
        "# RQ1 Summary",
        "",
        "Filter: closed-book answer is incorrect (`y_CB != o`).",
        "Outcome: correct-context answer is correct (`RCR`).",
        "",
        "## Model-Level Corrective Retrieval",
        "",
        "| Model | n | Corrected | RCR | Pearson(log10(RAS+1), RCR) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for summary in model_summaries(rows):
        lines.append(
            "| {model} | {n_closed_book_wrong} | {corrected} | {rcr:.3f} | "
            "{pearson_log_ras_rcr:.3f} |".format(**summary)
        )

    lines.extend(
        [
            "",
            "## Figure 2 Bins",
            "",
            "| Model | Bin | RAS range | n | CorrectionRate | 95% CI |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in binned_rows:
        lines.append(
            "| {model} | {bin} | {ras_min}-{ras_max} | {n} | "
            "{correction_rate:.3f} | [{ci95_low:.3f}, {ci95_high:.3f}] |".format(
                **row
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to generate PDF figures. Install it with "
            "`python -m pip install matplotlib` and rerun this script."
        ) from exc
    return plt


def write_figure(path: Path, binned_rows: list[dict]) -> None:
    plt = import_matplotlib()
    max_log_ras = max(math.log10(float(row["ras_mean"]) + 1) for row in binned_rows)
    colors = {
        "amber": "#0072B2",
        "redpajama": "#D55E00",
        "olmo": "#009E73",
        "olmo32": "#CC79A7",
    }
    labels = {
        "amber": "Amber",
        "redpajama": "RedPajama",
        "olmo": "OLMo",
        "olmo32": "OLMo-32B",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 3.01),
        gridspec_kw={"width_ratios": [2.1, 1.0]},
    )
    ax, ax_gap = axes
    gap_rows = []

    for model in sorted({row["model"] for row in binned_rows}):
        model_rows = sorted(
            [row for row in binned_rows if row["model"] == model],
            key=lambda row: float(row["ras_mean"]),
        )
        x = [math.log10(float(row["ras_mean"]) + 1) for row in model_rows]
        y = [float(row["correction_rate"]) for row in model_rows]
        low = float(model_rows[0]["correction_rate"])
        high = float(model_rows[-1]["correction_rate"])
        gap_rows.append((model, high - low))
        ax.plot(x, y, marker="o", label=labels.get(model, model), color=colors.get(model))

    ax.set_title("(a) Correction across RAS quantiles")
    ax.set_xlabel("log10(RAS + 1)")
    ax.set_ylabel("Retrieval Correction Rate")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(0, max_log_ras * 1.03)
    ax.grid(True)
    ax.legend()

    order = ["amber", "olmo", "redpajama", "olmo32"]
    gap_rows = sorted(
        gap_rows,
        key=lambda row: order.index(row[0]) if row[0] in order else 99,
    )
    x_labels = [labels.get(model, model) for model, _ in gap_rows]
    gaps = [gap for _, gap in gap_rows]
    bar_colors = [colors.get(model) for model, _ in gap_rows]
    ax_gap.bar(x_labels, gaps, color=bar_colors)
    ax_gap.axhline(0, color="black", linewidth=1)
    for idx, gap in enumerate(gaps):
        ax_gap.text(idx, gap - 0.004, f"{gap * 100:.1f} pp", ha="center", va="top", fontsize=9)
    ax_gap.set_title("(b) High-low RAS gap")
    ax_gap.set_ylabel(r"$\Delta$ RCR")
    ax_gap.set_ylim(min(gaps) - 0.025, 0.01)
    ax_gap.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bins < 1:
        raise ValueError("--bins must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_examples(args)
    if not rows:
        raise RuntimeError("No RQ1 examples found after closed-book-wrong filtering.")

    binned_rows = build_binned_rows(rows, args.bins)
    write_csv(args.output_dir / "rq1_examples.csv", rows)
    write_csv(args.output_dir / "rq1_bins.csv", binned_rows)
    write_summary(args.output_dir / "rq1_summary.md", rows, binned_rows)
    figure_path = args.output_dir / "figure2_correction_rate_vs_ras.pdf"
    write_figure(figure_path, binned_rows)
    for stale_figure in (
        args.output_dir / "figure1_correction_rate_vs_ras.png",
        args.output_dir / "figure2_correction_rate_vs_ras.png",
        args.output_dir / "figure2_ras_vs_correction.svg",
    ):
        if stale_figure.exists():
            stale_figure.unlink()
    (args.output_dir / "rq1_summary.json").write_text(
        json.dumps(
            {
                "models": list(args.models),
                "q_type": args.q_type,
                "n_examples": len(rows),
                "summaries": model_summaries(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {len(rows)} filtered examples to {args.output_dir}")
    print(f"[ok] wrote Figure 2 to {figure_path}")


if __name__ == "__main__":
    main()
