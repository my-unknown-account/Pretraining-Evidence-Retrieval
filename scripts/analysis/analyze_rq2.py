#!/usr/bin/env python3
"""Analyze RQ2: Does pretraining evidence protect against contradictory retrieval?

The analysis uses existing generations:
  - scripts/inference/closed_book/results/run_*_{model}_simple.json
  - scripts/inference/contradictory_context/results/run_*_{model}_simple.json

It filters to closed-book successes and measures whether contradictory evidence
overrides the answer.
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
DEFAULT_CONTRADICTORY_DIR = REPO_ROOT / "scripts" / "inference" / "contradictory_context" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq2"
DEFAULT_MODELS = ("amber", "redpajama", "olmo")
QUESTION_TYPE = "simple"
RUN_RE = re.compile(r"run_(\d+)_(.+)_(.+)\.json$")

TRUE_OBJECT = "TRUE_OBJECT"
FALSE_CONTEXT_OBJECT = "FALSE_CONTEXT_OBJECT"
OTHER = "OTHER"
VALID_LABELS = {TRUE_OBJECT, FALSE_CONTEXT_OBJECT, OTHER}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RQ2 contradictory-retrieval override rates against RAS."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--closed-book-dir", type=Path, default=DEFAULT_CLOSED_BOOK_DIR)
    parser.add_argument(
        "--contradictory-dir", type=Path, default=DEFAULT_CONTRADICTORY_DIR
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
        help="Number of within-model RAS quantile bins for Figure 3.",
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


def normalize_label(value) -> str:
    text = str(value).strip().upper()
    if text in VALID_LABELS:
        return text
    for label in VALID_LABELS:
        if label in text:
            return label
    return OTHER


def collect_examples(args: argparse.Namespace) -> list[dict]:
    dataset = load_json(args.dataset)
    rows = []

    for model in args.models:
        ds_model = dataset_key_for_model(model)
        if ds_model not in dataset:
            raise KeyError(f"Missing dataset section for model {model!r}: {ds_model!r}")

        cb_files = result_files(args.closed_book_dir, model, args.q_type)
        neg_files = result_files(args.contradictory_dir, model, args.q_type)
        shared_runs = sorted(set(cb_files) & set(neg_files))

        if not shared_runs:
            print(f"[warn] no paired runs for {model}")
            continue

        for run_id in shared_runs:
            closed_book = load_json(cb_files[run_id])
            contradictory = load_json(neg_files[run_id])
            shared_qids = sorted(set(closed_book) & set(contradictory))

            for qid in shared_qids:
                cb_item = closed_book[qid]
                if not is_true(cb_item.get("is_correct")):
                    continue

                neg_wrapper = contradictory[qid]
                if model not in neg_wrapper:
                    raise KeyError(f"Missing {model!r} in {neg_files[run_id]}:{qid}")
                neg_item = neg_wrapper[model]

                label = normalize_label(neg_item.get("is_correct"))
                fact = dataset[ds_model][qid]
                confidence = cb_item.get("confidence", {})

                rows.append(
                    {
                        "model": model,
                        "run_id": run_id,
                        "qid": qid,
                        "subject": fact.get("sub", ""),
                        "relation": fact.get("rel", ""),
                        "object": fact.get("obj", ""),
                        "false_object": fact.get("false_obj", ""),
                        "ras": int(fact.get("relation_support", 0)),
                        "lexical_so": int(fact.get("lexical_so", 0)),
                        "lexical_sro": int(fact.get("lexical_sro", 0)),
                        "negative_label": label,
                        "resistance": int(label == TRUE_OBJECT),
                        "override": int(label == FALSE_CONTEXT_OBJECT),
                        "other_degradation": int(label == OTHER),
                        "ror": int(label == FALSE_CONTEXT_OBJECT),
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

            total = len(group)
            overrides = sum(r["override"] for r in group)
            resistance = sum(r["resistance"] for r in group)
            other = sum(r["other_degradation"] for r in group)
            lo, hi = wilson_interval(overrides, total)
            ras_values = [r["ras"] for r in group]

            out.append(
                {
                    "model": model,
                    "bin": bin_id,
                    "ras_min": min(ras_values),
                    "ras_max": max(ras_values),
                    "ras_mean": round(mean(ras_values), 4),
                    "n": total,
                    "resistance_count": resistance,
                    "override_count": overrides,
                    "other_degradation_count": other,
                    "resistance_rate": round(resistance / total, 6),
                    "override_rate": round(overrides / total, 6),
                    "other_degradation_rate": round(other / total, 6),
                    "override_ci95_low": round(lo, 6),
                    "override_ci95_high": round(hi, 6),
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
        total = len(group)
        override = sum(r["override"] for r in group)
        resistance = sum(r["resistance"] for r in group)
        other = sum(r["other_degradation"] for r in group)
        support = [math.log10(r["ras"] + 1) for r in group]
        outcome = [r["override"] for r in group]
        summaries.append(
            {
                "model": model,
                "n_closed_book_correct": total,
                "resistance": resistance,
                "override": override,
                "other_degradation": other,
                "resistance_rate": resistance / total if total else float("nan"),
                "ror": override / total if total else float("nan"),
                "other_degradation_rate": other / total if total else float("nan"),
                "mean_ras": mean(r["ras"] for r in group) if group else float("nan"),
                "pearson_log_ras_ror": pearson(support, outcome),
            }
        )
    return summaries


def write_summary(path: Path, rows: list[dict], binned_rows: list[dict]) -> None:
    lines = [
        "# RQ2 Summary",
        "",
        "Filter: closed-book answer is correct (`y_CB = o`).",
        "Outcome: contradictory-context answer adopts the false object (`ROR`).",
        "",
        "## Model-Level Contradictory Retrieval",
        "",
        "| Model | n | Resistance o->o | Override o->o' | Other o->z | ROR | Pearson(log10(RAS+1), ROR) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in model_summaries(rows):
        lines.append(
            "| {model} | {n_closed_book_correct} | {resistance_rate:.3f} | "
            "{ror:.3f} | {other_degradation_rate:.3f} | {ror:.3f} | "
            "{pearson_log_ras_ror:.3f} |".format(**summary)
        )

    lines.extend(
        [
            "",
            "## Figure 3 Bins",
            "",
            "| Model | Bin | RAS range | n | Resistance | Override/ROR | Other | Override 95% CI |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in binned_rows:
        lines.append(
            "| {model} | {bin} | {ras_min}-{ras_max} | {n} | "
            "{resistance_rate:.3f} | {override_rate:.3f} | "
            "{other_degradation_rate:.3f} | [{override_ci95_low:.3f}, "
            "{override_ci95_high:.3f}] |".format(**row)
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

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.01), sharex=False)
    ax_override, ax_split = axes

    for model in sorted({row["model"] for row in binned_rows}):
        model_rows = sorted(
            [row for row in binned_rows if row["model"] == model],
            key=lambda row: float(row["ras_mean"]),
        )
        x = [math.log10(float(row["ras_mean"]) + 1) for row in model_rows]
        y = [float(row["override_rate"]) for row in model_rows]
        ax_override.plot(x, y, marker="o", label=labels.get(model, model), color=colors.get(model))

    by_bin: dict[int, list[dict]] = {}
    for row in binned_rows:
        by_bin.setdefault(int(row["bin"]), []).append(row)
    bin_ids = sorted(by_bin)
    resistance = [
        mean(float(row["resistance_rate"]) for row in by_bin[bin_id])
        for bin_id in bin_ids
    ]
    override = [
        mean(float(row["override_rate"]) for row in by_bin[bin_id])
        for bin_id in bin_ids
    ]
    other = [
        mean(float(row["other_degradation_rate"]) for row in by_bin[bin_id])
        for bin_id in bin_ids
    ]
    ax_split.stackplot(
        bin_ids,
        resistance,
        override,
        other,
        labels=["Resistance", "Corruption", "Other"],
        colors=["#009E73", "#D55E00", "#C9C9C9"],
        alpha=0.85,
    )
    ax_split.plot(bin_ids, override, marker="o", color="black", linewidth=1.8)

    ax_override.set_title("(a) Corruption across RAS quantiles")
    ax_override.set_xlabel("log10(RAS + 1)")
    ax_override.set_ylabel("Rate of Corruption (RCP)")
    ax_override.set_ylim(0.4, 1.0)
    ax_override.set_xlim(0, max_log_ras * 1.03)
    ax_override.grid(True)
    ax_override.legend()

    ax_split.set_title("(b) Outcome split, macro-average")
    ax_split.set_xlabel("RAS quantile bin (low to high)")
    ax_split.set_ylabel("Rate")
    ax_split.set_ylim(0.0, 1.0)
    ax_split.set_xlim(min(bin_ids), max(bin_ids))
    ax_split.set_xticks(bin_ids)
    ax_split.grid(True)
    ax_split.legend(loc="upper right")

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
        raise RuntimeError("No RQ2 examples found after closed-book-correct filtering.")

    binned_rows = build_binned_rows(rows, args.bins)
    write_csv(args.output_dir / "rq2_examples.csv", rows)
    write_csv(args.output_dir / "rq2_bins.csv", binned_rows)
    write_summary(args.output_dir / "rq2_summary.md", rows, binned_rows)
    figure_path = args.output_dir / "figure3_override_rate_vs_ras.pdf"
    write_figure(figure_path, binned_rows)
    for stale_figure in (
        args.output_dir / "figure2_override_rate_vs_ras.png",
        args.output_dir / "figure3_override_rate_vs_ras.png",
        args.output_dir / "figure3_ras_vs_override.svg",
    ):
        if stale_figure.exists():
            stale_figure.unlink()
    (args.output_dir / "rq2_summary.json").write_text(
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
    print(f"[ok] wrote Figure 3 to {figure_path}")


if __name__ == "__main__":
    main()
