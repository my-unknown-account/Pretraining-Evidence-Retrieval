#!/usr/bin/env python3
"""Run RQ1/RQ2/RQ3 analyses for OLMo vs. OLMo32.

This is an analysis-only experiment. It reuses existing closed-book,
correct-context, and contradictory-context generations. Section 5.4 is
intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq4"
RQ1_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "analyze_rq1.py"
RQ2_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "analyze_rq2.py"
MODELS = ("olmo", "olmo32")
SIGNALS = {
    "Lex-SO": ("lexical_so",),
    "Lex-SRO": ("lexical_sro",),
    "Token logprob": ("token_logprob",),
    "Verbalized confidence": ("verbalized_confidence",),
    "P(true)": ("p_true",),
    "Self-consistency": ("self_consistency",),
    "Relation-Aware Support": ("ras",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OLMo-vs-OLMo32 analyses for RQ1, RQ2, and RQ3."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=5)
    return parser.parse_args()


def run_analysis(script: Path, output_dir: Path, bins: int) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--models",
        *MODELS,
        "--output-dir",
        str(output_dir),
        "--bins",
        str(bins),
    ]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_rows(path: Path, outcome_col: str) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "model": row["model"],
                "run_id": row["run_id"],
                "qid": row["qid"],
                "outcome": int(row[outcome_col]),
            }
            for col in (
                "lexical_so",
                "lexical_sro",
                "ras",
                "token_logprob",
                "verbalized_confidence",
                "p_true",
                "self_consistency",
            ):
                item[col] = parse_float(row.get(col, ""))
            rows.append(item)
    return rows


def transformed_feature(row: dict, col: str) -> float:
    value = row[col]
    if col in {"lexical_so", "lexical_sro", "ras"}:
        return math.log1p(max(0.0, value))
    return value


def auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2
        positives = sum(label for _, label in pairs[i : j + 1])
        rank_sum += positives * avg_rank
        rank += j - i + 1
        i = j + 1

    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def signal_auc(rows: list[dict], cols: tuple[str, ...]) -> tuple[float, str]:
    labels = [row["outcome"] for row in rows]
    if sum(labels) == 0 or sum(labels) == len(labels):
        return float("nan"), "undefined"

    scores = [transformed_feature(row, cols[0]) for row in rows]
    raw_auc = auc(scores, labels)
    direction = "positive" if raw_auc >= 0.5 else "negative"
    return max(raw_auc, 1.0 - raw_auc), direction


def evaluate_scope(rows: list[dict], outcome: str, scope: str) -> list[dict]:
    output = []
    for signal, cols in SIGNALS.items():
        score, direction = signal_auc(rows, cols)
        output.append(
            {
                "signal": signal,
                "outcome": outcome,
                "scope": scope,
                "n": len(rows),
                "positive_rate": round(sum(r["outcome"] for r in rows) / len(rows), 6),
                "auc": round(score, 6),
                "direction": direction,
            }
        )
    return output


def evaluate_all(rows: list[dict], outcome: str) -> list[dict]:
    output = evaluate_scope(rows, outcome, "pooled")
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        output.extend(evaluate_scope(model_rows, outcome, model))
    return output


def macro_rows(rows: list[dict]) -> list[dict]:
    output = []
    for outcome in sorted({row["outcome"] for row in rows}):
        for signal in SIGNALS:
            model_rows = [
                row
                for row in rows
                if row["outcome"] == outcome
                and row["signal"] == signal
                and row["scope"] in MODELS
            ]
            output.append(
                {
                    "signal": signal,
                    "outcome": outcome,
                    "scope": "macro",
                    "n": sum(row["n"] for row in model_rows),
                    "positive_rate": round(
                        mean(row["positive_rate"] for row in model_rows), 6
                    ),
                    "auc": round(mean(row["auc"] for row in model_rows), 6),
                    "direction": "macro-average",
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table_md(path: Path, rows: list[dict], scope: str) -> None:
    table_rows = [row for row in rows if row["scope"] == scope]
    by_signal = {row["signal"]: {} for row in table_rows}
    for row in table_rows:
        by_signal[row["signal"]][row["outcome"]] = row

    lines = [
        "| Signal | Correction AUROC | Override AUROC |",
        "| --- | ---: | ---: |",
    ]
    for signal in SIGNALS:
        correction = by_signal[signal]["Correction"]["auc"]
        override = by_signal[signal]["Override"]["auc"]
        name = f"**{signal}**" if signal == "Relation-Aware Support" else signal
        lines.append(f"| {name} | {correction:.3f} | {override:.3f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rq3_outputs(output_dir: Path, rq1_examples: Path, rq2_examples: Path) -> None:
    rq3_dir = output_dir / "rq3"
    rq3_dir.mkdir(parents=True, exist_ok=True)

    correction_rows = read_rows(rq1_examples, "rcr")
    override_rows = read_rows(rq2_examples, "ror")
    rows = []
    rows.extend(evaluate_all(correction_rows, "Correction"))
    rows.extend(evaluate_all(override_rows, "Override"))
    rows.extend(macro_rows(rows))

    write_csv(rq3_dir / "table3_olmo_vs_olmo32_signal_auc.csv", rows)
    write_table_md(rq3_dir / "table3_olmo_vs_olmo32_signal_auc.md", rows, "macro")
    write_table_md(
        rq3_dir / "table3_olmo_vs_olmo32_signal_auc_pooled.md", rows, "pooled"
    )
    (rq3_dir / "rq3_olmo_vs_olmo32_summary.json").write_text(
        json.dumps(
            {
                "models": list(MODELS),
                "confidence_signals": [
                    "token_logprob",
                    "verbalized_confidence",
                    "p_true",
                    "self_consistency",
                ],
                "metric": "direction-adjusted rank AUROC",
                "section_5_4": "excluded",
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_model_summary(path: Path, metric_name: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in data["summaries"]:
        item = {"model": row["model"]}
        if metric_name == "rcr":
            item["n"] = row["n_closed_book_wrong"]
            item["rate"] = row["rcr"]
            item["trend"] = row["pearson_log_ras_rcr"]
        else:
            item["n"] = row["n_closed_book_correct"]
            item["rate"] = row["ror"]
            item["resistance_rate"] = row["resistance_rate"]
            item["other_degradation_rate"] = row["other_degradation_rate"]
            item["trend"] = row["pearson_log_ras_ror"]
        rows.append(item)
    return rows


def write_combined_summary(output_dir: Path) -> None:
    rq1_rows = read_model_summary(output_dir / "rq1" / "rq1_summary.json", "rcr")
    rq2_rows = read_model_summary(output_dir / "rq2" / "rq2_summary.json", "ror")
    rq3_table = (
        output_dir / "rq3" / "table3_olmo_vs_olmo32_signal_auc.md"
    ).read_text(encoding="utf-8").strip()

    lines = [
        "# RQ4: OLMo vs. OLMo32",
        "",
        "Models: `olmo`, `olmo32`.",
        "Analyses: RQ1, RQ2, and RQ3 only; Section 5.4 is excluded.",
        "",
        "## RQ1 Corrective Retrieval",
        "",
        "| Model | n closed-book wrong | RCR | Pearson(log10(RAS+1), RCR) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rq1_rows:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['rate']:.3f} | {row['trend']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## RQ2 Contradictory Retrieval",
            "",
            "| Model | n closed-book correct | Resistance | Override/ROR | Other | Pearson(log10(RAS+1), ROR) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rq2_rows:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['resistance_rate']:.3f} | "
            f"{row['rate']:.3f} | {row['other_degradation_rate']:.3f} | "
            f"{row['trend']:.3f} |"
        )

    lines.extend(["", "## RQ3 Signal Comparison", "", rq3_table, ""])
    (output_dir / "rq4_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rq1_dir = args.output_dir / "rq1"
    rq2_dir = args.output_dir / "rq2"
    run_analysis(RQ1_SCRIPT, rq1_dir, args.bins)
    run_analysis(RQ2_SCRIPT, rq2_dir, args.bins)
    write_rq3_outputs(
        args.output_dir,
        rq1_dir / "rq1_examples.csv",
        rq2_dir / "rq2_examples.csv",
    )
    write_combined_summary(args.output_dir)

    print(f"[ok] wrote RQ4 summary to {args.output_dir / 'rq4_summary.md'}")


if __name__ == "__main__":
    main()
