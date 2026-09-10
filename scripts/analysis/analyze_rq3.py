#!/usr/bin/env python3
"""Analyze RQ3: semantic support vs. lexical evidence and confidence.

This script compares existing signals on the existing RQ1/RQ2 outcomes. It does
not run any additional model generations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RQ1_EXAMPLES = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq1" / "rq1_examples.csv"
DEFAULT_RQ2_EXAMPLES = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq2" / "rq2_examples.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq3"
SIGNALS = {
    "Lex-SO": ("lexical_so",),
    "Lex-SRO": ("lexical_sro",),
    "Self-consistency": ("self_consistency",),
    "Relation-Aware Support": ("ras",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Table 3 for RQ3 using self-consistency as confidence."
    )
    parser.add_argument("--rq1-examples", type=Path, default=DEFAULT_RQ1_EXAMPLES)
    parser.add_argument("--rq2-examples", type=Path, default=DEFAULT_RQ2_EXAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rows(path: Path, outcome_col: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding RQ analysis first."
        )

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
            for col in ("lexical_so", "lexical_sro", "ras", "self_consistency"):
                item[col] = parse_float(row.get(col, ""))
            rows.append(item)
    return rows


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def percentile_ranks(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [0.5]

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2
        percentile = avg_rank / (len(values) - 1)
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = percentile
        i = j + 1
    return ranks


def signal_auc(rows: list[dict], cols: tuple[str, ...]) -> tuple[float, str]:
    labels = [row["outcome"] for row in rows]
    if sum(labels) == 0 or sum(labels) == len(labels):
        return float("nan"), "undefined"

    if len(cols) == 1:
        scores = [transformed_feature(row, cols[0]) for row in rows]
        raw_auc = auc(scores, labels)
        direction = "positive" if raw_auc >= 0.5 else "negative"
        return max(raw_auc, 1.0 - raw_auc), direction

    aligned_rank_sets = []
    directions = []
    for col in cols:
        scores = [transformed_feature(row, col) for row in rows]
        raw_auc = auc(scores, labels)
        ranks = percentile_ranks(scores)
        if raw_auc >= 0.5:
            aligned_rank_sets.append(ranks)
            directions.append(f"{col}:positive")
        else:
            aligned_rank_sets.append([1.0 - rank for rank in ranks])
            directions.append(f"{col}:negative")

    combined = [
        sum(rank_set[i] for rank_set in aligned_rank_sets) / len(aligned_rank_sets)
        for i in range(len(rows))
    ]
    raw_auc = auc(combined, labels)
    direction = "positive" if raw_auc >= 0.5 else "negative"
    return max(raw_auc, 1.0 - raw_auc), "; ".join(directions) or direction


def evaluate(
    rows: list[dict],
    outcome_name: str,
    scope: str,
) -> list[dict]:
    output = []
    for signal_name, cols in SIGNALS.items():
        score, direction = signal_auc(rows, cols)
        output.append(
            {
                "signal": signal_name,
                "outcome": outcome_name,
                "scope": scope,
                "n": len(rows),
                "positive_rate": round(sum(r["outcome"] for r in rows) / len(rows), 6),
                "auc": round(score, 6),
                "direction": direction,
            }
        )
    return output


def evaluate_all_scopes(rows: list[dict], outcome_name: str) -> list[dict]:
    output = evaluate(rows, outcome_name, "pooled")
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        output.extend(evaluate(model_rows, outcome_name, model))
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
                and row["scope"] not in {"pooled", "macro"}
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


def write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# RQ3 Summary",
        "",
        "Confidence signal: closed-book `self_consistency`.",
        "Main metric: direction-adjusted rank AUROC; values above 0.5 mean the signal separates the outcome classes.",
        "Main table reports macro-average AUROC across models.",
        "",
        "## Table 3",
        "",
        Path(path.parent / "table3_signal_auc.md").read_text(encoding="utf-8").strip(),
        "",
        "## Pooled Diagnostic",
        "",
        Path(path.parent / "table3_signal_auc_pooled.md").read_text(encoding="utf-8").strip(),
        "",
        "## Direction Notes",
        "",
        "| Scope | Signal | Outcome | Direction |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['signal']} | {row['outcome']} | {row['direction']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correction_rows = read_rows(args.rq1_examples, "rcr")
    override_rows = read_rows(args.rq2_examples, "ror")

    results = []
    results.extend(evaluate_all_scopes(correction_rows, "Correction"))
    results.extend(evaluate_all_scopes(override_rows, "Override"))
    results.extend(macro_rows(results))

    write_csv(args.output_dir / "table3_signal_auc.csv", results)
    write_table_md(args.output_dir / "table3_signal_auc.md", results, "macro")
    write_table_md(args.output_dir / "table3_signal_auc_pooled.md", results, "pooled")
    write_summary(args.output_dir / "rq3_summary.md", results)
    (args.output_dir / "rq3_summary.json").write_text(
        json.dumps(
            {
                "confidence_signal": "self_consistency",
                "metric": "direction-adjusted rank AUROC",
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote RQ3 table to {args.output_dir / 'table3_signal_auc.md'}")


if __name__ == "__main__":
    main()
