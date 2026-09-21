#!/usr/bin/env python3
"""Generate paper-ready result materials.

Outputs are written to scripts/analysis/materials:
  - Figure 2. Counterfactual-passage prompts.
  - Figure 3. Judging prompts.
  - Figure 4. Correction Rate vs RAS.
  - Figure 5. Corruption Rate vs RAS.
  - Table 1. Dataset/relation statistics.
  - Table 2. Models, corpora, evaluation sizes.
  - Table 3. RQ3 signal comparison.
  - Table 4. OLMo vs OLMo-32B.
  - Prompt figures as Markdown files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALS_DIR = REPO_ROOT / "scripts" / "analysis" / "materials"
DATASET_PATH = REPO_ROOT / "data" / "dataset.json"
RQ1_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq1"
RQ2_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq2"
RQ3_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq3"
RQ4_DIR = REPO_ROOT / "scripts" / "analysis" / "outputs" / "rq4"
FIGURE1 = REPO_ROOT / "paper" / "figure1_conceptual_framework.pdf"
RQ1_FIGURE = RQ1_DIR / "figure2_correction_rate_vs_ras.pdf"
RQ2_FIGURE = RQ2_DIR / "figure3_override_rate_vs_ras.pdf"

MODEL_LABELS = {
    "amber": "Amber",
    "redpajama": "RedPajama",
    "olmo": "OLMo",
    "olmo32": "OLMo-32B",
}

CORPUS_LABELS = {
    "amber": "Amber corpus",
    "redpajama": "RedPajama corpus",
    "olmo": "OLMo corpus",
    "olmo32": "OLMo corpus",
}

MODEL_COLORS = {
    "amber": "#0072B2",
    "redpajama": "#D55E00",
    "olmo": "#009E73",
    "olmo32": "#CC79A7",
}

SIGNALS_FOR_TABLE3 = (
    "Lex-SO",
    "Lex-SRO",
    "Self-consistency",
    "Relation-Aware Support",
)

PROMPT_FIGURES = {
    "figure3_prompt_judging.md": """# Figure 3. Judging prompts

## Correctness judge

System message:

```text
You are a strict evaluator. Output only true or false.
```

User message:

```text
Question: {question}
Answer: {ground_truth}
Candidate: {candidate_answer}
Is candidate correct?
```

## Contradictory-context outcome judge

System message:

```text
You are a strict evaluator. Output only one label: TRUE_OBJECT, FALSE_CONTEXT_OBJECT, or OTHER.
```

User message:

```text
Question: {question}
True object: {true_object}
False context object: {false_context_object}
Candidate: {candidate_answer}
Classify the candidate answer.
Return only TRUE_OBJECT if the candidate answer is the true object.
Return only FALSE_CONTEXT_OBJECT if the candidate answer is the false context object.
Return only OTHER otherwise.
```
""",
    "figure2_prompt_counterfactual_passages.md": """# Figure 2. Counterfactual-passage prompts

## Candidate proposal

```text
Select plausible controlled counterfactual objects for a factual-retrieval experiment.

You are given a true subject-relation-object fact and a list of candidate objects.
The candidate list has ALREADY been filtered so that known Wikidata values for this
subject-relation pair are removed.

Select up to five candidates, ranked best first.

A good candidate:
- has the correct semantic type and granularity for the relation;
- is plausible enough that a reader would not reject it as absurd on sight;
- is not simply an alias/equivalent of the true object;
- is not obviously another correct, historical, or disputed value;
- is similar in specificity to the true object when possible.

IMPORTANT: Do NOT demand proof of falsity here. This is a candidate-proposal stage.
A separate verifier will search for positive evidence that a proposed value might
actually be true. Prefer useful candidates over returning UNSUITABLE too readily.

You MUST select only QIDs from the supplied candidate list.
```

Input fields:

```text
subject, subject_qid, subject_wikipedia, relation, property,
true_object, true_object_qid, warning, multi_valued_relation,
candidate_objects
```

## Adversarial verification

```text
Perform an adversarial POSITIVE-EVIDENCE check for a candidate contradiction.

The candidate has already passed these automatic checks:
1. it is a known object type used with the same Wikidata relation elsewhere;
2. it is NOT listed among the subject's current Wikidata values for this relation.

Your task is NOT to prove a universal negative. Instead, actively look for reasons
the candidate should be REJECTED.

Reject if you find credible evidence that the candidate:
- is actually a true alternative value for this subject/relation;
- was historically true;
- is disputed/uncertain in a way that makes calling it false unsafe;
- has the wrong semantic type or granularity;
- is effectively an alias/equivalent/broader/narrower form of a correct value.

If, after checking, you find NO positive evidence that it is true, historical,
disputed, or type-incompatible, return PASS_NO_POSITIVE_EVIDENCE.

For multi-valued relations, search especially carefully for alternative true values.
Be concise in the reason.
```

Input fields:

```text
subject, subject_qid, subject_wikipedia, relation, property,
true_object, true_object_qid, candidate_object, candidate_qid, warning
```
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper/materials artifacts.")
    parser.add_argument("--output-dir", type=Path, default=MATERIALS_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rerun RQ1, RQ2, RQ3, and RQ4 analyses before collecting materials",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    require(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(cmd: list[str]) -> None:
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def refresh_experiments() -> None:
    run([sys.executable, "scripts/analysis/analyze_rq1.py"])
    run([sys.executable, "scripts/analysis/analyze_rq2.py"])
    run([sys.executable, "scripts/analysis/analyze_rq3.py"])
    run([sys.executable, "scripts/analysis/analyze_rq4.py"])


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def clean_stale_materials(output_dir: Path) -> None:
    for path in output_dir.glob("*.csv"):
        path.unlink()
    for path in output_dir.glob("*.svg"):
        path.unlink()
    for path in output_dir.glob("*.pdf"):
        path.unlink()
    for path in output_dir.glob("figure*.png"):
        path.unlink()
    for path in output_dir.glob("figure*_prompt*.md"):
        path.unlink()


def copy_figures(output_dir: Path) -> None:
    figures = [
        (FIGURE1, output_dir / "figure1_conceptual_framework.pdf"),
        (RQ1_FIGURE, output_dir / "figure4_correction_rate_vs_ras.pdf"),
        (RQ2_FIGURE, output_dir / "figure5_override_rate_vs_ras.pdf"),
    ]
    for src, dst in figures:
        if src.exists():
            shutil.copyfile(src, dst)


def markdown_table(rows: list[dict], headers: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in headers) + " |",
        "| "
        + " | ".join(
            "---" if not is_numeric_column(rows, key) else "---:"
            for key, _ in headers
        )
        + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[key]) for key, _ in headers) + " |")
    return "\n".join(lines)


def is_numeric_column(rows: list[dict], key: str) -> bool:
    return all(isinstance(row.get(key), int | float) for row in rows)


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def write_md_table(
    path: Path, title: str, rows: list[dict], headers: list[tuple[str, str]]
) -> None:
    text = f"# {title}\n\n{markdown_table(rows, headers)}\n"
    path.write_text(text, encoding="utf-8")


def table1_dataset_relation_stats(output_dir: Path) -> list[dict]:
    dataset = load_json(DATASET_PATH)
    models = ["amber", "redpajama", "olmo"]
    relation_counts = {
        model: Counter(item["rel"] for item in dataset[model].values())
        for model in models
    }
    relations = sorted(set().union(*(counts.keys() for counts in relation_counts.values())))

    rows = []
    for relation in relations:
        counts = [relation_counts[model][relation] for model in models]
        rows.append(
            {
                "relation": relation,
                "amber": counts[0],
                "redpajama": counts[1],
                "olmo": counts[2],
            }
        )

    rows.append(
        {
            "relation": "Total",
            "amber": len(dataset["amber"]),
            "redpajama": len(dataset["redpajama"]),
            "olmo": len(dataset["olmo"]),
        }
    )

    write_md_table(
        output_dir / "table1_dataset_relation_statistics.md",
        "Table 1. Dataset/relation statistics",
        rows,
        [
            ("relation", "Relation"),
            ("amber", "Amber"),
            ("redpajama", "RedPajama"),
            ("olmo", "OLMo"),
        ],
    )
    return rows


def count_json_records(pattern: str) -> int:
    total = 0
    for path in sorted(REPO_ROOT.glob(pattern)):
        total += len(load_json(path))
    return total


def table2_models_evaluation_sizes(output_dir: Path) -> list[dict]:
    dataset = load_json(DATASET_PATH)
    rq1 = {row["model"]: row for row in load_json(RQ1_DIR / "rq1_summary.json")["summaries"]}
    rq2 = {row["model"]: row for row in load_json(RQ2_DIR / "rq2_summary.json")["summaries"]}
    rq4_rq1 = {
        row["model"]: row
        for row in load_json(RQ4_DIR / "rq1" / "rq1_summary.json")["summaries"]
    }
    rq4_rq2 = {
        row["model"]: row
        for row in load_json(RQ4_DIR / "rq2" / "rq2_summary.json")["summaries"]
    }

    rows = []
    for model in ["amber", "redpajama", "olmo", "olmo32"]:
        dataset_key = "olmo" if model == "olmo32" else model
        rq1_row = rq4_rq1[model] if model == "olmo32" else rq1[model]
        rq2_row = rq4_rq2[model] if model == "olmo32" else rq2[model]
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "corpus": CORPUS_LABELS[model],
                "facts": len(dataset[dataset_key]),
                "closed_book_generations": count_json_records(
                    f"scripts/inference/closed_book/results/run_*_{model}_simple.json"
                ),
                "rq1_n": rq1_row["n_closed_book_wrong"],
                "rq2_n": rq2_row["n_closed_book_correct"],
            }
        )

    write_md_table(
        output_dir / "table2_models_corpora_evaluation_sizes.md",
        "Table 2. Models, corpora, evaluation sizes",
        rows,
        [
            ("model", "Model"),
            ("corpus", "Corpus"),
            ("facts", "Facts"),
            ("closed_book_generations", "Closed-book generations"),
            ("rq1_n", "RQ1 n"),
            ("rq2_n", "RQ2 n"),
        ],
    )
    return rows


def table3_signal_comparison(output_dir: Path) -> str:
    src_csv = RQ3_DIR / "table3_signal_auc.csv"
    require(src_csv)
    by_signal = {signal: {} for signal in SIGNALS_FOR_TABLE3}
    for row in read_csv_rows(src_csv):
        if row["scope"] in {"macro", "pooled"}:
            by_signal[row["signal"]][(row["scope"], row["outcome"])] = float(row["auc"])

    compact_rows = [
        {
            "signal": signal,
            "macro_correction_auc": by_signal[signal][("macro", "Correction")],
            "macro_override_auc": by_signal[signal][("macro", "Override")],
            "pooled_correction_auc": by_signal[signal][("pooled", "Correction")],
            "pooled_override_auc": by_signal[signal][("pooled", "Override")],
        }
        for signal in SIGNALS_FOR_TABLE3
    ]

    dst = output_dir / "table3_rq3_signal_comparison.md"
    write_md_table(
        dst,
        "Table 3. Discriminative ability of pretraining-evidence and confidence signals for corrective and contradictory retrieval",
        compact_rows,
        [
            ("signal", "Signal"),
            ("macro_correction_auc", "Macro Correction"),
            ("macro_override_auc", "Macro Override"),
            ("pooled_correction_auc", "Pooled Correction"),
            ("pooled_override_auc", "Pooled Override"),
        ],
    )
    return dst.read_text(encoding="utf-8").strip()


def table4_signal_rows() -> list[dict]:
    src_csv = RQ4_DIR / "rq3" / "table3_olmo_vs_olmo32_signal_auc.csv"
    require(src_csv)
    by_signal = {signal: {} for signal in SIGNALS_FOR_TABLE3}
    for row in read_csv_rows(src_csv):
        if row["scope"] == "macro":
            by_signal[row["signal"]][row["outcome"]] = float(row["auc"])

    return [
        {
            "signal": signal,
            "correction_auc": by_signal[signal]["Correction"],
            "override_auc": by_signal[signal]["Override"],
        }
        for signal in SIGNALS_FOR_TABLE3
    ]


def table4_olmo_vs_olmo32(output_dir: Path) -> str:
    rq1 = {
        row["model"]: row
        for row in load_json(RQ4_DIR / "rq1" / "rq1_summary.json")["summaries"]
    }
    rq2 = {
        row["model"]: row
        for row in load_json(RQ4_DIR / "rq2" / "rq2_summary.json")["summaries"]
    }

    corrective_rows = []
    contradictory_rows = []
    for model in ["olmo", "olmo32"]:
        corrective_rows.append(
            {
                "model": MODEL_LABELS[model],
                "rq1_n": rq1[model]["n_closed_book_wrong"],
                "rcr": rq1[model]["rcr"],
                "rcr_ras_corr": rq1[model]["pearson_log_ras_rcr"],
            }
        )
        contradictory_rows.append(
            {
                "model": MODEL_LABELS[model],
                "rq2_n": rq2[model]["n_closed_book_correct"],
                "resistance": rq2[model]["resistance_rate"],
                "ror": rq2[model]["ror"],
                "other": rq2[model]["other_degradation_rate"],
                "ror_ras_corr": rq2[model]["pearson_log_ras_ror"],
            }
        )

    parts = (
        "# Table 4. OLMo vs. OLMo-32B",
        "",
        "## Table 4(a). Corrective retrieval",
        "",
        markdown_table(
            corrective_rows,
            [
                ("model", "Model"),
                ("rq1_n", r"Closed-book wrong \(n\)"),
                ("rcr", "RCR"),
                ("rcr_ras_corr", r"\(r_{\log(RAS+1),RCR}\)"),
            ],
        ),
        "",
        "## Table 4(b). Contradictory retrieval",
        "",
        markdown_table(
            contradictory_rows,
            [
                ("model", "Model"),
                ("rq2_n", r"Closed-book correct \(n\)"),
                ("resistance", "Resistance"),
                ("ror", "ROR"),
                ("other", "Other"),
                ("ror_ras_corr", r"\(r_{\log(RAS+1),ROR}\)"),
            ],
        ),
        "",
        "## Table 4(c). Signal comparison",
        "",
        markdown_table(
            table4_signal_rows(),
            [
                ("signal", "Signal"),
                ("correction_auc", "Correction AUROC"),
                ("override_auc", "Override AUROC"),
            ],
        ),
        "",
    )
    dst = output_dir / "table4_olmo_vs_olmo32.md"
    dst.write_text("\n".join(parts), encoding="utf-8")
    return dst.read_text(encoding="utf-8").strip()


def write_manifest(output_dir: Path) -> None:
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    (output_dir / "manifest.json").write_text(
        json.dumps({"materials": files}, indent=2) + "\n", encoding="utf-8"
    )


def write_prompt_figures(output_dir: Path) -> None:
    for filename, text in PROMPT_FIGURES.items():
        (output_dir / filename).write_text(text.strip() + "\n", encoding="utf-8")


def write_combined(output_dir: Path) -> None:
    parts = [
        "# Paper Materials",
        "",
        "## Figure 1. Conceptual framework",
        "",
        "File: `figure1_conceptual_framework.pdf`",
        "",
        (output_dir / "figure2_prompt_counterfactual_passages.md").read_text(
            encoding="utf-8"
        ).strip(),
        "",
        (output_dir / "figure3_prompt_judging.md").read_text(encoding="utf-8").strip(),
        "",
        "## Figure 4. Correction Rate vs RAS",
        "",
        "File: `figure4_correction_rate_vs_ras.pdf`",
        "",
        "## Figure 5. Corruption Rate vs RAS",
        "",
        "File: `figure5_override_rate_vs_ras.pdf`",
        "",
        (output_dir / "table1_dataset_relation_statistics.md").read_text(encoding="utf-8").strip(),
        "",
        (output_dir / "table2_models_corpora_evaluation_sizes.md").read_text(encoding="utf-8").strip(),
        "",
        (output_dir / "table3_rq3_signal_comparison.md").read_text(encoding="utf-8").strip(),
        "",
        (output_dir / "table4_olmo_vs_olmo32.md").read_text(encoding="utf-8").strip(),
        "",
    ]
    (output_dir / "all_materials.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.refresh:
        refresh_experiments()

    for path in [
        DATASET_PATH,
        RQ1_DIR / "rq1_summary.json",
        RQ2_DIR / "rq2_summary.json",
        RQ3_DIR / "table3_signal_auc.csv",
        RQ4_DIR / "rq4_summary.md",
        RQ4_DIR / "rq3" / "table3_olmo_vs_olmo32_signal_auc.csv",
    ]:
        require(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_stale_materials(args.output_dir)
    copy_figures(args.output_dir)
    table1_dataset_relation_stats(args.output_dir)
    table2_models_evaluation_sizes(args.output_dir)
    table3_signal_comparison(args.output_dir)
    table4_olmo_vs_olmo32(args.output_dir)
    write_prompt_figures(args.output_dir)
    write_combined(args.output_dir)
    write_manifest(args.output_dir)

    print(f"[ok] wrote paper materials to {args.output_dir}")


if __name__ == "__main__":
    main()
