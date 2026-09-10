#!/usr/bin/env python3
"""Build the cleaned contradiction dataset.

The script joins corpus-specific exposure counts from
data/exposure_data.json with accepted contradiction fields from
data/contradictions.json, then writes data/dataset.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_KEYS = [
    "sub",
    "rel",
    "obj",
    "false_obj",
    "sub_entity",
    "sub_wiki_url",
    "sub_aliases",
    "rel_entity",
    "rel_aliases",
    "obj_entity",
    "obj_wiki_url",
    "obj_aliases",
    "false_obj_entity",
    "question",
    "correct_context",
    "contradictory_context",
    "relation_support",
    "lexical_so",
    "lexical_sro",
]

EXPOSURE_KEYS = [
    "relation_support",
    "lexical_so",
    "lexical_sro",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def qid_sort_key(qid: str) -> tuple[int, int | str]:
    prefix = "qid_"
    if qid.startswith(prefix) and qid[len(prefix) :].isdigit():
        return (0, int(qid[len(prefix) :]))
    return (1, qid)


def get_question(record: dict[str, Any], question_type: str) -> str | None:
    questions = record.get("questions", {})
    if not isinstance(questions, dict):
        return None

    question = questions.get(question_type)
    if question is not None:
        return question

    for fallback in ("simple", "template_based", "complex"):
        question = questions.get(fallback)
        if question is not None:
            return question

    return None


def build_record(
    exposure_record: dict[str, Any],
    contradiction_record: dict[str, Any],
    question_type: str,
) -> dict[str, Any]:
    contradiction = contradiction_record["contradiction"]

    values = {
        **{key: exposure_record.get(key, contradiction_record.get(key)) for key in OUTPUT_KEYS},
        "false_obj": contradiction.get("false_obj"),
        "false_obj_entity": contradiction.get("false_obj_entity"),
        "question": get_question(exposure_record, question_type),
        "correct_context": contradiction.get("correct_context"),
        "contradictory_context": contradiction.get("contradictory_context"),
        **{key: exposure_record.get(key) for key in EXPOSURE_KEYS},
    }
    return {key: values[key] for key in OUTPUT_KEYS}


def build_dataset(
    exposure_data: dict[str, dict[str, dict[str, Any]]],
    contradictions: dict[str, dict[str, Any]],
    question_type: str,
    accepted_only: bool,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    dataset: dict[str, dict[str, dict[str, Any]]] = {}
    stats = {
        "written": 0,
        "missing_contradiction": 0,
        "missing_contradiction_payload": 0,
        "not_accepted": 0,
    }

    for corpus, qid_records in exposure_data.items():
        if not isinstance(qid_records, dict):
            raise TypeError(f"Expected corpus {corpus!r} to contain a qid mapping")

        corpus_output: dict[str, dict[str, Any]] = {}
        for qid in sorted(qid_records, key=qid_sort_key):
            exposure_record = qid_records[qid]
            contradiction_record = contradictions.get(qid)

            if contradiction_record is None:
                stats["missing_contradiction"] += 1
                continue

            contradiction = contradiction_record.get("contradiction")
            if not isinstance(contradiction, dict):
                stats["missing_contradiction_payload"] += 1
                continue

            if accepted_only and contradiction.get("status") != "accepted":
                stats["not_accepted"] += 1
                continue

            corpus_output[qid] = build_record(
                exposure_record,
                contradiction_record,
                question_type,
            )
            stats["written"] += 1

        dataset[corpus] = corpus_output

    return dataset, stats


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description="Extract selected qid fields into data/dataset.json."
    )
    parser.add_argument(
        "--exposure",
        type=Path,
        default=repo_dir / "data" / "exposure_data.json",
        help="Path to exposure_data.json.",
    )
    parser.add_argument(
        "--contradictions",
        type=Path,
        default=repo_dir / "data" / "contradictions.json",
        help="Path to contradictions.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_dir / "data" / "dataset.json",
        help="Path for the cleaned output dataset.",
    )
    parser.add_argument(
        "--question-type",
        choices=("simple", "template_based", "complex"),
        default="simple",
        help="Question variant to store as the flat 'question' field.",
    )
    parser.add_argument(
        "--include-unaccepted",
        action="store_true",
        help="Include contradiction entries even when status is not accepted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exposure_data = load_json(args.exposure)
    contradictions = load_json(args.contradictions)
    if not isinstance(exposure_data, dict):
        raise TypeError("exposure_data.json must contain a JSON object")
    if not isinstance(contradictions, dict):
        raise TypeError("contradictions.json must contain a JSON object")

    dataset, stats = build_dataset(
        exposure_data=exposure_data,
        contradictions=contradictions,
        question_type=args.question_type,
        accepted_only=not args.include_unaccepted,
    )
    write_json(dataset, args.output)

    print(f"Wrote {stats['written']} records to {args.output}")
    for key, value in stats.items():
        if key != "written" and value:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
