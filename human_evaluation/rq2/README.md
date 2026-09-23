# RQ2 Human Evaluation

This folder audits the automatic judge used for RQ2. RQ2 asks whether a model keeps the true answer or follows a contradictory retrieved passage.

The paper uses this evaluation to validate the three-way contradictory-context
judge that separates resistance, corruption, and unrelated degradation.

## Labels

- `TRUE_OBJECT`: the candidate answer gives the true object, `o`.
- `FALSE_CONTEXT_OBJECT`: the candidate answer gives the false context object, `o'`.
- `OTHER`: the candidate answer gives neither target object.

## Contents

- `create_annotation_workbooks.py`: samples examples and creates annotation workbooks.
- `compute_agreement.py`: compares filled human labels with the original automatic judge labels.
- `results/rq2_examples.csv`: sampled-source metadata used by the agreement script.
- `results/*_filled.xlsx`: filled annotator workbooks.

## Create New Annotation Workbooks

This is only needed if you want to rerun the human evaluation:

```powershell
python .\create_annotation_workbooks.py
```

The script samples from `scripts/analysis/outputs/rq2/rq2_examples.csv` and joins raw contradictory-context generations from `scripts/inference/contradictory_context/results/`.

## Annotation Rule

Judge the `candidate_answer` for the `question` against the two target objects. Use `TRUE_OBJECT` for the true answer, `FALSE_CONTEXT_OBJECT` for the contradictory-context answer, and `OTHER` for unrelated, ambiguous, mixed, refusal, or different-main-claim answers.

When an answer contains extra explanation, label by the main answer.

## Compute Agreement

The included filled workbooks can be evaluated directly:

```powershell
python .\compute_agreement.py
```

Expected summary from the included workbooks:

```text
rq2_human_evaluation_annotator_1_filled.xlsx,300,287,95.67
rq2_human_evaluation_annotator_2_filled.xlsx,300,291,97.00
rq2_human_evaluation_annotator_3_filled.xlsx,300,273,91.00
OVERALL,900,851,94.56
```
