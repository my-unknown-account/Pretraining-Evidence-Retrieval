# RAS Human Evaluation

This folder contains the support-only human evaluation for relation-aware
support (`RAS`). Annotators judged whether sampled support passages really
support the target relation claim.

The paper uses this evaluation to validate the majority-vote automatic support
labeling procedure behind `RAS`.

## Contents

- `samples.json`: the sampled support passages.
- `make_excels.py`: creates one annotation workbook per qid.
- `compute_agreement.py`: computes annotator agreement and majority-vs-gold
  agreement.
- `results/`: filled workbooks from the annotators.

## Create New Workbooks

This is only needed if you want to rerun annotation:

```powershell
python .\make_excels.py
```

By default this writes unfilled workbooks to:

```text
workbooks/
  qid_6414.xlsx
  qid_1078.xlsx
  ...
```

Each workbook has one sheet per support passage.

## Filled Results

The included filled workbooks live in:

```text
results/
  Jamshid/
  Zahra/
  Kurosh/
```

## Compute Agreement

```powershell
python .\compute_agreement.py
```

Expected summary from the included workbooks:

```text
Jamshid,300,256,85.33,0
Kurosh,300,244,81.33,0
Zahra,300,251,83.67,0
majority_vs_gold,300,267,89.00
```
