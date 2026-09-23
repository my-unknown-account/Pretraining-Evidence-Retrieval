# RQ1: Corrective Retrieval vs. Pretraining Evidence

RQ1 measures when retrieval helps. The experiment focuses on facts the model gets wrong without context, then asks whether a correct retrieved passage repairs the answer.

Filter:

```text
y_CB != o
```

Metric:

```text
RCR = P(y_positive = o | y_CB != o)
```

The key question is whether low-RAS facts benefit more from corrective retrieval than high-RAS facts. If a model already has strong relation-aware pretraining evidence, retrieval should have less room to help.

## Paper Result

Correct retrieval repaired a large share of initially wrong answers:

- Amber: `RCR = 0.773`
- OLMo: `RCR = 0.889`
- RedPajama: `RCR = 0.844`

Across RAS bins, correction rates were lower in the highest-support group than
in the lowest-support group by 8.0 percentage points for Amber, 8.7 for OLMo,
and 10.0 for RedPajama.

## Run

```powershell
python scripts\inference\select_majority_results.py
python scripts\analysis\analyze_rq1.py
```

## Inputs

- `data/dataset.json`
- `scripts/inference/results.json`

## Outputs

Written to `scripts/analysis/outputs/rq1/`:

- `rq1_examples.csv`: filtered example-level rows.
- `rq1_bins.csv`: RAS-bin correction rates.
- `rq1_summary.md`: model-level rates and trend diagnostics.
- `figure2_correction_rate_vs_ras.pdf`: correction-rate plot.
