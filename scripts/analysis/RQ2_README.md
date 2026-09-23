# RQ2: Contradictory Retrieval vs. Pretraining Evidence

RQ2 measures when retrieval hurts. The experiment focuses on facts the model gets right without context, then asks whether a contradictory passage overrides the correct answer.

Filter:

```text
y_CB = o
```

Metric:

```text
ROR = P(y_negative = o')
```

Labels:

- `TRUE_OBJECT`: resistance, `o -> o`
- `FALSE_CONTEXT_OBJECT`: override, `o -> o'`
- `OTHER`: other degradation, `o -> z`

## Paper Result

Contradictory context frequently displaced initially correct answers:

- Amber: `ROR = 0.522`
- OLMo: `ROR = 0.749`
- RedPajama: `ROR = 0.603`

Override rates dropped from the lowest-RAS group to the highest-RAS group by
14.2 percentage points for Amber, 19.9 for OLMo, and 16.8 for RedPajama.

## Run

```powershell
python scripts\inference\select_majority_results.py
python scripts\analysis\analyze_rq2.py
```

## Inputs

- `data/dataset.json`
- `scripts/inference/results.json`

## Outputs

Written to `scripts/analysis/outputs/rq2/`:

- `rq2_examples.csv`: filtered example-level rows.
- `rq2_bins.csv`: RAS-bin resistance, override, and other rates.
- `rq2_summary.md`: model-level rates and trend diagnostics.
- `figure3_override_rate_vs_ras.pdf`: override-rate plot.
