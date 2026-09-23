# RQ4: OLMo-3-7B vs. OLMo-3-32B

RQ4 repeats the RQ1-RQ3 analysis family for `olmo` and `olmo32`. It isolates the model-scale comparison while keeping the same dataset, retrieval settings, and signal definitions.

Included models:

- `olmo`
- `olmo32`

## Paper Result

Corrective retrieval was similar across model scales, but contradictory
retrieval was not:

| Model | RCR | Resistance | Rcp | Other |
| --- | ---: | ---: | ---: | ---: |
| OLMo-3-7B | 0.889 | 0.063 | 0.744 | 0.192 |
| OLMo-3-32B | 0.879 | 0.170 | 0.620 | 0.210 |

The larger model was less likely to adopt the planted false object, even though
both OLMo models share the same corpus-derived evidence measurements.

## Run

```powershell
python scripts\inference\select_majority_results.py
python scripts\analysis\analyze_rq4.py
```

## Inputs

RQ4 reuses the same majority-selected inference file as RQ1 and RQ2:

- `scripts/inference/results.json`

## Outputs

Written to `scripts/analysis/outputs/rq4/`:

- `rq1/`: corrective retrieval analysis for OLMo-3-7B and OLMo-3-32B.
- `rq2/`: contradictory retrieval analysis for OLMo-3-7B and OLMo-3-32B.
- `rq3/`: semantic-vs-lexical signal comparison.
- `rq4_summary.md`: combined summary.
