# RQ4: OLMo vs. OLMo-32B

RQ4 repeats the RQ1-RQ3 analysis family for `olmo` and `olmo32`. It isolates the model-scale comparison while keeping the same dataset, retrieval settings, and signal definitions.

Included models:

- `olmo`
- `olmo32`

## Paper Result

Corrective retrieval was similar across model scales, but contradictory
retrieval was not:

| Model | RCR | Resistance | ROR | Other |
| --- | ---: | ---: | ---: | ---: |
| OLMo | 0.888 | 0.055 | 0.749 | 0.196 |
| OLMo-32B | 0.880 | 0.159 | 0.621 | 0.220 |

The larger model was less likely to adopt the planted false object, even though
both OLMo models share the same corpus-derived evidence measurements.

## Run

```powershell
python scripts\analysis\analyze_rq4.py
```

## Inputs

RQ4 reuses the same inference result directories as RQ1 and RQ2:

- `scripts/inference/closed_book/results/`
- `scripts/inference/correct_context/results/`
- `scripts/inference/contradictory_context/results/`

## Outputs

Written to `scripts/analysis/outputs/rq4/`:

- `rq1/`: corrective retrieval analysis for OLMo and OLMo-32B.
- `rq2/`: contradictory retrieval analysis for OLMo and OLMo-32B.
- `rq3/`: semantic-vs-lexical signal comparison.
- `rq4_summary.md`: combined summary.
