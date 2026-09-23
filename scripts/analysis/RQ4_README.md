# RQ4: OLMo-3-7B vs. OLMo-3-32B

RQ4 repeats the RQ1-RQ3 analysis family for `olmo` and `olmo32`. It isolates the model-scale comparison while keeping the same dataset, retrieval settings, and signal definitions.

Included models:

- `olmo` = OLMo-3-7B
- `olmo32` = OLMo-3-32B

## Paper Result

Corrective retrieval was similar across model scales, but contradictory
retrieval was not:

| Model | Closed-book wrong n | RCR | Closed-book correct n | Resistance | Rcp | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMo-3-7B | 9,489 | 0.879 | 1,249 | 0.072 | 0.744 | 0.184 |
| OLMo-3-32B | 9,489 | 0.878 | 1,249 | 0.251 | 0.530 | 0.219 |

The larger model was less likely to adopt the planted false object, even though
both OLMo models share the same corpus-derived evidence measurements. The table
uses only facts retained for both models after the RQ-specific closed-book
filter.

Paper signal-discrimination values for the shared-fact OLMo comparison:

| Signal | Correction AUROC | Override AUROC |
| --- | ---: | ---: |
| Lex-SO | 0.622 | 0.619 |
| Lex-SRO | 0.562 | 0.535 |
| Token logprob | 0.535 | 0.535 |
| Verbalized confidence | 0.550 | 0.531 |
| P(true) | 0.552 | 0.526 |
| Self-consistency | 0.575 | 0.545 |
| Relation-Aware Support | 0.606 | 0.621 |

The analysis script recomputes signal-discrimination values from the local
example files. The table above records the values reported in the paper.

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
