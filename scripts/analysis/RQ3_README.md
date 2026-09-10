# RQ3: Semantic vs. Lexical Evidence

RQ3 compares signals. It asks whether semantic pretraining evidence explains retrieval behavior better than lexical co-occurrence or closed-book confidence.

Signals:

- `Lex-SO`: subject-object lexical co-occurrence.
- `Lex-SRO`: subject-relation-object lexical co-occurrence.
- `Self-consistency`: closed-book confidence signal.
- `Relation-Aware Support`: semantic support, `RAS`.

The metric is direction-adjusted rank AUROC for predicting correction and contradictory override outcomes.

## Paper Result

`RAS` was the strongest individual signal for both behaviors:

| Signal | Correction macro AUROC | Override macro AUROC |
| --- | ---: | ---: |
| Lex-SO | 0.572 | 0.572 |
| Lex-SRO | 0.573 | 0.546 |
| Self-consistency | 0.562 | 0.527 |
| RAS | 0.607 | 0.593 |

The effect is meaningful but not deterministic: no single signal fully predicts
whether a model follows retrieved evidence on a specific generation trial.

## Run

Run RQ1 and RQ2 first, then:

```powershell
python scripts\analysis\analyze_rq3.py
```

## Inputs

- `scripts/analysis/outputs/rq1/rq1_examples.csv`
- `scripts/analysis/outputs/rq2/rq2_examples.csv`

## Outputs

Written to `scripts/analysis/outputs/rq3/`:

- `table3_signal_auc.csv`
- `table3_signal_auc.md`
- `table3_signal_auc_pooled.md`
- `rq3_summary.md`
