# RQ3: Semantic vs. Lexical Evidence

RQ3 compares signals. It asks whether semantic pretraining evidence explains retrieval behavior better than lexical co-occurrence or closed-book confidence.

Signals:

- `Lex-SO`: subject-object lexical co-occurrence.
- `Lex-SRO`: subject-relation-object lexical co-occurrence.
- `Token logprob`: closed-book confidence signal.
- `Verbalized confidence`: closed-book confidence signal.
- `P(true)`: closed-book confidence signal.
- `Self-consistency`: closed-book confidence signal.
- `Relation-Aware Support`: semantic support, `RAS`.

The metric is direction-adjusted rank AUROC for predicting correction and contradictory-context corruption outcomes.

## Paper Result

The signals separated outcomes modestly, but no single signal fully determined
whether a model followed supplied context:

| Signal | Macro Correction | Macro Override | Pooled Correction | Pooled Override |
| --- | ---: | ---: | ---: | ---: |
| Lex-SO | 0.583 | 0.581 | 0.553 | 0.557 |
| Lex-SRO | 0.572 | 0.548 | 0.570 | 0.543 |
| Token logprob | 0.563 | 0.535 | 0.615 | 0.534 |
| Verbalized confidence | 0.522 | 0.512 | 0.532 | 0.531 |
| P(true) | 0.541 | 0.547 | 0.543 | 0.521 |
| Self-consistency | 0.571 | 0.506 | 0.569 | 0.504 |
| Relation-Aware Support | 0.568 | 0.582 | 0.528 | 0.549 |

The effect is meaningful but not deterministic: no single signal fully predicts
whether a model follows retrieved evidence on a specific generation trial.

The analysis script recomputes AUROC values from the local RQ1/RQ2 example
files. The table above records the values reported in the paper.

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
