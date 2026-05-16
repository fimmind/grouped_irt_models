# Duolingo HLR grouped IRT results (balanced-accuracy optimization)

## Setup

- Dataset: `data/raw/duolingo_hlr/learning_traces.csv.gz`
- Language pair: `en->es`
- Dense evaluation matrix: `120 users x 1200 lexemes`
- Known/unknown label: `known = mean(p_recall per user-lexeme) >= 0.8`
- Missing user-lexeme cells: imputed via lexeme-majority known label
- Lexeme difficulty: mean `p_recall` per lexeme -> clipped `-logit` -> standardized
- Evaluation: per-user random reveal, then predict hidden words
- Repeats: `3` for final metrics

## Optimization scope

- Coarse stage: 20 models, `q = 50, 100`, 1 repeat
- Fine stage: top 6 models, `q = 50, 100, 1000`, 3 repeats
- Grouping families tested:
  - residual-response clusters
  - response-only clusters
  - residual-sign clusters
  - fastText clusters
  - residual x difficulty-bin interaction groups

## Best model by q

| q | best model | balanced_accuracy | pr_auc | auc |
| --- | --- | --- | --- | --- |
| 50 | `response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk` | 0.8370 | 0.9499 | 0.8972 |
| 100 | `residual16_s701_g16_tau1p6_c12p0_observed_ba_opt_shrunk` | 0.8461 | 0.9533 | 0.9051 |
| 1000 | `residual16_s701_g16_tau1p0_c20p0_observed_ba_opt` | 0.8619 | 0.9610 | 0.9196 |

## Comparison to matched Rasch baseline

Baseline used: `rasch_bal_ba` on the exact same data slice and protocol.

| q | Rasch BA | best grouped BA | BA gain |
| --- | --- | --- | --- |
| 50 | 0.7818 | 0.8370 | +0.0552 |
| 100 | 0.7887 | 0.8461 | +0.0575 |
| 1000 | 0.7969 | 0.8619 | +0.0650 |

## Notes

- On this Duolingo setup, grouped models consistently outperform Rasch in balanced accuracy.
- Best-performing grouped families are `response12` and `residual16` with shrunk BA-optimized thresholds.
- Full fine-stage ranking is stored in `duolingo_irt_optimization_summary.md`.
