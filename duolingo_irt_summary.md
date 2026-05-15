# Duolingo HTL benchmark (q=50)

## Setup

- Source: `data/raw/duolingo_hlr/learning_traces.csv.gz`.
- Language pair: `en->es`.
- Subset selection: top `500` users by row count and top `1000` lexemes by row count.
- Label construction: `known = (mean p_recall per user-lexeme >= 0.5)`.
- Missing user-lexeme cells were imputed with lexeme-majority known/unknown to form a dense matrix required by grouped IRT.
- Lexeme difficulty was estimated from Duolingo subset accuracy and transformed as standardized `-logit(accuracy)`.
- Evaluation protocol: per-user random reveal of `q=50` lexemes, hidden-set scoring, `3` repeats.

## Results

| rank | model | pr_auc | balanced_accuracy |
| --- | --- | --- | --- |
| 1 | residual_k12_seed701_tau1p4_g8_shrunk | 0.9918 | 0.5213 |
| 2 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.9916 | 0.5213 |
| 3 | residual_k12_seed701_tau1p6_g12_shrunk | 0.9918 | 0.5210 |

## Notes

- Rankings are sorted by balanced accuracy first, then PR-AUC.
- Dense matrix size used for this run: `500 x 1000`.