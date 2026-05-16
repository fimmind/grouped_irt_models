# Grouped residual IRT model comparison

## Setup

- Evaluation vocabulary: 11,952/11,996 Ehara word strings had Excel difficulty coverage.
- Binary label: Ehara `raw_score >= 4` is treated as known; scores 1-3 are treated as unknown/uncertain.
- Users: 16.
- Cold-start reveal counts: 50, 100, 1000.
- Repeats per user and reveal count: 5.
- Difficulty: Excel `Words.accuracy` converted to Rasch difficulty with clipped `-logit(accuracy)`, then standardized.
- Grouping in this run: residual-response clusters (simple k-means grouping over Rasch residual profiles).

## Model family

All grouped models use the spec's residual form:

`sigmoid(theta - b_i + lambda_q * Q_i @ delta_u)`

The optimizer first fits Rasch-only `theta`, then jointly fits `theta` and group residuals with Gaussian priors. `lambda_q = q / (q + c)` and `tau_delta < tau_theta` shrink group effects during cold start.

| model | groups | tau_theta | tau_delta | gate_c | fit_strategy | threshold_strategy |
| --- | --- | --- | --- | --- | --- | --- |
| excel_difficulty_only_fixed | 0 | n/a | n/a | n/a | none | fixed_0.50 |
| excel_difficulty_only_observed_ba_threshold | 0 | n/a | n/a | n/a | none | observed_ba_opt |
| rasch_std_fixed | 0 | 2.0000 | 0.5000 | 50.0000 | standard | fixed_0.50 |
| rasch_balanced_fit | 0 | 2.0000 | 0.5000 | 50.0000 | class_weighted | fixed_0.50 |
| rasch_balanced_fit_ba_threshold | 0 | 2.0000 | 0.5000 | 50.0000 | class_weighted | observed_ba_opt |
| residual_k8_bal_tau1p0_g20_shrunk | 8 | 2.0000 | 1.0000 | 20.0000 | class_weighted | observed_ba_opt_shrunk |
| residual_k8_std_tau1p0_g20_ba | 8 | 2.0000 | 1.0000 | 20.0000 | standard | observed_ba_opt |
| residual_k12_bal_tau1p0_g20_ba | 12 | 2.0000 | 1.0000 | 20.0000 | class_weighted | observed_ba_opt |
| residual_k12_seed1301_tau1p4_g12_shrunk | 12 | 2.0000 | 1.4000 | 12.0000 | class_weighted | observed_ba_opt_shrunk |
| residual_k12_seed701_tau1p4_g8_shrunk | 12 | 2.0000 | 1.4000 | 8.0000 | class_weighted | observed_ba_opt_shrunk |
| residual_k12_seed701_tau1p6_g12_shrunk | 12 | 2.0000 | 1.6000 | 12.0000 | class_weighted | observed_ba_opt_shrunk |
| residual_k12_seed1301_tau1p6_g8_shrunk | 12 | 2.0000 | 1.6000 | 8.0000 | class_weighted | observed_ba_opt_shrunk |

## Best model by reveal count

| q | model | log_loss | brier | auc | pr_auc | accuracy | balanced_accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | residual_k12_seed701_tau1p6_g12_shrunk | 0.3700 | 0.1121 | 0.9150 | 0.9635 | 0.8584 | 0.8303 | 0.1339 | 1169.5088 |
| 100 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.3477 | 0.1039 | 0.9267 | 0.9699 | 0.8653 | 0.8513 | 0.1284 | 1165.2375 |
| 1000 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.2881 | 0.0874 | 0.9418 | 0.9772 | 0.8738 | 0.8806 | 0.0978 | 965.8808 |

## Full aggregate results

| q | model | log_loss | brier | auc | pr_auc | accuracy | balanced_accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | residual_k12_seed701_tau1p6_g12_shrunk | 0.3700 | 0.1121 | 0.9150 | 0.9635 | 0.8584 | 0.8303 | 0.1339 | 1169.5088 |
| 50 | residual_k12_seed701_tau1p4_g8_shrunk | 0.3754 | 0.1139 | 0.9149 | 0.9631 | 0.8580 | 0.8288 | 0.1378 | 1194.8026 |
| 50 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.3817 | 0.1162 | 0.9142 | 0.9632 | 0.8565 | 0.8274 | 0.1408 | 1222.0365 |
| 50 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.3660 | 0.1111 | 0.9144 | 0.9643 | 0.8587 | 0.8273 | 0.1292 | 1146.6349 |
| 50 | residual_k12_bal_tau1p0_g20_ba | 0.4260 | 0.1328 | 0.9075 | 0.9558 | 0.8334 | 0.8093 | 0.1609 | 1411.4513 |
| 50 | residual_k8_bal_tau1p0_g20_shrunk | 0.4277 | 0.1334 | 0.9019 | 0.9562 | 0.8335 | 0.8083 | 0.1638 | 1433.7804 |
| 50 | residual_k8_std_tau1p0_g20_ba | 0.3726 | 0.1150 | 0.9033 | 0.9560 | 0.8313 | 0.7979 | 0.1041 | 423.7455 |
| 50 | excel_difficulty_only_observed_ba_threshold | 0.5833 | 0.1995 | 0.8571 | 0.9191 | 0.7559 | 0.7627 | 0.2235 | 2419.7424 |
| 50 | excel_difficulty_only_fixed | 0.5833 | 0.1995 | 0.8571 | 0.9191 | 0.6921 | 0.7587 | 0.2235 | 2419.7424 |
| 50 | rasch_balanced_fit_ba_threshold | 0.4322 | 0.1397 | 0.8571 | 0.9191 | 0.7731 | 0.7528 | 0.0906 | 477.8705 |
| 50 | rasch_balanced_fit | 0.4322 | 0.1397 | 0.8571 | 0.9191 | 0.7981 | 0.6484 | 0.0906 | 477.8705 |
| 50 | rasch_std_fixed | 0.4322 | 0.1397 | 0.8571 | 0.9191 | 0.7981 | 0.6484 | 0.0906 | 477.8705 |
| 100 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.3477 | 0.1039 | 0.9267 | 0.9699 | 0.8653 | 0.8513 | 0.1284 | 1165.2375 |
| 100 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.3367 | 0.1008 | 0.9262 | 0.9701 | 0.8670 | 0.8504 | 0.1180 | 1104.9947 |
| 100 | residual_k12_seed701_tau1p4_g8_shrunk | 0.3450 | 0.1031 | 0.9246 | 0.9689 | 0.8669 | 0.8494 | 0.1257 | 1136.6683 |
| 100 | residual_k12_seed701_tau1p6_g12_shrunk | 0.3388 | 0.1014 | 0.9241 | 0.9689 | 0.8675 | 0.8485 | 0.1198 | 1102.1642 |
| 100 | residual_k12_bal_tau1p0_g20_ba | 0.3815 | 0.1151 | 0.9279 | 0.9677 | 0.8574 | 0.8467 | 0.1553 | 1321.9263 |
| 100 | residual_k8_bal_tau1p0_g20_shrunk | 0.3991 | 0.1226 | 0.9142 | 0.9647 | 0.8449 | 0.8384 | 0.1589 | 1418.5275 |
| 100 | residual_k8_std_tau1p0_g20_ba | 0.3323 | 0.0998 | 0.9170 | 0.9649 | 0.8484 | 0.8335 | 0.0898 | 289.3280 |
| 100 | excel_difficulty_only_observed_ba_threshold | 0.5833 | 0.1995 | 0.8572 | 0.9191 | 0.7575 | 0.7677 | 0.2235 | 2410.6325 |
| 100 | rasch_balanced_fit_ba_threshold | 0.4284 | 0.1380 | 0.8572 | 0.9191 | 0.7703 | 0.7625 | 0.0860 | 372.7637 |
| 100 | excel_difficulty_only_fixed | 0.5833 | 0.1995 | 0.8572 | 0.9191 | 0.6921 | 0.7587 | 0.2235 | 2410.6325 |
| 100 | rasch_balanced_fit | 0.4284 | 0.1380 | 0.8572 | 0.9191 | 0.8026 | 0.6548 | 0.0860 | 372.7637 |
| 100 | rasch_std_fixed | 0.4284 | 0.1380 | 0.8572 | 0.9191 | 0.8026 | 0.6548 | 0.0860 | 372.7637 |
| 1000 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.2881 | 0.0874 | 0.9418 | 0.9772 | 0.8738 | 0.8806 | 0.0978 | 965.8808 |
| 1000 | residual_k12_seed701_tau1p6_g12_shrunk | 0.2878 | 0.0874 | 0.9410 | 0.9767 | 0.8726 | 0.8801 | 0.0971 | 960.0555 |
| 1000 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.2912 | 0.0882 | 0.9416 | 0.9770 | 0.8734 | 0.8798 | 0.0998 | 978.0995 |
| 1000 | residual_k12_seed701_tau1p4_g8_shrunk | 0.2908 | 0.0882 | 0.9407 | 0.9765 | 0.8720 | 0.8793 | 0.0989 | 971.6537 |
| 1000 | residual_k12_bal_tau1p0_g20_ba | 0.2987 | 0.0897 | 0.9438 | 0.9769 | 0.8728 | 0.8768 | 0.1039 | 995.8273 |
| 1000 | residual_k8_bal_tau1p0_g20_shrunk | 0.3470 | 0.1082 | 0.9224 | 0.9699 | 0.8473 | 0.8511 | 0.1201 | 1201.7774 |
| 1000 | residual_k8_std_tau1p0_g20_ba | 0.2757 | 0.0839 | 0.9240 | 0.9700 | 0.8550 | 0.8505 | 0.0397 | 81.9007 |
| 1000 | excel_difficulty_only_observed_ba_threshold | 0.5835 | 0.1996 | 0.8571 | 0.9191 | 0.7612 | 0.7760 | 0.2237 | 2228.9392 |
| 1000 | rasch_balanced_fit_ba_threshold | 0.4239 | 0.1364 | 0.8571 | 0.9191 | 0.7692 | 0.7715 | 0.0796 | 104.4894 |
| 1000 | excel_difficulty_only_fixed | 0.5835 | 0.1996 | 0.8571 | 0.9191 | 0.6920 | 0.7588 | 0.2237 | 2228.9392 |
| 1000 | rasch_balanced_fit | 0.4239 | 0.1364 | 0.8571 | 0.9191 | 0.8053 | 0.6533 | 0.0796 | 104.4894 |
| 1000 | rasch_std_fixed | 0.4239 | 0.1364 | 0.8571 | 0.9191 | 0.8053 | 0.6533 | 0.0796 | 104.4894 |

## Interpretation

- At q=50, `residual_k12_seed701_tau1p6_g12_shrunk` has the highest balanced accuracy (0.8303), improving over Rasch by 0.1819.
- At q=100, `residual_k12_seed1301_tau1p4_g12_shrunk` has the highest balanced accuracy (0.8513), improving over Rasch by 0.1965.
- At q=1000, `residual_k12_seed1301_tau1p6_g8_shrunk` has the highest balanced accuracy (0.8806), improving over Rasch by 0.2273.
- Ranking in this report is by balanced accuracy first (higher is better), then log loss.
- Residual-response groups are expected to perform strongly here because they are derived from the same response matrix. Treat those results as an optimistic upper-bound for learned behavioral groupings unless groups are rebuilt on a separate calibration sample.
- Balanced-accuracy-oriented threshold optimization can degrade calibration and probability quality. Use log loss and Brier score as guardrails when choosing production settings.

## Cross-dataset update (latest)

The results above are the original Ehara-centered benchmark. Additional optimization was run on Duolingo HLR and then transferred back to Ehara.

### Duolingo-optimized setup

- Dataset slice: `en->es`, dense `120 x 1200` matrix.
- Labeling: known if `mean(p_recall) >= 0.8`.
- Missing user-lexeme cells: lexeme-majority imputation.
- Difficulty: lexeme mean `p_recall`, transformed as standardized clipped `-logit`.

### Best grouped models on Duolingo by reveal count

| q | model | balanced_accuracy | pr_auc |
| --- | --- | --- | --- |
| 50 | response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk | 0.8370 | 0.9499 |
| 100 | residual16_s701_g16_tau1p6_c12p0_observed_ba_opt_shrunk | 0.8461 | 0.9533 |
| 1000 | residual16_s701_g16_tau1p0_c20p0_observed_ba_opt | 0.8619 | 0.9610 |

Matched Rasch baseline on the same Duolingo setup:

| q | rasch_bal_ba | best_grouped | BA gain |
| --- | --- | --- | --- |
| 50 | 0.7818 | 0.8370 | +0.0552 |
| 100 | 0.7887 | 0.8461 | +0.0575 |
| 1000 | 0.7969 | 0.8619 | +0.0650 |

### Transfer of the best shared model back to Ehara

The most robust shared model across both datasets is:

- `response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk`

Ehara performance (same `q` protocol) for that model:

| q | balanced_accuracy | pr_auc |
| --- | --- | --- |
| 50 | 0.8337 | 0.9628 |
| 100 | 0.8549 | 0.9703 |
| 1000 | 0.8862 | 0.9773 |

Ehara matched Rasch baseline (`rasch_bal_ba`) and gain:

| q | rasch_bal_ba | response12... | BA gain |
| --- | --- | --- | --- |
| 50 | 0.7407 | 0.8337 | +0.0931 |
| 100 | 0.7625 | 0.8549 | +0.0924 |
| 1000 | 0.7699 | 0.8862 | +0.1163 |
