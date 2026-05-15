# Grouped residual IRT model comparison

## Setup

- Evaluation vocabulary: 11,952/11,996 Ehara word strings had Excel difficulty coverage.
- Binary label: Ehara `raw_score >= 4` is treated as known; scores 1-3 are treated as unknown/uncertain.
- Users: 16.
- Cold-start reveal counts: 50, 100, 1000.
- Repeats per user and reveal count: 5.
- Difficulty: Excel `Words.accuracy` converted to Rasch difficulty with clipped `-logit(accuracy)`, then standardized.
<<<<<<< HEAD
- FastText: local skipgram model trained from the available vocabulary text, then clustered with KMeans.
- Fitting objective: asymmetric BCE with `known_loss_weight=0.9` and `unknown_loss_weight=1.1` to penalize false known predictions more than false unknown predictions.
- Benchmark focus: unknown-priority balanced accuracy `0.4 * known_recall + 0.6 * unknown_recall`.
=======
- Grouping in this run: residual-response clusters (simple k-means grouping over Rasch residual profiles).
>>>>>>> 2fb8341 (Improve residual IRT balanced accuracy for q50/q100 and add q1000 eval)

## Model family

All grouped models use the spec's residual form:

`sigmoid(theta - b_i + lambda_q * Q_i @ delta_u)`

The optimizer first fits Rasch-only `theta`, then jointly fits `theta` and group residuals with Gaussian priors. `lambda_q = q / (q + c)` and `tau_delta < tau_theta` shrink group effects during cold start.

<<<<<<< HEAD
| model | groups | tau_theta | tau_delta | gate_c | known_loss_weight | unknown_loss_weight |
| --- | --- | --- | --- | --- | --- | --- |
| excel_difficulty_only | 0 | n/a | n/a | n/a | n/a | n/a |
| rasch | 0 | 2.0000 | 0.5000 | 50.0000 | 0.9000 | 1.1000 |
| semantic_fasttext_k12 | 12 | 2.0000 | 0.5500 | 50.0000 | 0.9000 | 1.1000 |
| semantic_fasttext_k24 | 24 | 2.0000 | 0.5500 | 50.0000 | 0.9000 | 1.1000 |
| semantic_fasttext_k48 | 48 | 2.0000 | 0.4500 | 70.0000 | 0.9000 | 1.1000 |
| expert_tags | 20 | 2.0000 | 0.5500 | 50.0000 | 0.9000 | 1.1000 |
| residual_k12 | 12 | 2.0000 | 0.5500 | 50.0000 | 0.9000 | 1.1000 |
| residual_k24 | 24 | 2.0000 | 0.5500 | 50.0000 | 0.9000 | 1.1000 |
| residual_k48 | 48 | 2.0000 | 0.4500 | 70.0000 | 0.9000 | 1.1000 |
| all_groups_balanced | 68 | 2.0000 | 0.4500 | 50.0000 | 0.9000 | 1.1000 |
| all_groups_fast_gate | 68 | 2.0000 | 0.7000 | 20.0000 | 0.9000 | 1.1000 |
| all_groups_strong_shrinkage | 68 | 2.0000 | 0.3500 | 100.0000 | 0.9000 | 1.1000 |
=======
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
>>>>>>> 2fb8341 (Improve residual IRT balanced accuracy for q50/q100 and add q1000 eval)

## Best model by reveal count (unknown-priority balanced accuracy)

<<<<<<< HEAD
| q | model | unknown_priority_balanced_accuracy | balanced_accuracy | known_recall | unknown_recall | log_loss | brier | auc | accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | excel_difficulty_only | 0.7818 | 0.7587 | 0.6429 | 0.8744 | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2428.2544 |
| 20 | excel_difficulty_only | 0.7818 | 0.7587 | 0.6430 | 0.8744 | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2426.0938 |
| 50 | excel_difficulty_only | 0.7818 | 0.7586 | 0.6429 | 0.8744 | 0.5833 | 0.1995 | 0.8571 | 0.6920 | 0.2235 | 2420.1824 |
| 100 | excel_difficulty_only | 0.7818 | 0.7586 | 0.6428 | 0.8744 | 0.5834 | 0.1995 | 0.8571 | 0.6920 | 0.2236 | 2410.6763 |

## Full aggregate results

| q | model | unknown_priority_balanced_accuracy | balanced_accuracy | known_recall | unknown_recall | log_loss | brier | auc | accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | excel_difficulty_only | 0.7818 | 0.7587 | 0.6429 | 0.8744 | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2428.2544 |
| 10 | residual_k12 | 0.6269 | 0.6672 | 0.8687 | 0.4656 | 0.4488 | 0.1453 | 0.8575 | 0.7934 | 0.1096 | 890.2489 |
| 10 | residual_k24 | 0.6266 | 0.6670 | 0.8687 | 0.4653 | 0.4490 | 0.1453 | 0.8573 | 0.7932 | 0.1095 | 890.3135 |
| 10 | all_groups_fast_gate | 0.6266 | 0.6669 | 0.8687 | 0.4652 | 0.4490 | 0.1453 | 0.8573 | 0.7932 | 0.1095 | 890.4148 |
| 10 | residual_k48 | 0.6265 | 0.6668 | 0.8685 | 0.4651 | 0.4491 | 0.1454 | 0.8572 | 0.7931 | 0.1095 | 890.5525 |
| 10 | expert_tags | 0.6264 | 0.6668 | 0.8685 | 0.4651 | 0.4491 | 0.1454 | 0.8571 | 0.7930 | 0.1094 | 890.6507 |
| 10 | all_groups_balanced | 0.6264 | 0.6667 | 0.8685 | 0.4650 | 0.4492 | 0.1454 | 0.8571 | 0.7930 | 0.1095 | 890.6415 |
| 10 | semantic_fasttext_k24 | 0.6264 | 0.6667 | 0.8685 | 0.4650 | 0.4492 | 0.1454 | 0.8571 | 0.7930 | 0.1095 | 890.6887 |
| 10 | all_groups_strong_shrinkage | 0.6264 | 0.6667 | 0.8684 | 0.4650 | 0.4492 | 0.1454 | 0.8571 | 0.7930 | 0.1095 | 890.6623 |
| 10 | rasch | 0.6264 | 0.6667 | 0.8684 | 0.4650 | 0.4492 | 0.1454 | 0.8571 | 0.7930 | 0.1095 | 890.6676 |
| 10 | semantic_fasttext_k12 | 0.6264 | 0.6667 | 0.8684 | 0.4650 | 0.4492 | 0.1454 | 0.8571 | 0.7930 | 0.1095 | 890.5714 |
| 10 | semantic_fasttext_k48 | 0.6264 | 0.6667 | 0.8685 | 0.4650 | 0.4492 | 0.1454 | 0.8571 | 0.7930 | 0.1095 | 890.6273 |
| 20 | excel_difficulty_only | 0.7818 | 0.7587 | 0.6430 | 0.8744 | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2426.0938 |
| 20 | residual_k12 | 0.6236 | 0.6610 | 0.8477 | 0.4743 | 0.4414 | 0.1433 | 0.8592 | 0.7915 | 0.1032 | 771.9783 |
| 20 | residual_k24 | 0.6228 | 0.6602 | 0.8473 | 0.4731 | 0.4421 | 0.1436 | 0.8585 | 0.7909 | 0.1028 | 772.4849 |
| 20 | all_groups_fast_gate | 0.6223 | 0.6598 | 0.8473 | 0.4724 | 0.4424 | 0.1437 | 0.8579 | 0.7905 | 0.1025 | 772.4294 |
| 20 | expert_tags | 0.6218 | 0.6593 | 0.8469 | 0.4717 | 0.4429 | 0.1439 | 0.8571 | 0.7899 | 0.1022 | 773.3060 |
| 20 | residual_k48 | 0.6216 | 0.6591 | 0.8465 | 0.4717 | 0.4430 | 0.1440 | 0.8574 | 0.7901 | 0.1026 | 773.9580 |
| 20 | all_groups_balanced | 0.6215 | 0.6590 | 0.8464 | 0.4716 | 0.4432 | 0.1441 | 0.8572 | 0.7899 | 0.1025 | 774.2143 |
| 20 | all_groups_strong_shrinkage | 0.6214 | 0.6588 | 0.8462 | 0.4715 | 0.4433 | 0.1441 | 0.8571 | 0.7898 | 0.1025 | 774.4378 |
| 20 | semantic_fasttext_k48 | 0.6214 | 0.6588 | 0.8463 | 0.4714 | 0.4433 | 0.1441 | 0.8571 | 0.7898 | 0.1026 | 774.2410 |
| 20 | rasch | 0.6213 | 0.6588 | 0.8462 | 0.4715 | 0.4433 | 0.1441 | 0.8571 | 0.7898 | 0.1025 | 774.4961 |
| 20 | semantic_fasttext_k24 | 0.6213 | 0.6588 | 0.8463 | 0.4713 | 0.4433 | 0.1441 | 0.8571 | 0.7898 | 0.1026 | 773.9410 |
| 20 | semantic_fasttext_k12 | 0.6212 | 0.6587 | 0.8462 | 0.4712 | 0.4433 | 0.1441 | 0.8571 | 0.7897 | 0.1026 | 774.3550 |
| 50 | excel_difficulty_only | 0.7818 | 0.7586 | 0.6429 | 0.8744 | 0.5833 | 0.1995 | 0.8571 | 0.6920 | 0.2235 | 2420.1824 |
| 50 | residual_k12 | 0.6418 | 0.6789 | 0.8643 | 0.4934 | 0.4212 | 0.1342 | 0.8719 | 0.8124 | 0.0998 | 583.4955 |
| 50 | residual_k24 | 0.6361 | 0.6737 | 0.8616 | 0.4858 | 0.4257 | 0.1361 | 0.8681 | 0.8084 | 0.0980 | 586.7969 |
| 50 | all_groups_fast_gate | 0.6310 | 0.6688 | 0.8576 | 0.4799 | 0.4306 | 0.1383 | 0.8615 | 0.8037 | 0.0946 | 589.9192 |
| 50 | expert_tags | 0.6300 | 0.6674 | 0.8544 | 0.4804 | 0.4329 | 0.1393 | 0.8572 | 0.8013 | 0.0914 | 590.8279 |
| 50 | residual_k48 | 0.6284 | 0.6667 | 0.8581 | 0.4752 | 0.4325 | 0.1390 | 0.8603 | 0.8028 | 0.0951 | 592.7299 |
| 50 | all_groups_balanced | 0.6265 | 0.6648 | 0.8566 | 0.4731 | 0.4344 | 0.1398 | 0.8580 | 0.8011 | 0.0942 | 595.0206 |
| 50 | all_groups_strong_shrinkage | 0.6256 | 0.6641 | 0.8564 | 0.4718 | 0.4352 | 0.1401 | 0.8573 | 0.8006 | 0.0942 | 596.0029 |
| 50 | rasch | 0.6253 | 0.6638 | 0.8563 | 0.4713 | 0.4354 | 0.1402 | 0.8571 | 0.8004 | 0.0942 | 596.3639 |
| 50 | semantic_fasttext_k12 | 0.6251 | 0.6637 | 0.8568 | 0.4706 | 0.4353 | 0.1402 | 0.8571 | 0.8005 | 0.0941 | 594.4903 |
| 50 | semantic_fasttext_k48 | 0.6250 | 0.6636 | 0.8565 | 0.4707 | 0.4353 | 0.1402 | 0.8572 | 0.8003 | 0.0942 | 595.0904 |
| 50 | semantic_fasttext_k24 | 0.6249 | 0.6636 | 0.8569 | 0.4703 | 0.4352 | 0.1402 | 0.8572 | 0.8004 | 0.0942 | 594.6938 |
| 100 | excel_difficulty_only | 0.7818 | 0.7586 | 0.6428 | 0.8744 | 0.5834 | 0.1995 | 0.8571 | 0.6920 | 0.2236 | 2410.6763 |
| 100 | residual_k12 | 0.6881 | 0.7193 | 0.8754 | 0.5633 | 0.3908 | 0.1213 | 0.8944 | 0.8398 | 0.1042 | 495.2890 |
| 100 | residual_k24 | 0.6765 | 0.7087 | 0.8696 | 0.5477 | 0.4014 | 0.1257 | 0.8880 | 0.8325 | 0.1005 | 505.6997 |
| 100 | all_groups_fast_gate | 0.6558 | 0.6896 | 0.8584 | 0.5208 | 0.4192 | 0.1332 | 0.8681 | 0.8165 | 0.0896 | 521.6268 |
| 100 | residual_k48 | 0.6534 | 0.6879 | 0.8600 | 0.5157 | 0.4204 | 0.1335 | 0.8691 | 0.8168 | 0.0929 | 523.6376 |
| 100 | expert_tags | 0.6534 | 0.6858 | 0.8480 | 0.5236 | 0.4242 | 0.1356 | 0.8569 | 0.8087 | 0.0779 | 525.7768 |
| 100 | all_groups_balanced | 0.6449 | 0.6799 | 0.8550 | 0.5049 | 0.4278 | 0.1367 | 0.8603 | 0.8102 | 0.0884 | 531.7659 |
| 100 | all_groups_strong_shrinkage | 0.6421 | 0.6774 | 0.8539 | 0.5009 | 0.4301 | 0.1376 | 0.8582 | 0.8085 | 0.0882 | 534.3347 |
| 100 | rasch | 0.6407 | 0.6761 | 0.8534 | 0.4989 | 0.4313 | 0.1381 | 0.8571 | 0.8076 | 0.0881 | 535.6861 |
| 100 | semantic_fasttext_k48 | 0.6405 | 0.6761 | 0.8540 | 0.4982 | 0.4309 | 0.1380 | 0.8573 | 0.8077 | 0.0881 | 532.2730 |
| 100 | semantic_fasttext_k12 | 0.6404 | 0.6759 | 0.8532 | 0.4985 | 0.4312 | 0.1381 | 0.8566 | 0.8071 | 0.0874 | 533.2917 |
| 100 | semantic_fasttext_k24 | 0.6403 | 0.6760 | 0.8543 | 0.4977 | 0.4308 | 0.1379 | 0.8572 | 0.8076 | 0.0879 | 530.1760 |

## Interpretation

- At q=10, `excel_difficulty_only` has the highest unknown-priority balanced accuracy (0.7818), improving over Rasch by 0.1554.
- At q=20, `excel_difficulty_only` has the highest unknown-priority balanced accuracy (0.7818), improving over Rasch by 0.1605.
- At q=50, `excel_difficulty_only` has the highest unknown-priority balanced accuracy (0.7818), improving over Rasch by 0.1565.
- At q=100, `excel_difficulty_only` has the highest unknown-priority balanced accuracy (0.7818), improving over Rasch by 0.1411.
- Residual-response groups are expected to perform strongly here because they are derived from the same response matrix. Treat those results as an optimistic upper-bound for learned behavioral groupings unless groups are rebuilt on a separate calibration sample.
- FastText semantic clusters are locally trained from the provided vocabulary rather than a large pretrained corpus. They still satisfy the embedding-based grouping path, but production-quality semantic groups should use pretrained English fastText vectors.
- Accuracy is less informative than unknown-priority balanced accuracy, log loss, Brier score, calibration error, and vocabulary-size error for this app because calibrated probabilities are the output.
=======
| q | model | log_loss | brier | auc | accuracy | balanced_accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | residual_k12_seed701_tau1p6_g12_shrunk | 0.3700 | 0.1121 | 0.9150 | 0.8584 | 0.8303 | 0.1339 | 1169.5088 |
| 100 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.3477 | 0.1039 | 0.9267 | 0.8653 | 0.8513 | 0.1284 | 1165.2375 |
| 1000 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.2881 | 0.0874 | 0.9418 | 0.8738 | 0.8806 | 0.0978 | 965.8808 |

## Full aggregate results

| q | model | log_loss | brier | auc | accuracy | balanced_accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | residual_k12_seed701_tau1p6_g12_shrunk | 0.3700 | 0.1121 | 0.9150 | 0.8584 | 0.8303 | 0.1339 | 1169.5088 |
| 50 | residual_k12_seed701_tau1p4_g8_shrunk | 0.3754 | 0.1139 | 0.9149 | 0.8580 | 0.8288 | 0.1378 | 1194.8026 |
| 50 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.3817 | 0.1162 | 0.9142 | 0.8565 | 0.8274 | 0.1408 | 1222.0365 |
| 50 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.3660 | 0.1111 | 0.9144 | 0.8587 | 0.8273 | 0.1292 | 1146.6349 |
| 50 | residual_k12_bal_tau1p0_g20_ba | 0.4260 | 0.1328 | 0.9075 | 0.8334 | 0.8093 | 0.1609 | 1411.4513 |
| 50 | residual_k8_bal_tau1p0_g20_shrunk | 0.4277 | 0.1334 | 0.9019 | 0.8335 | 0.8083 | 0.1638 | 1433.7804 |
| 50 | residual_k8_std_tau1p0_g20_ba | 0.3726 | 0.1150 | 0.9033 | 0.8313 | 0.7979 | 0.1041 | 423.7455 |
| 50 | excel_difficulty_only_observed_ba_threshold | 0.5833 | 0.1995 | 0.8571 | 0.7559 | 0.7627 | 0.2235 | 2419.7424 |
| 50 | excel_difficulty_only_fixed | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.7587 | 0.2235 | 2419.7424 |
| 50 | rasch_balanced_fit_ba_threshold | 0.4322 | 0.1397 | 0.8571 | 0.7731 | 0.7528 | 0.0906 | 477.8705 |
| 50 | rasch_balanced_fit | 0.4322 | 0.1397 | 0.8571 | 0.7981 | 0.6484 | 0.0906 | 477.8705 |
| 50 | rasch_std_fixed | 0.4322 | 0.1397 | 0.8571 | 0.7981 | 0.6484 | 0.0906 | 477.8705 |
| 100 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.3477 | 0.1039 | 0.9267 | 0.8653 | 0.8513 | 0.1284 | 1165.2375 |
| 100 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.3367 | 0.1008 | 0.9262 | 0.8670 | 0.8504 | 0.1180 | 1104.9947 |
| 100 | residual_k12_seed701_tau1p4_g8_shrunk | 0.3450 | 0.1031 | 0.9246 | 0.8669 | 0.8494 | 0.1257 | 1136.6683 |
| 100 | residual_k12_seed701_tau1p6_g12_shrunk | 0.3388 | 0.1014 | 0.9241 | 0.8675 | 0.8485 | 0.1198 | 1102.1642 |
| 100 | residual_k12_bal_tau1p0_g20_ba | 0.3815 | 0.1151 | 0.9279 | 0.8574 | 0.8467 | 0.1553 | 1321.9263 |
| 100 | residual_k8_bal_tau1p0_g20_shrunk | 0.3991 | 0.1226 | 0.9142 | 0.8449 | 0.8384 | 0.1589 | 1418.5275 |
| 100 | residual_k8_std_tau1p0_g20_ba | 0.3323 | 0.0998 | 0.9170 | 0.8484 | 0.8335 | 0.0898 | 289.3280 |
| 100 | excel_difficulty_only_observed_ba_threshold | 0.5833 | 0.1995 | 0.8572 | 0.7575 | 0.7677 | 0.2235 | 2410.6325 |
| 100 | rasch_balanced_fit_ba_threshold | 0.4284 | 0.1380 | 0.8572 | 0.7703 | 0.7625 | 0.0860 | 372.7637 |
| 100 | excel_difficulty_only_fixed | 0.5833 | 0.1995 | 0.8572 | 0.6921 | 0.7587 | 0.2235 | 2410.6325 |
| 100 | rasch_balanced_fit | 0.4284 | 0.1380 | 0.8572 | 0.8026 | 0.6548 | 0.0860 | 372.7637 |
| 100 | rasch_std_fixed | 0.4284 | 0.1380 | 0.8572 | 0.8026 | 0.6548 | 0.0860 | 372.7637 |
| 1000 | residual_k12_seed1301_tau1p6_g8_shrunk | 0.2881 | 0.0874 | 0.9418 | 0.8738 | 0.8806 | 0.0978 | 965.8808 |
| 1000 | residual_k12_seed701_tau1p6_g12_shrunk | 0.2878 | 0.0874 | 0.9410 | 0.8726 | 0.8801 | 0.0971 | 960.0555 |
| 1000 | residual_k12_seed1301_tau1p4_g12_shrunk | 0.2912 | 0.0882 | 0.9416 | 0.8734 | 0.8798 | 0.0998 | 978.0995 |
| 1000 | residual_k12_seed701_tau1p4_g8_shrunk | 0.2908 | 0.0882 | 0.9407 | 0.8720 | 0.8793 | 0.0989 | 971.6537 |
| 1000 | residual_k12_bal_tau1p0_g20_ba | 0.2987 | 0.0897 | 0.9438 | 0.8728 | 0.8768 | 0.1039 | 995.8273 |
| 1000 | residual_k8_bal_tau1p0_g20_shrunk | 0.3470 | 0.1082 | 0.9224 | 0.8473 | 0.8511 | 0.1201 | 1201.7774 |
| 1000 | residual_k8_std_tau1p0_g20_ba | 0.2757 | 0.0839 | 0.9240 | 0.8550 | 0.8505 | 0.0397 | 81.9007 |
| 1000 | excel_difficulty_only_observed_ba_threshold | 0.5835 | 0.1996 | 0.8571 | 0.7612 | 0.7760 | 0.2237 | 2228.9392 |
| 1000 | rasch_balanced_fit_ba_threshold | 0.4239 | 0.1364 | 0.8571 | 0.7692 | 0.7715 | 0.0796 | 104.4894 |
| 1000 | excel_difficulty_only_fixed | 0.5835 | 0.1996 | 0.8571 | 0.6920 | 0.7588 | 0.2237 | 2228.9392 |
| 1000 | rasch_balanced_fit | 0.4239 | 0.1364 | 0.8571 | 0.8053 | 0.6533 | 0.0796 | 104.4894 |
| 1000 | rasch_std_fixed | 0.4239 | 0.1364 | 0.8571 | 0.8053 | 0.6533 | 0.0796 | 104.4894 |

## Interpretation

- At q=50, `residual_k12_seed701_tau1p6_g12_shrunk` has the highest balanced accuracy (0.8303), improving over Rasch by 0.1819.
- At q=100, `residual_k12_seed1301_tau1p4_g12_shrunk` has the highest balanced accuracy (0.8513), improving over Rasch by 0.1965.
- At q=1000, `residual_k12_seed1301_tau1p6_g8_shrunk` has the highest balanced accuracy (0.8806), improving over Rasch by 0.2273.
- Ranking in this report is by balanced accuracy first (higher is better), then log loss.
- Residual-response groups are expected to perform strongly here because they are derived from the same response matrix. Treat those results as an optimistic upper-bound for learned behavioral groupings unless groups are rebuilt on a separate calibration sample.
- Balanced-accuracy-oriented threshold optimization can degrade calibration and probability quality. Use log loss and Brier score as guardrails when choosing production settings.
>>>>>>> 2fb8341 (Improve residual IRT balanced accuracy for q50/q100 and add q1000 eval)
