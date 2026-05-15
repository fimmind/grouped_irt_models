# Grouped residual IRT model comparison

## Setup

- Evaluation vocabulary: 11,952/11,996 Ehara word strings had Excel difficulty coverage.
- Binary label: Ehara `raw_score >= 4` is treated as known; scores 1-3 are treated as unknown/uncertain.
- Users: 16.
- Cold-start reveal counts: 10, 20, 50, 100.
- Repeats per user and reveal count: 3.
- Difficulty: Excel `Words.accuracy` converted to Rasch difficulty with clipped `-logit(accuracy)`, then standardized.
- FastText: local skipgram model trained from the available vocabulary text, then clustered with KMeans.
- Fitting objective: asymmetric BCE with `known_loss_weight=0.9` and `unknown_loss_weight=1.1` to penalize false known predictions more than false unknown predictions.
- Benchmark focus: unknown-priority balanced accuracy `0.4 * known_recall + 0.6 * unknown_recall`.

## Model family

All grouped models use the spec's residual form:

`sigmoid(theta - b_i + lambda_q * Q_i @ delta_u)`

The optimizer first fits Rasch-only `theta`, then jointly fits `theta` and group residuals with Gaussian priors. `lambda_q = q / (q + c)` and `tau_delta < tau_theta` shrink group effects during cold start.

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

## Best model by reveal count (unknown-priority balanced accuracy)

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
