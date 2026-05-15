# Grouped residual IRT model comparison

## Setup

- Evaluation vocabulary: 11,952/11,996 Ehara word strings had Excel difficulty coverage.
- Binary label: Ehara `raw_score >= 4` is treated as known; scores 1-3 are treated as unknown/uncertain.
- Users: 16.
- Cold-start reveal counts: 10, 20, 50, 100.
- Repeats per user and reveal count: 3.
- Difficulty: Excel `Words.accuracy` converted to Rasch difficulty with clipped `-logit(accuracy)`, then standardized.
- FastText: local skipgram model trained from the available vocabulary text, then clustered with KMeans.

## Model family

All grouped models use the spec's residual form:

`sigmoid(theta - b_i + lambda_q * Q_i @ delta_u)`

The optimizer first fits Rasch-only `theta`, then jointly fits `theta` and group residuals with Gaussian priors. `lambda_q = q / (q + c)` and `tau_delta < tau_theta` shrink group effects during cold start.

| model | groups | tau_theta | tau_delta | gate_c |
| --- | --- | --- | --- | --- |
| excel_difficulty_only | 0 | n/a | n/a | n/a |
| rasch | 0 | 2.0000 | 0.5000 | 50.0000 |
| semantic_fasttext_k12 | 12 | 2.0000 | 0.5500 | 50.0000 |
| semantic_fasttext_k24 | 24 | 2.0000 | 0.5500 | 50.0000 |
| semantic_fasttext_k48 | 48 | 2.0000 | 0.4500 | 70.0000 |
| expert_tags | 20 | 2.0000 | 0.5500 | 50.0000 |
| residual_k12 | 12 | 2.0000 | 0.5500 | 50.0000 |
| residual_k24 | 24 | 2.0000 | 0.5500 | 50.0000 |
| residual_k48 | 48 | 2.0000 | 0.4500 | 70.0000 |
| all_groups_balanced | 68 | 2.0000 | 0.4500 | 50.0000 |
| all_groups_fast_gate | 68 | 2.0000 | 0.7000 | 20.0000 |
| all_groups_strong_shrinkage | 68 | 2.0000 | 0.3500 | 100.0000 |

## Best model by reveal count

| q | model | log_loss | brier | auc | accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | residual_k12 | 0.4674 | 0.1530 | 0.8575 | 0.7767 | 0.1272 | 1123.5411 |
| 20 | residual_k12 | 0.4427 | 0.1440 | 0.8592 | 0.7904 | 0.1069 | 804.6529 |
| 50 | residual_k12 | 0.4206 | 0.1345 | 0.8722 | 0.8102 | 0.1001 | 516.7472 |
| 100 | residual_k12 | 0.3889 | 0.1221 | 0.8954 | 0.8315 | 0.1011 | 325.3910 |

## Mathematical descriptions of the top models

### `residual_k12`

This was the best model at every tested reveal count. It is a Rasch model with 12 learned residual-response groups:

```text
P(y_{u,i}=1 | theta_u, delta_u) =
  sigmoid(theta_u - b_i + lambda_q * sum_{k=1}^{12} Q_{i,k} * delta_{u,k})
```

where:

- `u` is a learner and `i` is a word.
- `b_i = standardized(-logit(accuracy_i))` is the global word difficulty from the Excel accuracy table.
- `theta_u` is the learner's general vocabulary ability.
- `Q_{i,k}` is the soft membership of word `i` in residual-response group `k`.
- `delta_{u,k}` is learner `u`'s residual strength or weakness in group `k`.
- `lambda_q = q / (q + 50)` gates the grouped term by the number of revealed answers.

The residual groups are built from the response matrix rather than from word form alone. First, a full-data Rasch estimate is fitted for each learner:

```text
p_hat_{u,i} = sigmoid(theta_hat_u - b_i)
r_{u,i} = y_{u,i} - p_hat_{u,i}
```

Each word is represented by its centered residual profile across the 16 learners:

```text
z_i = normalize(r_{1:U,i} - mean_u(r_{u,i}))
```

KMeans clusters these word residual profiles into 12 clusters. Soft memberships are then computed from cosine similarity to cluster centers, keeping the top 3 memberships per word and normalizing each row of `Q`.

For a learner with observed answers `O_u`, the fitted parameters are the MAP solution:

```text
argmin_{theta_u, delta_u}
  sum_{(i,y_i) in O_u} BCE(y_i, sigmoid(theta_u - b_i + lambda_q * Q_i @ delta_u))
  + theta_u^2 / (2 * 2.0^2)
  + ||delta_u||_2^2 / (2 * 0.55^2)
```

This model performed best because the groups capture words that learners know or miss together after controlling for global difficulty.

### `residual_k24`

This is the same model form as `residual_k12`, but with 24 residual-response groups:

```text
P(y_{u,i}=1 | theta_u, delta_u) =
  sigmoid(theta_u - b_i + lambda_q * sum_{k=1}^{24} Q_{i,k} * delta_{u,k})
```

with the same priors and gate:

```text
theta_u ~ N(0, 2.0^2)
delta_{u,k} ~ N(0, 0.55^2)
lambda_q = q / (q + 50)
```

It was consistently second-best by log loss. The extra groups add resolution, but with only 10-100 revealed answers per learner they also increase the number of residual parameters. The stronger empirical result for `residual_k12` suggests that 12 groups are a better bias-variance tradeoff for this small 16-user evaluation set.

### `all_groups_fast_gate`

This was the strongest non-pure-residual model at low reveal counts and combines three group systems:

```text
Q_all = row_normalize([Q_semantic_fasttext_k24, Q_residual_k24, Q_expert_tags])
```

The prediction equation is:

```text
P(y_{u,i}=1 | theta_u, delta_u) =
  sigmoid(theta_u - b_i + lambda_q * Q_all_i @ delta_u)
```

with a faster gate and weaker shrinkage:

```text
theta_u ~ N(0, 2.0^2)
delta_{u,k} ~ N(0, 0.70^2)
lambda_q = q / (q + 20)
```

This makes group deviations active earlier than in the residual-only models. It helped relative to most semantic and expert-tag variants, but underperformed `residual_k12` because the combined 68-dimensional group vector is more parameter-heavy and includes weaker group sources.

## Full aggregate results

| q | model | log_loss | brier | auc | accuracy | calibration_error | vocabulary_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | residual_k12 | 0.4674 | 0.1530 | 0.8575 | 0.7767 | 0.1272 | 1123.5411 |
| 10 | residual_k24 | 0.4674 | 0.1531 | 0.8574 | 0.7766 | 0.1272 | 1123.5886 |
| 10 | all_groups_fast_gate | 0.4675 | 0.1531 | 0.8573 | 0.7766 | 0.1271 | 1123.7550 |
| 10 | residual_k48 | 0.4676 | 0.1531 | 0.8572 | 0.7765 | 0.1271 | 1123.7411 |
| 10 | expert_tags | 0.4676 | 0.1531 | 0.8571 | 0.7765 | 0.1271 | 1123.7883 |
| 10 | all_groups_balanced | 0.4676 | 0.1532 | 0.8572 | 0.7765 | 0.1271 | 1123.7675 |
| 10 | all_groups_strong_shrinkage | 0.4677 | 0.1532 | 0.8572 | 0.7765 | 0.1271 | 1123.7706 |
| 10 | semantic_fasttext_k12 | 0.4677 | 0.1532 | 0.8571 | 0.7765 | 0.1271 | 1123.8862 |
| 10 | semantic_fasttext_k24 | 0.4677 | 0.1532 | 0.8571 | 0.7765 | 0.1271 | 1123.8901 |
| 10 | semantic_fasttext_k48 | 0.4677 | 0.1532 | 0.8571 | 0.7765 | 0.1271 | 1123.7812 |
| 10 | rasch | 0.4677 | 0.1532 | 0.8571 | 0.7765 | 0.1271 | 1123.7718 |
| 10 | excel_difficulty_only | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2428.1332 |
| 20 | residual_k12 | 0.4427 | 0.1440 | 0.8592 | 0.7904 | 0.1069 | 804.6529 |
| 20 | residual_k24 | 0.4431 | 0.1441 | 0.8587 | 0.7900 | 0.1067 | 804.9056 |
| 20 | all_groups_fast_gate | 0.4436 | 0.1443 | 0.8580 | 0.7895 | 0.1062 | 805.4248 |
| 20 | residual_k48 | 0.4441 | 0.1445 | 0.8576 | 0.7892 | 0.1062 | 805.6477 |
| 20 | expert_tags | 0.4443 | 0.1446 | 0.8571 | 0.7890 | 0.1059 | 806.4631 |
| 20 | all_groups_balanced | 0.4445 | 0.1447 | 0.8572 | 0.7890 | 0.1061 | 806.3027 |
| 20 | all_groups_strong_shrinkage | 0.4446 | 0.1447 | 0.8571 | 0.7889 | 0.1061 | 806.4134 |
| 20 | semantic_fasttext_k24 | 0.4446 | 0.1447 | 0.8571 | 0.7888 | 0.1062 | 806.1034 |
| 20 | semantic_fasttext_k12 | 0.4446 | 0.1447 | 0.8571 | 0.7888 | 0.1061 | 806.5145 |
| 20 | semantic_fasttext_k48 | 0.4446 | 0.1447 | 0.8571 | 0.7889 | 0.1062 | 806.4426 |
| 20 | rasch | 0.4446 | 0.1447 | 0.8571 | 0.7889 | 0.1062 | 806.4333 |
| 20 | excel_difficulty_only | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2426.1284 |
| 50 | residual_k12 | 0.4206 | 0.1345 | 0.8722 | 0.8102 | 0.1001 | 516.7472 |
| 50 | residual_k24 | 0.4239 | 0.1359 | 0.8679 | 0.8071 | 0.0985 | 519.7320 |
| 50 | all_groups_fast_gate | 0.4293 | 0.1382 | 0.8616 | 0.8024 | 0.0952 | 523.7993 |
| 50 | residual_k48 | 0.4304 | 0.1386 | 0.8611 | 0.8018 | 0.0959 | 526.3950 |
| 50 | expert_tags | 0.4319 | 0.1393 | 0.8572 | 0.7996 | 0.0923 | 524.7100 |
| 50 | all_groups_balanced | 0.4332 | 0.1397 | 0.8581 | 0.7998 | 0.0949 | 529.2580 |
| 50 | all_groups_strong_shrinkage | 0.4340 | 0.1400 | 0.8574 | 0.7993 | 0.0948 | 530.3204 |
| 50 | semantic_fasttext_k24 | 0.4341 | 0.1401 | 0.8572 | 0.7993 | 0.0951 | 530.2896 |
| 50 | semantic_fasttext_k48 | 0.4342 | 0.1401 | 0.8572 | 0.7991 | 0.0949 | 530.0896 |
| 50 | rasch | 0.4343 | 0.1402 | 0.8571 | 0.7991 | 0.0949 | 530.7139 |
| 50 | semantic_fasttext_k12 | 0.4343 | 0.1402 | 0.8571 | 0.7991 | 0.0949 | 530.2716 |
| 50 | excel_difficulty_only | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2420.9244 |
| 100 | residual_k12 | 0.3889 | 0.1221 | 0.8954 | 0.8315 | 0.1011 | 325.3910 |
| 100 | residual_k24 | 0.3964 | 0.1253 | 0.8876 | 0.8246 | 0.0969 | 325.7277 |
| 100 | residual_k48 | 0.4141 | 0.1325 | 0.8704 | 0.8101 | 0.0894 | 336.9991 |
| 100 | all_groups_fast_gate | 0.4150 | 0.1328 | 0.8683 | 0.8101 | 0.0866 | 340.0080 |
| 100 | expert_tags | 0.4209 | 0.1354 | 0.8571 | 0.8046 | 0.0756 | 347.3316 |
| 100 | all_groups_balanced | 0.4240 | 0.1365 | 0.8605 | 0.8035 | 0.0855 | 343.1175 |
| 100 | all_groups_strong_shrinkage | 0.4264 | 0.1374 | 0.8583 | 0.8019 | 0.0852 | 343.9088 |
| 100 | semantic_fasttext_k24 | 0.4274 | 0.1378 | 0.8573 | 0.8011 | 0.0851 | 346.6214 |
| 100 | semantic_fasttext_k48 | 0.4275 | 0.1379 | 0.8574 | 0.8010 | 0.0851 | 344.8397 |
| 100 | semantic_fasttext_k12 | 0.4277 | 0.1380 | 0.8566 | 0.8009 | 0.0844 | 342.1778 |
| 100 | rasch | 0.4277 | 0.1380 | 0.8571 | 0.8010 | 0.0850 | 344.3344 |
| 100 | excel_difficulty_only | 0.5833 | 0.1995 | 0.8571 | 0.6921 | 0.2235 | 2409.8019 |

## Interpretation

- At q=10, `residual_k12` has the lowest log loss (0.4674), improving over Rasch by 0.0003.
- At q=20, `residual_k12` has the lowest log loss (0.4427), improving over Rasch by 0.0019.
- At q=50, `residual_k12` has the lowest log loss (0.4206), improving over Rasch by 0.0137.
- At q=100, `residual_k12` has the lowest log loss (0.3889), improving over Rasch by 0.0388.
- Residual-response groups are expected to perform strongly here because they are derived from the same response matrix. Treat those results as an optimistic upper-bound for learned behavioral groupings unless groups are rebuilt on a separate calibration sample.
- FastText semantic clusters are locally trained from the provided vocabulary rather than a large pretrained corpus. They still satisfy the embedding-based grouping path, but production-quality semantic groups should use pretrained English fastText vectors.
- Accuracy is less informative than log loss, Brier score, calibration error, and vocabulary-size error for this app because calibrated probabilities are the output.
