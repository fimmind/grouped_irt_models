# Best Cross-Dataset Model: `response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk`

This specification defines the grouped residual IRT model variant that currently gives the best overall transfer profile across both datasets (strongest on Ehara, and top-tier on Duolingo).

## 1) Data contract

Input must be a binary user-item matrix `Y` of shape `(U, I)` and an item difficulty vector `b` of shape `(I,)`.

- `Y[u, i] = 1` means user `u` knows item `i`.
- `Y[u, i] = 0` means unknown.
- `b[i]` is standardized Rasch difficulty for item `i`.

### Ehara labels/difficulty

- Known label: `raw_score >= 4`.
- Difficulty source: Excel `Words.accuracy`.
- Transform: `acc_clipped = clip(acc, 1e-4, 1 - 1e-4)`, `b_raw = -logit(acc_clipped)`, `b = standardize(b_raw)`.

### Duolingo labels/difficulty (optimized setup)

- Aggregate events to `(user_id, lexeme_id)` with `mean_p_recall`.
- Known label: `mean_p_recall >= 0.8`.
- Build dense matrix by lexeme-majority imputation for missing cells.
- Difficulty source: mean `p_recall` per lexeme.
- Same transform: clipped `-logit`, then standardize.

## 2) Group construction (`response12`)

Build 12 soft item groups from response patterns only:

1. Let `X = Y^T` (item-by-user matrix).
2. Mean-center each item row: `Xc[i] = X[i] - mean(X[i])`.
3. L2-normalize each row (if norm 0, use 1).
4. Run KMeans with `k=12`, `n_init=20`, fixed seed.
5. Compute cosine similarities between normalized item vectors and normalized cluster centers.
6. For each item:
   - take top-3 similar clusters,
   - softmax their similarities with temperature `6.0`,
   - assign those weights,
   - ensure hard cluster gets at least `0.5` weight,
   - renormalize row to sum 1.

This yields group matrix `Q` of shape `(I, 12)`.

## 3) Model form

For user `u`, item `i`:

- Residual term: `r_i = Q[i] dot delta_u`
- Gate: `lambda_q = q / (q + c)`
- Logit: `z_i = theta_u - b_i + lambda_q * r_i`
- Probability: `p_i = sigmoid(z_i)`

Parameters per user:

- scalar ability `theta_u`
- group vector `delta_u` in `R^12`

Fixed hyperparameters for this model:

- `tau_theta = 2.0`
- `tau_delta = 1.6`
- `c = 12.0`
- fit strategy: class-weighted
- threshold strategy: observed BA optimum (shrunk)

## 4) Per-user MAP fitting

Given observed items `O` (`|O| = q`), minimize:

`NLL + prior`

where

- weighted Bernoulli NLL on observed answers
- Gaussian priors:
  - `theta^2 / (2 * tau_theta^2)`
  - `||delta||^2 / (2 * tau_delta^2)`

### Class weighting

Let `y` be observed labels.

- `pos_rate = clip(mean(y), 0.05, 0.95)`
- `w_pos = 0.5 / pos_rate`
- `w_neg = 0.5 / (1 - pos_rate)`
- sample weight `w = y*w_pos + (1-y)*w_neg`

### Optimization procedure

1. Initialize `theta` by 1D Rasch MAP fit (with `delta=0`).
2. Initialize `delta = 0`.
3. Optimize jointly with L-BFGS-B using analytic gradients.
4. Use gate `lambda_q = q / (q + 12.0)`.

## 5) Threshold selection for binary prediction

After fitting each user on observed set:

1. Compute observed probabilities.
2. Search thresholds on `[0.10, 0.90]` with step `0.005`.
3. Pick threshold maximizing observed balanced accuracy.
4. If tie, choose value closest to `0.50`.
5. Shrink toward 0.5:
   - `shrink = q / (q + 30)`
   - `t_final = 0.5 + shrink * (t_opt - 0.5)`

Use `t_final` for hidden-item classification.

## 6) Evaluation protocol

For each user and each `q` (typically 50, 100, 1000):

1. Randomly permute items.
2. First `q` -> observed; rest -> hidden.
3. Fit per-user MAP on observed.
4. Predict hidden probabilities and hidden labels.
5. Aggregate metrics across users and repeats.

Metrics to report:

- balanced accuracy (primary)
- PR-AUC
- ROC-AUC
- accuracy
- log loss
- Brier score

## 7) Determinism requirements

- Fix all RNG seeds for:
  - split permutations
  - KMeans initialization
  - any embedding/grouping randomness
- Keep identical `q`, repeats, and preprocessing when comparing models.

## 8) Minimal implementation checklist

1. Build dense binary matrix `Y` and difficulty `b`.
2. Build `Q = response12` with soft top-3 memberships.
3. For each user split:
   - fit `(theta, delta)` with class-weighted MAP,
   - choose shrunk BA-opt threshold,
   - score hidden items.
4. Aggregate metrics over users/repeats.
5. Rank by balanced accuracy.

## 9) Reproducibility and debugging guide (when results drop to ~0.6)

If a reimplementation reports around `0.6`, the most common cause is protocol mismatch, not model failure.

### 9.1 Check metric identity first

- Primary metric is **balanced accuracy**, not plain accuracy.
- If only plain accuracy is reported, values are not comparable to this spec.

### 9.2 Verify dataset-specific labeling exactly

- Ehara: `known = (raw_score >= 4)`.
- Duolingo optimized run: `known = (mean p_recall per user-lexeme >= 0.8)`.
- Using `0.5` instead of `0.8` on Duolingo can strongly change class balance and BA.

### 9.3 Verify densification rule

- For Duolingo optimized setup: missing cells must be filled by **lexeme-majority** label.
- Different fill strategies (user-majority, blend, or dropping sparse cells) produce different results.

### 9.4 Verify exact data slice

- Duolingo reference slice: `en->es`, `top_users=120`, `top_lexemes=1200`.
- Different language pair/top-k is a different experiment.

### 9.5 Verify threshold strategy

Must use `observed_ba_opt_shrunk`:

1. Search threshold on observed items in `[0.10, 0.90]` with step `0.005`.
2. Maximize observed balanced accuracy.
3. Tie-break by closest to `0.50`.
4. Shrink to:
   - `shrink = q / (q + 30)`
   - `t_final = 0.5 + shrink * (t_opt - 0.5)`

Using fixed `0.5` is a common reason BA drops.

### 9.6 Verify `response12` group construction details

All of these must match:

- KMeans: `k=12`, `n_init=20`, fixed seed.
- Item profiles: mean-centered and L2-normalized.
- Soft assignment over **top-3** clusters.
- Softmax temperature: `6.0`.
- Hard-cluster minimum membership: `>= 0.5`.
- Final per-item row normalization to sum 1.

### 9.7 Verify fitting objective and hyperparameters

- Fit strategy: `class_weighted`.
- Priors:
  - `tau_theta = 2.0`
  - `tau_delta = 1.6`
- Gate:
  - `c = 12.0`
  - `lambda_q = q / (q + c)`
- Optimizer: L-BFGS-B with analytic gradients.
- Initialization: Rasch-only 1D MAP for `theta`, `delta = 0`.

### 9.8 Verify split protocol

For each user and each repeat:

1. Random permutation of items.
2. First `q` observed, remaining hidden.
3. Fit on observed only.
4. Evaluate on hidden only.

Any leakage or different split logic can distort results.

### 9.9 Recommended sanity checks

- Confirm `Q.shape == (I, 12)` and each row sums to `1.0 ± 1e-6`.
- Confirm class weighting values are finite and positive.
- Confirm optimized threshold is usually not exactly `0.5` for small `q`.
- Confirm metric aggregation is over all user-repeat hidden predictions.

### 9.10 Expected reference ranges (for correct implementation)

On the latest validated runs:

- Ehara (`response12_g12_tau1p6_c12p0_observed_ba_opt_shrunk`):
  - `q=50` BA ~ `0.83`
  - `q=100` BA ~ `0.85`
  - `q=1000` BA ~ `0.89`

- Duolingo optimized setup (`en->es`, `120x1200`, threshold `0.8`):
  - Best models BA ~ `0.84` (`q=50`)
  - BA ~ `0.85` (`q=100`)
  - BA ~ `0.86` (`q=1000`)

If observed BA is near `0.6`, start by checking Sections 9.1-9.6 in order.
