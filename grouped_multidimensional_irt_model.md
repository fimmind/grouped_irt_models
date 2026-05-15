# Grouped Multidimensional IRT for Vocabulary Knowledge Estimation

## 1. Purpose

This document describes a vocabulary knowledge estimation model that extends classical Rasch/IRT by tracking not only a learner's general vocabulary level, but also their strengths and weaknesses across semantically or functionally meaningful word groups.

The intended use case is an adaptive vocabulary estimation app. The app asks a learner a relatively small number of known/unknown questions and estimates:

1. the probability that the learner knows each word or word family;
2. the learner's total expected vocabulary size;
3. the learner's strengths across semantic, domain, register, and morphological groups;
4. which words should be asked next to reduce uncertainty efficiently.

The model is designed for cold-start use: it should work after only 10-50 user answers, while remaining extensible to richer user histories later.

## 2. Baseline: Rasch Vocabulary Model

The classical Rasch model represents each learner by a single scalar ability and each word by a single scalar difficulty.

For learner \(u\) and word \(i\), define:

\[
\theta_u \in \mathbb{R}
\]

as the learner's general vocabulary ability, and

\[
b_i \in \mathbb{R}
\]

as the difficulty of word \(i\). Larger \(b_i\) means the word is harder.

The probability that learner \(u\) knows word \(i\) is

\[
P(y_{u,i}=1)=\sigma(\theta_u-b_i),
\]

where

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

This model is robust and interpretable, but it assumes that vocabulary knowledge is essentially one-dimensional. That assumption is too restrictive. A learner may know scientific, academic, or programming vocabulary above their general level, while being weaker on domestic, informal, literary, or phrasal-verb vocabulary.

## 3. Main Extension: Grouped Multidimensional IRT

The proposed model keeps the Rasch backbone but adds group-specific learner deviations.

Each word belongs, possibly softly, to several groups. Examples of groups include:

- science;
- academic vocabulary;
- computing;
- business;
- food and cooking;
- emotions;
- animals;
- domestic life;
- literary vocabulary;
- slang or informal speech;
- Latinate/cognate-rich vocabulary;
- phrasal verbs;
- morphologically complex words.

Let

\[
q_{i,k}\in[0,1]
\]

denote the degree to which word \(i\) belongs to group \(k\). The matrix

\[
Q=(q_{i,k})
\]

is the word-group membership matrix.

For each learner \(u\), introduce group-specific deviations

\[
\delta_{u,k}\in\mathbb{R}.
\]

A positive \(\delta_{u,k}\) means the learner is stronger than expected in group \(k\), after controlling for general vocabulary level. A negative \(\delta_{u,k}\) means the learner is weaker than expected in that group.

The model is

\[
P(y_{u,i}=1)
=
\sigma\left(
\theta_u-b_i+
\lambda_q\sum_{k=1}^K q_{i,k}\delta_{u,k}
\right).
\]

Here:

- \(\theta_u\) is the learner's general vocabulary ability;
- \(b_i\) is the global difficulty of word \(i\);
- \(q_{i,k}\) is the membership of word \(i\) in group \(k\);
- \(\delta_{u,k}\) is the learner's strength or weakness in group \(k\);
- \(\lambda_q\in[0,1]\) controls how much the model trusts group-specific deviations after \(q\) observed answers.

This is best understood as a Rasch model plus a structured residual term.

## 4. Why Use a Gating Factor?

In cold start, group-specific estimates are noisy. If the learner answers one science word correctly, the model should not immediately conclude that the learner is strong in all science vocabulary.

Therefore the grouped term should be suppressed early and allowed to matter more as evidence accumulates.

A practical gate is

\[
\lambda_q
=
\sigma\left(\alpha(\log(1+q)-\tau)\right),
\]

where:

- \(q\) is the number of answered diagnostic items;
- \(\alpha\) controls how sharply the grouped model turns on;
- \(\tau\) controls the point at which the grouped model starts to matter.

For example, choose \(\tau\) so that group deviations are weak for the first 10-20 answers and become meaningful around 50-100 answers.

A simpler alternative is

\[
\lambda_q=\frac{q}{q+c},
\]

where \(c\) is a smoothing constant such as 30, 50, or 100.

## 5. Priors and Regularization

The model should be regularized strongly, especially in cold start.

Use Gaussian priors:

\[
\theta_u\sim N(0,\tau_\theta^2),
\]

\[
\delta_{u,k}\sim N(0,\tau_\delta^2),
\]

with

\[
\tau_\delta < \tau_\theta.
\]

This expresses the belief that general ability varies substantially, while group-specific deviations should usually be moderate unless there is strong evidence.

The MAP objective for a learner with observed answers

\[
O_u=\{(i_t,y_t)\}_{t=1}^q
\]

is

\[
\mathcal{L}(\theta_u,\delta_u)
=
\sum_{(i,y_i)\in O_u}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
-
\frac{\theta_u^2}{2\tau_\theta^2}
-
\sum_{k=1}^K\frac{\delta_{u,k}^2}{2\tau_\delta^2},
\]

where

\[
p_i=
\sigma\left(
\theta_u-b_i+
\lambda_q\sum_{k=1}^K q_{i,k}\delta_{u,k}
\right).
\]

## 6. Recommended Inference Procedure

For a production implementation, use a two-stage or warm-started optimization procedure.

### Stage 1: Rasch-only update

Estimate \(\theta_u\) using the Rasch model:

\[
P(y_{u,i}=1)=\sigma(\theta_u-b_i).
\]

This gives a stable global estimate even with few answers.

### Stage 2: Group residual update

Using the Rasch estimate as initialization, jointly update

\[
\theta_u,\delta_{u,1},\ldots,\delta_{u,K}.
\]

Use strong L2 regularization on \(\delta_u\). For small \(q\), either fix \(\delta_u=0\), use very strong shrinkage, or set \(\lambda_q\) near zero.

### Stage 3: Posterior uncertainty

Approximate uncertainty using the inverse Hessian at the MAP estimate, or use a diagonal approximation for speed.

For item selection, the most important quantity is uncertainty in \(\theta_u\) and in the group deviations relevant to candidate words.

## 7. Word Difficulty Calibration

The model requires word difficulties \(b_i\). These can be obtained in several ways.

### From empirical known rates

If a dataset gives the proportion \(r_i\) of learners who know word \(i\), convert it to Rasch difficulty by

\[
b_i=-\operatorname{logit}(r_i)
=-\log\frac{r_i}{1-r_i}.
\]

Use clipping to avoid infinities:

\[
r_i\leftarrow \min(1-\epsilon,\max(\epsilon,r_i)).
\]

### From frequency or prevalence ranks

If only frequency or prevalence rank is available, map ranks to a difficulty scale using a monotone transformation, for example:

\[
b_i = a\log(\operatorname{rank}_i)+c.
\]

The constants \(a,c\) should be calibrated against available known-rate data.

### Hybrid difficulty prior

For production, use a hybrid difficulty estimate:

\[
b_i
=
\rho_i b_i^{\text{emp}}
+(1-\rho_i)b_i^{\text{prior}},
\]

where

\[
\rho_i=\frac{n_i}{n_i+c}.
\]

Here:

- \(b_i^{\text{emp}}\) is empirical difficulty from user responses;
- \(b_i^{\text{prior}}\) is a prior from frequency, prevalence, CEFR level, word length, morphology, embeddings, and other features;
- \(n_i\) is the number of observed responses for word \(i\);
- \(c\) controls shrinkage.

This allows the system to handle both well-calibrated words and new or rare words.

## 8. How to Define Word Groups

The model should not depend on a single hard partition of vocabulary. Instead, define several overlapping group systems and combine them into one membership matrix \(Q\).

### 8.1 Base unit: lemma + POS or word family

Use either:

\[
(\text{lemma},\text{POS})
\]

or word families as items.

For recognition testing, lemma + POS is often a good compromise. It distinguishes major homographs such as noun/verb uses, but avoids full word-sense disambiguation.

Word families should be used for reporting vocabulary size and for avoiding redundant questions.

### 8.2 Embedding-based semantic clusters

Use pretrained embeddings to create semantic groups.

Recommended approach:

1. choose a vocabulary list;
2. map each word or lemma+POS to an embedding;
3. cluster nouns, verbs, adjectives, and adverbs separately;
4. use spherical k-means or graph clustering;
5. target roughly 30-100 semantic clusters for a large vocabulary;
6. represent cluster membership softly if possible.

For a first implementation, use about 50 semantic clusters.

### 8.3 Residual-response clusters

If user response data are available, build clusters from residual correlations.

First fit Rasch:

\[
\hat{p}_{u,i}=\sigma(\hat\theta_u-b_i).
\]

Compute residuals:

\[
r_{u,i}=y_{u,i}-\hat{p}_{u,i}.
\]

Then estimate word-word residual correlations. Words with positive residual correlation are words that tend to be known or unknown together beyond what general ability predicts.

Build a graph from these correlations and cluster it. These clusters are often more useful than purely semantic clusters because they capture actual learner behavior.

### 8.4 Expert and resource-based tags

Add interpretable tags from external resources:

- CEFR level;
- Academic Word List membership;
- domain labels such as science, medicine, law, business, computing, religion, sports;
- register labels such as formal, informal, literary, archaic, slang;
- morphological tags such as compound, derived form, prefix/suffix family;
- phrasal verb indicator;
- cognate-likelihood relative to the learner's L1, if known.

### 8.5 Book-specific or corpus-specific groups

For task-based vocabulary learning, build groups from a target book or domain corpus.

A useful approach is to build a lexical graph where:

- nodes are word families or lemma+POS items;
- edges connect semantically similar words;
- edge weights come from embedding cosine similarity, co-occurrence, or residual correlation;
- graph communities become groups;
- central nodes become good diagnostic candidates.

This is useful when the learner wants to prepare for a specific book, course, domain, or exam.

## 9. Example Membership Matrix

A word may belong to several groups at once.

| Word | Science | Academic | Cognate-like | Everyday | Emotion | Computing |
|---|---:|---:|---:|---:|---:|---:|
| oxygen | 0.9 | 0.5 | 0.8 | 0.1 | 0.0 | 0.0 |
| password | 0.1 | 0.0 | 0.1 | 0.6 | 0.0 | 0.9 |
| melancholy | 0.0 | 0.6 | 0.7 | 0.1 | 0.9 | 0.0 |
| kettle | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 |

This lets the model infer patterns such as:

- the learner is strong in academic/cognate vocabulary;
- the learner is weak in domestic everyday vocabulary;
- the learner knows computing words above their general level;
- the learner knows literary emotion words below their general level.

## 10. Prediction and Vocabulary Size Estimation

After fitting the learner parameters, predict word knowledge by

\[
\hat{p}_{u,i}=P(y_{u,i}=1).
\]

Do not estimate vocabulary size by thresholding words at \(\hat{p}_{u,i}\geq 0.5\). Instead, compute expected vocabulary size.

For word-level vocabulary:

\[
\widehat{V}_u
=
\sum_{i\in W}\hat{p}_{u,i}.
\]

For word-family vocabulary:

\[
\widehat{V}_u
=
\sum_{f\in F}P(\text{learner knows family } f).
\]

A simple family-level estimate is

\[
P(\text{knows family } f)
=
\max_{i\in f}\hat{p}_{u,i},
\]

or, more conservatively,

\[
P(\text{knows family } f)
=1-\prod_{i\in f}(1-\hat{p}_{u,i}).
\]

The correct choice depends on how "knowing a word family" is defined.

## 11. Adaptive Question Selection

The app should choose questions that reduce uncertainty quickly.

For Rasch, item information is

\[
I_i(\theta)=p_i(1-p_i).
\]

This is largest when

\[
p_i\approx 0.5,
\]

meaning the word is near the learner's estimated level.

However, in the grouped model, we also want coverage across semantic groups. A practical scoring function is

\[
S(i)
=
\alpha I_i(\hat\theta_u)
+
\beta C_i
+
\gamma U_i
-
\eta R_i.
\]

Here:

- \(I_i(\hat\theta_u)\) is Rasch information;
- \(C_i\) is coverage value for under-tested groups;
- \(U_i\) is uncertainty about the word or its groups;
- \(R_i\) is redundancy with already asked words;
- \(\alpha,\beta,\gamma,\eta\) are tunable weights.

A cold-start policy should emphasize coverage. Later, as the global ability estimate improves, the policy should emphasize information near the learner's ability boundary.

A good practical rule:

- first 10-20 questions: cover broad difficulty bands and semantic groups;
- next 20-50 questions: focus near the estimated ability frontier;
- after 50+ questions: probe group-specific deviations and uncertainty hotspots.

## 12. Training and Evaluation

### 12.1 Offline training

Use historical response data if available.

Fit:

1. word difficulties \(b_i\);
2. possibly word discriminations if using 2PL;
3. group membership matrix \(Q\), if supervised grouping is used;
4. hyperparameters \(\tau_\theta,\tau_\delta,\lambda_q\).

### 12.2 Evaluation tasks

Evaluate the model on held-out learners.

For each held-out learner:

1. hide most of their word responses;
2. reveal only \(q\) answers, for example \(q=10,20,30,50,100\);
3. estimate learner parameters;
4. predict the hidden responses.

Report performance as a function of \(q\).

### 12.3 Metrics

Use multiple metrics:

- AUC;
- log loss;
- Brier score;
- calibration error;
- accuracy at threshold 0.5;
- expected vocabulary size error;
- uncertainty interval coverage;
- speed of uncertainty reduction as \(q\) increases.

For an adaptive app, calibration and vocabulary-size error are more important than raw accuracy alone.

### 12.4 Ablations

Compare at least:

1. frequency-only baseline;
2. Rasch;
3. Rasch + semantic groups;
4. Rasch + expert tags;
5. Rasch + residual-response clusters;
6. Rasch + all groups;
7. optional neural residual model.

This will show whether the group structure actually improves prediction beyond the Rasch backbone.

## 13. Implementation Plan

### Phase 1: Rasch backbone

Implement fixed-difficulty Rasch estimation:

\[
P(y_{u,i}=1)=\sigma(\theta_u-b_i).
\]

Use known rates, prevalence, frequency, or existing difficulty data to initialize \(b_i\).

### Phase 2: Soft group matrix

Construct \(Q\) using:

- embedding clusters;
- word-family information;
- domain tags;
- register tags;
- difficulty bands;
- optional CEFR and academic lists.

### Phase 3: Grouped MAP estimator

Implement MAP estimation for:

\[
\theta_u,\delta_{u,1},\ldots,\delta_{u,K}.
\]

Start with diagonal or low-rank approximations for speed.

### Phase 4: Adaptive item selection

Use a score combining:

- information near current ability;
- group coverage;
- uncertainty;
- non-redundancy.

### Phase 5: Evaluation harness

Build a benchmark that simulates cold-start testing from historical response matrices.

Evaluate performance for several query budgets:

\[
q\in\{10,20,30,50,100,200\}.
\]

### Phase 6: Optional neural residual

Only after the grouped model is stable, add a small residual network:

\[
P(y_{u,i}=1)
=
\sigma\left(
\theta_u-b_i
+
\lambda_q\sum_k q_{i,k}\delta_{u,k}
+
\mu_q r_\phi(h_u,x_i)
\right).
\]

Here:

- \(h_u\) is a learned representation of the user's observed answers;
- \(x_i\) is the word feature vector;
- \(r_\phi\) is a neural residual;
- \(\mu_q\) gates the neural residual.

Keep \(\mu_q\) small in cold start.

## 14. Recommended Defaults

For a first implementation:

| Component | Recommendation |
|---|---|
| Item unit | lemma + POS |
| Reporting unit | word family |
| Difficulty source | L2 prevalence if available; otherwise frequency + CEFR + lexical features |
| Number of semantic clusters | 50 |
| Group membership | soft, normalized to sum at most 1-3 active groups per word |
| General prior | \(\theta_u\sim N(0,25)\) |
| Group prior | \(\delta_{u,k}\sim N(0,1)\) or smaller |
| Cold-start gate | \(\lambda_q=q/(q+50)\) |
| Initial questions | 30 by default |
| Query strategy | broad coverage first, then frontier refinement |
| Size estimate | sum of probabilities, not threshold count |
| Main benchmark | held-out response prediction by query budget |

## 15. Main Risks

### Overfitting group deviations

With too many groups and too few answers, the model will infer false strengths. Use strong shrinkage and a cold-start gate.

### Bad semantic clusters

Embedding clusters may group words by topical association rather than actual learner co-knowledge. Prefer residual-response clusters once data exist.

### Frequency dominates everything

Frequency/prevalence is a very strong predictor. The grouped model should be evaluated against a strong Rasch/frequency baseline, not against weak baselines.

### Hard partitions lose information

Many words belong to multiple groups. Use soft memberships whenever possible.

### Vocabulary knowledge is not binary

A learner may recognize a word without knowing its meaning deeply. If the app later supports graded labels, extend the Bernoulli model to an ordinal IRT model.

## 16. Summary

The proposed model is a structured extension of Rasch/IRT:

\[
P(y_{u,i}=1)
=
\sigma\left(
\theta_u-b_i+
\lambda_q\sum_k q_{i,k}\delta_{u,k}
\right).
\]

It preserves the strengths of Rasch:

- interpretability;
- calibration;
- robustness in cold start;
- efficient online updating.

It adds the ability to model:

- semantic-domain strengths;
- register-specific weaknesses;
- morphology and cognate effects;
- task/book/domain-specific vocabulary profiles;
- learner-specific deviations from the one-dimensional vocabulary scale.

The most important design principle is: use Rasch as the backbone, and treat group-specific abilities as regularized residuals. This gives a practical nonlinear or multidimensional vocabulary estimator without sacrificing cold-start stability.
