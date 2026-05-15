# grouped_irt_models

This repository includes grouped residual IRT experiments and evaluation scripts.

## Duolingo HLR dataset placement

Place the Duolingo Half-Life Regression dataset under:

`data/raw/duolingo_hlr/`

The expected layout is:

```text
data/
  raw/
    duolingo_hlr/
      learning_traces.csv.gz
      README.md
      DOWNLOAD_DATASET.md
      experiment.py
      evaluation.r
      lexeme_reference.txt
```

### What each item means

- `learning_traces.csv.gz`
  - Main event-level learning log.
  - Each row is one user/lexeme practice event with recall outcome and history context.
  - Key fields used by this repo:
    - `user_id`: learner identity
    - `lexeme_id`: vocabulary item identity
    - `p_recall`: observed recall quality in `[0, 1]`
    - `ui_language`, `learning_language`: language pair filter (for example `en->es`)
  - Our evaluator aggregates rows to a user-lexeme matrix and binarizes knowledge with `known = mean(p_recall) >= 0.5`.

- `lexeme_reference.txt`
  - Reference dictionary for lexeme tag components (POS/morphology markers).
  - Useful for interpreting `lexeme_string` tags.
  - Not required by the current grouped IRT benchmark pipeline.

- `README.md` (inside `duolingo_hlr`)
  - Upstream dataset documentation from Duolingo.
  - Defines column semantics and original experimental context.

- `DOWNLOAD_DATASET.md`
  - Notes on obtaining the raw dataset bundle.

- `experiment.py`, `evaluation.r`
  - Original baseline modeling/evaluation scripts released with the dataset.
  - Included for provenance and reference; not required to run this repo’s grouped IRT evaluator.

## Required file for benchmark

The Duolingo benchmark script in this repo requires:

- `data/raw/duolingo_hlr/learning_traces.csv.gz`

Other files are optional for our benchmark flow, but are part of the standard dataset bundle.

## Structural assumptions in this repo

- **Granularity**: data starts as repeated event rows, then gets aggregated to one cell per `(user_id, lexeme_id)`.
- **Matrix shape**: grouped IRT requires a dense user-item matrix; missing cells are imputed by lexeme-majority class.
- **Item difficulty source**: per-lexeme empirical known-rate is transformed to Rasch-style difficulty via standardized `-logit(accuracy)`.
- **Cold-start evaluation**: per user, `q` observed lexemes are revealed, model is fit, and the rest are predicted.

## Current benchmark command

```bash
PYTHONPATH=src .venv/bin/python -m grouped_irt_models.duolingo_eval \
  --duolingo data/raw/duolingo_hlr/learning_traces.csv.gz \
  --summary duolingo_irt_summary.py \
  --seed 42 \
  --q 50 \
  --repeats 3 \
  --top-users 500 \
  --top-lexemes 1000 \
  --language 'en->es'
```
