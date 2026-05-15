from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_irt_models.experiment import (
    Dataset,
    FitResult,
    ModelSpec,
    build_residual_clusters,
    clipped_logit_difficulty,
    evaluate_predictions,
    fit_grouped_map,
    format_float,
    predict_probability,
    select_threshold,
    standardize,
)


@dataclass(frozen=True)
class EvalResult:
    model: str
    pr_auc: float
    balanced_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duolingo", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--top-users", type=int, required=True)
    parser.add_argument("--top-lexemes", type=int, required=True)
    parser.add_argument("--language", required=True)
    return parser.parse_args()


def top3_q50_models(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_full_aggregate = False
    models: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line.startswith("## Full aggregate results"):
            in_full_aggregate = True
            continue
        if line.startswith("## ") and in_full_aggregate:
            break
        if not in_full_aggregate:
            continue
        if not line.startswith("| 50 |"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            continue
        model_name = parts[2]
        if model_name == "model":
            continue
        if model_name in seen:
            continue
        seen.add(model_name)
        models.append(model_name)
        if len(models) == 3:
            break
    if len(models) != 3:
        raise ValueError("Failed to parse top-3 q=50 models from grouped_irt_summary.md")
    return models


def load_duolingo_dataset(path: Path, language: str, top_users: int, top_lexemes: int) -> Dataset:
    frame = pd.read_csv(path, compression="gzip")
    frame = frame[(frame["ui_language"].astype(str) + "->" + frame["learning_language"].astype(str)) == language].copy()
    if frame.empty:
        raise ValueError(f"No rows for language pair: {language}")

    frame = frame[["user_id", "lexeme_id", "p_recall"]].dropna().copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["lexeme_id"] = frame["lexeme_id"].astype(str)
    frame["p_recall"] = frame["p_recall"].astype(float).clip(0.0, 1.0)

    user_counts = frame.groupby("user_id").size().sort_values(ascending=False)
    lexeme_counts = frame.groupby("lexeme_id").size().sort_values(ascending=False)
    kept_users = set(user_counts.head(top_users).index.tolist())
    kept_lexemes = set(lexeme_counts.head(top_lexemes).index.tolist())

    frame = frame[frame["user_id"].isin(kept_users) & frame["lexeme_id"].isin(kept_lexemes)].copy()
    if frame.empty:
        raise ValueError("Filtering by top users/lexemes produced an empty dataset")

    agg = frame.groupby(["user_id", "lexeme_id"], as_index=False).agg(p_recall=("p_recall", "mean"))
    agg["known"] = agg["p_recall"] >= 0.5

    user_ids = sorted(agg["user_id"].unique().tolist())
    lexemes = sorted(agg["lexeme_id"].unique().tolist())

    known_pivot = agg.pivot(index="user_id", columns="lexeme_id", values="known").reindex(index=user_ids, columns=lexemes)
    lexeme_accuracy = agg.groupby("lexeme_id", as_index=True)["known"].mean().reindex(lexemes)

    for lexeme in lexemes:
        column_value = bool(lexeme_accuracy.loc[lexeme] >= 0.5)
        known_pivot[lexeme] = known_pivot[lexeme].fillna(column_value)

    if known_pivot.isna().any().any():
        raise ValueError("Dense known-matrix construction failed")

    accuracy = np.clip(lexeme_accuracy.to_numpy(dtype=np.float64), 1e-4, 1 - 1e-4)
    difficulties = standardize(clipped_logit_difficulty(accuracy, 1e-4))

    return Dataset(
        user_ids=user_ids,
        words=lexemes,
        known=known_pivot.to_numpy(dtype=np.bool_),
        difficulties=difficulties,
        excel_coverage=f"Duolingo {language}: users={len(user_ids)}, lexemes={len(lexemes)}",
    )


def build_model_spec_by_name(dataset: Dataset, model_name: str) -> ModelSpec:
    if model_name == "residual_k12_seed701_tau1p6_g12_shrunk":
        matrix = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 701)
        return ModelSpec(model_name, matrix, 2.0, 1.6, 12.0, "class_weighted", "observed_ba_opt_shrunk")
    if model_name == "residual_k12_seed701_tau1p4_g8_shrunk":
        matrix = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 701)
        return ModelSpec(model_name, matrix, 2.0, 1.4, 8.0, "class_weighted", "observed_ba_opt_shrunk")
    if model_name == "residual_k12_seed1301_tau1p4_g12_shrunk":
        matrix = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 1301)
        return ModelSpec(model_name, matrix, 2.0, 1.4, 12.0, "class_weighted", "observed_ba_opt_shrunk")
    raise ValueError(f"Unsupported model name in top-3 list: {model_name}")


def evaluate_model(dataset: Dataset, spec: ModelSpec, q: int, repeats: int, seed: int) -> EvalResult:
    rng = np.random.default_rng(seed)
    pr_aucs: list[float] = []
    balanced_accuracies: list[float] = []
    word_count = len(dataset.words)
    if q >= word_count:
        raise ValueError(f"q={q} must be smaller than word count={word_count}")

    for repeat in range(repeats):
        for user_index in range(len(dataset.user_ids)):
            permutation = rng.permutation(word_count)
            observed = permutation[:q]
            hidden = permutation[q:]
            observed_y = dataset.known[user_index, observed].astype(np.float64)
            fit: FitResult = fit_grouped_map(
                observed_y,
                dataset.difficulties[observed],
                spec.group_matrix[observed],
                spec.tau_theta,
                spec.tau_delta,
                spec.gate_c,
                spec.fit_strategy,
            )
            observed_prob = predict_probability(fit, dataset.difficulties[observed], spec.group_matrix[observed])
            threshold = select_threshold(observed_y, observed_prob, spec.threshold_strategy)
            hidden_prob = predict_probability(fit, dataset.difficulties[hidden], spec.group_matrix[hidden])
            hidden_y = dataset.known[user_index, hidden].astype(np.float64)
            _, _, _, pr_auc, _, balanced_accuracy, _, _ = evaluate_predictions(hidden_y, hidden_prob, threshold)
            pr_aucs.append(pr_auc)
            balanced_accuracies.append(balanced_accuracy)
            _ = repeat

    return EvalResult(
        model=spec.name,
        pr_auc=float(np.nanmean(np.array(pr_aucs, dtype=np.float64))),
        balanced_accuracy=float(np.nanmean(np.array(balanced_accuracies, dtype=np.float64))),
    )


def write_summary(
    output_path: Path,
    dataset: Dataset,
    language: str,
    q: int,
    repeats: int,
    top_users: int,
    top_lexemes: int,
    results: list[EvalResult],
) -> None:
    ordered = sorted(results, key=lambda item: (item.balanced_accuracy, item.pr_auc), reverse=True)
    lines = [
        "# Duolingo HTL benchmark (q=50)",
        "",
        "## Setup",
        "",
        f"- Source: `data/raw/duolingo_hlr/learning_traces.csv.gz`.",
        f"- Language pair: `{language}`.",
        f"- Subset selection: top `{top_users}` users by row count and top `{top_lexemes}` lexemes by row count.",
        "- Label construction: `known = (mean p_recall per user-lexeme >= 0.5)`.",
        "- Missing user-lexeme cells were imputed with lexeme-majority known/unknown to form a dense matrix required by grouped IRT.",
        "- Lexeme difficulty was estimated from Duolingo subset accuracy and transformed as standardized `-logit(accuracy)`.",
        f"- Evaluation protocol: per-user random reveal of `q={q}` lexemes, hidden-set scoring, `{repeats}` repeats.",
        "",
        "## Results",
        "",
        "| rank | model | pr_auc | balanced_accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for index, row in enumerate(ordered, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    row.model,
                    format_float(row.pr_auc, 4),
                    format_float(row.balanced_accuracy, 4),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Rankings are sorted by balanced accuracy first, then PR-AUC.",
            f"- Dense matrix size used for this run: `{len(dataset.user_ids)} x {len(dataset.words)}`.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    top_models = top3_q50_models(Path("grouped_irt_summary.md"))
    dataset = load_duolingo_dataset(
        Path(args.duolingo),
        str(args.language),
        int(args.top_users),
        int(args.top_lexemes),
    )
    specs = [build_model_spec_by_name(dataset, model_name) for model_name in top_models]
    results = [evaluate_model(dataset, spec, int(args.q), int(args.repeats), int(args.seed)) for spec in specs]
    write_summary(
        Path(args.summary),
        dataset,
        str(args.language),
        int(args.q),
        int(args.repeats),
        int(args.top_users),
        int(args.top_lexemes),
        results,
    )


if __name__ == "__main__":
    main()
