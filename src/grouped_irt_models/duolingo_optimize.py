from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_irt_models.experiment import (
    Dataset,
    Metrics,
    ModelSpec,
    aggregate_metrics,
    build_difficulty_bin_matrix,
    build_interaction_groups,
    build_residual_clusters,
    build_residual_sign_clusters,
    build_response_clusters,
    clipped_logit_difficulty,
    combine_group_matrices,
    evaluate_model,
    format_float,
    soft_cluster_membership,
    standardize,
    train_fasttext_embeddings,
)


@dataclass(frozen=True)
class LoadedDuolingo:
    dataset: Dataset
    lexeme_strings: list[str]


@dataclass(frozen=True)
class RankedModel:
    model: str
    mean_ba: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duolingo", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-users", type=int, required=True)
    parser.add_argument("--top-lexemes", type=int, required=True)
    parser.add_argument("--known-threshold", type=float, required=True)
    parser.add_argument("--impute-strategy", required=True)
    parser.add_argument("--difficulty-source", required=True)
    parser.add_argument("--coarse-repeats", type=int, required=True)
    parser.add_argument("--fine-repeats", type=int, required=True)
    parser.add_argument("--top-models", type=int, required=True)
    parser.add_argument("--q-values", nargs="+", type=int, required=True)
    return parser.parse_args()


def load_duolingo_dense(
    path: Path,
    language: str,
    top_users: int,
    top_lexemes: int,
    known_threshold: float,
    impute_strategy: str,
    difficulty_source: str,
) -> LoadedDuolingo:
    frame = pd.read_csv(
        path,
        compression="gzip",
        usecols=["user_id", "lexeme_id", "lexeme_string", "p_recall", "ui_language", "learning_language"],
    )
    pair = frame["ui_language"].astype(str) + "->" + frame["learning_language"].astype(str)
    frame = frame[pair == language].copy()
    if frame.empty:
        raise ValueError(f"No rows for language pair: {language}")

    frame = frame[["user_id", "lexeme_id", "lexeme_string", "p_recall"]].dropna().copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["lexeme_id"] = frame["lexeme_id"].astype(str)
    frame["lexeme_string"] = frame["lexeme_string"].astype(str)
    frame["p_recall"] = frame["p_recall"].astype(float).clip(0.0, 1.0)

    user_counts = frame.groupby("user_id").size().sort_values(ascending=False)
    lexeme_counts = frame.groupby("lexeme_id").size().sort_values(ascending=False)
    kept_users = set(user_counts.head(top_users).index.tolist())
    kept_lexemes = set(lexeme_counts.head(top_lexemes).index.tolist())
    frame = frame[frame["user_id"].isin(kept_users) & frame["lexeme_id"].isin(kept_lexemes)].copy()
    if frame.empty:
        raise ValueError("Filtering by top-users/top-lexemes produced an empty frame")

    agg = frame.groupby(["user_id", "lexeme_id"], as_index=False).agg(p_recall=("p_recall", "mean"))
    agg["known"] = agg["p_recall"] >= known_threshold

    lexeme_names = (
        frame.groupby("lexeme_id")["lexeme_string"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )

    user_ids = sorted(agg["user_id"].unique().tolist())
    lexemes = sorted(agg["lexeme_id"].unique().tolist())
    lexeme_strings = [lexeme_names.get(lexeme, lexeme) for lexeme in lexemes]

    known_pivot = (
        agg.pivot(index="user_id", columns="lexeme_id", values="known")
        .reindex(index=user_ids, columns=lexemes)
        .astype("float64")
    )
    lexeme_known_rate = agg.groupby("lexeme_id", as_index=True)["known"].mean().reindex(lexemes).to_numpy(dtype=np.float64)
    user_known_rate = agg.groupby("user_id", as_index=True)["known"].mean().reindex(user_ids).to_numpy(dtype=np.float64)

    matrix = known_pivot.to_numpy(dtype=np.float64)
    missing = np.isnan(matrix)
    if impute_strategy == "lexeme_majority":
        fill_values = (lexeme_known_rate >= known_threshold).astype(np.float64)[np.newaxis, :]
    elif impute_strategy == "user_majority":
        fill_values = (user_known_rate >= known_threshold).astype(np.float64)[:, np.newaxis]
    elif impute_strategy == "blend_majority":
        blend = 0.5 * user_known_rate[:, np.newaxis] + 0.5 * lexeme_known_rate[np.newaxis, :]
        fill_values = (blend >= known_threshold).astype(np.float64)
    else:
        raise ValueError(f"Unknown impute strategy: {impute_strategy}")

    dense = np.where(missing, fill_values, matrix)
    if np.isnan(dense).any():
        raise ValueError("Dense matrix still contains NaN after imputation")

    if difficulty_source == "known_rate":
        difficulty_accuracy = lexeme_known_rate
    elif difficulty_source == "recall_mean":
        difficulty_accuracy = (
            agg.groupby("lexeme_id", as_index=True)["p_recall"].mean().reindex(lexemes).to_numpy(dtype=np.float64)
        )
    else:
        raise ValueError(f"Unknown difficulty source: {difficulty_source}")

    clipped = np.clip(difficulty_accuracy, 1e-4, 1.0 - 1e-4)
    difficulties = standardize(clipped_logit_difficulty(clipped, 1e-4))

    dataset = Dataset(
        user_ids=user_ids,
        words=lexemes,
        known=dense.astype(np.bool_),
        difficulties=difficulties,
        excel_coverage=(
            f"Duolingo {language}: users={len(user_ids)}, lexemes={len(lexemes)}, "
            f"known_threshold={known_threshold}, impute={impute_strategy}, difficulty={difficulty_source}"
        ),
    )
    return LoadedDuolingo(dataset=dataset, lexeme_strings=lexeme_strings)


def build_group_matrices(dataset: Dataset, lexeme_strings: list[str], seed: int) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    empty = np.zeros((len(dataset.words), 0), dtype=np.float64)
    matrices["rasch"] = empty

    residual_12_701 = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 701)
    residual_12_1301 = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 1301)
    residual_16_701 = build_residual_clusters(dataset.known, dataset.difficulties, 16, 2.0, 701)
    response_12 = build_response_clusters(dataset.known, 12, seed + 412)
    response_16 = build_response_clusters(dataset.known, 16, seed + 416)
    residual_sign_12 = build_residual_sign_clusters(dataset.known, dataset.difficulties, 12, 2.0, seed + 512)

    matrices["residual12_s701"] = residual_12_701
    matrices["residual12_s1301"] = residual_12_1301
    matrices["residual16_s701"] = residual_16_701
    matrices["response12"] = response_12
    matrices["response16"] = response_16
    matrices["residual_sign12"] = residual_sign_12

    fasttext_vectors = train_fasttext_embeddings(lexeme_strings, dataset.difficulties, seed + 777)
    fasttext_12 = soft_cluster_membership(fasttext_vectors, 12, seed + 912, 8.0)
    fasttext_16 = soft_cluster_membership(fasttext_vectors, 16, seed + 916, 8.0)
    matrices["fasttext12"] = fasttext_12
    matrices["fasttext16"] = fasttext_16

    bins_8 = build_difficulty_bin_matrix(dataset.difficulties, 8)
    matrices["difficulty_bins8"] = bins_8
    matrices["residual12_x_bins8"] = build_interaction_groups(residual_12_701, bins_8)
    matrices["residual12_plus_fasttext12"] = combine_group_matrices([residual_12_701, fasttext_12])

    return matrices


def build_model_specs(matrices: dict[str, np.ndarray]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    specs.append(ModelSpec("rasch_std_fixed", matrices["rasch"], 2.0, 0.5, 50.0, "standard", "fixed_0.50"))
    specs.append(
        ModelSpec(
            "rasch_bal_ba",
            matrices["rasch"],
            2.0,
            0.5,
            50.0,
            "class_weighted",
            "observed_ba_opt",
        )
    )

    selected_matrices = [
        "residual12_s701",
        "residual16_s701",
        "response12",
        "residual_sign12",
        "fasttext12",
        "residual12_x_bins8",
    ]

    presets = [
        (1.0, 20.0, "class_weighted", "observed_ba_opt"),
        (1.4, 8.0, "class_weighted", "observed_ba_opt"),
        (1.6, 12.0, "class_weighted", "observed_ba_opt_shrunk"),
    ]

    for matrix_name in selected_matrices:
        matrix = matrices[matrix_name]
        if matrix_name == "rasch":
            continue
        group_count = int(matrix.shape[1])
        for tau_delta, gate_c, fit_strategy, threshold_strategy in presets:
            name = (
                f"{matrix_name}_g{group_count}_tau{str(tau_delta).replace('.', 'p')}_"
                f"c{str(gate_c).replace('.', 'p')}_{threshold_strategy}"
            )
            specs.append(
                ModelSpec(
                    name,
                    matrix,
                    2.0,
                    tau_delta,
                    gate_c,
                    fit_strategy,
                    threshold_strategy,
                )
            )

    return specs


def rank_models_from_coarse(metrics: list[Metrics], top_models: int) -> list[RankedModel]:
    aggregate = aggregate_metrics(metrics)
    grouped = (
        aggregate.groupby("model", as_index=False)
        .agg(mean_ba=("balanced_accuracy", "mean"))
        .sort_values("mean_ba", ascending=False)
        .reset_index(drop=True)
    )
    ranked: list[RankedModel] = []
    for _, row in grouped.head(top_models).iterrows():
        ranked.append(RankedModel(model=str(row["model"]), mean_ba=float(row["mean_ba"])))
    return ranked


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(format_float(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    output_path: Path,
    dataset: Dataset,
    coarse_aggregate: pd.DataFrame,
    fine_aggregate: pd.DataFrame,
    ranked: list[RankedModel],
    args: argparse.Namespace,
) -> None:
    best_by_q = (
        fine_aggregate.sort_values(["q", "balanced_accuracy", "pr_auc"], ascending=[True, False, False])
        .groupby("q", as_index=False)
        .first()
    )
    ranked_frame = pd.DataFrame([rank.__dict__ for rank in ranked])
    lines = [
        "# Duolingo grouped IRT balanced-accuracy optimization",
        "",
        "## Setup",
        "",
        f"- Dataset: `{args.duolingo}`",
        f"- Language pair: `{args.language}`",
        f"- Dense matrix size: `{len(dataset.user_ids)} x {len(dataset.words)}`",
        f"- Known threshold: `{args.known_threshold}`",
        f"- Imputation strategy: `{args.impute_strategy}`",
        f"- Difficulty source: `{args.difficulty_source}`",
        f"- q values: `{', '.join(str(value) for value in args.q_values)}`",
        f"- Coarse repeats: `{args.coarse_repeats}`, fine repeats: `{args.fine_repeats}`",
        f"- Candidate models: `{coarse_aggregate['model'].nunique()}`",
        "",
        "## Coarse stage ranking (mean BA over q=50,100)",
        "",
        markdown_table(ranked_frame, ["model", "mean_ba"]),
        "",
        "## Best model per q (fine stage)",
        "",
        markdown_table(
            best_by_q[
                [
                    "q",
                    "model",
                    "balanced_accuracy",
                    "pr_auc",
                    "auc",
                    "accuracy",
                    "log_loss",
                    "brier",
                ]
            ],
            ["q", "model", "balanced_accuracy", "pr_auc", "auc", "accuracy", "log_loss", "brier"],
        ),
        "",
        "## Fine stage full results",
        "",
        markdown_table(
            fine_aggregate[
                [
                    "q",
                    "model",
                    "balanced_accuracy",
                    "pr_auc",
                    "auc",
                    "accuracy",
                    "log_loss",
                    "brier",
                ]
            ],
            ["q", "model", "balanced_accuracy", "pr_auc", "auc", "accuracy", "log_loss", "brier"],
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    print("loading dataset", flush=True)
    loaded = load_duolingo_dense(
        Path(args.duolingo),
        str(args.language),
        int(args.top_users),
        int(args.top_lexemes),
        float(args.known_threshold),
        str(args.impute_strategy),
        str(args.difficulty_source),
    )
    print(
        f"dataset ready: users={len(loaded.dataset.user_ids)} words={len(loaded.dataset.words)}",
        flush=True,
    )
    print("building group matrices (includes fasttext training)", flush=True)
    matrices = build_group_matrices(loaded.dataset, loaded.lexeme_strings, int(args.seed))
    print(f"group matrices ready: {', '.join(sorted(matrices.keys()))}", flush=True)
    all_specs = build_model_specs(matrices)
    print(
        f"loaded dataset: users={len(loaded.dataset.user_ids)} words={len(loaded.dataset.words)} "
        f"models={len(all_specs)}",
        flush=True,
    )

    coarse_metrics: list[Metrics] = []
    coarse_q_values = [value for value in args.q_values if value in {50, 100}]
    for index, spec in enumerate(all_specs, start=1):
        print(f"coarse [{index}/{len(all_specs)}] {spec.name}", flush=True)
        coarse_metrics.extend(
            evaluate_model(
                loaded.dataset,
                spec,
                coarse_q_values,
                int(args.coarse_repeats),
                int(args.seed),
            )
        )
    coarse_aggregate = aggregate_metrics(coarse_metrics)

    ranked = rank_models_from_coarse(coarse_metrics, int(args.top_models))
    ranked_names = {entry.model for entry in ranked}
    fine_specs = [spec for spec in all_specs if spec.name in ranked_names]
    print(f"fine stage models={len(fine_specs)}", flush=True)

    fine_metrics: list[Metrics] = []
    for index, spec in enumerate(fine_specs, start=1):
        print(f"fine [{index}/{len(fine_specs)}] {spec.name}", flush=True)
        fine_metrics.extend(
            evaluate_model(
                loaded.dataset,
                spec,
                args.q_values,
                int(args.fine_repeats),
                int(args.seed),
            )
        )
    fine_aggregate = aggregate_metrics(fine_metrics)

    write_summary(Path(args.summary), loaded.dataset, coarse_aggregate, fine_aggregate, ranked, args)
    print(fine_aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
