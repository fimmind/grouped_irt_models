from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import fasttext
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Dataset:
    user_ids: list[str]
    words: list[str]
    known: BoolArray
    difficulties: FloatArray
    excel_coverage: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group_matrix: FloatArray
    tau_theta: float
    tau_delta: float
    gate_c: float
    fit_strategy: str
    threshold_strategy: str


@dataclass(frozen=True)
class DifficultyOnlySpec:
    name: str
    threshold_strategy: str


@dataclass(frozen=True)
class FitResult:
    theta: float
    delta: FloatArray
    gate: float


@dataclass(frozen=True)
class Metrics:
    model: str
    q: int
    repeat: int
    log_loss: float
    brier: float
    auc: float
    pr_auc: float
    accuracy: float
    balanced_accuracy: float
    calibration_error: float
    vocabulary_mae: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--word-difficulty", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--q-values", nargs="+", type=int, required=True)
    return parser.parse_args()


def clipped_logit_difficulty(accuracy: FloatArray, epsilon: float) -> FloatArray:
    clipped = np.clip(accuracy, epsilon, 1.0 - epsilon)
    return -logit(clipped)


def standardize(values: FloatArray) -> FloatArray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0.0:
        raise ValueError("Cannot standardize a constant vector.")
    return (values - mean) / std


def load_dataset(responses_path: Path, word_difficulty_path: Path) -> Dataset:
    responses = pd.read_csv(responses_path)
    required_response_columns = {"user_id", "word", "raw_score"}
    if not required_response_columns.issubset(responses.columns):
        raise ValueError(f"responses CSV must contain columns {sorted(required_response_columns)}")

    responses = responses.dropna(subset=["word"]).copy()
    responses["word"] = responses["word"].map(lambda value: str(value).strip().lower())
    responses = responses[responses["word"] != "nan"].copy()
    responses["raw_score"] = responses["raw_score"].astype(int)
    responses = (
        responses.groupby(["user_id", "word"], as_index=False)
        .agg(raw_score=("raw_score", "max"))
        .copy()
    )
    responses["known"] = responses["raw_score"] >= 4

    word_info = pd.read_excel(word_difficulty_path, sheet_name="Words", usecols=["spelling", "accuracy"])
    word_info = word_info.dropna(subset=["spelling", "accuracy"]).copy()
    word_info["word"] = word_info["spelling"].map(lambda value: str(value).strip().lower())
    word_info = word_info.drop_duplicates(subset=["word"], keep="first")

    available_words = set(word_info["word"])
    response_words = set(responses["word"])
    usable_words = sorted(response_words & available_words)
    if not usable_words:
        raise ValueError("No overlap between response words and Excel word difficulty rows.")

    filtered = responses[responses["word"].isin(usable_words)].copy()
    pivot = filtered.pivot(index="user_id", columns="word", values="known")
    pivot = pivot.reindex(columns=usable_words)
    if pivot.isna().any().any():
        missing_count = int(pivot.isna().sum().sum())
        raise ValueError(f"Response matrix must be dense after filtering; missing cells: {missing_count}")

    difficulty_frame = word_info.set_index("word").reindex(usable_words)
    accuracy = difficulty_frame["accuracy"].to_numpy(dtype=np.float64)
    difficulties = standardize(clipped_logit_difficulty(accuracy, 1e-4))

    coverage = f"{len(usable_words):,}/{len(response_words):,} Ehara word strings had Excel difficulty coverage"
    return Dataset(
        user_ids=[str(value) for value in pivot.index.tolist()],
        words=usable_words,
        known=pivot.to_numpy(dtype=np.bool_),
        difficulties=difficulties,
        excel_coverage=coverage,
    )


def build_fasttext_corpus(words: list[str], difficulties: FloatArray, output_path: Path) -> None:
    ordered_indices = np.argsort(difficulties)
    difficulty_lines = [words[index] for index in ordered_indices]
    morphology_lines: list[str] = []
    suffix_groups: dict[str, list[str]] = {}
    prefix_groups: dict[str, list[str]] = {}
    for word in words:
        clean = word.replace("-", " ")
        morphology_lines.append(clean)
        if len(word) >= 5:
            suffix_groups.setdefault(word[-3:], []).append(word)
            prefix_groups.setdefault(word[:3], []).append(word)

    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(difficulty_lines), 80):
            handle.write(" ".join(difficulty_lines[start : start + 80]))
            handle.write("\n")
        for line in morphology_lines:
            handle.write(line)
            handle.write("\n")
        for group in suffix_groups.values():
            if len(group) >= 3:
                handle.write(" ".join(group[:200]))
                handle.write("\n")
        for group in prefix_groups.values():
            if len(group) >= 3:
                handle.write(" ".join(group[:200]))
                handle.write("\n")


def train_fasttext_embeddings(words: list[str], difficulties: FloatArray, seed: int) -> FloatArray:
    np.random.seed(seed)
    with tempfile.TemporaryDirectory() as directory:
        corpus_path = Path(directory) / "fasttext_corpus.txt"
        build_fasttext_corpus(words, difficulties, corpus_path)
        model = fasttext.train_unsupervised(
            str(corpus_path),
            model="skipgram",
            dim=50,
            epoch=35,
            minn=2,
            maxn=6,
            minCount=1,
            thread=1,
            verbose=0,
        )
        vectors = np.vstack([model.get_word_vector(word) for word in words]).astype(np.float64)

    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0.0] = 1.0
    return vectors / norms[:, np.newaxis]


def soft_cluster_membership(vectors: FloatArray, group_count: int, seed: int, temperature: float) -> FloatArray:
    if group_count <= 1:
        raise ValueError(f"group_count must be greater than 1, got {group_count}")
    kmeans = KMeans(n_clusters=group_count, n_init=20, random_state=seed)
    labels = kmeans.fit_predict(vectors)
    centers = kmeans.cluster_centers_.astype(np.float64)
    center_norms = np.linalg.norm(centers, axis=1)
    center_norms[center_norms == 0.0] = 1.0
    centers = centers / center_norms[:, np.newaxis]
    similarities = vectors @ centers.T
    top_count = min(3, group_count)
    top_indices = np.argpartition(similarities, -top_count, axis=1)[:, -top_count:]
    matrix = np.zeros((vectors.shape[0], group_count), dtype=np.float64)
    for row_index in range(vectors.shape[0]):
        columns = top_indices[row_index]
        logits = similarities[row_index, columns] * temperature
        weights = np.exp(logits - np.max(logits))
        weights = weights / np.sum(weights)
        matrix[row_index, columns] = weights
        matrix[row_index, labels[row_index]] = max(matrix[row_index, labels[row_index]], 0.5)
        matrix[row_index] = matrix[row_index] / np.sum(matrix[row_index])
    return matrix


def rasch_neg_log_posterior(theta: float, y: FloatArray, b: FloatArray, tau_theta: float) -> float:
    logits = theta - b
    log_likelihood = np.sum(y * np.logaddexp(0.0, -logits) + (1.0 - y) * np.logaddexp(0.0, logits))
    penalty = theta * theta / (2.0 * tau_theta * tau_theta)
    return float(log_likelihood + penalty)


def fit_rasch_theta(y: FloatArray, b: FloatArray, tau_theta: float) -> float:
    result = minimize_scalar(
        lambda theta: rasch_neg_log_posterior(float(theta), y, b, tau_theta),
        bounds=(-6.0, 6.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not result.success:
        raise RuntimeError(f"Rasch theta optimization failed: {result.message}")
    return float(result.x)


def rational_gate(observed_count: int, gate_c: float) -> float:
    return float(observed_count / (observed_count + gate_c))


def fit_grouped_map(
    y: FloatArray,
    b: FloatArray,
    q_matrix: FloatArray,
    tau_theta: float,
    tau_delta: float,
    gate_c: float,
    fit_strategy: str,
) -> FitResult:
    theta_start = fit_rasch_theta(y, b, tau_theta)
    gate = rational_gate(int(y.shape[0]), gate_c)
    group_count = int(q_matrix.shape[1])
    if group_count == 0 or gate <= 1e-12:
        return FitResult(theta=theta_start, delta=np.zeros(group_count, dtype=np.float64), gate=gate)

    initial = np.zeros(group_count + 1, dtype=np.float64)
    initial[0] = theta_start

    if fit_strategy == "standard":
        sample_weights = np.ones_like(y)
    elif fit_strategy == "class_weighted":
        positive_rate = float(np.clip(np.mean(y), 0.05, 0.95))
        weight_positive = 0.5 / positive_rate
        weight_negative = 0.5 / (1.0 - positive_rate)
        sample_weights = y * weight_positive + (1.0 - y) * weight_negative
    else:
        raise ValueError(f"Unknown fit_strategy: {fit_strategy}")

    def objective(params: FloatArray) -> tuple[float, FloatArray]:
        theta = float(params[0])
        delta = params[1:]
        residual = q_matrix @ delta
        logits = theta - b + gate * residual
        probabilities = expit(logits)
        nll = np.sum(
            sample_weights * (y * np.logaddexp(0.0, -logits) + (1.0 - y) * np.logaddexp(0.0, logits))
        )
        penalty = theta * theta / (2.0 * tau_theta * tau_theta)
        penalty += float(np.dot(delta, delta)) / (2.0 * tau_delta * tau_delta)
        errors = sample_weights * (probabilities - y)
        gradient = np.empty_like(params)
        gradient[0] = np.sum(errors) + theta / (tau_theta * tau_theta)
        gradient[1:] = gate * (q_matrix.T @ errors) + delta / (tau_delta * tau_delta)
        return float(nll + penalty), gradient

    result = minimize(
        fun=lambda params: objective(params)[0],
        x0=initial,
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Grouped MAP optimization failed: {result.message}")
    return FitResult(theta=float(result.x[0]), delta=result.x[1:].astype(np.float64), gate=gate)


def predict_probability(fit: FitResult, b: FloatArray, q_matrix: FloatArray) -> FloatArray:
    if q_matrix.shape[1] == 0:
        return expit(fit.theta - b)
    return expit(fit.theta - b + fit.gate * (q_matrix @ fit.delta))


def fit_full_rasch_thetas(known: BoolArray, difficulties: FloatArray, tau_theta: float) -> FloatArray:
    return np.array(
        [
            fit_rasch_theta(known[user_index].astype(np.float64), difficulties, tau_theta)
            for user_index in range(known.shape[0])
        ],
        dtype=np.float64,
    )


def build_residual_clusters(
    known: BoolArray,
    difficulties: FloatArray,
    group_count: int,
    tau_theta: float,
    seed: int,
) -> FloatArray:
    theta = fit_full_rasch_thetas(known, difficulties, tau_theta)
    logits = theta[:, np.newaxis] - difficulties[np.newaxis, :]
    residuals = known.astype(np.float64) - expit(logits)
    word_profiles = residuals.T
    profile_means = np.mean(word_profiles, axis=1, keepdims=True)
    centered_profiles = word_profiles - profile_means
    profile_norms = np.linalg.norm(centered_profiles, axis=1)
    profile_norms[profile_norms == 0.0] = 1.0
    normalized_profiles = centered_profiles / profile_norms[:, np.newaxis]
    return soft_cluster_membership(normalized_profiles, group_count, seed, 8.0)


def build_response_clusters(known: BoolArray, group_count: int, seed: int) -> FloatArray:
    word_profiles = known.astype(np.float64).T
    profile_means = np.mean(word_profiles, axis=1, keepdims=True)
    centered_profiles = word_profiles - profile_means
    profile_norms = np.linalg.norm(centered_profiles, axis=1)
    profile_norms[profile_norms == 0.0] = 1.0
    normalized_profiles = centered_profiles / profile_norms[:, np.newaxis]
    return soft_cluster_membership(normalized_profiles, group_count, seed, 6.0)


def build_residual_sign_clusters(
    known: BoolArray,
    difficulties: FloatArray,
    group_count: int,
    tau_theta: float,
    seed: int,
) -> FloatArray:
    theta = fit_full_rasch_thetas(known, difficulties, tau_theta)
    logits = theta[:, np.newaxis] - difficulties[np.newaxis, :]
    residuals = known.astype(np.float64) - expit(logits)
    signs = np.where(residuals >= 0.0, 1.0, -1.0).T
    profile_means = np.mean(signs, axis=1, keepdims=True)
    centered_profiles = signs - profile_means
    profile_norms = np.linalg.norm(centered_profiles, axis=1)
    profile_norms[profile_norms == 0.0] = 1.0
    normalized_profiles = centered_profiles / profile_norms[:, np.newaxis]
    return soft_cluster_membership(normalized_profiles, group_count, seed, 6.0)


def binary_column(values: Iterable[bool]) -> FloatArray:
    return np.array([1.0 if value else 0.0 for value in values], dtype=np.float64)


def build_expert_tag_matrix(words: list[str], difficulties: FloatArray) -> FloatArray:
    columns: list[FloatArray] = []
    labels: list[str] = []

    quantiles = np.quantile(difficulties, np.linspace(0.0, 1.0, 9))
    for bucket_index in range(8):
        lower = quantiles[bucket_index]
        upper = quantiles[bucket_index + 1]
        if bucket_index == 7:
            mask = (difficulties >= lower) & (difficulties <= upper)
        else:
            mask = (difficulties >= lower) & (difficulties < upper)
        columns.append(mask.astype(np.float64))
        labels.append(f"difficulty_q{bucket_index + 1}")

    morphology_specs: list[tuple[str, Callable[[str], bool]]] = [
        ("short", lambda word: len(word) <= 4),
        ("long", lambda word: len(word) >= 10),
        ("hyphenated", lambda word: "-" in word),
        ("apostrophe", lambda word: "'" in word),
        ("latinate_tion_sion", lambda word: word.endswith("tion") or word.endswith("sion")),
        ("latinate_ity", lambda word: word.endswith("ity")),
        ("academic_ism_ist", lambda word: word.endswith("ism") or word.endswith("ist")),
        ("adverb_ly", lambda word: word.endswith("ly")),
        ("gerund_ing", lambda word: word.endswith("ing")),
        ("past_ed", lambda word: word.endswith("ed")),
        ("agent_er_or", lambda word: word.endswith("er") or word.endswith("or")),
        ("negative_prefix", lambda word: word.startswith(("un", "in", "im", "ir", "il", "dis", "non"))),
        ("re_prefix", lambda word: word.startswith("re") and len(word) >= 5),
        ("compound_like", lambda word: any(part in word for part in ["house", "work", "book", "water", "land"])),
    ]
    for label, predicate in morphology_specs:
        values = binary_column(predicate(word) for word in words)
        if float(np.sum(values)) > 0.0:
            columns.append(values)
            labels.append(label)

    matrix = np.column_stack(columns).astype(np.float64)
    column_sums = np.sum(matrix, axis=0)
    keep_mask = column_sums > 0.0
    if not bool(np.all(keep_mask)):
        dropped = [labels[index] for index, keep in enumerate(keep_mask) if not keep]
        raise ValueError(f"Expert tag columns unexpectedly empty: {dropped}")
    row_sums = np.sum(matrix, axis=1)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums[:, np.newaxis]


def combine_group_matrices(matrices: list[FloatArray]) -> FloatArray:
    if not matrices:
        raise ValueError("At least one matrix is required.")
    row_count = matrices[0].shape[0]
    for matrix in matrices:
        if matrix.shape[0] != row_count:
            raise ValueError("All group matrices must have the same number of rows.")
    combined = np.column_stack(matrices).astype(np.float64)
    row_sums = np.sum(combined, axis=1)
    row_sums[row_sums == 0.0] = 1.0
    return combined / row_sums[:, np.newaxis]


def build_difficulty_bin_matrix(difficulties: FloatArray, bin_count: int) -> FloatArray:
    quantiles = np.quantile(difficulties, np.linspace(0.0, 1.0, bin_count + 1))
    columns: list[FloatArray] = []
    for index in range(bin_count):
        lower = quantiles[index]
        upper = quantiles[index + 1]
        if index == bin_count - 1:
            mask = (difficulties >= lower) & (difficulties <= upper)
        else:
            mask = (difficulties >= lower) & (difficulties < upper)
        columns.append(mask.astype(np.float64))
    matrix = np.column_stack(columns).astype(np.float64)
    row_sums = np.sum(matrix, axis=1)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums[:, np.newaxis]


def build_interaction_groups(base_groups: FloatArray, bin_groups: FloatArray) -> FloatArray:
    columns: list[FloatArray] = []
    for bin_index in range(bin_groups.shape[1]):
        bin_values = bin_groups[:, bin_index][:, np.newaxis]
        columns.append(base_groups * bin_values)
    interaction = np.column_stack(columns).astype(np.float64)
    row_sums = np.sum(interaction, axis=1)
    row_sums[row_sums == 0.0] = 1.0
    return interaction / row_sums[:, np.newaxis]


def balanced_accuracy_from_predictions(y_true: FloatArray, predicted_positive: BoolArray) -> float:
    actual_positive = y_true >= 0.5
    actual_negative = ~actual_positive
    if int(np.sum(actual_positive)) == 0 or int(np.sum(actual_negative)) == 0:
        return float("nan")
    true_positive_rate = float(np.mean(predicted_positive[actual_positive]))
    true_negative_rate = float(np.mean(~predicted_positive[actual_negative]))
    return 0.5 * (true_positive_rate + true_negative_rate)


def select_threshold(
    observed_y: FloatArray,
    observed_probabilities: FloatArray,
    threshold_strategy: str,
) -> float:
    if threshold_strategy == "fixed_0.50":
        return 0.50
    if threshold_strategy == "observed_rate":
        return float(np.clip(np.mean(observed_y), 0.10, 0.90))
    if threshold_strategy in {"observed_ba_opt", "observed_ba_opt_shrunk"}:
        threshold_candidates = np.linspace(0.10, 0.90, 161)
        scores: list[tuple[float, float]] = []
        for threshold in threshold_candidates:
            predicted = observed_probabilities >= threshold
            score = balanced_accuracy_from_predictions(observed_y, predicted)
            if math.isnan(score):
                continue
            scores.append((score, float(threshold)))
        if not scores:
            return 0.50
        best_score = max(score for score, _ in scores)
        best_thresholds = [threshold for score, threshold in scores if abs(score - best_score) <= 1e-12]
        optimal_threshold = min(best_thresholds, key=lambda value: abs(value - 0.50))
        if threshold_strategy == "observed_ba_opt":
            return float(optimal_threshold)
        answer_count = int(observed_y.shape[0])
        shrink_gate = float(answer_count / (answer_count + 30))
        return float(0.50 + shrink_gate * (optimal_threshold - 0.50))
    raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}")


def calibration_error(y_true: FloatArray, y_prob: FloatArray, bin_count: int) -> float:
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    total = float(y_true.shape[0])
    error = 0.0
    for index in range(bin_count):
        lower = bins[index]
        upper = bins[index + 1]
        if index == bin_count - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        count = int(np.sum(mask))
        if count == 0:
            continue
        observed = float(np.mean(y_true[mask]))
        expected = float(np.mean(y_prob[mask]))
        error += (count / total) * abs(observed - expected)
    return float(error)


def evaluate_predictions(
    y_true: FloatArray, y_prob: FloatArray, threshold: float
) -> tuple[float, float, float, float, float, float, float, float]:
    epsilon = 1e-9
    clipped = np.clip(y_prob, epsilon, 1.0 - epsilon)
    log_loss = -float(np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))
    brier = float(np.mean((y_true - y_prob) ** 2.0))
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        auc = float("nan")
        pr_auc = float("nan")
    predicted_positive = y_prob >= threshold
    actual_positive = y_true >= 0.5
    accuracy = float(np.mean(predicted_positive == actual_positive))
    balanced_accuracy = balanced_accuracy_from_predictions(y_true, predicted_positive)
    ece = calibration_error(y_true, y_prob, 10)
    vocabulary_mae = abs(float(np.sum(y_prob)) - float(np.sum(y_true)))
    return log_loss, brier, auc, pr_auc, accuracy, balanced_accuracy, ece, vocabulary_mae


def evaluate_model(
    dataset: Dataset,
    spec: ModelSpec,
    q_values: list[int],
    repeats: int,
    seed: int,
) -> list[Metrics]:
    rng = np.random.default_rng(seed)
    metrics: list[Metrics] = []
    word_count = len(dataset.words)
    for q in q_values:
        if q >= word_count:
            raise ValueError(f"q={q} must be smaller than word count {word_count}")
        for repeat in range(repeats):
            for user_index, user_id in enumerate(dataset.user_ids):
                permutation = rng.permutation(word_count)
                observed_indices = permutation[:q]
                hidden_indices = permutation[q:]
                observed_y = dataset.known[user_index, observed_indices].astype(np.float64)
                observed_b = dataset.difficulties[observed_indices]
                observed_q = spec.group_matrix[observed_indices]
                fit = fit_grouped_map(
                    observed_y,
                    observed_b,
                    observed_q,
                    spec.tau_theta,
                    spec.tau_delta,
                    spec.gate_c,
                    spec.fit_strategy,
                )
                observed_probabilities = predict_probability(fit, observed_b, observed_q)
                threshold = select_threshold(observed_y, observed_probabilities, spec.threshold_strategy)
                hidden_probabilities = predict_probability(
                    fit,
                    dataset.difficulties[hidden_indices],
                    spec.group_matrix[hidden_indices],
                )
                hidden_y = dataset.known[user_index, hidden_indices].astype(np.float64)
                log_loss, brier, auc, pr_auc, accuracy, balanced_accuracy, ece, vocabulary_mae = evaluate_predictions(
                    hidden_y,
                    hidden_probabilities,
                    threshold,
                )
                metrics.append(
                    Metrics(
                        model=spec.name,
                        q=q,
                        repeat=repeat,
                        log_loss=log_loss,
                        brier=brier,
                        auc=auc,
                        pr_auc=pr_auc,
                        accuracy=accuracy,
                        balanced_accuracy=balanced_accuracy,
                        calibration_error=ece,
                        vocabulary_mae=vocabulary_mae,
                    )
                )
                _ = user_id
    return metrics


def evaluate_difficulty_only_model(
    dataset: Dataset,
    spec: DifficultyOnlySpec,
    q_values: list[int],
    repeats: int,
    seed: int,
) -> list[Metrics]:
    rng = np.random.default_rng(seed)
    metrics: list[Metrics] = []
    word_count = len(dataset.words)
    base_probabilities = expit(-dataset.difficulties)
    for q in q_values:
        if q >= word_count:
            raise ValueError(f"q={q} must be smaller than word count {word_count}")
        for repeat in range(repeats):
            for user_index in range(len(dataset.user_ids)):
                permutation = rng.permutation(word_count)
                observed_indices = permutation[:q]
                hidden_indices = permutation[q:]
                observed_probabilities = base_probabilities[observed_indices]
                observed_y = dataset.known[user_index, observed_indices].astype(np.float64)
                threshold = select_threshold(observed_y, observed_probabilities, spec.threshold_strategy)
                hidden_probabilities = base_probabilities[hidden_indices]
                hidden_y = dataset.known[user_index, hidden_indices].astype(np.float64)
                log_loss, brier, auc, pr_auc, accuracy, balanced_accuracy, ece, vocabulary_mae = evaluate_predictions(
                    hidden_y,
                    hidden_probabilities,
                    threshold,
                )
                metrics.append(
                    Metrics(
                        model=spec.name,
                        q=q,
                        repeat=repeat,
                        log_loss=log_loss,
                        brier=brier,
                        auc=auc,
                        pr_auc=pr_auc,
                        accuracy=accuracy,
                        balanced_accuracy=balanced_accuracy,
                        calibration_error=ece,
                        vocabulary_mae=vocabulary_mae,
                    )
                )
    return metrics


def aggregate_metrics(metrics: list[Metrics]) -> pd.DataFrame:
    frame = pd.DataFrame([metric.__dict__ for metric in metrics])
    grouped = frame.groupby(["model", "q"], as_index=False).agg(
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        auc=("auc", "mean"),
        pr_auc=("pr_auc", "mean"),
        accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        calibration_error=("calibration_error", "mean"),
        vocabulary_mae=("vocabulary_mae", "mean"),
    )
    return grouped.sort_values(["q", "balanced_accuracy", "log_loss"], ascending=[True, False, True]).reset_index(
        drop=True
    )


def format_float(value: float, precision: int) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


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
    aggregate: pd.DataFrame,
    baseline_specs: list[DifficultyOnlySpec],
    model_specs: list[ModelSpec],
    q_values: list[int],
    repeats: int,
) -> None:
    best_by_q = aggregate.sort_values(["q", "balanced_accuracy", "log_loss"], ascending=[True, False, True]).groupby(
        "q", as_index=False
    ).first()
    baseline_rows = [
        {
            "model": spec.name,
            "groups": 0,
            "tau_theta": "n/a",
            "tau_delta": "n/a",
            "gate_c": "n/a",
            "fit_strategy": "none",
            "threshold_strategy": spec.threshold_strategy,
        }
        for spec in baseline_specs
    ]
    learner_rows = [
        {
            "model": spec.name,
            "groups": int(spec.group_matrix.shape[1]),
            "tau_theta": spec.tau_theta,
            "tau_delta": spec.tau_delta,
            "gate_c": spec.gate_c,
            "fit_strategy": spec.fit_strategy,
            "threshold_strategy": spec.threshold_strategy,
        }
        for spec in model_specs
    ]
    model_rows = pd.DataFrame(baseline_rows + learner_rows)

    lines = [
        "# Grouped residual IRT model comparison",
        "",
        "## Setup",
        "",
        f"- Evaluation vocabulary: {dataset.excel_coverage}.",
        "- Binary label: Ehara `raw_score >= 4` is treated as known; scores 1-3 are treated as unknown/uncertain.",
        f"- Users: {len(dataset.user_ids)}.",
        f"- Cold-start reveal counts: {', '.join(str(value) for value in q_values)}.",
        f"- Repeats per user and reveal count: {repeats}.",
        "- Difficulty: Excel `Words.accuracy` converted to Rasch difficulty with clipped `-logit(accuracy)`, then standardized.",
        "- Grouping in this run: residual-response clusters (simple k-means grouping over Rasch residual profiles).",
        "",
        "## Model family",
        "",
        "All grouped models use the spec's residual form:",
        "",
        "`sigmoid(theta - b_i + lambda_q * Q_i @ delta_u)`",
        "",
        "The optimizer first fits Rasch-only `theta`, then jointly fits `theta` and group residuals with Gaussian priors. "
        "`lambda_q = q / (q + c)` and `tau_delta < tau_theta` shrink group effects during cold start.",
        "",
        markdown_table(
            model_rows,
            [
                "model",
                "groups",
                "tau_theta",
                "tau_delta",
                "gate_c",
                "fit_strategy",
                "threshold_strategy",
            ],
        ),
        "",
        "## Best model by reveal count",
        "",
        markdown_table(
            best_by_q[
                [
                    "q",
                    "model",
                    "log_loss",
                    "brier",
                    "auc",
                    "pr_auc",
                    "accuracy",
                    "balanced_accuracy",
                    "calibration_error",
                    "vocabulary_mae",
                ]
            ],
            [
                "q",
                "model",
                "log_loss",
                "brier",
                "auc",
                "pr_auc",
                "accuracy",
                "balanced_accuracy",
                "calibration_error",
                "vocabulary_mae",
            ],
        ),
        "",
        "## Full aggregate results",
        "",
        markdown_table(
            aggregate[
                [
                    "q",
                    "model",
                    "log_loss",
                    "brier",
                    "auc",
                    "pr_auc",
                    "accuracy",
                    "balanced_accuracy",
                    "calibration_error",
                    "vocabulary_mae",
                ]
            ],
            [
                "q",
                "model",
                "log_loss",
                "brier",
                "auc",
                "pr_auc",
                "accuracy",
                "balanced_accuracy",
                "calibration_error",
                "vocabulary_mae",
            ],
        ),
        "",
        "## Interpretation",
        "",
    ]

    rasch = aggregate[aggregate["model"] == "rasch_std_fixed"]
    best = best_by_q
    for _, best_row in best.iterrows():
        q = int(best_row["q"])
        rasch_row = rasch[rasch["q"] == q].iloc[0]
        improvement = float(best_row["balanced_accuracy"] - rasch_row["balanced_accuracy"])
        lines.append(
            f"- At q={q}, `{best_row['model']}` has the highest balanced accuracy "
            f"({format_float(float(best_row['balanced_accuracy']), 4)}), improving over Rasch by "
            f"{format_float(improvement, 4)}."
        )

    lines.extend(
        [
            "- Ranking in this report is by balanced accuracy first (higher is better), then log loss.",
            "- Residual-response groups are expected to perform strongly here because they are derived from the same response matrix. "
            "Treat those results as an optimistic upper-bound for learned behavioral groupings unless groups are rebuilt on a separate calibration sample.",
            "- Balanced-accuracy-oriented threshold optimization can degrade calibration and probability quality. "
            "Use log loss and Brier score as guardrails when choosing production settings.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_model_specs(dataset: Dataset, seed: int) -> list[ModelSpec]:
    empty = np.zeros((len(dataset.words), 0), dtype=np.float64)
    residual_8 = build_residual_clusters(dataset.known, dataset.difficulties, 8, 2.0, seed + 108)
    residual_12 = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, seed + 112)
    residual_12_seed1301 = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 1301)
    residual_12_seed701 = build_residual_clusters(dataset.known, dataset.difficulties, 12, 2.0, 701)

    return [
        ModelSpec("rasch_std_fixed", empty, 2.0, 0.5, 50.0, "standard", "fixed_0.50"),
        ModelSpec("rasch_balanced_fit", empty, 2.0, 0.5, 50.0, "class_weighted", "fixed_0.50"),
        ModelSpec("rasch_balanced_fit_ba_threshold", empty, 2.0, 0.5, 50.0, "class_weighted", "observed_ba_opt"),
        ModelSpec("residual_k8_bal_tau1p0_g20_shrunk", residual_8, 2.0, 1.00, 20.0, "class_weighted", "observed_ba_opt_shrunk"),
        ModelSpec("residual_k8_std_tau1p0_g20_ba", residual_8, 2.0, 1.00, 20.0, "standard", "observed_ba_opt"),
        ModelSpec("residual_k12_bal_tau1p0_g20_ba", residual_12, 2.0, 1.00, 20.0, "class_weighted", "observed_ba_opt"),
        ModelSpec("residual_k12_seed1301_tau1p4_g12_shrunk", residual_12_seed1301, 2.0, 1.40, 12.0, "class_weighted", "observed_ba_opt_shrunk"),
        ModelSpec("residual_k12_seed701_tau1p4_g8_shrunk", residual_12_seed701, 2.0, 1.40, 8.0, "class_weighted", "observed_ba_opt_shrunk"),
        ModelSpec("residual_k12_seed701_tau1p6_g12_shrunk", residual_12_seed701, 2.0, 1.60, 12.0, "class_weighted", "observed_ba_opt_shrunk"),
        ModelSpec("residual_k12_seed1301_tau1p6_g8_shrunk", residual_12_seed1301, 2.0, 1.60, 8.0, "class_weighted", "observed_ba_opt_shrunk"),
    ]


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.responses), Path(args.word_difficulty))
    baseline_specs = [
        DifficultyOnlySpec("excel_difficulty_only_fixed", "fixed_0.50"),
        DifficultyOnlySpec("excel_difficulty_only_observed_ba_threshold", "observed_ba_opt"),
    ]
    model_specs = build_model_specs(dataset, int(args.seed))
    all_metrics: list[Metrics] = []
    for spec in baseline_specs:
        print(f"evaluating {spec.name} (no learner adaptation)", flush=True)
        all_metrics.extend(
            evaluate_difficulty_only_model(dataset, spec, args.q_values, int(args.repeats), int(args.seed))
        )
    for spec in model_specs:
        print(f"evaluating {spec.name} ({spec.group_matrix.shape[1]} groups)", flush=True)
        all_metrics.extend(evaluate_model(dataset, spec, args.q_values, int(args.repeats), int(args.seed)))
    aggregate = aggregate_metrics(all_metrics)
    write_summary(
        Path(args.summary),
        dataset,
        aggregate,
        baseline_specs,
        model_specs,
        args.q_values,
        int(args.repeats),
    )
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
