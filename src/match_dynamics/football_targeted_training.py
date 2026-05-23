from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .evaluation import evaluate_binary
from .football_lstm_ablation import (
    aggregate_feature_ranking,
    confusion_matrix_rows_for_prob,
    fit_and_evaluate_subset,
)
from .football_lstm_training import _prediction_dict, load_sequence_npz
from .football_threshold_tuning import FINAL_THRESHOLDS


TARGETS = ["home_scores_next_half", "away_scores_next_half"]


def _load_targeted_data(
    data_dir: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    X_train, y_train_home, y_train_away, feature_columns = load_sequence_npz(
        data_dir / "football_targeted_train_sequences.npz"
    )
    X_val, y_val_home, y_val_away, _ = load_sequence_npz(
        data_dir / "football_targeted_val_sequences.npz"
    )
    X_test, y_test_home, y_test_away, _ = load_sequence_npz(
        data_dir / "football_targeted_test_sequences.npz"
    )
    data = {
        "train": (X_train, y_train_home, y_train_away),
        "val": (X_val, y_val_home, y_val_away),
        "test": (X_test, y_test_home, y_test_away),
    }
    return data, feature_columns


def _final_threshold_metrics(
    model: tf.keras.Model,
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_indices: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    confusion_rows = []
    for split, (X, y_home, y_away) in data.items():
        X_sub = X[:, :, feature_indices]
        pred = _prediction_dict(model.predict(X_sub, batch_size=batch_size, verbose=0))
        for target, y_true, prob in [
            ("home_scores_next_half", y_home, pred["home_scores_next_half"]),
            ("away_scores_next_half", y_away, pred["away_scores_next_half"]),
        ]:
            row = evaluate_binary(
                y_true=y_true,
                prob=prob,
                name=f"targeted_top50_{target}_final_threshold",
                threshold=FINAL_THRESHOLDS[target],
            )
            row["split"] = split
            row["target"] = target
            row["threshold_mode"] = "final_fixed"
            rows.append(row)
            confusion_rows.extend(
                confusion_matrix_rows_for_prob(
                    split=split,
                    target=target,
                    y_true=y_true,
                    prob=prob,
                    threshold=FINAL_THRESHOLDS[target],
                    threshold_mode="final_fixed",
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def _baseline_comparison(
    targeted_metrics: pd.DataFrame,
    baseline_metrics_path: Path,
) -> pd.DataFrame:
    if not baseline_metrics_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_metrics_path)
    baseline_test = baseline[
        baseline["split"].eq("test") & baseline["threshold_mode"].eq("final_fixed")
    ].copy()
    targeted_test = targeted_metrics[
        targeted_metrics["split"].eq("test") & targeted_metrics["threshold_mode"].eq("final_fixed")
    ].copy()
    rows = []
    for target in TARGETS:
        base_row = baseline_test[baseline_test["target"].eq(target)].iloc[0]
        new_row = targeted_test[targeted_test["target"].eq(target)].iloc[0]
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "mae",
            "mse",
        ]:
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "baseline_top50": float(base_row[metric]),
                    "targeted_top50": float(new_row[metric]),
                    "delta_targeted_minus_baseline": float(new_row[metric] - base_row[metric]),
                }
            )
    return pd.DataFrame(rows)


def run_targeted_top50_training(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    baseline_metrics_path: Path,
    epochs: int = 25,
    batch_size: int = 32,
) -> dict[str, pd.DataFrame]:
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data, feature_columns = _load_targeted_data(data_dir)
    X_train, y_train_home, y_train_away = data["train"]
    ranking = aggregate_feature_ranking(X_train, y_train_home, y_train_away, feature_columns)
    selected = ranking.head(50).copy()
    feature_indices = selected["feature_index"].to_numpy(dtype=int)

    ranking.to_csv(metrics_dir / "targeted_feature_ranking.csv", index=False)
    selected.to_csv(metrics_dir / "targeted_top_50_selected_features.csv", index=False)

    metrics, summary = fit_and_evaluate_subset(
        feature_count=50,
        feature_indices=feature_indices,
        data=data,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        epochs=epochs,
        batch_size=batch_size,
        artifact_prefix="targeted_top50",
    )

    model = tf.keras.models.load_model(models_dir / "targeted_top50_top_50.keras")
    final_metrics, confusion = _final_threshold_metrics(model, data, feature_indices, batch_size)
    comparison = _baseline_comparison(final_metrics, baseline_metrics_path)

    metrics.to_csv(metrics_dir / "targeted_top50_metrics_threshold_0_5.csv", index=False)
    summary.to_csv(metrics_dir / "targeted_top50_training_summary.csv", index=False)
    final_metrics.to_csv(metrics_dir / "targeted_top50_final_threshold_metrics.csv", index=False)
    confusion.to_csv(metrics_dir / "targeted_top50_confusion_matrices.csv", index=False)
    comparison.to_csv(metrics_dir / "targeted_top50_vs_baseline_comparison.csv", index=False)

    return {
        "ranking": ranking,
        "selected_features": selected,
        "metrics_threshold_0_5": metrics,
        "training_summary": summary,
        "final_threshold_metrics": final_metrics,
        "confusion_matrices": confusion,
        "comparison": comparison,
    }
