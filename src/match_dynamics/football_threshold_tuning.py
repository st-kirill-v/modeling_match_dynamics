from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .evaluation import evaluate_binary
from .football_lstm_training import _prediction_dict, load_sequence_npz


THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)
TARGETS = {
    "home_scores_next_half": "home_output",
    "away_scores_next_half": "away_output",
}
PRECISION_CONSTRAINTS = {
    "home_scores_next_half": 0.60,
    "away_scores_next_half": 0.55,
}


def load_top_feature_indices(selected_features_path: Path) -> np.ndarray:
    selected = pd.read_csv(selected_features_path)
    if "feature_index" not in selected.columns:
        raise ValueError(f"`feature_index` column not found in {selected_features_path}")
    return selected["feature_index"].to_numpy(dtype=int)


def threshold_curve(y_true: np.ndarray, prob: np.ndarray, target: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        row = evaluate_binary(
            y_true=y_true,
            prob=prob,
            name=f"top50_threshold_{target}",
            threshold=float(threshold),
        )
        row["target"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def plot_threshold_curves(
    curves: pd.DataFrame,
    target: str,
    output_path: Path,
    precision_constraint: float | None = None,
) -> None:
    target_df = curves[curves["target"].eq(target)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["f1", "precision", "recall"]:
        ax.plot(target_df["threshold"], target_df[metric], label=metric)
    if precision_constraint is not None:
        ax.axhline(
            precision_constraint,
            linestyle="--",
            color="black",
            alpha=0.7,
            label=f"precision constraint={precision_constraint:.2f}",
        )
    ax.set_title(f"Threshold tuning: {target}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0.10, 0.90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def best_thresholds_by_f1(validation_curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        target_df = validation_curves[validation_curves["target"].eq(target)].copy()
        best_idx = target_df["f1"].idxmax()
        rows.append(target_df.loc[best_idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def balanced_thresholds_by_f1(validation_curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, min_precision in PRECISION_CONSTRAINTS.items():
        target_df = validation_curves[validation_curves["target"].eq(target)].copy()
        candidates = target_df[target_df["precision"].ge(min_precision)].copy()
        if candidates.empty:
            best_idx = target_df["f1"].idxmax()
            selected = target_df.loc[best_idx].copy()
            selected["constraint_satisfied"] = False
            selected["precision_constraint"] = min_precision
        else:
            best_idx = candidates["f1"].idxmax()
            selected = candidates.loc[best_idx].copy()
            selected["constraint_satisfied"] = True
            selected["precision_constraint"] = min_precision
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_fixed_and_tuned(
    y_true_by_target: dict[str, np.ndarray],
    prob_by_target: dict[str, np.ndarray],
    best_thresholds: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    rows = []
    best_map = dict(zip(best_thresholds["target"], best_thresholds["threshold"]))
    for target in TARGETS:
        y_true = y_true_by_target[target]
        prob = prob_by_target[target]
        for label, threshold in [("default_0.5", 0.5), ("tuned", float(best_map[target]))]:
            row = evaluate_binary(
                y_true=y_true,
                prob=prob,
                name=f"top50_{label}_{target}",
                threshold=threshold,
            )
            row["split"] = split
            row["target"] = target
            row["threshold_mode"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_threshold_modes(
    y_true_by_target: dict[str, np.ndarray],
    prob_by_target: dict[str, np.ndarray],
    threshold_sets: dict[str, pd.DataFrame],
    split: str,
) -> pd.DataFrame:
    threshold_maps = {
        name: dict(zip(thresholds["target"], thresholds["threshold"]))
        for name, thresholds in threshold_sets.items()
    }
    rows = []
    for target in TARGETS:
        y_true = y_true_by_target[target]
        prob = prob_by_target[target]
        for label, threshold_map in threshold_maps.items():
            threshold = 0.5 if label == "default_0.5" else float(threshold_map[target])
            row = evaluate_binary(
                y_true=y_true,
                prob=prob,
                name=f"top50_{label}_{target}",
                threshold=threshold,
            )
            row["split"] = split
            row["target"] = target
            row["threshold_mode"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    default = test[test["threshold_mode"].eq("default_0.5")].set_index("target")
    tuned = test[test["threshold_mode"].eq("tuned")].set_index("target")
    rows = []
    for target in TARGETS:
        for metric in ["precision", "recall", "f1"]:
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "threshold_0_5": float(default.loc[target, metric]),
                    "best_threshold": float(tuned.loc[target, metric]),
                    "delta": float(tuned.loc[target, metric] - default.loc[target, metric]),
                }
            )
    return pd.DataFrame(rows)


def balanced_threshold_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"].eq("test")].copy()
    pivot = test.set_index(["target", "threshold_mode"])
    rows = []
    for target in TARGETS:
        for metric in ["precision", "recall", "f1"]:
            default_value = float(pivot.loc[(target, "default_0.5"), metric])
            previous_value = float(pivot.loc[(target, "previous_f1_tuned"), metric])
            balanced_value = float(pivot.loc[(target, "balanced"), metric])
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "threshold_0_5": default_value,
                    "previous_tuned": previous_value,
                    "balanced": balanced_value,
                    "balanced_minus_0_5": balanced_value - default_value,
                    "balanced_minus_previous": balanced_value - previous_value,
                }
            )
    return pd.DataFrame(rows)


def run_football_threshold_tuning(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    selected_features_path: Path,
    batch_size: int = 32,
) -> dict[str, pd.DataFrame]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "feature_ablation_fast_top_50.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Top-50 LSTM model not found: {model_path}")

    feature_indices = load_top_feature_indices(selected_features_path)
    X_val, y_val_home, y_val_away, _ = load_sequence_npz(data_dir / "football_val_sequences.npz")
    X_test, y_test_home, y_test_away, _ = load_sequence_npz(
        data_dir / "football_test_sequences.npz"
    )
    X_val = X_val[:, :, feature_indices]
    X_test = X_test[:, :, feature_indices]

    model = tf.keras.models.load_model(model_path)
    val_pred = _prediction_dict(model.predict(X_val, batch_size=batch_size, verbose=0))
    test_pred = _prediction_dict(model.predict(X_test, batch_size=batch_size, verbose=0))

    y_val = {
        "home_scores_next_half": y_val_home,
        "away_scores_next_half": y_val_away,
    }
    y_test = {
        "home_scores_next_half": y_test_home,
        "away_scores_next_half": y_test_away,
    }

    validation_curves = pd.concat(
        [threshold_curve(y_val[target], val_pred[target], target) for target in TARGETS],
        ignore_index=True,
    )
    best_thresholds = best_thresholds_by_f1(validation_curves)
    validation_metrics = evaluate_fixed_and_tuned(y_val, val_pred, best_thresholds, "val")
    test_metrics = evaluate_fixed_and_tuned(y_test, test_pred, best_thresholds, "test")
    metrics = pd.concat([validation_metrics, test_metrics], ignore_index=True)
    comparison = threshold_comparison(metrics)

    validation_curves.to_csv(metrics_dir / "threshold_validation_curves.csv", index=False)
    best_thresholds.to_csv(metrics_dir / "best_thresholds.csv", index=False)
    metrics.to_csv(metrics_dir / "threshold_metrics.csv", index=False)
    test_metrics.to_csv(metrics_dir / "tuned_test_metrics.csv", index=False)
    comparison.to_csv(metrics_dir / "threshold_0_5_vs_tuned_comparison.csv", index=False)

    pd.DataFrame(
        {
            "split": ["validation", "test"],
            "matches": [len(X_val), len(X_test)],
            "timesteps": [X_val.shape[1], X_test.shape[1]],
            "feature_count": [X_val.shape[2], X_test.shape[2]],
            "model_path": [str(model_path), str(model_path)],
            "selected_features_path": [str(selected_features_path), str(selected_features_path)],
        }
    ).to_csv(metrics_dir / "threshold_tuning_diagnostics.csv", index=False)

    for target in TARGETS:
        plot_threshold_curves(
            validation_curves,
            target,
            figures_dir / f"threshold_curves_{target}.png",
        )

    return {
        "validation_curves": validation_curves,
        "best_thresholds": best_thresholds,
        "metrics": metrics,
        "comparison": comparison,
        "tuned_test_metrics": test_metrics,
    }


def run_balanced_football_threshold_tuning(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    selected_features_path: Path,
    batch_size: int = 32,
) -> dict[str, pd.DataFrame]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "feature_ablation_fast_top_50.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Top-50 LSTM model not found: {model_path}")

    feature_indices = load_top_feature_indices(selected_features_path)
    X_val, y_val_home, y_val_away, _ = load_sequence_npz(data_dir / "football_val_sequences.npz")
    X_test, y_test_home, y_test_away, _ = load_sequence_npz(
        data_dir / "football_test_sequences.npz"
    )
    X_val = X_val[:, :, feature_indices]
    X_test = X_test[:, :, feature_indices]

    model = tf.keras.models.load_model(model_path)
    val_pred = _prediction_dict(model.predict(X_val, batch_size=batch_size, verbose=0))
    test_pred = _prediction_dict(model.predict(X_test, batch_size=batch_size, verbose=0))

    y_val = {
        "home_scores_next_half": y_val_home,
        "away_scores_next_half": y_val_away,
    }
    y_test = {
        "home_scores_next_half": y_test_home,
        "away_scores_next_half": y_test_away,
    }

    validation_curves = pd.concat(
        [threshold_curve(y_val[target], val_pred[target], target) for target in TARGETS],
        ignore_index=True,
    )
    previous_thresholds = best_thresholds_by_f1(validation_curves)
    balanced_thresholds = balanced_thresholds_by_f1(validation_curves)
    default_thresholds = pd.DataFrame([{"target": target, "threshold": 0.5} for target in TARGETS])
    threshold_sets = {
        "default_0.5": default_thresholds,
        "previous_f1_tuned": previous_thresholds,
        "balanced": balanced_thresholds,
    }
    validation_metrics = evaluate_threshold_modes(y_val, val_pred, threshold_sets, "val")
    test_metrics = evaluate_threshold_modes(y_test, test_pred, threshold_sets, "test")
    metrics = pd.concat([validation_metrics, test_metrics], ignore_index=True)
    comparison = balanced_threshold_comparison(metrics)

    validation_curves.to_csv(metrics_dir / "balanced_threshold_validation_curves.csv", index=False)
    previous_thresholds.to_csv(metrics_dir / "previous_f1_tuned_thresholds.csv", index=False)
    balanced_thresholds.to_csv(metrics_dir / "balanced_thresholds.csv", index=False)
    metrics.to_csv(metrics_dir / "balanced_threshold_metrics.csv", index=False)
    test_metrics.to_csv(metrics_dir / "balanced_tuned_test_metrics.csv", index=False)
    comparison.to_csv(metrics_dir / "balanced_threshold_comparison.csv", index=False)

    for target in TARGETS:
        plot_threshold_curves(
            validation_curves,
            target,
            figures_dir / f"balanced_threshold_curves_{target}.png",
            precision_constraint=PRECISION_CONSTRAINTS[target],
        )

    return {
        "validation_curves": validation_curves,
        "previous_thresholds": previous_thresholds,
        "balanced_thresholds": balanced_thresholds,
        "metrics": metrics,
        "comparison": comparison,
        "balanced_test_metrics": test_metrics,
    }
