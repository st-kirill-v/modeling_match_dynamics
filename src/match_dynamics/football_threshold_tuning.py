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


def plot_threshold_curves(curves: pd.DataFrame, target: str, output_path: Path) -> None:
    target_df = curves[curves["target"].eq(target)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["f1", "precision", "recall"]:
        ax.plot(target_df["threshold"], target_df[metric], label=metric)
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
