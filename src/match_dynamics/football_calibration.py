from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .evaluation import evaluate_binary
from .football_lstm_training import _prediction_dict
from .football_threshold_tuning import FINAL_THRESHOLDS, TARGETS, load_top_feature_indices


CALIBRATION_METHODS = ["raw", "platt", "isotonic"]


def load_sequence_npz_with_ids(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return (
        data["X"],
        data["y_home"],
        data["y_away"],
        data["match_ids"],
        data["feature_columns"],
    )


def fit_platt_calibrator(y_val: np.ndarray, prob_val: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(prob_val.reshape(-1, 1), y_val.astype(int))
    return calibrator


def fit_isotonic_calibrator(y_val: np.ndarray, prob_val: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(prob_val, y_val.astype(int))
    return calibrator


def apply_calibrators(
    prob: np.ndarray,
    platt: LogisticRegression,
    isotonic: IsotonicRegression,
) -> dict[str, np.ndarray]:
    return {
        "raw": np.clip(prob, 0.0, 1.0),
        "platt": np.clip(platt.predict_proba(prob.reshape(-1, 1))[:, 1], 0.0, 1.0),
        "isotonic": np.clip(isotonic.predict(prob), 0.0, 1.0),
    }


def build_metrics(
    y_test: dict[str, np.ndarray],
    calibrated_test: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for method in CALIBRATION_METHODS:
            row = evaluate_binary(
                y_true=y_test[target],
                prob=calibrated_test[target][method],
                name=f"top50_lstm_{target}_{method}",
                threshold=FINAL_THRESHOLDS[target],
            )
            row["target"] = target
            row["calibration_method"] = method
            row["split"] = "test"
            rows.append(row)
    return pd.DataFrame(rows)


def build_predictions(
    match_ids: np.ndarray,
    y_test: dict[str, np.ndarray],
    calibrated_test: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        target_df = pd.DataFrame(
            {
                "match_id": match_ids,
                "target": target,
                "true_label": y_test[target].astype(int),
                "threshold": FINAL_THRESHOLDS[target],
            }
        )
        for method in CALIBRATION_METHODS:
            prob = calibrated_test[target][method]
            target_df[f"{method}_prob"] = prob
            target_df[f"{method}_pred"] = (prob >= FINAL_THRESHOLDS[target]).astype(int)
        rows.append(target_df)
    return pd.concat(rows, ignore_index=True)


def build_calibration_bins(
    y_test: dict[str, np.ndarray],
    calibrated_test: dict[str, dict[str, np.ndarray]],
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for method in CALIBRATION_METHODS:
            prob_true, prob_pred = calibration_curve(
                y_test[target],
                calibrated_test[target][method],
                n_bins=n_bins,
                strategy="quantile",
            )
            for idx, (mean_predicted_prob, observed_rate) in enumerate(zip(prob_pred, prob_true)):
                rows.append(
                    {
                        "target": target,
                        "calibration_method": method,
                        "bin": idx + 1,
                        "mean_predicted_probability": mean_predicted_prob,
                        "observed_event_rate": observed_rate,
                    }
                )
    return pd.DataFrame(rows)


def plot_calibration_curve(
    bins: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    target_bins = bins[bins["target"].eq(target)].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for method in CALIBRATION_METHODS:
        method_bins = target_bins[target_bins["calibration_method"].eq(method)]
        if method_bins.empty:
            continue
        ax.plot(
            method_bins["mean_predicted_probability"],
            method_bins["observed_event_rate"],
            marker="o",
            label=method,
        )
    ax.set_title(f"Football LSTM calibration: {target}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = metrics.set_index(["target", "calibration_method"])
    for target in TARGETS:
        raw = indexed.loc[(target, "raw")]
        for method in ["platt", "isotonic"]:
            current = indexed.loc[(target, method)]
            for metric in [
                "brier",
                "log_loss",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]:
                rows.append(
                    {
                        "target": target,
                        "metric": metric,
                        "raw": float(raw[metric]),
                        "calibrated_method": method,
                        "calibrated": float(current[metric]),
                        "delta_calibrated_minus_raw": float(current[metric] - raw[metric]),
                    }
                )
    return pd.DataFrame(rows)


def run_football_probability_calibration(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    calibrators_dir: Path,
    selected_features_path: Path,
    batch_size: int = 32,
) -> dict[str, pd.DataFrame]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    calibrators_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "feature_ablation_fast_top_50.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Top-50 LSTM model not found: {model_path}")

    feature_indices = load_top_feature_indices(selected_features_path)
    X_val, y_val_home, y_val_away, val_match_ids, _ = load_sequence_npz_with_ids(
        data_dir / "football_val_sequences.npz"
    )
    X_test, y_test_home, y_test_away, test_match_ids, _ = load_sequence_npz_with_ids(
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

    calibrated_val: dict[str, dict[str, np.ndarray]] = {}
    calibrated_test: dict[str, dict[str, np.ndarray]] = {}
    calibrator_rows = []
    for target in TARGETS:
        platt = fit_platt_calibrator(y_val[target], val_pred[target])
        isotonic = fit_isotonic_calibrator(y_val[target], val_pred[target])

        joblib.dump(platt, calibrators_dir / f"platt_{target}.pkl")
        joblib.dump(isotonic, calibrators_dir / f"isotonic_{target}.pkl")

        calibrated_val[target] = apply_calibrators(val_pred[target], platt, isotonic)
        calibrated_test[target] = apply_calibrators(test_pred[target], platt, isotonic)
        calibrator_rows.append(
            {
                "target": target,
                "platt_path": str(calibrators_dir / f"platt_{target}.pkl"),
                "isotonic_path": str(calibrators_dir / f"isotonic_{target}.pkl"),
                "validation_matches": len(val_match_ids),
                "test_matches": len(test_match_ids),
                "threshold_after_calibration": FINAL_THRESHOLDS[target],
            }
        )

    metrics = build_metrics(y_test, calibrated_test)
    comparison = build_comparison(metrics)
    predictions = build_predictions(test_match_ids, y_test, calibrated_test)
    bins = build_calibration_bins(y_test, calibrated_test)
    calibrators = pd.DataFrame(calibrator_rows)
    diagnostics = pd.DataFrame(
        [
            {
                "model_path": str(model_path),
                "selected_features_path": str(selected_features_path),
                "validation_shape": str(X_val.shape),
                "test_shape": str(X_test.shape),
                "feature_count": X_test.shape[2],
                "calibration_fit_split": "validation",
                "final_evaluation_split": "test",
            }
        ]
    )

    metrics.to_csv(metrics_dir / "calibration_metrics.csv", index=False)
    comparison.to_csv(metrics_dir / "calibration_comparison.csv", index=False)
    predictions.to_csv(metrics_dir / "calibrated_predictions.csv", index=False)
    bins.to_csv(metrics_dir / "calibration_bins.csv", index=False)
    calibrators.to_csv(metrics_dir / "calibrator_paths.csv", index=False)
    diagnostics.to_csv(metrics_dir / "calibration_diagnostics.csv", index=False)

    for target in TARGETS:
        plot_calibration_curve(
            bins,
            target,
            figures_dir / f"calibration_curve_{target}.png",
        )

    return {
        "metrics": metrics,
        "comparison": comparison,
        "predictions": predictions,
        "calibration_bins": bins,
        "calibrators": calibrators,
        "diagnostics": diagnostics,
    }
