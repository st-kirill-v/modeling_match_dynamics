from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_class_weight


def compute_weights(y: np.ndarray) -> dict:
    classes = np.unique(y.astype(int))
    if len(classes) == 1:
        return {int(classes[0]): 1.0}
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.astype(int))
    return dict(zip(classes, weights))


def top_decile_lift(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob)
    cutoff = np.quantile(prob, 0.90)
    selected = y_true[prob >= cutoff]
    if len(selected) == 0 or y_true.mean() == 0:
        return np.nan
    return float(selected.mean() / y_true.mean())


def find_best_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    metric: str = "f1",
) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def threshold_candidates(y_val: np.ndarray, prob_val: np.ndarray) -> dict[str, float]:
    return {
        "fixed_0_5": 0.5,
        "best_f1": find_best_threshold(y_val, prob_val, metric="f1"),
        "best_balanced_accuracy": find_best_threshold(
            y_val,
            prob_val,
            metric="balanced_accuracy",
        ),
    }


def evaluate_binary(
    y_true: np.ndarray,
    prob: np.ndarray,
    name: str,
    threshold: float = 0.5,
    threshold_mode: str = "fixed_0_5",
) -> dict:
    pred = (prob >= threshold).astype(int)
    mse = mean_squared_error(y_true, prob)
    return {
        "model": name,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
        "mae": mean_absolute_error(y_true, prob),
        "mse": mse,
        "rmse": mse**0.5,
        "brier": brier_score_loss(y_true, prob),
        "top_decile_lift": top_decile_lift(y_true, prob),
    }


def calibrate_probabilities(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    prob_test: np.ndarray,
) -> np.ndarray:
    y_val = np.asarray(y_val).astype(int)
    prob_val = np.asarray(prob_val).astype(float)
    prob_test = np.asarray(prob_test).astype(float)
    if len(np.unique(y_val)) < 2:
        return np.clip(prob_test, 0.0, 1.0)

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(prob_val, y_val)
    return np.clip(calibrator.predict(prob_test), 0.0, 1.0)


def evaluate_regression(y_true: np.ndarray, pred: np.ndarray, name: str) -> dict:
    mse = mean_squared_error(y_true, pred)
    return {
        "model": name,
        "mae": mean_absolute_error(y_true, pred),
        "mse": mse,
        "rmse": mse**0.5,
        "r2": r2_score(y_true, pred) if len(y_true) > 1 else np.nan,
    }


def calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    calib = pd.DataFrame({"y": y_true, "prob": prob})
    calib["bin"] = pd.qcut(calib["prob"], q=bins, duplicates="drop")
    return calib.groupby("bin", observed=True).agg(
        mean_prob=("prob", "mean"), event_rate=("y", "mean"), n=("y", "size")
    )


def confusion_frame(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    return pd.DataFrame(
        confusion_matrix(y_true, (prob >= threshold).astype(int)),
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
