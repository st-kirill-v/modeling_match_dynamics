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
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray, name: str) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, prob)
        if len(np.unique(y_true)) > 1
        else np.nan,
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, prob),
        "top_decile_lift": top_decile_lift(y_true, prob),
    }


def calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    calib = pd.DataFrame({"y": y_true, "prob": prob})
    calib["bin"] = pd.qcut(calib["prob"], q=bins, duplicates="drop")
    return calib.groupby("bin", observed=True).agg(
        mean_prob=("prob", "mean"), event_rate=("y", "mean"), n=("y", "size")
    )


def confusion_frame(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        confusion_matrix(y_true, (prob >= 0.5).astype(int)),
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
