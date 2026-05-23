from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from .football_lstm_training import (
    _prediction_dict,
    evaluate_split,
    load_sequence_npz,
    plot_pr_curve,
    plot_roc_curve,
    save_history,
)
from .evaluation import evaluate_binary
from .football_threshold_tuning import FINAL_THRESHOLDS


FEATURE_COUNTS = [100, 75, 50, 25]


def build_ablation_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="first_half_sequence")
    x = tf.keras.layers.LSTM(64, return_sequences=False, name="lstm_64")(inputs)
    x = tf.keras.layers.Dropout(0.4, name="dropout_04")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_32_relu")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_03")(x)
    home_output = tf.keras.layers.Dense(1, activation="sigmoid", name="home_output")(x)
    away_output = tf.keras.layers.Dense(1, activation="sigmoid", name="away_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=[home_output, away_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={"home_output": "binary_crossentropy", "away_output": "binary_crossentropy"},
        metrics={
            "home_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "away_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        },
    )
    return model


def confusion_matrix_rows_for_prob(
    split: str,
    target: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    threshold_mode: str,
) -> list[dict]:
    matrix = confusion_matrix(y_true, (prob >= threshold).astype(int), labels=[0, 1])
    rows = []
    for true_label in [0, 1]:
        for predicted_label in [0, 1]:
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "threshold_mode": threshold_mode,
                    "threshold": threshold,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[true_label, predicted_label]),
                }
            )
    return rows


def aggregate_feature_ranking(
    X_train: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    feature_columns: np.ndarray,
) -> pd.DataFrame:
    flat_x = X_train.reshape(-1, X_train.shape[-1]).astype("float64")
    y_home_rep = np.repeat(y_home.astype("float64"), X_train.shape[1])
    y_away_rep = np.repeat(y_away.astype("float64"), X_train.shape[1])

    rows = []
    for idx, feature in enumerate(feature_columns.astype(str)):
        values = flat_x[:, idx]
        if np.std(values) == 0:
            home_corr = 0.0
            away_corr = 0.0
        else:
            home_corr = np.corrcoef(values, y_home_rep)[0, 1]
            away_corr = np.corrcoef(values, y_away_rep)[0, 1]
            home_corr = 0.0 if np.isnan(home_corr) else float(home_corr)
            away_corr = 0.0 if np.isnan(away_corr) else float(away_corr)
        rows.append(
            {
                "feature": feature,
                "feature_index": idx,
                "home_corr": home_corr,
                "away_corr": away_corr,
                "home_abs_corr": abs(home_corr),
                "away_abs_corr": abs(away_corr),
                "aggregate_abs_corr": (abs(home_corr) + abs(away_corr)) / 2,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("aggregate_abs_corr", ascending=False).reset_index(drop=True)
    )


def plot_ablation_loss(history: tf.keras.callbacks.History, title: str, output_path: Path) -> None:
    hist = history.history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["loss"], label="train_loss")
    ax.plot(hist["val_loss"], label="val_loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fit_and_evaluate_subset(
    feature_count: int,
    feature_indices: np.ndarray,
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    epochs: int,
    batch_size: int,
    artifact_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_set = f"top_{feature_count}"
    X_train, y_train_home, y_train_away = data["train"]
    X_val, y_val_home, y_val_away = data["val"]
    X_test, y_test_home, y_test_away = data["test"]

    X_train_sub = X_train[:, :, feature_indices]
    X_val_sub = X_val[:, :, feature_indices]
    X_test_sub = X_test[:, :, feature_indices]

    model = build_ablation_lstm((X_train_sub.shape[1], X_train_sub.shape[2]))
    history = model.fit(
        X_train_sub,
        {"home_output": y_train_home, "away_output": y_train_away},
        validation_data=(X_val_sub, {"home_output": y_val_home, "away_output": y_val_away}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )
    model.save(models_dir / f"{artifact_prefix}_{feature_set}.keras")
    save_history(
        history,
        metrics_dir / f"{artifact_prefix}_{feature_set}_history.csv",
        metrics_dir / f"{artifact_prefix}_{feature_set}_history.json",
    )
    plot_ablation_loss(
        history,
        f"Football LSTM Feature Ablation: {feature_set}",
        figures_dir / f"{artifact_prefix}_{feature_set}_loss.png",
    )

    metric_rows = []
    for split, X, y_home, y_away in [
        ("train", X_train_sub, y_train_home, y_train_away),
        ("val", X_val_sub, y_val_home, y_val_away),
        ("test", X_test_sub, y_test_home, y_test_away),
    ]:
        pred = _prediction_dict(model.predict(X, batch_size=batch_size, verbose=0))
        rows = evaluate_split(
            split,
            y_home,
            y_away,
            pred["home_scores_next_half"],
            pred["away_scores_next_half"],
        )
        for row in rows:
            row["feature_set"] = feature_set
            row["feature_count"] = feature_count
        metric_rows.extend(rows)
        if split == "test":
            for target, y_true, prob in [
                ("home_scores_next_half", y_home, pred["home_scores_next_half"]),
                ("away_scores_next_half", y_away, pred["away_scores_next_half"]),
            ]:
                plot_roc_curve(
                    y_true,
                    prob,
                    f"Feature ablation {feature_set} ROC: {target}",
                    figures_dir / f"{artifact_prefix}_{feature_set}_{target}_roc.png",
                )
                plot_pr_curve(
                    y_true,
                    prob,
                    f"Feature ablation {feature_set} PR: {target}",
                    figures_dir / f"{artifact_prefix}_{feature_set}_{target}_pr.png",
                )

    hist_df = pd.DataFrame(history.history)
    best_idx = int(hist_df["val_loss"].idxmin())
    summary = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "feature_count": feature_count,
                "best_epoch": best_idx + 1,
                "best_val_loss": float(hist_df.loc[best_idx, "val_loss"]),
                "final_train_loss": float(hist_df["loss"].iloc[-1]),
                "final_val_loss": float(hist_df["val_loss"].iloc[-1]),
                "overfitting_gap": float(hist_df["val_loss"].iloc[-1] - hist_df["loss"].iloc[-1]),
            }
        ]
    )
    return pd.DataFrame(metric_rows), summary


def build_comparison(
    metrics_df: pd.DataFrame, summaries_df: pd.DataFrame, metrics_dir: Path
) -> pd.DataFrame:
    rows = []
    baseline_path = metrics_dir / "baseline_lstm_metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline = baseline[baseline["split"].eq("test")].copy()
        baseline["feature_set"] = "all_238_baseline"
        baseline["feature_count"] = 238
        baseline["best_epoch"] = pd.NA
        baseline["final_train_loss"] = pd.NA
        baseline["final_val_loss"] = pd.NA
        baseline["overfitting_gap"] = pd.NA
        rows.append(baseline)

    test_metrics = metrics_df[metrics_df["split"].eq("test")].merge(
        summaries_df[
            [
                "feature_set",
                "best_epoch",
                "final_train_loss",
                "final_val_loss",
                "overfitting_gap",
            ]
        ],
        on="feature_set",
        how="left",
    )
    rows.append(test_metrics)
    comparison = pd.concat(rows, ignore_index=True)
    keep = [
        "feature_set",
        "feature_count",
        "target",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mae",
        "mse",
        "best_epoch",
        "final_train_loss",
        "final_val_loss",
        "overfitting_gap",
    ]
    return comparison[[col for col in keep if col in comparison.columns]]


def run_football_feature_ablation(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    epochs: int = 25,
    batch_size: int = 32,
    artifact_prefix: str = "feature_ablation_fast",
) -> dict[str, pd.DataFrame]:
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train_home, y_train_away, feature_columns = load_sequence_npz(
        data_dir / "football_train_sequences.npz"
    )
    X_val, y_val_home, y_val_away, _ = load_sequence_npz(data_dir / "football_val_sequences.npz")
    X_test, y_test_home, y_test_away, _ = load_sequence_npz(
        data_dir / "football_test_sequences.npz"
    )
    ranking = aggregate_feature_ranking(X_train, y_train_home, y_train_away, feature_columns)
    ranking.to_csv(metrics_dir / f"{artifact_prefix}_feature_ranking.csv", index=False)

    data = {
        "train": (X_train, y_train_home, y_train_away),
        "val": (X_val, y_val_home, y_val_away),
        "test": (X_test, y_test_home, y_test_away),
    }
    all_metrics = []
    summaries = []
    for feature_count in FEATURE_COUNTS:
        selected = ranking.head(feature_count).copy()
        selected.to_csv(
            metrics_dir / f"{artifact_prefix}_top_{feature_count}_selected_features.csv",
            index=False,
        )
        metrics, summary = fit_and_evaluate_subset(
            feature_count=feature_count,
            feature_indices=selected["feature_index"].to_numpy(),
            data=data,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            epochs=epochs,
            batch_size=batch_size,
            artifact_prefix=artifact_prefix,
        )
        all_metrics.append(metrics)
        summaries.append(summary)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summaries_df = pd.concat(summaries, ignore_index=True)
    comparison = build_comparison(metrics_df, summaries_df, metrics_dir)

    metrics_df.to_csv(metrics_dir / f"{artifact_prefix}_metrics.csv", index=False)
    summaries_df.to_csv(metrics_dir / f"{artifact_prefix}_training_summary.csv", index=False)
    comparison.to_csv(metrics_dir / f"{artifact_prefix}_comparison.csv", index=False)
    return {
        "metrics": metrics_df,
        "training_summary": summaries_df,
        "comparison": comparison,
        "ranking": ranking,
    }


def run_top50_retrain(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    selected_features_path: Path,
    epochs: int = 25,
    batch_size: int = 32,
    artifact_prefix: str = "top50_retrain",
) -> dict[str, pd.DataFrame]:
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train_home, y_train_away, _ = load_sequence_npz(
        data_dir / "football_train_sequences.npz"
    )
    X_val, y_val_home, y_val_away, _ = load_sequence_npz(data_dir / "football_val_sequences.npz")
    X_test, y_test_home, y_test_away, _ = load_sequence_npz(
        data_dir / "football_test_sequences.npz"
    )
    selected = pd.read_csv(selected_features_path)
    feature_indices = selected["feature_index"].to_numpy(dtype=int)
    data = {
        "train": (X_train, y_train_home, y_train_away),
        "val": (X_val, y_val_home, y_val_away),
        "test": (X_test, y_test_home, y_test_away),
    }
    metrics, summary = fit_and_evaluate_subset(
        feature_count=len(feature_indices),
        feature_indices=feature_indices,
        data=data,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        epochs=epochs,
        batch_size=batch_size,
        artifact_prefix=artifact_prefix,
    )

    model = tf.keras.models.load_model(
        models_dir / f"{artifact_prefix}_top_{len(feature_indices)}.keras"
    )
    fixed_rows = []
    confusion_rows = []
    for split, X, y_home, y_away in [
        ("train", X_train[:, :, feature_indices], y_train_home, y_train_away),
        ("val", X_val[:, :, feature_indices], y_val_home, y_val_away),
        ("test", X_test[:, :, feature_indices], y_test_home, y_test_away),
    ]:
        pred = _prediction_dict(model.predict(X, batch_size=batch_size, verbose=0))
        for target, y_true, prob in [
            ("home_scores_next_half", y_home, pred["home_scores_next_half"]),
            ("away_scores_next_half", y_away, pred["away_scores_next_half"]),
        ]:
            row = evaluate_binary(
                y_true=y_true,
                prob=prob,
                name=f"{artifact_prefix}_{target}_final_threshold",
                threshold=FINAL_THRESHOLDS[target],
            )
            row["split"] = split
            row["target"] = target
            row["threshold_mode"] = "final_fixed"
            fixed_rows.append(row)
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
    fixed_threshold_metrics = pd.DataFrame(fixed_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    metrics.to_csv(metrics_dir / f"{artifact_prefix}_metrics.csv", index=False)
    summary.to_csv(metrics_dir / f"{artifact_prefix}_training_summary.csv", index=False)
    fixed_threshold_metrics.to_csv(
        metrics_dir / f"{artifact_prefix}_final_threshold_metrics.csv",
        index=False,
    )
    confusion_df.to_csv(metrics_dir / f"{artifact_prefix}_confusion_matrices.csv", index=False)
    return {
        "metrics": metrics,
        "training_summary": summary,
        "final_threshold_metrics": fixed_threshold_metrics,
        "confusion_matrices": confusion_df,
        "selected_features": selected,
    }
