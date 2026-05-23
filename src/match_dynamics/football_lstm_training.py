from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from .evaluation import evaluate_binary


THRESHOLD = 0.5


def load_sequence_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y_home"], data["y_away"], data["feature_columns"]


def build_baseline_multioutput_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="first_half_sequence")
    x = tf.keras.layers.LSTM(128, return_sequences=False, name="lstm_128")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="dropout_03")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64_relu")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_02")(x)
    home_output = tf.keras.layers.Dense(1, activation="sigmoid", name="home_output")(x)
    away_output = tf.keras.layers.Dense(1, activation="sigmoid", name="away_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=[home_output, away_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            "home_output": "binary_crossentropy",
            "away_output": "binary_crossentropy",
        },
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


def _prediction_dict(predictions) -> dict[str, np.ndarray]:
    if isinstance(predictions, dict):
        return {
            "home_scores_next_half": predictions["home_output"].ravel(),
            "away_scores_next_half": predictions["away_output"].ravel(),
        }
    return {
        "home_scores_next_half": predictions[0].ravel(),
        "away_scores_next_half": predictions[1].ravel(),
    }


def evaluate_split(
    split: str,
    y_home: np.ndarray,
    y_away: np.ndarray,
    prob_home: np.ndarray,
    prob_away: np.ndarray,
) -> list[dict]:
    rows = []
    for target, y_true, prob in [
        ("home_scores_next_half", y_home, prob_home),
        ("away_scores_next_half", y_away, prob_away),
    ]:
        row = evaluate_binary(
            y_true=y_true,
            prob=prob,
            name=f"baseline_lstm_{target}",
            threshold=THRESHOLD,
        )
        row["split"] = split
        row["target"] = target
        rows.append(row)
    return rows


def save_history(history: tf.keras.callbacks.History, output_csv: Path, output_json: Path) -> None:
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in history.history.items()}, f, indent=2)


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_loss_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    hist = history.history
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist["loss"], label="train_loss")
    ax.plot(hist["val_loss"], label="val_loss")
    for key in [
        "home_output_loss",
        "away_output_loss",
        "val_home_output_loss",
        "val_away_output_loss",
    ]:
        if key in hist:
            ax.plot(hist[key], linestyle="--", alpha=0.75, label=key)
    ax.set_title("Baseline Football LSTM: Train/Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary crossentropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, prob: np.ndarray, title: str, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, prob)
    score = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={score:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, prob: np.ndarray, title: str, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    score = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC={score:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    prob: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    pred = (prob >= THRESHOLD).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_figures(
    split: str,
    y_home: np.ndarray,
    y_away: np.ndarray,
    prob_home: np.ndarray,
    prob_away: np.ndarray,
    figures_dir: Path,
) -> None:
    for target, y_true, prob in [
        ("home_scores_next_half", y_home, prob_home),
        ("away_scores_next_half", y_away, prob_away),
    ]:
        prefix = figures_dir / f"{split}_{target}"
        plot_roc_curve(
            y_true, prob, f"{split} ROC: {target}", prefix.with_name(prefix.name + "_roc.png")
        )
        plot_pr_curve(
            y_true, prob, f"{split} PR: {target}", prefix.with_name(prefix.name + "_pr.png")
        )
        plot_confusion_matrix(
            y_true,
            prob,
            f"{split} Confusion Matrix: {target}",
            prefix.with_name(prefix.name + "_confusion.png"),
        )


def confusion_matrix_rows(
    split: str,
    y_home: np.ndarray,
    y_away: np.ndarray,
    prob_home: np.ndarray,
    prob_away: np.ndarray,
) -> list[dict]:
    rows = []
    for target, y_true, prob in [
        ("home_scores_next_half", y_home, prob_home),
        ("away_scores_next_half", y_away, prob_away),
    ]:
        matrix = confusion_matrix(y_true, (prob >= THRESHOLD).astype(int), labels=[0, 1])
        for true_label in [0, 1]:
            for predicted_label in [0, 1]:
                rows.append(
                    {
                        "split": split,
                        "target": target,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_label, predicted_label]),
                    }
                )
    return rows


def overfitting_report(
    history: tf.keras.callbacks.History, metrics_df: pd.DataFrame
) -> pd.DataFrame:
    hist = pd.DataFrame(history.history)
    best_idx = int(hist["val_loss"].idxmin())
    test_rows = metrics_df[metrics_df["split"].eq("test")].set_index("target")
    val_rows = metrics_df[metrics_df["split"].eq("val")].set_index("target")
    rows = [
        {"metric": "best_epoch", "value": best_idx + 1},
        {"metric": "best_val_loss", "value": float(hist.loc[best_idx, "val_loss"])},
        {"metric": "final_train_loss", "value": float(hist["loss"].iloc[-1])},
        {"metric": "final_val_loss", "value": float(hist["val_loss"].iloc[-1])},
        {
            "metric": "loss_gap_final_val_minus_train",
            "value": float(hist["val_loss"].iloc[-1] - hist["loss"].iloc[-1]),
        },
    ]
    for target in ["home_scores_next_half", "away_scores_next_half"]:
        if target in test_rows.index and target in val_rows.index:
            rows.append(
                {
                    "metric": f"{target}_test_minus_val_pr_auc",
                    "value": float(
                        test_rows.loc[target, "pr_auc"] - val_rows.loc[target, "pr_auc"]
                    ),
                }
            )
            rows.append(
                {
                    "metric": f"{target}_test_minus_val_roc_auc",
                    "value": float(
                        test_rows.loc[target, "roc_auc"] - val_rows.loc[target, "roc_auc"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def train_baseline_football_lstm(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    epochs: int = 25,
    batch_size: int = 32,
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

    model = build_baseline_multioutput_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    save_model_summary(model, models_dir / "baseline_lstm_summary.txt")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        {"home_output": y_train_home, "away_output": y_train_away},
        validation_data=(X_val, {"home_output": y_val_home, "away_output": y_val_away}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(models_dir / "baseline_multioutput_lstm.keras")
    save_history(
        history,
        metrics_dir / "baseline_lstm_history.csv",
        metrics_dir / "baseline_lstm_history.json",
    )
    plot_loss_curves(history, figures_dir / "baseline_lstm_loss_curves.png")

    metric_rows = []
    confusion_rows = []
    predictions = {}
    for split, X, y_home, y_away in [
        ("train", X_train, y_train_home, y_train_away),
        ("val", X_val, y_val_home, y_val_away),
        ("test", X_test, y_test_home, y_test_away),
    ]:
        pred = _prediction_dict(model.predict(X, batch_size=batch_size, verbose=0))
        predictions[split] = pred
        metric_rows.extend(
            evaluate_split(
                split,
                y_home,
                y_away,
                pred["home_scores_next_half"],
                pred["away_scores_next_half"],
            )
        )
        confusion_rows.extend(
            confusion_matrix_rows(
                split,
                y_home,
                y_away,
                pred["home_scores_next_half"],
                pred["away_scores_next_half"],
            )
        )
        save_evaluation_figures(
            split,
            y_home,
            y_away,
            pred["home_scores_next_half"],
            pred["away_scores_next_half"],
            figures_dir,
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df[
        [
            "split",
            "target",
            "model",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "log_loss",
            "brier",
            "mae",
            "mse",
            "rmse",
            "top_decile_lift",
        ]
    ]
    metrics_df.to_csv(metrics_dir / "baseline_lstm_metrics.csv", index=False)
    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(metrics_dir / "baseline_lstm_confusion_matrices.csv", index=False)

    shapes = pd.DataFrame(
        [
            {"name": "X_train", "shape": str(X_train.shape)},
            {"name": "X_val", "shape": str(X_val.shape)},
            {"name": "X_test", "shape": str(X_test.shape)},
            {"name": "y_train_home", "shape": str(y_train_home.shape)},
            {"name": "y_train_away", "shape": str(y_train_away.shape)},
            {"name": "feature_columns", "shape": str(feature_columns.shape)},
        ]
    )
    shapes.to_csv(metrics_dir / "baseline_lstm_shapes.csv", index=False)

    overfit = overfitting_report(history, metrics_df)
    overfit.to_csv(metrics_dir / "baseline_lstm_overfitting_report.csv", index=False)
    return {
        "metrics": metrics_df,
        "confusion": confusion_df,
        "history": pd.DataFrame(history.history),
        "shapes": shapes,
        "overfitting": overfit,
    }
