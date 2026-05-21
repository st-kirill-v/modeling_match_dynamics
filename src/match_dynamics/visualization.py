from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve


def save_football_training_curves(histories: dict, targets: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, target in zip(axes[0], targets):
        h = histories[target]
        ax.plot(h.history["loss"], label="train")
        ax.plot(h.history["val_loss"], label="val")
        ax.set_title(f"Football loss: {target}")
        ax.legend()
    for ax, target in zip(axes[1], targets):
        h = histories[target]
        ax.plot(h.history["accuracy"], label="train")
        ax.plot(h.history["val_accuracy"], label="val")
        ax.set_title(f"Football accuracy: {target}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_football_error_curves(histories: dict, targets: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, target in zip(axes[0], targets):
        h = histories[target]
        if "mse" in h.history:
            ax.plot(h.history["mse"], label="train")
            ax.plot(h.history["val_mse"], label="val")
        ax.set_title(f"Football MSE: {target}")
        ax.legend()
    for ax, target in zip(axes[1], targets):
        h = histories[target]
        if "mae" in h.history:
            ax.plot(h.history["mae"], label="train")
            ax.plot(h.history["val_mae"], label="val")
        ax.set_title(f"Football MAE: {target}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    y_true,
    prob,
    title: str,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y_true, (prob >= threshold).astype(int)),
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pr_curve(y_true, prob, title: str, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_calibration_curve(calib_plot: pd.DataFrame, title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    plt.plot(calib_plot["mean_prob"], calib_plot["event_rate"], marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_correlation_heatmap(
    df: pd.DataFrame, cols: list[str], title: str, output_path: Path
) -> None:
    fig = plt.figure(figsize=(14, 9))
    sns.heatmap(df[cols].corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importance(
    importance: pd.DataFrame, title: str, output_path: Path, top_n: int = 25
) -> None:
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(data=importance.head(top_n), x="importance", y="feature")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
