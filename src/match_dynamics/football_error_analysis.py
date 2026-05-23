from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .football_lstm_training import _prediction_dict, load_sequence_npz
from .football_threshold_tuning import FINAL_THRESHOLDS, load_top_feature_indices


TARGETS = ["home_scores_next_half", "away_scores_next_half"]


def _safe_qcut(values: pd.Series, labels: list[str]) -> pd.Series:
    try:
        return pd.qcut(values, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series([labels[len(labels) // 2]] * len(values), index=values.index)


def load_top50_predictions(
    data_dir: Path,
    models_dir: Path,
    selected_features_path: Path,
    batch_size: int = 32,
) -> pd.DataFrame:
    model_path = models_dir / "feature_ablation_fast_top_50.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Top-50 LSTM model not found: {model_path}")

    X_test, y_home, y_away, feature_columns = load_sequence_npz(
        data_dir / "football_test_sequences.npz"
    )
    match_ids = np.load(data_dir / "football_test_sequences.npz", allow_pickle=True)["match_ids"]
    selected_indices = load_top_feature_indices(selected_features_path)
    X_test_top50 = X_test[:, :, selected_indices]

    model = tf.keras.models.load_model(model_path)
    pred = _prediction_dict(model.predict(X_test_top50, batch_size=batch_size, verbose=0))
    rows = []
    for target, y_true in [
        ("home_scores_next_half", y_home),
        ("away_scores_next_half", y_away),
    ]:
        prob = pred[target]
        threshold = FINAL_THRESHOLDS[target]
        y_pred = (prob >= threshold).astype(int)
        for idx, match_id in enumerate(match_ids):
            true_label = int(y_true[idx])
            pred_label = int(y_pred[idx])
            if true_label == 1 and pred_label == 1:
                outcome = "TP"
            elif true_label == 0 and pred_label == 0:
                outcome = "TN"
            elif true_label == 0 and pred_label == 1:
                outcome = "FP"
            else:
                outcome = "FN"
            rows.append(
                {
                    "id_odsp": str(match_id),
                    "target": target,
                    "threshold": threshold,
                    "probability": float(prob[idx]),
                    "predicted_label": pred_label,
                    "true_label": true_label,
                    "outcome": outcome,
                    "feature_count": len(selected_indices),
                    "available_feature_columns": len(feature_columns),
                }
            )
    return pd.DataFrame(rows)


def build_match_context(test_csv: Path, match_ids: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    df = df[df["id_odsp"].isin(match_ids)].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.sort_values(["id_odsp", "time"], kind="mergesort")

    last = df.groupby("id_odsp", sort=False).tail(1).copy()
    last = last.set_index("id_odsp")
    sums = df.groupby("id_odsp", sort=False).agg(
        total_events_sum=("total_events_proxy", "sum"),
        home_pressure_sum=("home_pressure_minute", "sum"),
        away_pressure_sum=("away_pressure_minute", "sum"),
        home_attempt_sum=("home_attempt", "sum"),
        away_attempt_sum=("away_attempt", "sum"),
    )
    context = last.join(sums, how="left").reset_index()

    score_diff = context["first_half_score_diff"].fillna(
        context["home_first_half_score"].fillna(0) - context["away_first_half_score"].fillna(0)
    )
    context["first_half_score_diff_segment"] = np.select(
        [score_diff.gt(0), score_diff.eq(0), score_diff.lt(0)],
        ["home_leading", "draw", "away_leading"],
        default="unknown",
    )

    total_first_half_score = context["home_first_half_score"].fillna(0) + context[
        "away_first_half_score"
    ].fillna(0)
    context["first_half_score_segment"] = np.select(
        [total_first_half_score.eq(0), total_first_half_score.eq(1), total_first_half_score.ge(2)],
        ["0-0", "low_scoring_1_goal", "goal_already_happened_2plus"],
        default="unknown",
    )

    pressure_diff = context["home_pressure_sum"].fillna(0) - context["away_pressure_sum"].fillna(0)
    pressure_abs_threshold = pressure_diff.abs().quantile(0.50)
    context["pressure_segment"] = np.select(
        [
            pressure_diff.gt(pressure_abs_threshold),
            pressure_diff.lt(-pressure_abs_threshold),
            pressure_diff.abs().le(pressure_abs_threshold),
        ],
        ["high_home_pressure", "high_away_pressure", "balanced_pressure"],
        default="unknown",
    )

    strength_diff = context.get(
        "team_attack_strength_diff_last_10", pd.Series(0, index=context.index)
    )
    strength_abs_threshold = strength_diff.abs().quantile(0.50)
    context["team_strength_segment"] = np.select(
        [
            strength_diff.gt(strength_abs_threshold),
            strength_diff.lt(-strength_abs_threshold),
            strength_diff.abs().le(strength_abs_threshold),
        ],
        ["home_stronger", "away_stronger", "balanced_teams"],
        default="unknown",
    )

    context["match_activity_segment"] = _safe_qcut(
        context["total_events_sum"].fillna(0),
        labels=["low_event_intensity", "medium_event_intensity", "high_event_intensity"],
    )
    return context[
        [
            "id_odsp",
            "date",
            "ht",
            "at",
            "home_first_half_score",
            "away_first_half_score",
            "first_half_score_diff",
            "home_first_half_pressure",
            "away_first_half_pressure",
            "first_half_pressure_diff",
            "total_events_sum",
            "team_attack_strength_diff_last_10",
            "first_half_score_diff_segment",
            "first_half_score_segment",
            "pressure_segment",
            "team_strength_segment",
            "match_activity_segment",
        ]
    ]


def segment_metrics(predictions: pd.DataFrame, segment_columns: list[str]) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        target_df = predictions[predictions["target"].eq(target)].copy()
        for segment_col in segment_columns:
            for segment_value, group in target_df.groupby(segment_col, dropna=False):
                y_true = group["true_label"].to_numpy()
                y_pred = group["predicted_label"].to_numpy()
                rows.append(
                    {
                        "target": target,
                        "segment_type": segment_col,
                        "segment_value": segment_value,
                        "count": len(group),
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, zero_division=0),
                        "recall": recall_score(y_true, y_pred, zero_division=0),
                        "f1": f1_score(y_true, y_pred, zero_division=0),
                        "TP": int(group["outcome"].eq("TP").sum()),
                        "TN": int(group["outcome"].eq("TN").sum()),
                        "FP": int(group["outcome"].eq("FP").sum()),
                        "FN": int(group["outcome"].eq("FN").sum()),
                    }
                )
    return pd.DataFrame(rows)


def top_error_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        target_df = predictions[predictions["target"].eq(target)].copy()
        fp = (
            target_df[target_df["outcome"].eq("FP")]
            .sort_values("probability", ascending=False)
            .head(20)
        )
        fn = (
            target_df[target_df["outcome"].eq("FN")]
            .sort_values("probability", ascending=True)
            .head(20)
        )
        rows.append(fp.assign(error_rank_type="top_false_positive_highest_probability"))
        rows.append(fn.assign(error_rank_type="top_false_negative_lowest_probability"))
    return pd.concat(rows, ignore_index=True)


def build_report(
    predictions: pd.DataFrame,
    segments: pd.DataFrame,
    top_errors: pd.DataFrame,
) -> str:
    lines = [
        "FOOTBALL LSTM ERROR ANALYSIS REPORT",
        "",
        "Model: feature_ablation_fast_top_50.keras",
        "Thresholds: home_scores_next_half=0.47, away_scores_next_half=0.39",
        "Split: test only",
        "",
        "Outcome counts:",
    ]
    outcome_counts = (
        predictions.groupby(["target", "outcome"]).size().unstack(fill_value=0).reset_index()
    )
    lines.append(outcome_counts.to_string(index=False))
    lines.extend(["", "Worst segments by F1:"])
    worst = segments.sort_values("f1", ascending=True).head(20)
    lines.append(
        worst[
            [
                "target",
                "segment_type",
                "segment_value",
                "count",
                "precision",
                "recall",
                "f1",
                "FP",
                "FN",
            ]
        ].to_string(index=False)
    )
    lines.extend(["", "Top error cases preview:"])
    lines.append(
        top_errors[
            [
                "target",
                "error_rank_type",
                "id_odsp",
                "date",
                "ht",
                "at",
                "probability",
                "true_label",
                "predicted_label",
                "outcome",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    return "\n".join(lines)


def run_football_error_analysis(
    data_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    reports_dir: Path,
    selected_features_path: Path,
    batch_size: int = 32,
) -> dict[str, pd.DataFrame | str]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(data_dir / "football_test_sequences.npz", allow_pickle=True)
    match_ids = npz["match_ids"]
    predictions = load_top50_predictions(data_dir, models_dir, selected_features_path, batch_size)
    context = build_match_context(data_dir / "football_merged_test.csv", match_ids)
    predictions = predictions.merge(context, on="id_odsp", how="left")

    segment_cols = [
        "first_half_score_diff_segment",
        "first_half_score_segment",
        "pressure_segment",
        "team_strength_segment",
        "match_activity_segment",
    ]
    segments = segment_metrics(predictions, segment_cols)
    errors = top_error_cases(predictions)
    report = build_report(predictions, segments, errors)

    segments.to_csv(metrics_dir / "error_analysis_segments.csv", index=False)
    errors.to_csv(metrics_dir / "error_analysis_top_errors.csv", index=False)
    predictions.to_csv(metrics_dir / "error_analysis_predictions.csv", index=False)
    (reports_dir / "error_analysis_report.txt").write_text(report, encoding="utf-8")

    return {
        "predictions": predictions,
        "segments": segments,
        "top_errors": errors,
        "report": report,
    }
