from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .football_sequence_dataset import SequenceDatasetPaths, build_football_sequence_dataset


TARGETED_FEATURES = [
    "is_0_0_at_half",
    "low_first_half_tempo",
    "high_first_half_tempo",
    "strong_home_low_pressure",
    "strong_away_low_pressure",
    "home_stronger_but_not_dominating",
    "away_stronger_but_not_dominating",
    "balanced_strength_match",
    "strong_home_and_pressure",
    "strong_away_and_pressure",
]

TARGETS = ["home_scores_next_half", "away_scores_next_half"]


@dataclass(frozen=True)
class TargetedFeaturePaths:
    input_csv: Path
    output_csv: Path
    report_dir: Path
    sequence_paths: SequenceDatasetPaths


def _temporal_train_match_ids(df: pd.DataFrame, n_train: int = 7000) -> list[str]:
    matches = (
        df[["id_odsp", "date"]]
        .drop_duplicates("id_odsp")
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
        .sort_values(["date", "id_odsp"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(matches) < n_train:
        raise ValueError(f"Need at least {n_train} matches, found {len(matches)}.")
    return matches.iloc[:n_train]["id_odsp"].astype(str).tolist()


def _match_level(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "id_odsp",
        "date",
        "home_first_half_score",
        "away_first_half_score",
        "home_first_half_attempts",
        "away_first_half_attempts",
        "home_first_half_pressure",
        "away_first_half_pressure",
        "first_half_pressure_diff",
        "team_attack_strength_diff_last_10",
        "team_form_diff_last_10",
        "home_scores_next_half",
        "away_scores_next_half",
    ]
    missing = [col for col in keep_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for targeted features: {missing}")
    return df[keep_cols].drop_duplicates("id_odsp").copy()


def add_targeted_error_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    train_ids = set(_temporal_train_match_ids(output))
    match_df = _match_level(output)
    match_df["total_first_half_activity"] = (
        match_df["home_first_half_attempts"]
        + match_df["away_first_half_attempts"]
        + match_df["home_first_half_pressure"]
        + match_df["away_first_half_pressure"]
    )

    train_match_df = match_df[match_df["id_odsp"].astype(str).isin(train_ids)].copy()
    low_tempo_threshold = float(train_match_df["total_first_half_activity"].quantile(0.25))
    high_tempo_threshold = float(train_match_df["total_first_half_activity"].quantile(0.75))
    balanced_strength_threshold = float(train_match_df["team_form_diff_last_10"].abs().median())

    match_df["is_0_0_at_half"] = (
        match_df["home_first_half_score"].eq(0) & match_df["away_first_half_score"].eq(0)
    ).astype("int8")
    match_df["low_first_half_tempo"] = (
        match_df["total_first_half_activity"].le(low_tempo_threshold)
    ).astype("int8")
    match_df["high_first_half_tempo"] = (
        match_df["total_first_half_activity"].ge(high_tempo_threshold)
    ).astype("int8")
    match_df["strong_home_low_pressure"] = (
        match_df["team_attack_strength_diff_last_10"].gt(0)
        & match_df["first_half_pressure_diff"].le(0)
    ).astype("int8")
    match_df["strong_away_low_pressure"] = (
        match_df["team_attack_strength_diff_last_10"].lt(0)
        & match_df["first_half_pressure_diff"].ge(0)
    ).astype("int8")
    match_df["home_stronger_but_not_dominating"] = (
        match_df["team_form_diff_last_10"].gt(0) & match_df["first_half_pressure_diff"].le(0)
    ).astype("int8")
    match_df["away_stronger_but_not_dominating"] = (
        match_df["team_form_diff_last_10"].lt(0) & match_df["first_half_pressure_diff"].ge(0)
    ).astype("int8")
    match_df["balanced_strength_match"] = (
        match_df["team_form_diff_last_10"].abs().le(balanced_strength_threshold)
    ).astype("int8")
    match_df["strong_home_and_pressure"] = (
        match_df["team_attack_strength_diff_last_10"].gt(0)
        & match_df["first_half_pressure_diff"].gt(0)
    ).astype("int8")
    match_df["strong_away_and_pressure"] = (
        match_df["team_attack_strength_diff_last_10"].lt(0)
        & match_df["first_half_pressure_diff"].lt(0)
    ).astype("int8")

    output = output.merge(match_df[["id_odsp", *TARGETED_FEATURES]], on="id_odsp", how="left")
    for feature in TARGETED_FEATURES:
        output[feature] = output[feature].fillna(0).astype("int8")

    thresholds = pd.DataFrame(
        [
            {
                "threshold_name": "low_first_half_tempo_p25",
                "value": low_tempo_threshold,
                "fit_split": "train_first_7000_matches_by_date",
            },
            {
                "threshold_name": "high_first_half_tempo_p75",
                "value": high_tempo_threshold,
                "fit_split": "train_first_7000_matches_by_date",
            },
            {
                "threshold_name": "balanced_strength_abs_median",
                "value": balanced_strength_threshold,
                "fit_split": "train_first_7000_matches_by_date",
            },
        ]
    )
    return output, thresholds


def targeted_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    match_df = df[["id_odsp", *TARGETED_FEATURES, *TARGETS]].drop_duplicates("id_odsp")
    for feature in TARGETED_FEATURES:
        for target in TARGETS:
            corr = match_df[feature].corr(match_df[target])
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": 0.0 if pd.isna(corr) else float(corr),
                    "abs_correlation": 0.0 if pd.isna(corr) else float(abs(corr)),
                    "positive_rate": float(match_df[feature].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "abs_correlation"], ascending=[True, False])


def validate_targeted_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in TARGETED_FEATURES:
        values = set(df[feature].dropna().unique().tolist())
        rows.append(
            {
                "feature": feature,
                "binary_0_1": values.issubset({0, 1}),
                "null_count": int(df[feature].isna().sum()),
                "positive_rows": int(df[feature].sum()),
                "positive_matches": int(
                    df[["id_odsp", feature]].drop_duplicates("id_odsp")[feature].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_targeted_feature_build(paths: TargetedFeaturePaths) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(paths.input_csv)
    input_shape = df.shape
    output, thresholds = add_targeted_error_features(df)
    output_shape = output.shape

    validation = validate_targeted_features(output)
    correlations = targeted_feature_correlations(output)
    summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_shape[0]},
            {"metric": "input_columns", "value": input_shape[1]},
            {"metric": "output_rows", "value": output_shape[0]},
            {"metric": "output_columns", "value": output_shape[1]},
            {"metric": "matches", "value": output["id_odsp"].nunique(dropna=True)},
            {"metric": "created_targeted_features", "value": len(TARGETED_FEATURES)},
            {"metric": "total_null_values", "value": int(output.isna().sum().sum())},
            {
                "metric": "new_feature_null_values",
                "value": int(output[TARGETED_FEATURES].isna().sum().sum()),
            },
            {
                "metric": "non_finite_new_feature_values",
                "value": int((~np.isfinite(output[TARGETED_FEATURES])).sum().sum()),
            },
        ]
    )

    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(paths.output_csv, index=False)
    sequence_reports = build_football_sequence_dataset(paths.output_csv, paths.sequence_paths)

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "summary": summary,
        "thresholds": thresholds,
        "validation": validation,
        "correlations": correlations,
        "created_features": pd.DataFrame({"feature": TARGETED_FEATURES}),
        "sequence_diagnostics": sequence_reports["diagnostics"],
        "sequence_target_distribution": sequence_reports["target_distribution"],
        "sequence_length_stats": sequence_reports["sequence_length_stats"],
    }
    for name, report in reports.items():
        report.to_csv(paths.report_dir / f"football_targeted_{name}.csv", index=False)
    return reports
