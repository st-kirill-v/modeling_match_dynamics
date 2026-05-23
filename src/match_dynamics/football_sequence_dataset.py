from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = ["home_scores_next_half", "away_scores_next_half"]

EXCLUDED_LEAKAGE_COLUMNS = [
    "fthg",
    "ftag",
    "final_score",
    "second_half_home_goals",
    "second_half_away_goals",
    "home_scores_next_half",
    "away_scores_next_half",
]

EXCLUDED_METADATA_COLUMNS = [
    "id_odsp",
    "date",
    "league",
    "season",
    "country",
    "ht",
    "at",
    "home_max_history_date",
    "away_max_history_date",
]


@dataclass(frozen=True)
class SequenceDatasetPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    train_npz: Path
    val_npz: Path
    test_npz: Path
    scaler: Path
    report_dir: Path


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(EXCLUDED_LEAKAGE_COLUMNS + EXCLUDED_METADATA_COLUMNS)
    blocked_prefixes = ("odd_",)
    feature_cols = []
    for col in df.columns:
        if col in excluded or col.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def temporal_match_split(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    matches = (
        df[["id_odsp", "date"]]
        .drop_duplicates("id_odsp")
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
        .sort_values(["date", "id_odsp"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(matches) < 9000:
        raise ValueError(f"Need at least 9000 matches for requested split, found {len(matches)}.")
    train_ids = matches.iloc[:7000]["id_odsp"].tolist()
    val_ids = matches.iloc[7000:8000]["id_odsp"].tolist()
    test_ids = matches.iloc[-1000:]["id_odsp"].tolist()
    used = set(train_ids + val_ids + test_ids)
    unused_ids = matches.loc[~matches["id_odsp"].isin(used), "id_odsp"].tolist()
    return train_ids, val_ids, test_ids, unused_ids


def validate_targets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        exists = target in df.columns
        binary = exists and set(df[target].dropna().unique()).issubset({0, 1})
        constant_per_match = (
            exists and df.groupby("id_odsp")[target].nunique(dropna=False).max() == 1
        )
        rows.append(
            {
                "target": target,
                "exists": exists,
                "binary_0_1": bool(binary),
                "constant_per_match": bool(constant_per_match),
            }
        )
    if {"home_scores_next_half", "second_half_home_goals"}.issubset(df.columns):
        rows.append(
            {
                "target": "home_scores_next_half_matches_second_half_home_goals",
                "exists": True,
                "binary_0_1": True,
                "constant_per_match": bool(
                    df["home_scores_next_half"]
                    .eq(df["second_half_home_goals"].gt(0).astype(int))
                    .all()
                ),
            }
        )
    if {"away_scores_next_half", "second_half_away_goals"}.issubset(df.columns):
        rows.append(
            {
                "target": "away_scores_next_half_matches_second_half_away_goals",
                "exists": True,
                "binary_0_1": True,
                "constant_per_match": bool(
                    df["away_scores_next_half"]
                    .eq(df["second_half_away_goals"].gt(0).astype(int))
                    .all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _make_split_df(df: pd.DataFrame, match_ids: list[str]) -> pd.DataFrame:
    split = df[df["id_odsp"].isin(match_ids) & df["time"].le(45)].copy()
    order = pd.DataFrame({"id_odsp": match_ids, "_match_order": range(len(match_ids))})
    split = split.merge(order, on="id_odsp", how="left")
    split = split.sort_values(["_match_order", "time"], kind="mergesort").drop(
        columns="_match_order"
    )
    return split.reset_index(drop=True)


def sequence_length_report(split_name: str, split_df: pd.DataFrame) -> pd.DataFrame:
    lengths = split_df.groupby("id_odsp").size()
    stats = lengths.describe()
    rows = [{"split": split_name, "metric": key, "value": value} for key, value in stats.items()]
    rows.append(
        {"split": split_name, "metric": "all_sequences_len_45", "value": bool(lengths.eq(45).all())}
    )
    return pd.DataFrame(rows)


def _build_arrays(
    split_df: pd.DataFrame,
    match_ids: list[str],
    feature_cols: list[str],
    scaler: StandardScaler,
    fit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = split_df[feature_cols].astype("float32").to_numpy()
    scaled = scaler.fit_transform(values) if fit else scaler.transform(values)

    expected_rows = len(match_ids) * 45
    if scaled.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} minute rows, got {scaled.shape[0]}.")
    X = scaled.astype("float32").reshape(len(match_ids), 45, len(feature_cols))

    target_df = split_df.drop_duplicates("id_odsp").set_index("id_odsp")
    y_home = target_df.loc[match_ids, "home_scores_next_half"].astype("int8").to_numpy()
    y_away = target_df.loc[match_ids, "away_scores_next_half"].astype("int8").to_numpy()
    for idx, match_id in enumerate(match_ids):
        match_rows = split_df[split_df["id_odsp"].eq(match_id)].sort_values("time")
        if len(match_rows) != 45:
            raise ValueError(f"Match {match_id} has {len(match_rows)} rows, expected 45.")
    return X, y_home, y_away


def target_distribution(split_name: str, y_home: np.ndarray, y_away: np.ndarray) -> pd.DataFrame:
    rows = []
    for target, values in [("home_scores_next_half", y_home), ("away_scores_next_half", y_away)]:
        counts = pd.Series(values).value_counts().sort_index()
        for value, count in counts.items():
            rows.append(
                {"split": split_name, "target": target, "value": int(value), "matches": int(count)}
            )
    return pd.DataFrame(rows)


def build_football_sequence_dataset(
    input_path: Path,
    paths: SequenceDatasetPaths,
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(input_path)
    input_rows = len(df)
    input_matches = df["id_odsp"].nunique(dropna=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.sort_values(["date", "id_odsp", "time"], kind="mergesort").reset_index(drop=True)

    target_checks = validate_targets(df)
    if not target_checks[["exists", "binary_0_1", "constant_per_match"]].all().all():
        raise ValueError(f"Target validation failed:\n{target_checks.to_string(index=False)}")

    train_ids, val_ids, test_ids, unused_ids = temporal_match_split(df)
    train_df = _make_split_df(df, train_ids)
    val_df = _make_split_df(df, val_ids)
    test_df = _make_split_df(df, test_ids)

    feature_cols = select_feature_columns(train_df)
    leakage_in_features = sorted(set(feature_cols) & set(EXCLUDED_LEAKAGE_COLUMNS))
    if leakage_in_features:
        raise ValueError(f"Leakage columns found in features: {leakage_in_features}")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not split_df["time"].le(45).all():
            raise ValueError(f"{split_name} contains rows with time > 45.")
        if split_df[feature_cols].isna().sum().sum() != 0:
            raise ValueError(f"{split_name} features contain NaN.")

    scaler = StandardScaler()
    X_train, y_train_home, y_train_away = _build_arrays(
        train_df, train_ids, feature_cols, scaler, fit=True
    )
    X_val, y_val_home, y_val_away = _build_arrays(val_df, val_ids, feature_cols, scaler, fit=False)
    X_test, y_test_home, y_test_away = _build_arrays(
        test_df, test_ids, feature_cols, scaler, fit=False
    )

    paths.train_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.scaler.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(paths.train_csv, index=False)
    val_df.to_csv(paths.val_csv, index=False)
    test_df.to_csv(paths.test_csv, index=False)
    joblib.dump(scaler, paths.scaler)

    np.savez_compressed(
        paths.train_npz,
        X=X_train,
        y_home=y_train_home,
        y_away=y_train_away,
        match_ids=np.array(train_ids),
        feature_columns=np.array(feature_cols),
    )
    np.savez_compressed(
        paths.val_npz,
        X=X_val,
        y_home=y_val_home,
        y_away=y_val_away,
        match_ids=np.array(val_ids),
        feature_columns=np.array(feature_cols),
    )
    np.savez_compressed(
        paths.test_npz,
        X=X_test,
        y_home=y_test_home,
        y_away=y_test_away,
        match_ids=np.array(test_ids),
        feature_columns=np.array(feature_cols),
    )

    seq_stats = pd.concat(
        [
            sequence_length_report("train", train_df),
            sequence_length_report("val", val_df),
            sequence_length_report("test", test_df),
        ],
        ignore_index=True,
    )
    dist = pd.concat(
        [
            target_distribution("train", y_train_home, y_train_away),
            target_distribution("val", y_val_home, y_val_away),
            target_distribution("test", y_test_home, y_test_away),
        ],
        ignore_index=True,
    )
    diagnostics = pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_rows},
            {"metric": "input_matches", "value": input_matches},
            {"metric": "train_matches", "value": len(train_ids)},
            {"metric": "val_matches", "value": len(val_ids)},
            {"metric": "test_matches", "value": len(test_ids)},
            {"metric": "unused_gap_matches_between_val_and_test", "value": len(unused_ids)},
            {"metric": "sequence_length", "value": 45},
            {"metric": "num_features", "value": len(feature_cols)},
            {"metric": "X_train_shape", "value": str(X_train.shape)},
            {"metric": "X_val_shape", "value": str(X_val.shape)},
            {"metric": "X_test_shape", "value": str(X_test.shape)},
            {"metric": "leakage_columns_in_features", "value": ", ".join(leakage_in_features)},
            {"metric": "nan_in_X_train", "value": int(np.isnan(X_train).sum())},
            {"metric": "nan_in_X_val", "value": int(np.isnan(X_val).sum())},
            {"metric": "nan_in_X_test", "value": int(np.isnan(X_test).sum())},
            {"metric": "max_time_train", "value": int(train_df["time"].max())},
            {"metric": "max_time_val", "value": int(val_df["time"].max())},
            {"metric": "max_time_test", "value": int(test_df["time"].max())},
        ]
    )
    feature_table = pd.DataFrame({"feature": feature_cols})
    leakage_table = pd.DataFrame(
        {"excluded_column": EXCLUDED_LEAKAGE_COLUMNS + EXCLUDED_METADATA_COLUMNS}
    )

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "diagnostics": diagnostics,
        "target_distribution": dist,
        "sequence_length_stats": seq_stats,
        "feature_columns": feature_table,
        "excluded_columns": leakage_table,
        "target_checks": target_checks,
    }
    for name, report in reports.items():
        report.to_csv(paths.report_dir / f"football_sequence_{name}.csv", index=False)
    return reports
