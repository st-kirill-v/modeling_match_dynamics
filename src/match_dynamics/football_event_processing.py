from __future__ import annotations

from pathlib import Path

import pandas as pd


EVENT_TYPE2_FEATURES = {
    "is_key_pass": 12,
    "is_failed_through_ball": 13,
    "is_sending_off": 14,
    "is_own_goal": 15,
}

SHOT_PLACE_FEATURES = {
    "shot_corner": [3, 4, 12, 13],
    "shot_central": [5, 11],
    "shot_off_target": [1, 6, 8, 9, 10],
    "shot_hit_bar": [7],
    "shot_blocked": [2],
}

SHOT_OUTCOME_FEATURES = {
    "is_on_target": 1,
    "is_off_target": 2,
    "is_blocked": 3,
    "is_hit_bar": 4,
}

SITUATION_FEATURES = {
    "is_open_play": 1,
    "is_set_piece": 2,
    "is_corner_situation": 3,
    "is_free_kick": 4,
}

BODYPART_FEATURES = {
    "shot_right_foot": 1,
    "shot_left_foot": 2,
    "shot_head": 3,
}

LOCATION_FEATURES = {
    "is_box_zone": [3, 9, 10, 11, 12, 13, 14],
    "is_left_wing_zone": [4],
    "is_right_wing_zone": [5],
    "is_long_range_zone": [15, 16, 17, 18],
    "is_attacking_half": [1],
    "is_difficult_angle": [6, 7, 8],
}

EVENT_TYPE_FEATURES = {
    "is_announcement": 0,
    "is_attempt": 1,
    "is_corner": 2,
    "is_foul": 3,
    "is_yellow_card": 4,
    "is_second_yellow_card": 5,
    "is_red_card": 6,
    "is_substitution": 7,
    "is_free_kick_won": 8,
    "is_offside": 9,
    "is_hand_ball": 10,
    "is_penalty_conceded": 11,
}

ASSIST_METHOD_FEATURES = {
    "assist_none": 0,
    "assist_pass": 1,
    "assist_cross": 2,
    "assist_headed_pass": 3,
    "assist_through_ball": 4,
}

HOME_AWAY_SPLIT_FEATURES = {
    "is_attempt": "attempt",
    "is_corner": "corner",
    "is_foul": "foul",
    "is_yellow_card": "yellow_card",
    "is_red_card": "red_card",
    "is_free_kick_won": "free_kick_won",
    "is_offside": "offside",
    "is_hand_ball": "hand_ball",
    "is_penalty_conceded": "penalty_conceded",
    "is_key_pass": "key_pass",
    "is_own_goal": "own_goal",
    "is_on_target": "on_target",
    "is_off_target": "off_target",
    "is_blocked": "blocked",
    "is_hit_bar": "hit_bar",
    "shot_corner": "shot_corner",
    "shot_central": "shot_central",
    "shot_off_target": "shot_off_target",
    "shot_hit_bar": "shot_hit_bar",
    "shot_blocked": "shot_blocked",
    "is_box_zone": "box_zone",
    "is_left_wing_zone": "left_wing_zone",
    "is_right_wing_zone": "right_wing_zone",
    "is_long_range_zone": "long_range_zone",
    "is_attacking_half": "attacking_half",
    "is_difficult_angle": "difficult_angle",
    "assist_pass": "assist_pass",
    "assist_cross": "assist_cross",
    "assist_headed_pass": "assist_headed_pass",
    "assist_through_ball": "assist_through_ball",
}

DROP_COLUMNS = [
    "player_in",
    "player_out",
    "odd_over",
    "odd_under",
    "odd_bts",
    "odd_bts_n",
    "event_type2",
    "shot_place",
    "shot_outcome",
    "situation",
    "bodypart",
    "player2",
    "location",
]

SECOND_PASS_DROP_COLUMNS = [
    "event_type",
    "assist_method",
    "Unnamed: 0",
]

COMMENT_ROWS = [
    {
        "topic": "player",
        "comment": (
            "player is kept as a categorical identifier. It has high cardinality and is not used "
            "as a numeric LSTM feature. It can be removed, target-encoded, embedded, or handled "
            "separately later."
        ),
    },
    {
        "topic": "binary_features",
        "comment": (
            "NaN source values are not globally filled. Binary features use equality/isin checks, "
            "so missing source values naturally become 0 when the specific event condition is absent."
        ),
    },
]

MINUTE_METADATA_COLUMNS = [
    "date",
    "league",
    "season",
    "country",
    "ht",
    "at",
    "fthg",
    "ftag",
    "odd_h",
    "odd_d",
    "odd_a",
    "final_score",
]

TARGET_COLUMNS = [
    "home_scores_next_half",
    "away_scores_next_half",
]

AUXILIARY_TARGET_COLUMNS = [
    "second_half_home_goals",
    "second_half_away_goals",
]

LEAKAGE_COLUMNS = [
    "fthg",
    "ftag",
    "final_score",
    "second_half_home_goals",
    "second_half_away_goals",
]

BOOKMAKER_PRIOR_COLUMNS = [
    "odd_h",
    "odd_d",
    "odd_a",
]

MINUTE_METADATA_ROLE_COLUMNS = [
    "id_odsp",
    "date",
    "league",
    "season",
    "country",
    "ht",
    "at",
    "time",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _binary_equals(df: pd.DataFrame, source_col: str, value: int) -> pd.Series:
    if source_col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int8")
    return _to_numeric(df[source_col]).eq(value).astype("int8")


def _binary_isin(df: pd.DataFrame, source_col: str, values: list[int]) -> pd.Series:
    if source_col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int8")
    return _to_numeric(df[source_col]).isin(values).astype("int8")


def process_football_merged_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    created: list[str] = []
    dropped: list[str] = []

    out["is_substitution"] = _binary_equals(out, "event_type", 7)
    created.append("is_substitution")

    for feature, value in EVENT_TYPE2_FEATURES.items():
        out[feature] = _binary_equals(out, "event_type2", value)
        created.append(feature)

    for feature, values in SHOT_PLACE_FEATURES.items():
        out[feature] = _binary_isin(out, "shot_place", values)
        created.append(feature)

    for feature, value in SHOT_OUTCOME_FEATURES.items():
        out[feature] = _binary_equals(out, "shot_outcome", value)
        created.append(feature)

    for feature, value in SITUATION_FEATURES.items():
        out[feature] = _binary_equals(out, "situation", value)
        created.append(feature)

    for feature, value in BODYPART_FEATURES.items():
        out[feature] = _binary_equals(out, "bodypart", value)
        created.append(feature)

    if "player2" in out.columns:
        out["has_player2"] = out["player2"].notna().astype("int8")
    else:
        out["has_player2"] = 0
    created.append("has_player2")

    # Keep player as a categorical identifier, not as a numeric LSTM feature.
    # It has high cardinality and can be removed, encoded, or embedded separately later.
    if "player" in out.columns:
        out["player"] = out["player"].fillna("unknown_player")

    for feature, values in LOCATION_FEATURES.items():
        out[feature] = _binary_isin(out, "location", values)
        created.append(feature)

    existing_drop_cols = [col for col in DROP_COLUMNS if col in out.columns]
    if existing_drop_cols:
        out = out.drop(columns=existing_drop_cols)
        dropped.extend(existing_drop_cols)

    log = pd.DataFrame(
        [{"action": "created", "column": col} for col in created]
        + [{"action": "dropped", "column": col} for col in dropped]
        + [
            {"action": "comment", "column": row["topic"], "details": row["comment"]}
            for row in COMMENT_ROWS
        ]
    )
    return out, log


def binary_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("is_", "shot_", "has_", "assist_")
    return [col for col in df.columns if col.startswith(prefixes)]


def process_event_type_assist_and_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    created: list[str] = []
    dropped: list[str] = []
    updated: list[str] = []

    if "event_type" in out.columns:
        for feature, value in EVENT_TYPE_FEATURES.items():
            action_target = updated if feature in out.columns else created
            out[feature] = _binary_equals(out, "event_type", value)
            action_target.append(feature)

    if "assist_method" in out.columns:
        for feature, value in ASSIST_METHOD_FEATURES.items():
            action_target = updated if feature in out.columns else created
            out[feature] = _binary_equals(out, "assist_method", value)
            action_target.append(feature)

    existing_drop_cols = [col for col in SECOND_PASS_DROP_COLUMNS if col in out.columns]
    if "adv_stats" in out.columns and out["adv_stats"].nunique(dropna=True) <= 1:
        existing_drop_cols.append("adv_stats")
    if existing_drop_cols:
        out = out.drop(columns=existing_drop_cols)
        dropped.extend(existing_drop_cols)

    full_duplicate_count = int(out.duplicated().sum())
    if full_duplicate_count:
        out = out.drop_duplicates().copy()

    out = cast_processed_types(out)
    out = out.sort_values(["id_odsp", "sort_order"], kind="mergesort").reset_index(drop=True)

    log = pd.DataFrame(
        [{"action": "created", "column": col} for col in created]
        + [{"action": "updated", "column": col} for col in updated]
        + [{"action": "dropped", "column": col} for col in dropped]
        + [
            {
                "action": "deduplicated",
                "column": "__full_rows__",
                "details": f"Removed {full_duplicate_count} full duplicate rows.",
            }
        ]
        + [
            {"action": "comment", "column": row["topic"], "details": row["comment"]}
            for row in COMMENT_ROWS
        ]
    )
    return out, log


def cast_processed_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in binary_columns(out):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int8")

    for col in ["time", "sort_order", "season", "fthg", "ftag"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    string_cols = [
        "id_odsp",
        "id_event",
        "text",
        "player",
        "event_team",
        "opponent",
        "ht",
        "at",
        "league",
        "country",
        "final_score",
    ]
    for col in string_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def binary_feature_validation(df: pd.DataFrame, binary_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in binary_cols:
        if col not in df.columns:
            rows.append(
                {
                    "feature": col,
                    "exists": False,
                    "valid_0_1_only": False,
                    "unique_values": "",
                    "null_count": pd.NA,
                }
            )
            continue
        values = sorted(pd.Series(df[col].dropna().unique()).astype(str).tolist())
        rows.append(
            {
                "feature": col,
                "exists": True,
                "valid_0_1_only": set(df[col].dropna().unique()).issubset({0, 1}),
                "unique_values": ", ".join(values),
                "null_count": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def impossible_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    checks: list[tuple[str, pd.Series]] = []
    if "time" in df.columns:
        time = pd.to_numeric(df["time"], errors="coerce")
        checks.append(("time_outside_0_130", time.lt(0) | time.gt(130)))
    if "side" in df.columns:
        checks.append(("side_not_1_2", ~pd.to_numeric(df["side"], errors="coerce").isin([1, 2])))
    if "is_goal" in df.columns:
        checks.append(
            ("is_goal_not_0_1", ~pd.to_numeric(df["is_goal"], errors="coerce").isin([0, 1]))
        )
    if "fthg" in df.columns:
        checks.append(("fthg_negative", pd.to_numeric(df["fthg"], errors="coerce").lt(0)))
    if "ftag" in df.columns:
        checks.append(("ftag_negative", pd.to_numeric(df["ftag"], errors="coerce").lt(0)))
    if "sort_order" in df.columns:
        checks.append(
            ("sort_order_non_positive", pd.to_numeric(df["sort_order"], errors="coerce").le(0))
        )

    existing_check_names = {name for name, _ in checks}
    for col in binary_columns(df):
        check_name = f"{col}_not_0_1"
        if check_name in existing_check_names:
            continue
        bad = ~pd.to_numeric(df[col], errors="coerce").isin([0, 1])
        checks.append((check_name, bad))

    return pd.DataFrame(
        [
            {
                "check": name,
                "violations": int(mask.fillna(False).sum()),
                "has_warning": bool(mask.fillna(False).any()),
            }
            for name, mask in checks
        ]
    )


def duplicate_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    id_event_duplicates = (
        int(df_after["id_event"].duplicated().sum()) if "id_event" in df_after else 0
    )
    return pd.DataFrame(
        [
            {
                "check": "full_duplicate_rows_before_drop",
                "value": int(df_before.duplicated().sum()),
            },
            {"check": "full_duplicate_rows_after_drop", "value": int(df_after.duplicated().sum())},
            {"check": "id_event_duplicate_rows", "value": id_event_duplicates},
        ]
    )


def duplicate_id_event_report(df: pd.DataFrame) -> pd.DataFrame:
    if "id_event" not in df.columns:
        return pd.DataFrame(columns=["id_event", "count"])
    counts = df["id_event"].value_counts(dropna=False)
    return counts[counts > 1].reset_index().rename(columns={"index": "id_event", "count": "count"})


def temporal_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    if not {"id_odsp", "sort_order", "time"}.issubset(df.columns):
        return pd.DataFrame()
    work = df[["id_odsp", "sort_order", "time"]].copy()
    work["sort_order"] = pd.to_numeric(work["sort_order"], errors="coerce")
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    sorted_work = work.sort_values(["id_odsp", "sort_order"], kind="mergesort")
    sort_diff = sorted_work.groupby("id_odsp")["sort_order"].diff()
    time_diff = sorted_work.groupby("id_odsp")["time"].diff()
    sort_problem = sort_diff.lt(0).fillna(False)
    time_problem = time_diff.lt(-5).fillna(False)
    duplicate_sort_order = sorted_work.duplicated(["id_odsp", "sort_order"], keep=False)
    return pd.DataFrame(
        [
            {
                "check": "sort_order_decreases_after_sort",
                "problem_rows": int(sort_problem.sum()),
                "problem_matches": int(sorted_work.loc[sort_problem, "id_odsp"].nunique()),
            },
            {
                "check": "duplicate_sort_order_within_match",
                "problem_rows": int(duplicate_sort_order.sum()),
                "problem_matches": int(sorted_work.loc[duplicate_sort_order, "id_odsp"].nunique()),
            },
            {
                "check": "time_decreases_by_more_than_5_minutes",
                "problem_rows": int(time_problem.sum()),
                "problem_matches": int(sorted_work.loc[time_problem, "id_odsp"].nunique()),
            },
        ]
    )


def event_flag_counts(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in EVENT_TYPE_FEATURES if col in df.columns]
    return pd.DataFrame({"feature": cols, "count": [int(df[col].sum()) for col in cols]})


def side_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "side" not in df.columns:
        return pd.DataFrame()
    return df["side"].value_counts(dropna=False).rename_axis("side").reset_index(name="count")


def goals_by_side(df: pd.DataFrame) -> pd.DataFrame:
    if not {"side", "is_goal"}.issubset(df.columns):
        return pd.DataFrame()
    return (
        df.groupby("side", dropna=False)["is_goal"]
        .sum()
        .reset_index()
        .rename(columns={"is_goal": "goals"})
    )


def events_per_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "id_odsp" not in df.columns:
        return pd.DataFrame()
    counts = df.groupby("id_odsp", dropna=False).size()
    stats = counts.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    return pd.DataFrame({"metric": stats.index, "value": stats.values})


def time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return pd.DataFrame()
    bins = [-1, 0, 15, 30, 45, 60, 75, 90, 105, 130]
    labels = ["0", "1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "91-105", "106-130"]
    bucket = pd.cut(pd.to_numeric(df["time"], errors="coerce"), bins=bins, labels=labels)
    return (
        bucket.value_counts(dropna=False)
        .sort_index()
        .rename_axis("time_bucket")
        .reset_index(name="count")
    )


def duplicate_feature_checks(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("shot_blocked", "is_blocked"),
        ("shot_hit_bar", "is_hit_bar"),
    ]
    rows = []
    for left, right in pairs:
        if left not in df.columns or right not in df.columns:
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "exists": False,
                    "equals_rate": pd.NA,
                    "identical": False,
                    "correlation": pd.NA,
                }
            )
            continue
        left_s = pd.to_numeric(df[left], errors="coerce")
        right_s = pd.to_numeric(df[right], errors="coerce")
        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "exists": True,
                "equals_rate": float(left_s.eq(right_s).mean()),
                "identical": bool(left_s.equals(right_s)),
                "correlation": float(left_s.corr(right_s))
                if left_s.nunique() > 1 and right_s.nunique() > 1
                else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def minute_level_count_columns(df: pd.DataFrame) -> list[str]:
    protected = {
        *MINUTE_METADATA_ROLE_COLUMNS,
        *LEAKAGE_COLUMNS,
        *TARGET_COLUMNS,
        *AUXILIARY_TARGET_COLUMNS,
        *BOOKMAKER_PRIOR_COLUMNS,
    }
    numeric_like = []
    for col in df.columns:
        if col in protected:
            continue
        if col.startswith(("home_", "away_")) or col.startswith("score_diff_"):
            numeric_like.append(col)
    return numeric_like


def goal_semantics_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["home_goal", "away_goal"]:
        exists = col in df.columns
        rows.append(
            {
                "column": col,
                "exists": exists,
                "meaning": (
                    "Current-minute first-half goal count aggregated from event-level is_goal "
                    "and side before the second-half targets were attached."
                    if exists
                    else "Column is absent."
                ),
                "leakage_status": "not_leakage_current_minute_event"
                if exists
                else "not_applicable",
                "action": "kept" if exists else "not_applicable",
                "total_count": int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
                if exists
                else 0,
                "nonzero_rows": int(pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).sum())
                if exists
                else 0,
            }
        )
    return pd.DataFrame(rows)


def date_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "metric": "date_exists",
                    "value": False,
                }
            ]
        )
    date = pd.to_datetime(df["date"], errors="coerce")
    sorted_keys = df.assign(date=date).sort_values(["date", "id_odsp", "time"], kind="mergesort")
    is_sorted = df[["id_odsp", "time"]].reset_index(drop=True).equals(
        sorted_keys[["id_odsp", "time"]].reset_index(drop=True)
    ) and date.reset_index(drop=True).equals(sorted_keys["date"].reset_index(drop=True))
    return pd.DataFrame(
        [
            {"metric": "date_exists", "value": True},
            {"metric": "date_nulls", "value": int(date.isna().sum())},
            {"metric": "min_date", "value": str(date.min().date()) if date.notna().any() else ""},
            {"metric": "max_date", "value": str(date.max().date()) if date.notna().any() else ""},
            {"metric": "unique_dates", "value": int(date.nunique(dropna=True))},
            {"metric": "sorted_by_date_id_time_before_cleanup", "value": bool(is_sorted)},
        ]
    )


def column_role_report(df: pd.DataFrame) -> pd.DataFrame:
    metadata = [col for col in MINUTE_METADATA_ROLE_COLUMNS if col in df.columns]
    target = [col for col in TARGET_COLUMNS if col in df.columns]
    leakage = [col for col in [*LEAKAGE_COLUMNS, *BOOKMAKER_PRIOR_COLUMNS] if col in df.columns]
    excluded = set(metadata + target + leakage)
    safe = [
        col
        for col in df.columns
        if col not in excluded and not pd.api.types.is_object_dtype(df[col])
    ]
    rows = []
    for role, columns in [
        ("safe_feature", safe),
        ("metadata", metadata),
        ("leakage_or_not_safe_feature", leakage),
        ("target", target),
    ]:
        for col in columns:
            rows.append({"role": role, "column": col, "dtype": str(df[col].dtype)})
    return pd.DataFrame(rows)


def cleanup_type_checks(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    binary_or_count = [
        col for col in feature_cols if col.startswith(("home_", "away_")) and col in df.columns
    ]
    object_features = [
        col
        for col in feature_cols
        if col in df.columns
        and (pd.api.types.is_object_dtype(df[col]) or str(df[col].dtype) == "string")
    ]
    rows = [
        {
            "check": "binary_or_count_features_int8_int16",
            "passed": all(str(df[col].dtype) in {"int8", "int16"} for col in binary_or_count),
            "details": ", ".join(
                sorted(
                    [
                        f"{col}:{df[col].dtype}"
                        for col in binary_or_count
                        if str(df[col].dtype) not in {"int8", "int16"}
                    ]
                )[:50]
            ),
        },
        {
            "check": "time_numeric",
            "passed": "time" in df.columns and pd.api.types.is_numeric_dtype(df["time"]),
            "details": str(df["time"].dtype) if "time" in df.columns else "missing",
        },
        {
            "check": "date_datetime64_in_memory",
            "passed": "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]),
            "details": str(df["date"].dtype) if "date" in df.columns else "missing",
        },
        {
            "check": "target_columns_numeric",
            "passed": all(
                col in df.columns and pd.api.types.is_numeric_dtype(df[col])
                for col in TARGET_COLUMNS
            ),
            "details": ", ".join(
                f"{col}:{df[col].dtype}" for col in TARGET_COLUMNS if col in df.columns
            ),
        },
        {
            "check": "object_columns_not_used_as_numeric_features",
            "passed": len(object_features) == 0,
            "details": ", ".join(object_features),
        },
    ]
    return pd.DataFrame(rows)


def final_minute_diagnostics(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows_per_match = (
        df.groupby("id_odsp").size() if "id_odsp" in df.columns else pd.Series(dtype=int)
    )
    dtype_counts = (
        df[feature_cols].dtypes.astype(str).value_counts() if feature_cols else pd.Series(dtype=int)
    )
    rows = [
        {"metric": "final_rows", "value": df.shape[0]},
        {"metric": "final_columns", "value": df.shape[1]},
        {
            "metric": "number_of_matches",
            "value": int(df["id_odsp"].nunique()) if "id_odsp" in df else 0,
        },
        {"metric": "feature_columns", "value": len(feature_cols)},
        {"metric": "target_columns", "value": len([c for c in TARGET_COLUMNS if c in df.columns])},
    ]
    for metric, value in rows_per_match.describe().items():
        rows.append({"metric": f"rows_per_match_{metric}", "value": value})
    for dtype, count in dtype_counts.items():
        rows.append({"metric": f"feature_dtype_{dtype}", "value": int(count)})
    return pd.DataFrame(rows)


def feature_target_correlations(
    df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    roles = column_role_report(df)
    feature_cols = roles.loc[roles["role"].eq("safe_feature"), "column"].tolist()
    rows = []
    for target in [col for col in TARGET_COLUMNS if col in df.columns]:
        target_s = pd.to_numeric(df[target], errors="coerce")
        for feature in feature_cols:
            feature_s = pd.to_numeric(df[feature], errors="coerce")
            if feature_s.nunique(dropna=True) <= 1 or target_s.nunique(dropna=True) <= 1:
                corr = pd.NA
            else:
                corr = feature_s.corr(target_s)
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "correlation": corr,
                    "abs_correlation": abs(corr) if pd.notna(corr) else pd.NA,
                }
            )
    corr_df = pd.DataFrame(rows).dropna(subset=["abs_correlation"])
    if corr_df.empty:
        return corr_df
    return (
        corr_df.sort_values(["target", "abs_correlation"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def processed_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in [col for col in TARGET_COLUMNS if col in df.columns]:
        counts = df[target].value_counts(dropna=False).sort_index()
        match_counts = df.drop_duplicates("id_odsp")[target].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "target": target,
                    "value": value,
                    "minute_rows": int(count),
                    "matches": int(match_counts.get(value, 0)),
                }
            )
    return pd.DataFrame(rows)


def processed_target_diagnostics(
    df: pd.DataFrame,
    log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    roles = column_role_report(df)
    feature_cols = roles.loc[roles["role"].eq("safe_feature"), "column"].tolist()
    created_home_away = [
        col
        for col in df.columns
        if col.startswith(("home_", "away_"))
        and col not in [*TARGET_COLUMNS, *AUXILIARY_TARGET_COLUMNS]
    ]
    dropped_generic = []
    if log is not None and not log.empty and {"action", "column"}.issubset(log.columns):
        dropped_generic = log.loc[log["action"].eq("dropped"), "column"].dropna().tolist()
    else:
        dropped_generic = [col for col in HOME_AWAY_SPLIT_FEATURES if col not in df.columns]
    matches = df["id_odsp"].nunique(dropna=True) if "id_odsp" in df.columns else 0
    rows = [
        {"metric": "input_shape", "value": f"{df.shape[0]} rows x {df.shape[1]} columns"},
        {"metric": "output_shape", "value": f"{df.shape[0]} rows x {df.shape[1]} columns"},
        {"metric": "number_of_matches", "value": matches},
        {"metric": "number_of_minute_rows", "value": df.shape[0]},
        {"metric": "expected_rows_matches_x_45", "value": matches * 45},
        {"metric": "feature_columns", "value": len(feature_cols)},
        {"metric": "target_columns", "value": len([c for c in TARGET_COLUMNS if c in df.columns])},
        {"metric": "created_home_away_features", "value": len(created_home_away)},
        {"metric": "dropped_generic_features", "value": len(dropped_generic)},
        {"metric": "null_values_after_aggregation", "value": int(df.isna().sum().sum())},
        {
            "metric": "target_recalculation_note",
            "value": (
                "football_merged_processed.csv contains only first-half minutes 1-45. "
                "Targets are present in this dataset, but strict recalculation with time > 45 "
                "requires the full event-level source."
            ),
        },
    ]
    return pd.DataFrame(rows)


def save_processed_target_reports(
    df: pd.DataFrame,
    report_dir: Path,
    audit_dir: Path | None = None,
    log: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    report_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "feature_target_correlations": feature_target_correlations(df),
        "target_distribution": processed_target_distribution(df),
        "target_diagnostics": processed_target_diagnostics(df, log),
        "sample_rows": df.head(20),
    }
    reports["feature_target_correlations"].to_csv(
        report_dir / "football_feature_target_correlations.csv", index=False
    )
    reports["target_distribution"].to_csv(
        report_dir / "football_target_distribution.csv", index=False
    )
    reports["target_diagnostics"].to_csv(
        report_dir / "football_target_diagnostics.csv", index=False
    )
    reports["sample_rows"].to_csv(report_dir / "football_processed_sample_rows.csv", index=False)
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        for name, table in reports.items():
            table.to_csv(audit_dir / f"football_merged_processed_{name}.csv", index=False)
    return reports


def cleanup_minute_level_processed(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    out = df.copy()
    dropped = []

    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
        dropped.append("Unnamed: 0")

    date_before = date_diagnostics(out)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in ["time", "season", "fthg", "ftag"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in [*TARGET_COLUMNS, *AUXILIARY_TARGET_COLUMNS]:
        if col in out.columns:
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .fillna(0)
                .astype("int16" if col.startswith("second_half") else "int8")
            )

    for col in minute_level_count_columns(out):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int16")

    for col in ["id_odsp", "league", "country", "ht", "at", "final_score"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    if {"date", "id_odsp", "time"}.issubset(out.columns):
        out = out.sort_values(["date", "id_odsp", "time"], kind="mergesort").reset_index(drop=True)

    roles = column_role_report(out)
    feature_cols = roles.loc[roles["role"].eq("safe_feature"), "column"].tolist()
    null_counts = (
        out.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "null_count"})
        .query("null_count > 0")
    )
    feature_nulls = (
        out[feature_cols]
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "null_count"})
        .query("null_count > 0")
        if feature_cols
        else pd.DataFrame(columns=["column", "null_count"])
    )

    cleanup_log = pd.DataFrame(
        [
            {"action": "dropped", "column": col, "details": "Old pandas index column."}
            for col in dropped
        ]
        + [
            {
                "action": "sorted",
                "column": "date / id_odsp / time",
                "details": "Dataset sorted chronologically for later temporal split.",
            }
        ]
    )
    reports = {
        "cleanup_log": cleanup_log,
        "date_diagnostics_before_cleanup": date_before,
        "date_diagnostics_after_cleanup": date_diagnostics(out),
        "goal_semantics": goal_semantics_report(out),
        "column_roles": roles,
        "type_checks": cleanup_type_checks(out, feature_cols),
        "null_counts": null_counts,
        "feature_null_counts": feature_nulls,
        "final_diagnostics": final_minute_diagnostics(out, feature_cols),
    }
    return out, reports


def build_minute_level_from_processed_events(
    event_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = event_df.copy()
    created: list[str] = []
    dropped: list[str] = []

    if "text" in work.columns:
        work = work.drop(columns=["text"])
        dropped.append("text")

    side = (
        pd.to_numeric(work["side"], errors="coerce")
        if "side" in work.columns
        else pd.Series(pd.NA, index=work.index)
    )

    for source_col, base_name in HOME_AWAY_SPLIT_FEATURES.items():
        if source_col not in work.columns:
            continue
        source = pd.to_numeric(work[source_col], errors="coerce").fillna(0)
        home_col = f"home_{base_name}"
        away_col = f"away_{base_name}"
        work[home_col] = source.where(side.eq(1), 0).astype("int16")
        work[away_col] = source.where(side.eq(2), 0).astype("int16")
        created.extend([home_col, away_col])

    if "is_goal" in work.columns:
        goal = pd.to_numeric(work["is_goal"], errors="coerce").fillna(0)
        work["home_goal"] = goal.where(side.eq(1), 0).astype("int16")
        work["away_goal"] = goal.where(side.eq(2), 0).astype("int16")
        created.extend(["home_goal", "away_goal"])

    split_source_cols = [col for col in HOME_AWAY_SPLIT_FEATURES if col in work.columns]
    if split_source_cols:
        work = work.drop(columns=split_source_cols)
        dropped.extend(split_source_cols)

    work["time"] = pd.to_numeric(work["time"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["id_odsp", "time"]).copy()
    work["time"] = work["time"].astype(int)

    second_half = work[work["time"].gt(45)]
    targets = second_half.groupby("id_odsp", as_index=False).agg(
        second_half_home_goals=("home_goal", "sum"),
        second_half_away_goals=("away_goal", "sum"),
    )
    targets["home_scores_next_half"] = targets["second_half_home_goals"].gt(0).astype("int8")
    targets["away_scores_next_half"] = targets["second_half_away_goals"].gt(0).astype("int8")

    first_half = work[work["time"].between(1, 45)].copy()
    count_cols = [col for col in created if col in first_half.columns]
    agg_spec = {col: (col, "sum") for col in count_cols}
    for col in [c for c in MINUTE_METADATA_COLUMNS if c in first_half.columns]:
        agg_spec[col] = (col, "first")

    minute = first_half.groupby(["id_odsp", "time"], as_index=False).agg(**agg_spec)

    match_ids = pd.Index(work["id_odsp"].dropna().unique(), name="id_odsp")
    full_index = pd.MultiIndex.from_product([match_ids, range(1, 46)], names=["id_odsp", "time"])
    minute = (
        minute.set_index(["id_odsp", "time"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["id_odsp", "time"], kind="mergesort")
    )

    for col in count_cols:
        minute[col] = minute[col].fillna(0).astype("int16")

    metadata = work.sort_values(["id_odsp", "time"], kind="mergesort").drop_duplicates("id_odsp")[
        ["id_odsp", *[c for c in MINUTE_METADATA_COLUMNS if c in work.columns]]
    ]
    metadata_cols = [c for c in metadata.columns if c != "id_odsp"]
    if metadata_cols:
        minute = minute.drop(columns=[c for c in metadata_cols if c in minute.columns])
        minute = minute.merge(metadata, on="id_odsp", how="left")

    minute = minute.merge(targets, on="id_odsp", how="left")
    target_cols = [
        "second_half_home_goals",
        "second_half_away_goals",
        "home_scores_next_half",
        "away_scores_next_half",
    ]
    for col in target_cols:
        minute[col] = minute[col].fillna(0).astype("int16" if "goals" in col else "int8")

    if {"home_goal", "away_goal"}.issubset(minute.columns):
        minute["home_score_first_half_so_far"] = minute.groupby("id_odsp")["home_goal"].cumsum()
        minute["away_score_first_half_so_far"] = minute.groupby("id_odsp")["away_goal"].cumsum()
        minute["score_diff_first_half_so_far"] = (
            minute["home_score_first_half_so_far"] - minute["away_score_first_half_so_far"]
        )

    log = pd.DataFrame(
        [{"action": "created", "column": col} for col in created]
        + [{"action": "dropped", "column": col} for col in dropped]
        + [
            {
                "action": "created",
                "column": "minute_grid_1_45",
                "details": "Added every minute 1-45 for each match; missing event counts are 0.",
            },
            {
                "action": "created",
                "column": "home_scores_next_half / away_scores_next_half",
                "details": "Targets computed from second-half goals before filtering to first-half minutes.",
            },
        ]
    )
    return minute, log


def minute_level_summary(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    log: pd.DataFrame,
    saved_path: Path,
) -> pd.DataFrame:
    created = log.loc[log["action"].eq("created"), "column"].dropna().tolist()
    dropped = log.loc[log["action"].eq("dropped"), "column"].dropna().tolist()
    matches = output_df["id_odsp"].nunique(dropna=True) if "id_odsp" in output_df else 0
    complete_grid_rows = matches * 45
    return pd.DataFrame(
        [
            {"metric": "input_event_rows", "value": input_df.shape[0]},
            {"metric": "input_event_columns", "value": input_df.shape[1]},
            {"metric": "output_minute_rows", "value": output_df.shape[0]},
            {"metric": "output_minute_columns", "value": output_df.shape[1]},
            {"metric": "unique_matches", "value": matches},
            {"metric": "expected_first_half_grid_rows", "value": complete_grid_rows},
            {"metric": "created_columns_count", "value": len(created)},
            {"metric": "dropped_columns_count", "value": len(dropped)},
            {"metric": "saved_path", "value": str(saved_path)},
        ]
    )


def processing_summary(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    log: pd.DataFrame,
) -> pd.DataFrame:
    created = log.loc[log["action"].eq("created"), "column"].dropna().tolist()
    dropped = log.loc[log["action"].eq("dropped"), "column"].dropna().tolist()
    return pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_df.shape[0]},
            {"metric": "input_columns", "value": input_df.shape[1]},
            {"metric": "output_rows", "value": output_df.shape[0]},
            {"metric": "output_columns", "value": output_df.shape[1]},
            {"metric": "created_features_count", "value": len(created)},
            {"metric": "dropped_columns_count", "value": len(dropped)},
            {
                "metric": "null_values_after_transformation",
                "value": int(output_df.isna().sum().sum()),
            },
        ]
    )


def second_pass_summary(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    log: pd.DataFrame,
    impossible: pd.DataFrame,
    duplicates: pd.DataFrame,
    saved_path: Path,
) -> pd.DataFrame:
    created = log.loc[log["action"].eq("created"), "column"].dropna().tolist()
    updated = log.loc[log["action"].eq("updated"), "column"].dropna().tolist()
    dropped = log.loc[log["action"].eq("dropped"), "column"].dropna().tolist()
    return pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_df.shape[0]},
            {"metric": "input_columns", "value": input_df.shape[1]},
            {"metric": "output_rows", "value": output_df.shape[0]},
            {"metric": "output_columns", "value": output_df.shape[1]},
            {"metric": "created_columns_count", "value": len(created)},
            {"metric": "updated_columns_count", "value": len(updated)},
            {"metric": "dropped_columns_count", "value": len(dropped)},
            {"metric": "impossible_value_violations", "value": int(impossible["violations"].sum())},
            {
                "metric": "id_event_duplicate_rows",
                "value": int(
                    duplicates.loc[duplicates["check"].eq("id_event_duplicate_rows"), "value"].sum()
                ),
            },
            {"metric": "saved_path", "value": str(saved_path)},
        ]
    )


def save_football_merged_processed_outputs(
    input_path: Path,
    output_path: Path,
    audit_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"football_merged.csv was not found: {input_path}")
    input_df = pd.read_csv(input_path)
    first_pass_df, first_log = process_football_merged_events(input_df)
    event_df, second_log = process_event_type_assist_and_quality(first_pass_df)
    output_df, minute_log = build_minute_level_from_processed_events(event_df)
    output_df, cleanup_reports = cleanup_minute_level_processed(output_df)
    log = pd.concat([first_log, second_log, minute_log], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    created_cols = (
        log.loc[log["action"].isin(["created", "updated"]), "column"].dropna().unique().tolist()
    )
    binary_cols = binary_columns(output_df)
    validation = binary_feature_validation(output_df, binary_cols)
    if not validation.empty and not validation["valid_0_1_only"].all():
        invalid = validation.loc[~validation["valid_0_1_only"], "feature"].tolist()
        raise ValueError(f"Non-binary values found in created features: {invalid}")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        log.to_csv(audit_dir / "football_merged_processed_feature_log.csv", index=False)
        processing_summary(input_df, output_df, log).to_csv(
            audit_dir / "football_merged_processed_summary.csv", index=False
        )
        validation.to_csv(
            audit_dir / "football_merged_processed_binary_validation.csv", index=False
        )
        preview_cols = [col for col in created_cols if col in output_df.columns]
        output_df[preview_cols].head(5).to_csv(
            audit_dir / "football_merged_processed_new_features_head.csv", index=False
        )
        output_df.head(20).to_csv(audit_dir / "football_merged_processed_head.csv", index=False)
        save_second_pass_reports(first_pass_df, event_df, second_log, output_path, audit_dir)
        minute_log.to_csv(audit_dir / "football_merged_processed_minute_level_log.csv", index=False)
        minute_level_summary(event_df, output_df, minute_log, output_path).to_csv(
            audit_dir / "football_merged_processed_minute_level_summary.csv", index=False
        )
        save_minute_cleanup_reports(cleanup_reports, audit_dir)
        save_processed_target_reports(
            output_df,
            audit_dir.parent.parent / "reports",
            audit_dir,
            log,
        )
    return output_df, log


def save_minute_cleanup_reports(reports: dict[str, pd.DataFrame], audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name, table in reports.items():
        table.to_csv(audit_dir / f"football_merged_processed_{name}.csv", index=False)


def save_second_pass_reports(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    log: pd.DataFrame,
    output_path: Path,
    audit_dir: Path,
) -> None:
    impossible = impossible_values_summary(output_df)
    duplicates = duplicate_summary(input_df, output_df)
    second_pass_summary(input_df, output_df, log, impossible, duplicates, output_path).to_csv(
        audit_dir / "football_merged_processed_second_pass_summary.csv", index=False
    )
    log.to_csv(audit_dir / "football_merged_processed_second_pass_log.csv", index=False)
    impossible.to_csv(audit_dir / "football_merged_processed_impossible_values.csv", index=False)
    duplicates.to_csv(audit_dir / "football_merged_processed_duplicate_summary.csv", index=False)
    duplicate_id_event_report(output_df).to_csv(
        audit_dir / "football_merged_processed_duplicate_id_event_report.csv", index=False
    )
    temporal_consistency_report(output_df).to_csv(
        audit_dir / "football_merged_processed_temporal_consistency.csv", index=False
    )
    event_flag_counts(output_df).to_csv(
        audit_dir / "football_merged_processed_event_flag_counts.csv", index=False
    )
    side_event_counts(output_df).to_csv(
        audit_dir / "football_merged_processed_side_event_counts.csv", index=False
    )
    goals_by_side(output_df).to_csv(
        audit_dir / "football_merged_processed_goals_by_side.csv", index=False
    )
    events_per_match_stats(output_df).to_csv(
        audit_dir / "football_merged_processed_events_per_match_stats.csv", index=False
    )
    time_distribution(output_df).to_csv(
        audit_dir / "football_merged_processed_time_distribution.csv", index=False
    )
    duplicate_feature_checks(output_df).to_csv(
        audit_dir / "football_merged_processed_duplicate_feature_checks.csv", index=False
    )


def update_existing_football_merged_processed(
    input_path: Path,
    output_path: Path,
    audit_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"football_merged_processed.csv was not found: {input_path}")
    input_df = pd.read_csv(input_path)
    output_df, log = process_event_type_assist_and_quality(input_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    checked_cols = (
        log.loc[log["action"].isin(["created", "updated"]), "column"].dropna().unique().tolist()
    )
    validation = binary_feature_validation(output_df, checked_cols)
    if not validation["valid_0_1_only"].all():
        invalid = validation.loc[~validation["valid_0_1_only"], "feature"].tolist()
        raise ValueError(f"Non-binary values found in created/updated features: {invalid}")

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        save_second_pass_reports(input_df, output_df, log, output_path, audit_dir)
        validation.to_csv(
            audit_dir / "football_merged_processed_second_pass_binary_validation.csv",
            index=False,
        )
        output_df[checked_cols].head(5).to_csv(
            audit_dir / "football_merged_processed_second_pass_new_features_head.csv",
            index=False,
        )
    return output_df, log
