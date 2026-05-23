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


def save_football_merged_processed_outputs(
    input_path: Path,
    output_path: Path,
    audit_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"football_merged.csv was not found: {input_path}")
    input_df = pd.read_csv(input_path)
    output_df, log = process_football_merged_events(input_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    created_cols = log.loc[log["action"].eq("created"), "column"].dropna().tolist()
    validation = binary_feature_validation(output_df, created_cols)
    if not validation["valid_0_1_only"].all():
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
        output_df[created_cols].head(5).to_csv(
            audit_dir / "football_merged_processed_new_features_head.csv", index=False
        )
        output_df.head(20).to_csv(audit_dir / "football_merged_processed_head.csv", index=False)
    return output_df, log
