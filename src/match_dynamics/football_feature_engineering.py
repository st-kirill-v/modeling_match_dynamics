from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["home_scores_next_half", "away_scores_next_half"]
LEAKAGE_COLUMNS = [
    "home_scores_next_half",
    "away_scores_next_half",
    "second_half_home_goals",
    "second_half_away_goals",
    "fthg",
    "ftag",
    "final_score",
]

ATTACK_FEATURES = [
    "home_attempt",
    "away_attempt",
    "home_on_target",
    "away_on_target",
    "home_key_pass",
    "away_key_pass",
    "home_corner",
    "away_corner",
    "home_box_zone",
    "away_box_zone",
    "home_attacking_half",
    "away_attacking_half",
    "home_assist_cross",
    "away_assist_cross",
    "home_assist_through_ball",
    "away_assist_through_ball",
]

DISCIPLINE_FEATURES = [
    "home_foul",
    "away_foul",
    "home_yellow_card",
    "away_yellow_card",
    "home_red_card",
    "away_red_card",
    "home_second_yellow_card",
    "away_second_yellow_card",
    "home_penalty_conceded",
    "away_penalty_conceded",
]

GOAL_FEATURES = ["home_goal", "away_goal"]

ROLLING_BASE_FEATURES = [
    "home_attempt",
    "away_attempt",
    "home_on_target",
    "away_on_target",
    "home_key_pass",
    "away_key_pass",
    "home_corner",
    "away_corner",
    "home_box_zone",
    "away_box_zone",
    "home_attacking_half",
    "away_attacking_half",
    "home_foul",
    "away_foul",
    "home_yellow_card",
    "away_yellow_card",
]

DIFF_PAIRS = {
    "attempt_diff": ("home_attempt", "away_attempt"),
    "on_target_diff": ("home_on_target", "away_on_target"),
    "key_pass_diff": ("home_key_pass", "away_key_pass"),
    "corner_diff": ("home_corner", "away_corner"),
    "box_zone_diff": ("home_box_zone", "away_box_zone"),
    "attacking_half_diff": ("home_attacking_half", "away_attacking_half"),
    "foul_diff": ("home_foul", "away_foul"),
    "yellow_card_diff": ("home_yellow_card", "away_yellow_card"),
    "red_card_diff": ("home_red_card", "away_red_card"),
}

ROLLING_DIFFS = {
    "attempt_diff": ("home_attempt", "away_attempt"),
    "on_target_diff": ("home_on_target", "away_on_target"),
    "key_pass_diff": ("home_key_pass", "away_key_pass"),
    "corner_diff": ("home_corner", "away_corner"),
    "box_zone_diff": ("home_box_zone", "away_box_zone"),
    "attacking_half_diff": ("home_attacking_half", "away_attacking_half"),
    "foul_diff": ("home_foul", "away_foul"),
}

WINDOWS = [3, 5, 10]


def _rolling_sum_by_match(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        df.groupby("id_odsp", sort=False)[col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).sum())
        .astype("int16")
    )


def _available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def _missing(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col not in df.columns]


def _safe_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(LEAKAGE_COLUMNS + ["time", "season"])
    blocked_prefixes = ("odd_",)
    cols = []
    for col in df.columns:
        if col in blocked or col.startswith(blocked_prefixes):
            continue
        if col.startswith(("home_", "away_")) and pd.api.types.is_numeric_dtype(df[col]):
            if "_scores_next_half" in col or col.startswith("second_half_"):
                continue
            cols.append(col)
    return cols


def add_advanced_football_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    input_shape = df.shape
    input_matches = df["id_odsp"].nunique(dropna=True)
    out = df.copy()
    created: list[str] = []
    skipped: list[dict[str, str]] = []

    required = ["id_odsp", "time", *TARGET_COLUMNS]
    missing_required = _missing(required, out)
    if missing_required:
        raise ValueError(f"Required columns are missing: {missing_required}")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        sort_cols = ["date", "id_odsp", "time"]
    else:
        sort_cols = ["id_odsp", "time"]
    out["time"] = pd.to_numeric(out["time"], errors="coerce").astype("Int64")
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    expected_groups = {
        "attack": ATTACK_FEATURES,
        "discipline": DISCIPLINE_FEATURES,
        "goal": GOAL_FEATURES,
    }
    for group, columns in expected_groups.items():
        for col in _missing(columns, out):
            skipped.append({"group": group, "column": col, "reason": "missing_in_input"})

    for col in _available(ATTACK_FEATURES + DISCIPLINE_FEATURES + GOAL_FEATURES, out):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int16")

    for col in _available(ROLLING_BASE_FEATURES, out):
        for window in WINDOWS:
            new_col = f"{col}_last_{window}min"
            out[new_col] = _rolling_sum_by_match(out, col, window)
            created.append(new_col)

    for new_col, (home_col, away_col) in DIFF_PAIRS.items():
        if home_col in out.columns and away_col in out.columns:
            out[new_col] = (
                pd.to_numeric(out[home_col], errors="coerce").fillna(0)
                - pd.to_numeric(out[away_col], errors="coerce").fillna(0)
            ).astype("int16")
            created.append(new_col)
        else:
            skipped.append(
                {
                    "group": "relative_current",
                    "column": new_col,
                    "reason": f"missing {home_col} or {away_col}",
                }
            )

    for base_name, (home_col, away_col) in ROLLING_DIFFS.items():
        for window in WINDOWS:
            home_roll = f"{home_col}_last_{window}min"
            away_roll = f"{away_col}_last_{window}min"
            new_col = f"{base_name}_last_{window}min"
            if home_roll in out.columns and away_roll in out.columns:
                out[new_col] = (out[home_roll] - out[away_roll]).astype("int16")
                created.append(new_col)
            else:
                skipped.append(
                    {
                        "group": "relative_rolling",
                        "column": new_col,
                        "reason": f"missing {home_roll} or {away_roll}",
                    }
                )

    home_attack_parts = _available(
        ["home_attempt", "home_key_pass", "home_corner", "home_box_zone", "home_attacking_half"],
        out,
    )
    away_attack_parts = _available(
        ["away_attempt", "away_key_pass", "away_corner", "away_box_zone", "away_attacking_half"],
        out,
    )
    out["home_attack_events"] = out[home_attack_parts].sum(axis=1).astype("int16")
    out["away_attack_events"] = out[away_attack_parts].sum(axis=1).astype("int16")
    out["total_attack_events"] = (out["home_attack_events"] + out["away_attack_events"]).astype(
        "int16"
    )
    event_count_cols = _safe_numeric_feature_columns(out)
    out["total_events_proxy"] = out[event_count_cols].sum(axis=1).astype("int16")
    created.extend(
        ["home_attack_events", "away_attack_events", "total_attack_events", "total_events_proxy"]
    )

    for window in WINDOWS:
        for source, prefix in [
            ("home_attack_events", "home_attack_intensity"),
            ("away_attack_events", "away_attack_intensity"),
            ("total_attack_events", "total_attack_intensity"),
            ("total_events_proxy", "total_events"),
        ]:
            new_col = f"{prefix}_last_{window}min"
            out[new_col] = _rolling_sum_by_match(out, source, window)
            created.append(new_col)

    out["attack_intensity_diff"] = (out["home_attack_events"] - out["away_attack_events"]).astype(
        "int16"
    )
    created.append("attack_intensity_diff")
    for window in WINDOWS:
        new_col = f"attack_intensity_diff_last_{window}min"
        out[new_col] = (
            out[f"home_attack_intensity_last_{window}min"]
            - out[f"away_attack_intensity_last_{window}min"]
        ).astype("int16")
        created.append(new_col)

    home_pressure_parts = _available(
        ["home_attempt", "home_on_target", "home_key_pass", "home_corner", "home_box_zone"],
        out,
    )
    away_pressure_parts = _available(
        ["away_attempt", "away_on_target", "away_key_pass", "away_corner", "away_box_zone"],
        out,
    )
    out["home_pressure_minute"] = out[home_pressure_parts].sum(axis=1).gt(0).astype("int8")
    out["away_pressure_minute"] = out[away_pressure_parts].sum(axis=1).gt(0).astype("int8")
    created.extend(["home_pressure_minute", "away_pressure_minute"])
    for window in WINDOWS:
        for side in ["home", "away"]:
            new_col = f"{side}_pressure_last_{window}min"
            out[new_col] = _rolling_sum_by_match(out, f"{side}_pressure_minute", window)
            created.append(new_col)
        diff_col = f"pressure_diff_last_{window}min"
        out[diff_col] = (
            out[f"home_pressure_last_{window}min"] - out[f"away_pressure_last_{window}min"]
        ).astype("int16")
        created.append(diff_col)

    out["home_dominating_last_5min"] = out["pressure_diff_last_5min"].gt(0).astype("int8")
    out["away_dominating_last_5min"] = out["pressure_diff_last_5min"].lt(0).astype("int8")
    out["home_strong_pressure_last_5min"] = out["home_pressure_last_5min"].ge(3).astype("int8")
    out["away_strong_pressure_last_5min"] = out["away_pressure_last_5min"].ge(3).astype("int8")
    created.extend(
        [
            "home_dominating_last_5min",
            "away_dominating_last_5min",
            "home_strong_pressure_last_5min",
            "away_strong_pressure_last_5min",
        ]
    )

    if "score_diff_first_half_so_far" not in out.columns:
        if {"home_goal", "away_goal"}.issubset(out.columns):
            out["home_score_first_half_so_far"] = out.groupby("id_odsp", sort=False)[
                "home_goal"
            ].cumsum()
            out["away_score_first_half_so_far"] = out.groupby("id_odsp", sort=False)[
                "away_goal"
            ].cumsum()
            out["score_diff_first_half_so_far"] = (
                out["home_score_first_half_so_far"] - out["away_score_first_half_so_far"]
            )
            created.extend(
                [
                    "home_score_first_half_so_far",
                    "away_score_first_half_so_far",
                    "score_diff_first_half_so_far",
                ]
            )
        else:
            skipped.append(
                {
                    "group": "game_state",
                    "column": "score_diff_first_half_so_far",
                    "reason": "missing score diff and goal columns",
                }
            )

    if "score_diff_first_half_so_far" in out.columns:
        score_diff = pd.to_numeric(out["score_diff_first_half_so_far"], errors="coerce").fillna(0)
        out["home_leading"] = score_diff.gt(0).astype("int8")
        out["away_leading"] = score_diff.lt(0).astype("int8")
        out["draw_state"] = score_diff.eq(0).astype("int8")
        created.extend(["home_leading", "away_leading", "draw_state"])

    if {"home_red_card", "away_red_card"}.issubset(out.columns):
        out["home_red_cards_so_far"] = out.groupby("id_odsp", sort=False)["home_red_card"].cumsum()
        out["away_red_cards_so_far"] = out.groupby("id_odsp", sort=False)["away_red_card"].cumsum()
        out["red_card_diff_so_far"] = out["home_red_cards_so_far"] - out["away_red_cards_so_far"]
        created.extend(["home_red_cards_so_far", "away_red_cards_so_far", "red_card_diff_so_far"])
    if {"home_yellow_card", "away_yellow_card"}.issubset(out.columns):
        out["home_yellow_cards_so_far"] = out.groupby("id_odsp", sort=False)[
            "home_yellow_card"
        ].cumsum()
        out["away_yellow_cards_so_far"] = out.groupby("id_odsp", sort=False)[
            "away_yellow_card"
        ].cumsum()
        out["yellow_card_diff_so_far"] = (
            out["home_yellow_cards_so_far"] - out["away_yellow_cards_so_far"]
        )
        created.extend(
            ["home_yellow_cards_so_far", "away_yellow_cards_so_far", "yellow_card_diff_so_far"]
        )
    if {
        "home_yellow_cards_so_far",
        "away_yellow_cards_so_far",
        "home_red_cards_so_far",
        "away_red_cards_so_far",
    }.issubset(out.columns):
        out["home_card_pressure"] = (
            out["home_yellow_cards_so_far"] + 2 * out["home_red_cards_so_far"]
        ).astype("int16")
        out["away_card_pressure"] = (
            out["away_yellow_cards_so_far"] + 2 * out["away_red_cards_so_far"]
        ).astype("int16")
        out["card_pressure_diff"] = (out["home_card_pressure"] - out["away_card_pressure"]).astype(
            "int16"
        )
        created.extend(["home_card_pressure", "away_card_pressure", "card_pressure_diff"])

    out["minute_norm"] = (pd.to_numeric(out["time"], errors="coerce") / 45).astype("float32")
    out["early_first_half"] = out["time"].le(15).astype("int8")
    out["mid_first_half"] = out["time"].between(16, 30).astype("int8")
    out["late_first_half"] = out["time"].between(31, 45).astype("int8")
    created.extend(["minute_norm", "early_first_half", "mid_first_half", "late_first_half"])

    # CSV cannot preserve datetime dtype, but keeping ISO date strings makes reload deterministic.
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    reports = build_feature_engineering_reports(
        input_shape=input_shape,
        input_matches=input_matches,
        output=out,
        created=created,
        skipped=skipped,
    )
    return out, reports


def feature_target_correlations(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    blocked = set(LEAKAGE_COLUMNS + ["time", "season"])
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in blocked and not col.startswith("odd_")
    ]
    rows = []
    for target in [col for col in TARGET_COLUMNS if col in df.columns]:
        target_s = pd.to_numeric(df[target], errors="coerce")
        for feature in numeric_cols:
            feature_s = pd.to_numeric(df[feature], errors="coerce")
            if feature_s.nunique(dropna=True) <= 1 or target_s.nunique(dropna=True) <= 1:
                continue
            corr = feature_s.corr(target_s)
            if pd.isna(corr):
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df
    return (
        corr_df.sort_values(["target", "abs_correlation"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    match_level = df.drop_duplicates("id_odsp") if "id_odsp" in df.columns else df
    for target in [col for col in TARGET_COLUMNS if col in df.columns]:
        minute_counts = df[target].value_counts(dropna=False).sort_index()
        match_counts = match_level[target].value_counts(dropna=False).sort_index()
        for value, minute_rows in minute_counts.items():
            rows.append(
                {
                    "target": target,
                    "value": int(value) if pd.notna(value) else value,
                    "minute_rows": int(minute_rows),
                    "matches": int(match_counts.get(value, 0)),
                }
            )
    return pd.DataFrame(rows)


def validate_feature_engineering(
    input_shape: tuple[int, int],
    input_matches: int,
    output: pd.DataFrame,
    created: list[str],
) -> pd.DataFrame:
    created_existing = [col for col in created if col in output.columns]
    binary_flags = [
        col
        for col in created_existing
        if col.endswith("_minute")
        or col
        in {
            "home_dominating_last_5min",
            "away_dominating_last_5min",
            "home_strong_pressure_last_5min",
            "away_strong_pressure_last_5min",
            "home_leading",
            "away_leading",
            "draw_state",
            "early_first_half",
            "mid_first_half",
            "late_first_half",
        }
    ]
    rolling_checks = []
    for col in ["home_attempt", "away_attempt", "home_pressure_minute"]:
        roll_col = f"{col}_last_5min"
        if col in output.columns and roll_col in output.columns:
            expected = _rolling_sum_by_match(output, col, 5)
            rolling_checks.append(bool(expected.equals(output[roll_col].astype(expected.dtype))))
    checks = [
        {
            "check": "new_features_no_nan",
            "passed": bool(output[created_existing].isna().sum().sum() == 0),
            "details": int(output[created_existing].isna().sum().sum()),
        },
        {
            "check": "rolling_features_are_causal",
            "passed": all(rolling_checks) if rolling_checks else True,
            "details": f"checked {len(rolling_checks)} rolling columns against causal recompute",
        },
        {
            "check": "binary_flags_0_1_only",
            "passed": all(
                set(output[col].dropna().unique()).issubset({0, 1}) for col in binary_flags
            ),
            "details": f"checked {len(binary_flags)} binary flags",
        },
        {
            "check": "no_infinite_values_in_new_features",
            "passed": bool(
                np.isfinite(output[created_existing].select_dtypes(include=[np.number])).all().all()
            ),
            "details": "",
        },
        {
            "check": "row_count_unchanged",
            "passed": output.shape[0] == input_shape[0],
            "details": f"{input_shape[0]} -> {output.shape[0]}",
        },
        {
            "check": "match_count_unchanged",
            "passed": output["id_odsp"].nunique(dropna=True) == input_matches,
            "details": f"{input_matches} -> {output['id_odsp'].nunique(dropna=True)}",
        },
    ]
    return pd.DataFrame(checks)


def build_feature_engineering_reports(
    input_shape: tuple[int, int],
    input_matches: int,
    output: pd.DataFrame,
    created: list[str],
    skipped: list[dict[str, str]],
) -> dict[str, pd.DataFrame]:
    target_dist = target_distribution(output)
    corr = feature_target_correlations(output, top_n=50)
    validation = validate_feature_engineering(input_shape, input_matches, output, created)
    summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_shape[0]},
            {"metric": "input_columns", "value": input_shape[1]},
            {"metric": "output_rows", "value": output.shape[0]},
            {"metric": "output_columns", "value": output.shape[1]},
            {"metric": "matches", "value": output["id_odsp"].nunique(dropna=True)},
            {"metric": "created_features", "value": len(set(created))},
            {"metric": "skipped_missing_features", "value": len(skipped)},
            {"metric": "null_values_total", "value": int(output.isna().sum().sum())},
        ]
    )
    return {
        "summary": summary,
        "created_features": pd.DataFrame({"feature": sorted(set(created))}),
        "skipped_features": pd.DataFrame(skipped),
        "validation": validation,
        "target_distribution": target_dist,
        "feature_target_correlations": corr,
        "sample_rows": output.head(20),
    }


def save_feature_engineering_reports(
    reports: dict[str, pd.DataFrame],
    report_dir: Path,
    audit_dir: Path | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = "football_merged_feature_engineering"
    for name, table in reports.items():
        table.to_csv(report_dir / f"{prefix}_{name}.csv", index=False)
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        for name, table in reports.items():
            table.to_csv(audit_dir / f"{prefix}_{name}.csv", index=False)


def _team_history_records(match_df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "id_odsp": match_df["id_odsp"],
            "date": match_df["date"],
            "team": match_df["ht"],
            "is_home": 1,
            "goals_scored": match_df["fthg"],
            "goals_conceded": match_df["ftag"],
            "second_half_goals": match_df["second_half_home_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "id_odsp": match_df["id_odsp"],
            "date": match_df["date"],
            "team": match_df["at"],
            "is_home": 0,
            "goals_scored": match_df["ftag"],
            "goals_conceded": match_df["fthg"],
            "second_half_goals": match_df["second_half_away_goals"],
        }
    )
    records = pd.concat([home, away], ignore_index=True)
    records["points"] = np.select(
        [
            records["goals_scored"].gt(records["goals_conceded"]),
            records["goals_scored"].eq(records["goals_conceded"]),
        ],
        [3, 1],
        default=0,
    )
    records["clean_sheet"] = records["goals_conceded"].eq(0).astype("int8")
    records["scored"] = records["goals_scored"].gt(0).astype("int8")
    return records.sort_values(["team", "date", "id_odsp"], kind="mergesort").reset_index(drop=True)


def _historical_stats_for_team(team_records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for current_date, same_day in team_records.groupby("date", sort=True):
        history = team_records[team_records["date"].lt(current_date)]
        for _, row in same_day.iterrows():
            item = {
                "id_odsp": row["id_odsp"],
                "team": row["team"],
                "date": current_date,
                "history_matches_available": int(len(history)),
                "max_history_date": history["date"].max() if not history.empty else pd.NaT,
            }
            for window in [5, 10]:
                tail = history.tail(window)
                suffix = f"last_{window}"
                if tail.empty:
                    item[f"avg_goals_scored_{suffix}"] = 0.0
                    item[f"avg_goals_conceded_{suffix}"] = 0.0
                    item[f"points_{suffix}"] = 0.0
                    item[f"clean_sheet_rate_{suffix}"] = 0.0
                    item[f"scored_rate_{suffix}"] = 0.0
                    item[f"avg_second_half_goals_{suffix}"] = 0.0
                else:
                    item[f"avg_goals_scored_{suffix}"] = float(tail["goals_scored"].mean())
                    item[f"avg_goals_conceded_{suffix}"] = float(tail["goals_conceded"].mean())
                    item[f"points_{suffix}"] = float(tail["points"].sum())
                    item[f"clean_sheet_rate_{suffix}"] = float(tail["clean_sheet"].mean())
                    item[f"scored_rate_{suffix}"] = float(tail["scored"].mean())
                    item[f"avg_second_half_goals_{suffix}"] = float(
                        tail["second_half_goals"].mean()
                    )
            rows.append(item)
    return pd.DataFrame(rows)


def _build_historical_team_strength(match_df: pd.DataFrame) -> pd.DataFrame:
    records = _team_history_records(match_df)
    history = pd.concat(
        [_historical_stats_for_team(team_df) for _, team_df in records.groupby("team", sort=False)],
        ignore_index=True,
    )

    home_history = history.rename(
        columns={
            "team": "ht",
            "history_matches_available": "home_history_matches_available",
            "max_history_date": "home_max_history_date",
            "avg_goals_scored_last_5": "home_team_avg_goals_scored_last_5",
            "avg_goals_scored_last_10": "home_team_avg_goals_scored_last_10",
            "avg_goals_conceded_last_5": "home_team_avg_goals_conceded_last_5",
            "avg_goals_conceded_last_10": "home_team_avg_goals_conceded_last_10",
            "points_last_5": "home_team_points_last_5",
            "points_last_10": "home_team_points_last_10",
            "clean_sheet_rate_last_5": "home_team_clean_sheet_rate_last_5",
            "clean_sheet_rate_last_10": "home_team_clean_sheet_rate_last_10",
            "scored_rate_last_5": "home_team_scored_rate_last_5",
            "scored_rate_last_10": "home_team_scored_rate_last_10",
            "avg_second_half_goals_last_5": "home_team_avg_second_half_goals_last_5",
            "avg_second_half_goals_last_10": "home_team_avg_second_half_goals_last_10",
        }
    )
    away_history = history.rename(
        columns={
            "team": "at",
            "history_matches_available": "away_history_matches_available",
            "max_history_date": "away_max_history_date",
            "avg_goals_scored_last_5": "away_team_avg_goals_scored_last_5",
            "avg_goals_scored_last_10": "away_team_avg_goals_scored_last_10",
            "avg_goals_conceded_last_5": "away_team_avg_goals_conceded_last_5",
            "avg_goals_conceded_last_10": "away_team_avg_goals_conceded_last_10",
            "points_last_5": "away_team_points_last_5",
            "points_last_10": "away_team_points_last_10",
            "clean_sheet_rate_last_5": "away_team_clean_sheet_rate_last_5",
            "clean_sheet_rate_last_10": "away_team_clean_sheet_rate_last_10",
            "scored_rate_last_5": "away_team_scored_rate_last_5",
            "scored_rate_last_10": "away_team_scored_rate_last_10",
            "avg_second_half_goals_last_5": "away_team_avg_second_half_goals_last_5",
            "avg_second_half_goals_last_10": "away_team_avg_second_half_goals_last_10",
        }
    )

    keep_home = ["id_odsp", "ht"] + [c for c in home_history.columns if c.startswith("home_")]
    keep_away = ["id_odsp", "at"] + [c for c in away_history.columns if c.startswith("away_")]
    features = match_df[["id_odsp", "date", "ht", "at"]].copy()
    features = features.merge(home_history[keep_home], on=["id_odsp", "ht"], how="left")
    features = features.merge(away_history[keep_away], on=["id_odsp", "at"], how="left")

    for window in [5, 10]:
        features[f"team_attack_strength_diff_last_{window}"] = (
            features[f"home_team_avg_goals_scored_last_{window}"]
            - features[f"away_team_avg_goals_scored_last_{window}"]
        )
        features[f"team_defense_strength_diff_last_{window}"] = (
            features[f"home_team_avg_goals_conceded_last_{window}"]
            - features[f"away_team_avg_goals_conceded_last_{window}"]
        )
        features[f"team_form_diff_last_{window}"] = (
            features[f"home_team_points_last_{window}"]
            - features[f"away_team_points_last_{window}"]
        )
        features[f"team_second_half_scoring_diff_last_{window}"] = (
            features[f"home_team_avg_second_half_goals_last_{window}"]
            - features[f"away_team_avg_second_half_goals_last_{window}"]
        )
    return features


def _first_half_match_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "home_attempt",
        "away_attempt",
        "home_pressure_minute",
        "away_pressure_minute",
        "home_goal",
        "away_goal",
    ]
    existing = [col for col in required_cols if col in df.columns]
    grouped = df.groupby("id_odsp", as_index=False)[existing].sum()
    rename_map = {
        "home_attempt": "home_first_half_attempts",
        "away_attempt": "away_first_half_attempts",
        "home_pressure_minute": "home_first_half_pressure",
        "away_pressure_minute": "away_first_half_pressure",
        "home_goal": "home_first_half_score",
        "away_goal": "away_first_half_score",
    }
    grouped = grouped.rename(columns=rename_map)
    if {"home_first_half_score", "away_first_half_score"}.issubset(grouped.columns):
        grouped["first_half_score_diff"] = (
            grouped["home_first_half_score"] - grouped["away_first_half_score"]
        )
    if {"home_first_half_attempts", "away_first_half_attempts"}.issubset(grouped.columns):
        grouped["first_half_attempt_diff"] = (
            grouped["home_first_half_attempts"] - grouped["away_first_half_attempts"]
        )
    if {"home_first_half_pressure", "away_first_half_pressure"}.issubset(grouped.columns):
        grouped["first_half_pressure_diff"] = (
            grouped["home_first_half_pressure"] - grouped["away_first_half_pressure"]
        )
    return grouped


def add_historical_team_strength_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    input_shape = df.shape
    input_matches = df["id_odsp"].nunique(dropna=True)
    out = df.copy()
    required = [
        "id_odsp",
        "date",
        "time",
        "ht",
        "at",
        "fthg",
        "ftag",
        "second_half_home_goals",
        "second_half_away_goals",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Required historical feature columns are missing: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["date", "id_odsp", "time"], kind="mergesort").reset_index(drop=True)
    match_df = out.drop_duplicates("id_odsp").copy()
    for col in ["fthg", "ftag", "second_half_home_goals", "second_half_away_goals"]:
        match_df[col] = pd.to_numeric(match_df[col], errors="coerce").fillna(0)

    historical = _build_historical_team_strength(match_df)
    first_half = _first_half_match_aggregates(out)
    feature_df = historical.merge(first_half, on="id_odsp", how="left")
    protected_metadata = {"id_odsp", "date", "ht", "at"}
    drop_before_merge = [
        col for col in feature_df.columns if col in out.columns and col not in protected_metadata
    ]
    out = out.drop(columns=drop_before_merge, errors="ignore").merge(
        feature_df.drop(columns=["date", "ht", "at"], errors="ignore"),
        on="id_odsp",
        how="left",
    )
    out = out.sort_values(["date", "id_odsp", "time"], kind="mergesort").reset_index(drop=True)

    created = [col for col in feature_df.columns if col not in {"id_odsp", "date", "ht", "at"}]
    history_date_cols = [col for col in created if col.endswith("_max_history_date")]
    for col in created:
        if col in out.columns and col not in history_date_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("float32")
    for col in history_date_cols:
        if col in out.columns:
            out[col] = (
                pd.to_datetime(out[col], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("no_history")
            )
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    reports = build_historical_team_strength_reports(
        input_shape=input_shape,
        input_matches=input_matches,
        output=out,
        created=created,
    )
    return out, reports


def build_historical_team_strength_reports(
    input_shape: tuple[int, int],
    input_matches: int,
    output: pd.DataFrame,
    created: list[str],
) -> dict[str, pd.DataFrame]:
    output_dates = pd.to_datetime(output["date"], errors="coerce")
    home_history_raw = output.get("home_max_history_date", pd.Series(index=output.index, dtype=str))
    away_history_raw = output.get("away_max_history_date", pd.Series(index=output.index, dtype=str))
    home_history_dates = pd.to_datetime(
        home_history_raw.where(home_history_raw.ne("no_history")),
        errors="coerce",
        format="%Y-%m-%d",
    )
    away_history_dates = pd.to_datetime(
        away_history_raw.where(away_history_raw.ne("no_history")),
        errors="coerce",
        format="%Y-%m-%d",
    )
    home_leakage = home_history_dates.notna() & home_history_dates.ge(output_dates)
    away_leakage = away_history_dates.notna() & away_history_dates.ge(output_dates)
    validation = pd.DataFrame(
        [
            {
                "check": "historical_home_dates_strictly_past",
                "passed": not bool(home_leakage.any()),
                "details": int(home_leakage.sum()),
            },
            {
                "check": "historical_away_dates_strictly_past",
                "passed": not bool(away_leakage.any()),
                "details": int(away_leakage.sum()),
            },
            {
                "check": "row_count_unchanged",
                "passed": output.shape[0] == input_shape[0],
                "details": f"{input_shape[0]} -> {output.shape[0]}",
            },
            {
                "check": "match_count_unchanged",
                "passed": output["id_odsp"].nunique(dropna=True) == input_matches,
                "details": f"{input_matches} -> {output['id_odsp'].nunique(dropna=True)}",
            },
            {
                "check": "created_features_no_nan",
                "passed": bool(output[created].isna().sum().sum() == 0),
                "details": int(output[created].isna().sum().sum()),
            },
        ]
    )
    if not validation["passed"].all():
        raise ValueError(
            f"Historical leakage/validation failure:\n{validation.to_string(index=False)}"
        )

    summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_shape[0]},
            {"metric": "input_columns", "value": input_shape[1]},
            {"metric": "output_rows", "value": output.shape[0]},
            {"metric": "output_columns", "value": output.shape[1]},
            {"metric": "matches", "value": output["id_odsp"].nunique(dropna=True)},
            {"metric": "created_historical_features", "value": len(created)},
            {"metric": "null_values_total", "value": int(output.isna().sum().sum())},
            {
                "metric": "matches_without_home_history",
                "value": int(
                    output.drop_duplicates("id_odsp")["home_history_matches_available"].eq(0).sum()
                ),
            },
            {
                "metric": "matches_without_away_history",
                "value": int(
                    output.drop_duplicates("id_odsp")["away_history_matches_available"].eq(0).sum()
                ),
            },
        ]
    )
    return {
        "historical_summary": summary,
        "historical_created_features": pd.DataFrame({"feature": created}),
        "historical_validation": validation,
        "target_distribution": target_distribution(output),
        "feature_target_correlations": feature_target_correlations(output, top_n=50),
        "sample_rows": output.head(20),
    }
