from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import FOOTBALL_HALF_CUTOFF, RANDOM_STATE


def build_proxy_xg(football_events: pd.DataFrame) -> pd.DataFrame:
    out = football_events.copy()
    shot_cols = [
        "location",
        "bodypart",
        "assist_method",
        "situation",
        "fast_break",
        "shot_place",
        "shot_outcome",
    ]
    shot_df = out.loc[out["is_attempt"] == 1].copy()
    if shot_df.empty or shot_df["is_goal_fixed"].nunique() < 2:
        out["proxy_xg"] = 0.0
        return out

    shot_X = pd.get_dummies(shot_df[shot_cols].fillna(-1).astype(str))
    shot_y = shot_df["is_goal_fixed"].astype(int)
    shot_xg_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    shot_xg_model.fit(shot_X, shot_y)
    shot_df["proxy_xg"] = shot_xg_model.predict_proba(shot_X)[:, 1]

    out["proxy_xg"] = 0.0
    out.loc[shot_df.index, "proxy_xg"] = shot_df["proxy_xg"]
    return out


def preprocess_football_events(df_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    football_events = df_events.copy().rename(columns={"id_odsp": "match_id"})
    football_events["time"] = football_events["time"].clip(1, 90).astype(int)
    football_events["text"] = football_events["text"].fillna("")
    football_events["event_type2"] = football_events["event_type2"].fillna("NA")
    football_events["sort_order"] = football_events["sort_order"].fillna(
        football_events.groupby("match_id").cumcount() + 1
    )

    goal_text = football_events["text"].str.startswith("Goal!") | football_events[
        "text"
    ].str.startswith("Own Goal")
    football_events["is_goal_fixed"] = (
        (football_events["event_type"] == 1) & goal_text
    ).astype(int)
    football_events["is_attempt"] = (football_events["event_type"] == 1).astype(int)
    football_events["is_corner"] = (football_events["event_type"] == 2).astype(int)
    football_events["is_foul"] = (football_events["event_type"] == 3).astype(int)
    football_events["is_yellow"] = (football_events["event_type"] == 4).astype(int)
    football_events["is_red"] = (football_events["event_type"] == 6).astype(int)
    football_events["is_offside"] = (football_events["event_type"] == 9).astype(int)
    football_events["is_key_pass"] = (
        football_events["event_type2"].astype(str).eq("12.0").astype(int)
    )
    football_events["is_fast_break"] = football_events["fast_break"].fillna(0).astype(int)

    for prefix, side_value in [("home", 1), ("away", 2)]:
        football_events[f"{prefix}_event"] = (football_events["side"] == side_value).astype(int)
        football_events[f"{prefix}_goal"] = (
            (football_events["side"] == side_value) & (football_events["is_goal_fixed"] == 1)
        ).astype(int)
        football_events[f"{prefix}_attempt"] = (
            (football_events["side"] == side_value) & (football_events["is_attempt"] == 1)
        ).astype(int)
        football_events[f"{prefix}_corner"] = (
            (football_events["side"] == side_value) & (football_events["is_corner"] == 1)
        ).astype(int)
        football_events[f"{prefix}_foul"] = (
            (football_events["side"] == side_value) & (football_events["is_foul"] == 1)
        ).astype(int)
        football_events[f"{prefix}_key_pass"] = (
            (football_events["side"] == side_value) & (football_events["is_key_pass"] == 1)
        ).astype(int)
        football_events[f"{prefix}_offside"] = (
            (football_events["side"] == side_value) & (football_events["is_offside"] == 1)
        ).astype(int)

    home_team_map = (
        football_events.loc[football_events["side"] == 1, ["match_id", "event_team"]]
        .dropna()
        .drop_duplicates("match_id")
        .rename(columns={"event_team": "home_team"})
    )
    away_team_map = (
        football_events.loc[football_events["side"] == 2, ["match_id", "event_team"]]
        .dropna()
        .drop_duplicates("match_id")
        .rename(columns={"event_team": "away_team"})
    )
    team_map = home_team_map.merge(away_team_map, on="match_id", how="outer")

    football_events = build_proxy_xg(football_events)
    football_events["home_xg"] = np.where(
        football_events["side"] == 1, football_events["proxy_xg"], 0.0
    )
    football_events["away_xg"] = np.where(
        football_events["side"] == 2, football_events["proxy_xg"], 0.0
    )

    football_minute = make_minute_level(football_events, team_map)
    football_model_df = football_minute.loc[
        football_minute["time"] <= FOOTBALL_HALF_CUTOFF
    ].copy()
    return football_minute, football_model_df


def make_minute_level(football_events: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        "home_goal",
        "away_goal",
        "home_event",
        "away_event",
        "home_attempt",
        "away_attempt",
        "home_corner",
        "away_corner",
        "home_foul",
        "away_foul",
        "home_key_pass",
        "away_key_pass",
        "home_offside",
        "away_offside",
        "is_yellow",
        "is_red",
        "is_fast_break",
    ]
    football_minute = football_events.groupby(["match_id", "time"], as_index=False).agg(
        **{c: (c, "sum") for c in agg_cols},
        home_xg=("home_xg", "sum"),
        away_xg=("away_xg", "sum"),
        sort_order_max=("sort_order", "max"),
        sort_order_min=("sort_order", "min"),
        event_rows=("id_event", "count"),
    )

    full_idx = pd.MultiIndex.from_product(
        [football_minute["match_id"].unique(), range(1, 91)], names=["match_id", "time"]
    )
    football_minute = (
        football_minute.set_index(["match_id", "time"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
        .sort_values(["match_id", "time"])
    )
    football_minute = football_minute.merge(team_map, on="match_id", how="left")
    football_minute["home_team"] = football_minute["home_team"].fillna("UnknownHome")
    football_minute["away_team"] = football_minute["away_team"].fillna("UnknownAway")

    football_minute["home_score"] = football_minute.groupby("match_id")["home_goal"].cumsum()
    football_minute["away_score"] = football_minute.groupby("match_id")["away_goal"].cumsum()
    football_minute["score_diff"] = football_minute["home_score"] - football_minute["away_score"]
    football_minute["minutes_remaining"] = 90 - football_minute["time"]
    football_minute["is_draw"] = (football_minute["score_diff"] == 0).astype(int)
    football_minute["home_leading"] = (football_minute["score_diff"] > 0).astype(int)
    football_minute["away_leading"] = (football_minute["score_diff"] < 0).astype(int)
    football_minute["abs_score_diff"] = football_minute["score_diff"].abs()
    football_minute["event_activity"] = (
        football_minute["home_event"] + football_minute["away_event"]
    )
    football_minute["xg_diff"] = football_minute["home_xg"] - football_minute["away_xg"]
    football_minute["time_sin"] = np.sin(2 * np.pi * football_minute["time"] / 90)
    football_minute["time_cos"] = np.cos(2 * np.pi * football_minute["time"] / 90)
    football_minute["events_per_minute"] = football_minute["event_rows"]
    football_minute["sort_order_velocity"] = (
        football_minute["sort_order_max"] - football_minute["sort_order_min"]
    ).clip(lower=0)

    add_rolling_features(football_minute)
    add_targets(football_minute)
    return football_minute


def add_rolling_features(football_minute: pd.DataFrame) -> None:
    rolling_cols = [
        "home_event",
        "away_event",
        "home_attempt",
        "away_attempt",
        "home_corner",
        "away_corner",
        "home_key_pass",
        "away_key_pass",
        "home_offside",
        "away_offside",
        "home_xg",
        "away_xg",
        "is_yellow",
        "is_red",
        "is_fast_break",
        "event_activity",
        "events_per_minute",
        "sort_order_velocity",
    ]
    for col in rolling_cols:
        for window in [5, 10]:
            football_minute[f"{col}_last_{window}min"] = football_minute.groupby("match_id")[
                col
            ].transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())

    momentum_cols = [
        "event_activity",
        "home_xg",
        "away_xg",
        "home_key_pass",
        "away_key_pass",
        "home_corner",
        "away_corner",
        "events_per_minute",
    ]
    for col in momentum_cols:
        last_5 = football_minute.groupby("match_id")[col].transform(
            lambda s: s.rolling(5, min_periods=1).sum()
        )
        previous_5 = (
            football_minute.groupby("match_id")[col]
            .transform(lambda s: s.shift(5).rolling(5, min_periods=1).sum())
            .fillna(0)
        )
        football_minute[f"{col}_momentum_5min"] = last_5 - previous_5

    football_minute["home_event_share_last_5min"] = football_minute[
        "home_event_last_5min"
    ] / (
        football_minute["home_event_last_5min"]
        + football_minute["away_event_last_5min"]
        + 1e-6
    )
    football_minute["away_event_share_last_5min"] = (
        1 - football_minute["home_event_share_last_5min"]
    )
    football_minute["xg_diff_last_10min"] = (
        football_minute["home_xg_last_10min"] - football_minute["away_xg_last_10min"]
    )
    football_minute["pressure_diff_last_5min"] = (
        1.5
        * (football_minute["home_attempt_last_5min"] - football_minute["away_attempt_last_5min"])
        + 1.2
        * (
            football_minute["home_key_pass_last_5min"]
            - football_minute["away_key_pass_last_5min"]
        )
        + 0.8
        * (football_minute["home_corner_last_5min"] - football_minute["away_corner_last_5min"])
        + 2.0 * (football_minute["home_xg_last_5min"] - football_minute["away_xg_last_5min"])
    )


def add_targets(football_minute: pd.DataFrame) -> None:
    second_half = (
        football_minute.loc[football_minute["time"] >= 46]
        .groupby("match_id", as_index=False)
        .agg(
            second_half_home_goals=("home_goal", "sum"),
            second_half_away_goals=("away_goal", "sum"),
        )
    )
    merged = football_minute.merge(second_half, on="match_id", how="left")
    football_minute["second_half_home_goals"] = merged["second_half_home_goals"].fillna(0)
    football_minute["second_half_away_goals"] = merged["second_half_away_goals"].fillna(0)
    football_minute["home_scores_next_half"] = (
        football_minute["second_half_home_goals"] > 0
    ).astype(int)
    football_minute["away_scores_next_half"] = (
        football_minute["second_half_away_goals"] > 0
    ).astype(int)


def build_team_strength(train_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    match_level = train_df.sort_values(["match_id", "time"]).groupby("match_id", as_index=False).tail(1)
    home = match_level[
        ["home_team", "home_score", "away_score", "home_xg", "away_xg"]
    ].rename(
        columns={
            "home_team": "team",
            "home_score": "goals_for",
            "away_score": "goals_against",
            "home_xg": "xg_for",
            "away_xg": "xg_against",
        }
    )
    away = match_level[
        ["away_team", "away_score", "home_score", "away_xg", "home_xg"]
    ].rename(
        columns={
            "away_team": "team",
            "away_score": "goals_for",
            "home_score": "goals_against",
            "away_xg": "xg_for",
            "home_xg": "xg_against",
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    stats = long.groupby("team").agg(
        attack_strength=("goals_for", "mean"),
        defense_strength=("goals_against", "mean"),
    )
    return stats, float(long["goals_for"].mean()), float(long["goals_against"].mean())


def add_team_strength(
    df: pd.DataFrame, stats: pd.DataFrame, global_attack: float, global_defense: float
) -> pd.DataFrame:
    out = df.copy()
    out = out.join(stats.add_prefix("home_"), on="home_team")
    out = out.join(stats.add_prefix("away_"), on="away_team")
    out["home_attack_strength"] = out["home_attack_strength"].fillna(global_attack)
    out["away_attack_strength"] = out["away_attack_strength"].fillna(global_attack)
    out["home_defense_strength"] = out["home_defense_strength"].fillna(global_defense)
    out["away_defense_strength"] = out["away_defense_strength"].fillna(global_defense)
    out["team_attack_diff"] = out["home_attack_strength"] - out["away_attack_strength"]
    out["team_defense_diff"] = out["away_defense_strength"] - out["home_defense_strength"]
    return out


def football_feature_importance(
    football_model_df: pd.DataFrame, feature_cols: list[str], target: str
) -> pd.DataFrame:
    rf_sample = football_model_df.sample(
        min(80000, len(football_model_df)), random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(rf_sample[feature_cols], rf_sample[target])
    return pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
