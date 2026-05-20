from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import NBA_MAX_GAMES, NBA_MOMENT_STRIDE, RANDOM_STATE


def parse_nba_tracking_features(
    json_files: list[Path],
    max_games: int = NBA_MAX_GAMES,
    moment_stride: int = NBA_MOMENT_STRIDE,
) -> pd.DataFrame:
    records = []
    for fpath in json_files[:max_games]:
        with fpath.open(encoding="utf-8") as f:
            game = json.load(f)
        game_id = str(game.get("gameid", fpath.stem))
        for event in game.get("events", []):
            event_id = event.get("eventId")
            moments = event.get("moments", [])[::moment_stride]
            for moment in moments:
                if len(moment) < 6:
                    continue
                period, game_clock, shot_clock, players = moment[0], moment[2], moment[3], moment[5]
                rows, ball_x, ball_y = [], np.nan, np.nan
                for player in players:
                    if len(player) < 4:
                        continue
                    team_id, player_id, x, y = player[:4]
                    if team_id == -1 or player_id == -1:
                        ball_x, ball_y = x, y
                    else:
                        rows.append((x, y))
                if not rows:
                    continue
                coords = np.array(rows, dtype=float)
                xs, ys = coords[:, 0], coords[:, 1]
                distances = np.sqrt((xs - 47.0) ** 2 + (ys - 25.0) ** 2)
                left_hoop = np.sqrt((xs - 5.25) ** 2 + (ys - 25.0) ** 2)
                right_hoop = np.sqrt((xs - 88.75) ** 2 + (ys - 25.0) ** 2)
                if not np.isnan(ball_x) and not np.isnan(ball_y):
                    ball_hoop_dist = min(
                        np.sqrt((ball_x - 5.25) ** 2 + (ball_y - 25.0) ** 2),
                        np.sqrt((ball_x - 88.75) ** 2 + (ball_y - 25.0) ** 2),
                    )
                else:
                    ball_hoop_dist = np.nan
                records.append(
                    {
                        "match_id": game_id,
                        "possession_id": f"{game_id}_{event_id}",
                        "event_id": event_id,
                        "period": period,
                        "game_clock": game_clock,
                        "shot_clock": np.nan if shot_clock is None else float(shot_clock),
                        "avg_distance": distances.mean(),
                        "std_distance": distances.std(),
                        "spread_x": xs.max() - xs.min(),
                        "spread_y": ys.max() - ys.min(),
                        "ball_x": ball_x,
                        "ball_y": ball_y,
                        "players_count": len(rows),
                        "ball_hoop_dist": ball_hoop_dist,
                        "min_player_hoop_dist": min(left_hoop.min(), right_hoop.min()),
                        "players_near_hoop": (left_hoop < 10).sum() + (right_hoop < 10).sum(),
                        "low_shot_clock": 0 if shot_clock is None else int(float(shot_clock) <= 7),
                    }
                )
    return pd.DataFrame(records)


def build_nba_possession_proxy(nba_tracking_raw: pd.DataFrame) -> pd.DataFrame:
    if nba_tracking_raw.empty:
        return pd.DataFrame()

    raw = nba_tracking_raw.copy()
    raw["frame_order"] = raw.groupby("possession_id").cumcount()
    nba_possession_df = raw.groupby(["match_id", "possession_id"], as_index=False).agg(
        period=("period", "max"),
        possession_duration_proxy=("frame_order", "max"),
        avg_distance=("avg_distance", "mean"),
        std_distance=("std_distance", "mean"),
        spread_x=("spread_x", "mean"),
        spread_y=("spread_y", "mean"),
        ball_x=("ball_x", "mean"),
        ball_y=("ball_y", "mean"),
        shot_clock_start=("shot_clock", "max"),
        shot_clock_end=("shot_clock", "min"),
        players_count=("players_count", "mean"),
        ball_hoop_dist=("ball_hoop_dist", "min"),
        min_player_hoop_dist=("min_player_hoop_dist", "min"),
        players_near_hoop=("players_near_hoop", "max"),
        low_shot_clock=("low_shot_clock", "max"),
    )
    nba_possession_df = nba_possession_df.fillna(nba_possession_df.median(numeric_only=True))
    nba_possession_df["intensity"] = (
        nba_possession_df["avg_distance"].rank(pct=True)
        + nba_possession_df["std_distance"].rank(pct=True)
        + nba_possession_df["spread_x"].rank(pct=True)
        + nba_possession_df["spread_y"].rank(pct=True)
    ) / 4
    nba_possession_df["possession_is_dangerous"] = (
        (nba_possession_df["ball_hoop_dist"] <= 8)
        & (nba_possession_df["min_player_hoop_dist"] <= 6)
        & (
            (nba_possession_df["low_shot_clock"] == 1)
            | (nba_possession_df["players_near_hoop"] >= 3)
            | (
                nba_possession_df["intensity"]
                >= nba_possession_df["intensity"].quantile(0.90)
            )
        )
    ).astype(int)
    return nba_possession_df


def preprocess_nba(json_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = parse_nba_tracking_features(json_files)
    possession = build_nba_possession_proxy(raw)
    return raw, possession


def nba_feature_importance(
    nba_possession_df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> pd.DataFrame:
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(nba_possession_df[feature_cols], nba_possession_df[target_col])
    return pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
