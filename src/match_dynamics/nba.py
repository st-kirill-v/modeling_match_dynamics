from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import NBA_MAX_GAMES, NBA_MOMENT_STRIDE, RANDOM_STATE


def normalize_game_id(value) -> str:
    return str(int(value)).zfill(10)


def game_id_to_int(value: str) -> int:
    return int(value)


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


def parse_nba_event_level_movement(
    json_files: list[Path],
    max_games: int = NBA_MAX_GAMES,
    moment_stride: int = NBA_MOMENT_STRIDE,
) -> pd.DataFrame:
    records = []
    for fpath in json_files[:max_games]:
        with fpath.open(encoding="utf-8") as f:
            game = json.load(f)
        game_id = normalize_game_id(game.get("gameid", fpath.stem))
        for event in game.get("events", []):
            event_id = event.get("eventId")
            moments = event.get("moments", [])
            sampled = moments[::moment_stride]
            moment_records = []
            for moment in sampled:
                features = _moment_features(moment)
                if features is not None:
                    moment_records.append(features)
            if not moment_records:
                continue
            mdf = pd.DataFrame(moment_records)
            records.append(
                {
                    "game_id": game_id,
                    "game_id_int": game_id_to_int(game_id),
                    "event_id": int(event_id),
                    "period": int(mdf["period"].max()),
                    "movement_moments_total": len(moments),
                    "movement_moments_sampled": len(mdf),
                    "game_clock_start": float(mdf["game_clock"].max()),
                    "game_clock_end": float(mdf["game_clock"].min()),
                    "shot_clock_start": float(mdf["shot_clock"].max()),
                    "shot_clock_end": float(mdf["shot_clock"].min()),
                    "avg_distance": float(mdf["avg_distance"].mean()),
                    "std_distance": float(mdf["std_distance"].mean()),
                    "spread_x": float(mdf["spread_x"].mean()),
                    "spread_y": float(mdf["spread_y"].mean()),
                    "ball_x": float(mdf["ball_x"].mean()),
                    "ball_y": float(mdf["ball_y"].mean()),
                    "ball_hoop_dist": float(mdf["ball_hoop_dist"].min()),
                    "min_player_hoop_dist": float(mdf["min_player_hoop_dist"].min()),
                    "players_near_hoop": float(mdf["players_near_hoop"].max()),
                    "low_shot_clock": int(mdf["low_shot_clock"].max()),
                }
            )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["intensity"] = (
        out["avg_distance"].rank(pct=True)
        + out["std_distance"].rank(pct=True)
        + out["spread_x"].rank(pct=True)
        + out["spread_y"].rank(pct=True)
    ) / 4
    return out


def _moment_features(moment: list) -> dict | None:
    if len(moment) < 6:
        return None
    period, game_clock, shot_clock, players = moment[0], moment[2], moment[3], moment[5]
    rows, ball_x, ball_y = [], np.nan, np.nan
    for player in players:
        if len(player) < 4:
            continue
        team_id, player_id, x, y = player[:4]
        if team_id == -1 or player_id == -1:
            ball_x, ball_y = float(x), float(y)
        else:
            rows.append((float(x), float(y)))
    if not rows:
        return None
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
    return {
        "period": period,
        "game_clock": float(game_clock),
        "shot_clock": np.nan if shot_clock is None else float(shot_clock),
        "avg_distance": distances.mean(),
        "std_distance": distances.std(),
        "spread_x": xs.max() - xs.min(),
        "spread_y": ys.max() - ys.min(),
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_hoop_dist": ball_hoop_dist,
        "min_player_hoop_dist": min(left_hoop.min(), right_hoop.min()),
        "players_near_hoop": (left_hoop < 10).sum() + (right_hoop < 10).sum(),
        "low_shot_clock": 0 if shot_clock is None else int(float(shot_clock) <= 7),
    }


def load_nba_events(events_dir: Path, game_ids: list[str]) -> pd.DataFrame:
    frames = []
    for game_id in game_ids:
        path = events_dir / f"{game_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["game_id"] = df["GAME_ID"].map(normalize_game_id)
        df["game_id_int"] = df["GAME_ID"].astype(int)
        df["event_id"] = df["EVENTNUM"].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_nba_shots(shots_path: Path, game_ids: list[str]) -> pd.DataFrame:
    game_ints = {game_id_to_int(gid) for gid in game_ids}
    shots = pd.read_csv(shots_path)
    shots = shots[shots["GAME_ID"].astype(int).isin(game_ints)].copy()
    shots["game_id"] = shots["GAME_ID"].map(normalize_game_id)
    shots["game_id_int"] = shots["GAME_ID"].astype(int)
    shots["event_id"] = shots["GAME_EVENT_ID"].astype(int)
    return shots


def build_nba_matched_dataset(
    movement_df: pd.DataFrame,
    events_df: pd.DataFrame,
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    if movement_df.empty:
        return movement_df

    event_cols = [
        "game_id",
        "game_id_int",
        "event_id",
        "EVENTMSGTYPE",
        "EVENTMSGACTIONTYPE",
        "PCTIMESTRING",
        "HOMEDESCRIPTION",
        "VISITORDESCRIPTION",
        "SCORE",
        "SCOREMARGIN",
        "PLAYER1_ID",
        "PLAYER1_NAME",
        "PLAYER1_TEAM_ID",
        "PLAYER1_TEAM_ABBREVIATION",
        "PLAYER2_ID",
        "PLAYER2_NAME",
        "PLAYER2_TEAM_ID",
        "PLAYER2_TEAM_ABBREVIATION",
    ]
    events_small = events_df[[c for c in event_cols if c in events_df.columns]].copy()
    matched = movement_df.merge(events_small, on=["game_id", "game_id_int", "event_id"], how="left")

    shot_cols = [
        "game_id",
        "game_id_int",
        "event_id",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_NAME",
        "EVENT_TYPE",
        "ACTION_TYPE",
        "SHOT_TYPE",
        "SHOT_ZONE_BASIC",
        "SHOT_ZONE_AREA",
        "SHOT_ZONE_RANGE",
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "SHOT_ATTEMPTED_FLAG",
        "SHOT_MADE_FLAG",
    ]
    shots_small = shots_df[[c for c in shot_cols if c in shots_df.columns]].copy()
    shots_small = shots_small.rename(
        columns={
            "PLAYER_ID": "SHOT_PLAYER_ID",
            "PLAYER_NAME": "SHOT_PLAYER_NAME",
            "TEAM_ID": "SHOT_TEAM_ID",
            "TEAM_NAME": "SHOT_TEAM_NAME",
            "EVENT_TYPE": "SHOT_EVENT_TYPE",
        }
    )
    matched = matched.merge(shots_small, on=["game_id", "game_id_int", "event_id"], how="left")

    event_type = matched["EVENTMSGTYPE"].fillna(-1).astype(int)
    matched["shot_attempt"] = event_type.isin([1, 2]).astype(int)
    matched["shot_made"] = (event_type == 1).astype(int)
    matched["shot_missed"] = (event_type == 2).astype(int)
    matched["free_throw"] = (event_type == 3).astype(int)
    matched["rebound"] = (event_type == 4).astype(int)
    matched["turnover"] = (event_type == 5).astype(int)
    matched["foul"] = (event_type == 6).astype(int)
    matched["has_shot_chart_row"] = matched["SHOT_MADE_FLAG"].notna().astype(int)
    matched["shot_made_from_shots"] = matched["SHOT_MADE_FLAG"].fillna(0).astype(int)
    matched["points_from_shot"] = np.where(
        matched["shot_made_from_shots"].eq(1),
        np.where(matched["SHOT_TYPE"].astype(str).str.startswith("3PT"), 3, 2),
        0,
    )
    matched["scoring_event"] = ((matched["shot_made"] == 1) | (matched["free_throw"] == 1)).astype(
        int
    )
    return matched


def parse_score(score) -> tuple[float, float]:
    if pd.isna(score):
        return np.nan, np.nan
    parts = str(score).replace(" ", "").split("-")
    if len(parts) != 2:
        return np.nan, np.nan
    # NBA API play-by-play SCORE is visitor-home, e.g. ORL at WAS starts 2-0 after visitor scores.
    return float(parts[0]), float(parts[1])


def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["SCORE"].apply(parse_score)
    out["visitor_score_from_score"] = [p[0] for p in parsed]
    out["home_score_from_score"] = [p[1] for p in parsed]
    out = out.sort_values(["game_id", "period", "event_id"])
    out["visitor_score_ffill"] = (
        out.groupby("game_id")["visitor_score_from_score"].ffill().fillna(0)
    )
    out["home_score_ffill"] = out.groupby("game_id")["home_score_from_score"].ffill().fillna(0)
    out["score_diff_home"] = out["home_score_ffill"] - out["visitor_score_ffill"]
    return out


def build_nba_final_score_checkpoint_dataset(
    matched_df: pd.DataFrame,
    checkpoint_seconds: float = 300.0,
) -> pd.DataFrame:
    df = add_score_columns(matched_df)
    rows = []
    movement_cols = [
        "game_clock_start",
        "shot_clock_start",
        "shot_clock_end",
        "avg_distance",
        "std_distance",
        "spread_x",
        "spread_y",
        "ball_hoop_dist",
        "min_player_hoop_dist",
        "players_near_hoop",
        "intensity",
    ]
    event_count_cols = [
        "shot_attempt",
        "shot_made",
        "shot_missed",
        "free_throw",
        "turnover",
        "foul",
    ]

    for game_id, group in df.groupby("game_id"):
        group = group.sort_values(["period", "event_id"]).copy()
        scored = group.dropna(subset=["SCORE"])
        if scored.empty:
            continue
        final_visitor = float(scored.iloc[-1]["visitor_score_from_score"])
        final_home = float(scored.iloc[-1]["home_score_from_score"])

        fourth = group[group["period"].eq(4)]
        checkpoint_candidates = fourth[fourth["game_clock_start"].le(checkpoint_seconds)]
        if checkpoint_candidates.empty:
            # TODO: Overtime games and games with missing 4Q tracking may need richer handling.
            continue
        checkpoint = checkpoint_candidates.iloc[0]
        history = group[
            (group["period"].lt(4))
            | ((group["period"].eq(4)) & (group["event_id"].le(checkpoint["event_id"])))
        ]
        last_state = history.iloc[-1]

        row = {
            "game_id": game_id,
            "checkpoint_event_id": int(checkpoint["event_id"]),
            "checkpoint_seconds_remaining": float(checkpoint["game_clock_start"]),
            "current_home_score": float(last_state["home_score_ffill"]),
            "current_visitor_score": float(last_state["visitor_score_ffill"]),
            "current_score_diff_home": float(last_state["score_diff_home"]),
            "current_total_score": float(
                last_state["home_score_ffill"] + last_state["visitor_score_ffill"]
            ),
            "final_home_score": final_home,
            "final_visitor_score": final_visitor,
            "final_score_diff_home": final_home - final_visitor,
            "final_total_score": final_home + final_visitor,
            "score_diff_change_after_checkpoint": (final_home - final_visitor)
            - float(last_state["score_diff_home"]),
            "home_win": int(final_home > final_visitor),
        }
        for col in event_count_cols:
            row[f"{col}_before_checkpoint"] = float(history[col].sum())
        for col in movement_cols:
            row[f"{col}_at_checkpoint"] = float(last_state[col])
            row[f"{col}_mean_before_checkpoint"] = float(history[col].mean())
        rows.append(row)

    return pd.DataFrame(rows)


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
            | (nba_possession_df["intensity"] >= nba_possession_df["intensity"].quantile(0.90))
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
