from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def ensure_uv_environment() -> None:
    if os.environ.get("MATCH_DYNAMICS_UV_BOOTSTRAPPED") == "1":
        return
    try:
        import pandas  # noqa: F401
        import py7zr  # noqa: F401
    except ModuleNotFoundError:
        uv_path = shutil.which("uv")
        if uv_path is None:
            print("Install uv first, then run: uv sync --python 3.13", file=sys.stderr)
            raise SystemExit(1)
        env = os.environ.copy()
        env["MATCH_DYNAMICS_UV_BOOTSTRAPPED"] = "1"
        cmd = [uv_path, "run", "python", str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT, env=env))


ensure_uv_environment()

import pandas as pd
import py7zr

from match_dynamics.nba import (
    build_nba_matched_dataset,
    load_nba_events,
    load_nba_shots,
    parse_nba_event_level_movement,
)


RAW_BASE = "https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data"
API_DATA_URL = "https://api.github.com/repos/sealneaward/nba-movement-data/contents/data"


def raw_url(relative_path: str) -> str:
    return f"{RAW_BASE}/{quote(relative_path, safe='/')}"


def download_file(relative_path: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path}")
        return
    print(f"[download] {relative_path}")
    with urllib.request.urlopen(raw_url(relative_path), timeout=180) as response:
        with output_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    print(f"           {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def list_archive_names(min_count: int) -> list[str]:
    with urllib.request.urlopen(API_DATA_URL, timeout=60) as response:
        payload = json.load(response)
    names = sorted(item["name"] for item in payload if item["name"].endswith(".7z"))
    if len(names) < min_count:
        raise RuntimeError(f"Only {len(names)} archives were listed, requested {min_count}.")
    return names


def extract_json_from_archive(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        names = [name for name in archive.getnames() if name.lower().endswith(".json")]
    if not names:
        raise FileNotFoundError(f"No JSON inside {archive_path}")
    json_name = names[0]
    output_path = extract_dir / json_name
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    print(f"[extract] {archive_path.name} -> {json_name}")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extract(path=extract_dir, targets=[json_name])
    return output_path


def game_id_from_json(json_path: Path) -> str:
    with json_path.open(encoding="utf-8") as f:
        game = json.load(f)
    return str(game.get("gameid", json_path.stem)).zfill(10)


def build_dataset(
    max_games: int, moment_stride: int, output_dir: Path, output_csv: Path
) -> pd.DataFrame:
    archives_dir = output_dir / "archives"
    events_dir = output_dir / "events"
    shots_dir = output_dir / "shots"
    extracted_dir = output_dir / "extracted_json"

    archive_names = list_archive_names(max_games)
    json_files, game_ids = [], []

    print(f"[1/6] Downloading movement archives until {max_games} valid games are ready...")
    for name in archive_names:
        if len(json_files) >= max_games:
            break
        archive_path = archives_dir / name
        try:
            download_file(name, archive_path)
            json_path = extract_json_from_archive(archive_path, extracted_dir)
            game_id = game_id_from_json(json_path)
        except Exception as exc:
            print(f"[skip-bad] {name}: {exc}")
            continue
        json_files.append(json_path)
        game_ids.append(game_id)

    if len(json_files) < max_games:
        raise RuntimeError(f"Only {len(json_files)} valid movement JSON files were prepared.")

    print("[2/6] Downloading matching play-by-play events...")
    for game_id in game_ids:
        download_file(f"events/{game_id}.csv", events_dir / f"{game_id}.csv")

    print("[3/6] Downloading shots.csv...")
    shots_path = shots_dir / "shots.csv"
    download_file("shots/shots.csv", shots_path)

    print("[4/6] Parsing event-level movement features...")
    movement_df = parse_nba_event_level_movement(
        json_files, max_games=max_games, moment_stride=moment_stride
    )

    print("[5/6] Loading labels from events and shots...")
    events_df = load_nba_events(events_dir, game_ids)
    shots_df = load_nba_shots(shots_path, game_ids)

    print("[6/6] Joining movement + events + shots...")
    matched = build_nba_matched_dataset(movement_df, events_df, shots_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_csv, index=False)
    return matched


def print_summary(df: pd.DataFrame, output_csv: Path) -> None:
    print("\nSaved:", output_csv)
    print("Shape:", df.shape)
    print("Games:", df["game_id"].nunique())
    label_cols = [
        "shot_attempt",
        "shot_made",
        "shot_missed",
        "free_throw",
        "turnover",
        "foul",
        "has_shot_chart_row",
        "scoring_event",
    ]
    print("\nLabel positive counts:")
    print(df[label_cols].sum().to_string())
    print("\nHead:")
    head_cols = [
        "game_id",
        "event_id",
        "period",
        "game_clock_start",
        "shot_clock_start",
        "ball_hoop_dist",
        "players_near_hoop",
        "EVENTMSGTYPE",
        "PCTIMESTRING",
        "HOMEDESCRIPTION",
        "VISITORDESCRIPTION",
        "SHOT_TYPE",
        "SHOT_DISTANCE",
        "SHOT_MADE_FLAG",
        "shot_attempt",
        "shot_made",
        "turnover",
        "foul",
    ]
    print(df[[c for c in head_cols if c in df.columns]].head(20).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build matched NBA movement + events + shots dataset without cloning the full repo."
    )
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--moment-stride", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nba_matched"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/nba_matched_events_200.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = build_dataset(args.max_games, args.moment_stride, args.output_dir, args.output_csv)
    print_summary(dataset, args.output_csv)
