from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_uv_environment() -> None:
    if os.environ.get("MATCH_DYNAMICS_UV_BOOTSTRAPPED") == "1":
        return
    try:
        import pandas  # noqa: F401
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


def show_csv_heads(sample_dir: Path, rows: int) -> None:
    for section in ["events", "shots"]:
        print(f"\n{'=' * 100}\n{section.upper()}\n{'=' * 100}")
        for path in sorted((sample_dir / section).glob("*.csv")):
            df = pd.read_csv(path)
            print(f"\nFILE: {path}")
            print(f"SHAPE: {df.shape}")
            print(f"COLUMNS: {list(df.columns)}")
            print(df.head(rows).to_string(index=False))


def show_movement_json_summary(sample_dir: Path, rows: int) -> None:
    print(f"\n{'=' * 100}\nMOVEMENT JSON\n{'=' * 100}")
    for path in sorted((sample_dir / "extracted_sample").glob("*.json")):
        with path.open(encoding="utf-8") as f:
            game = json.load(f)
        events = game.get("events", [])
        print(f"\nFILE: {path}")
        print(f"gameid={game.get('gameid')} gamedate={game.get('gamedate')} events={len(events)}")
        for event in events[:rows]:
            moments = event.get("moments", [])
            print(
                f"eventId={event.get('eventId')} "
                f"moments={len(moments)} "
                f"first_moment={moments[0][:5] if moments else None}"
            )
            if moments and len(moments[0]) > 5 and moments[0][5]:
                print(f"first player/ball row={moments[0][5][0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show head() for downloaded NBA sample files.")
    parser.add_argument("--sample-dir", type=Path, default=Path("data/nba_sample"))
    parser.add_argument("--rows", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    show_csv_heads(args.sample_dir, args.rows)
    show_movement_json_summary(args.sample_dir, args.rows)
