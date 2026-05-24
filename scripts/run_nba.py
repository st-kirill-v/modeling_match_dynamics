from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

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

from match_dynamics.config import ProjectConfig
from match_dynamics.nba_merge_pipeline import (
    NbaMergePaths,
    build_nba_events_shots_merge,
    download_nba_sources,
    run_nba_merge_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download limited NBA movement/events/shots_fixed data and merge events + shots."
    )
    parser.add_argument("--max-archives", type=int, default=500)
    parser.add_argument("--max-events", type=int, default=500)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Merged output CSV path. Defaults to data/nba/nba_events_shots_merged_<max-events>.csv.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already downloaded data/nba files and only rebuild merge reports.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download limited source files; do not build the merged dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    nba_dir = cfg.data_dir / "nba"
    output_csv = args.output_csv or nba_dir / f"nba_events_shots_merged_{args.max_events}.csv"
    paths = NbaMergePaths(
        data_dir=nba_dir,
        movement_dir=nba_dir / "movement",
        events_dir=nba_dir / "events",
        shots_dir=nba_dir / "shots",
        merged_csv=output_csv,
        audit_dir=cfg.output_dir / "audits" / "data_quality",
        report_dir=cfg.output_dir / "reports" / "nba_merge",
    )
    if args.download_only:
        inventory = download_nba_sources(
            paths,
            max_archives=args.max_archives,
            max_events=args.max_events,
        )
        print("\nDownload inventory:")
        print(inventory.to_string(index=False))
        print("\nNBA DOWNLOAD COMPLETE")
        return

    if args.skip_download:
        reports = build_nba_events_shots_merge(paths, max_events=args.max_events)
    else:
        reports = run_nba_merge_pipeline(
            paths,
            max_archives=args.max_archives,
            max_events=args.max_events,
        )

    print("\nNBA merge summary:")
    print(reports["merged_summary"].to_string(index=False))
    print("\nMerge diagnostics:")
    print(reports["merge_diagnostics"].to_string(index=False))
    print("\nMovement sample:")
    print(reports["movement_head"].to_string(index=False))
    print("\nEvents head:")
    print(reports["events_head"].head(10).to_string(index=False))
    print("\nShots fixed head:")
    print(reports["shots_fixed_head"].head(10).to_string(index=False))
    print("\nMerged head:")
    print(reports["merged_head"].head(10).to_string(index=False))
    print("\nTop missing columns:")
    print(
        reports["merged_top_missing_columns"][
            ["column", "dtype", "null_count", "null_percent", "unique_count"]
        ]
        .head(20)
        .to_string(index=False)
    )
    print("\nSaved merged dataset:")
    print(paths.merged_csv)
    print("\nNBA MERGE PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
