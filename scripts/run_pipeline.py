from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def ensure_uv_environment() -> None:
    """Re-run through uv when the script is started with a bare Python interpreter."""
    if os.environ.get("MATCH_DYNAMICS_UV_BOOTSTRAPPED") == "1":
        return

    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        uv_path = shutil.which("uv")
        if uv_path is None:
            print(
                "Dependencies are not installed for this Python. Install uv first, then run:\n"
                "  uv sync --python 3.13\n"
                "  uv run python scripts\\run_pipeline.py --football-path "
                '"D:\\Учеба\\Глубокое обучение (DL)\\Football Events.zip" --skip-nba-download',
                file=sys.stderr,
            )
            raise SystemExit(1)

        env = os.environ.copy()
        env["MATCH_DYNAMICS_UV_BOOTSTRAPPED"] = "1"
        cmd = [uv_path, "run", "python", str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT, env=env))


ensure_uv_environment()

from match_dynamics.config import ProjectConfig
from match_dynamics.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run match dynamics modeling pipeline.")
    parser.add_argument(
        "--football-path",
        type=Path,
        default=None,
        help="Path to events.csv or Football Events.zip.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Local data directory.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output directory."
    )
    parser.add_argument(
        "--nba-repo-dir", type=Path, default=None, help="Existing nba-movement-data repository."
    )
    parser.add_argument(
        "--nba-extract-dir", type=Path, default=None, help="Directory for extracted NBA JSON files."
    )
    parser.add_argument(
        "--nba-json-dir",
        type=Path,
        default=None,
        help="Directory containing extracted NBA JSON files.",
    )
    parser.add_argument(
        "--nba-matched-path",
        type=Path,
        default=None,
        help="Prepared NBA matched CSV. Default: data/processed/nba_matched_events_200.csv.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="LSTM epochs.")
    parser.add_argument("--main-window", type=int, default=10, help="Main LSTM window in minutes.")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip Football LSTM training.")
    parser.add_argument(
        "--skip-nba-download",
        action="store_true",
        help="Legacy option. Main pipeline now uses prepared NBA matched CSV and does not download NBA.",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Train Football all/top20/top30/top40 feature-set comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig(
        football_path=args.football_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        nba_repo_dir=args.nba_repo_dir,
        nba_extract_dir=args.nba_extract_dir,
        nba_json_dir=args.nba_json_dir,
        nba_matched_path=args.nba_matched_path,
        epochs=args.epochs,
        main_window=args.main_window,
        feature_selection=args.feature_selection,
        skip_lstm=args.skip_lstm,
        skip_nba_download=args.skip_nba_download,
    )
    result = run_pipeline(cfg)
    print("Pipeline finished.")
    print("Best model:", result["best_model"])
    print("Metrics saved to:", cfg.metrics_dir / "metrics.csv")
    print(result["metrics_df"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
