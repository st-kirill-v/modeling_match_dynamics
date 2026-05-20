from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from match_dynamics.config import ProjectConfig
from match_dynamics.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run match dynamics modeling pipeline.")
    parser.add_argument("--football-path", type=Path, default=None, help="Path to events.csv or Football Events.zip.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Local data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--nba-repo-dir", type=Path, default=None, help="Existing nba-movement-data repository.")
    parser.add_argument("--nba-extract-dir", type=Path, default=None, help="Directory for extracted NBA JSON files.")
    parser.add_argument("--nba-json-dir", type=Path, default=None, help="Directory containing extracted NBA JSON files.")
    parser.add_argument("--epochs", type=int, default=10, help="LSTM epochs. Lab default is 10.")
    parser.add_argument("--main-window", type=int, default=20, help="Main LSTM window in minutes.")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip Football LSTM training.")
    parser.add_argument("--skip-nba-download", action="store_true", help="Do not clone/download NBA Movement Data.")
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
        epochs=args.epochs,
        main_window=args.main_window,
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
