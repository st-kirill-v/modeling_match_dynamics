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
    if os.environ.get("MATCH_DYNAMICS_UV_BOOTSTRAPPED") == "1":
        return
    try:
        import numpy  # noqa: F401
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
from match_dynamics.pipeline import run_football_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Football-only match dynamics pipeline.")
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
    parser.add_argument("--epochs", type=int, default=10, help="LSTM epochs. Lab default is 10.")
    parser.add_argument("--main-window", type=int, default=20, help="Main LSTM window in minutes.")
    parser.add_argument(
        "--compare-windows",
        action="store_true",
        help="Train and compare all configured Football LSTM windows.",
    )
    parser.add_argument("--skip-lstm", action="store_true", help="Skip Football LSTM training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig(
        football_path=args.football_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        main_window=args.main_window,
        compare_windows=args.compare_windows,
        skip_lstm=args.skip_lstm,
    )
    result = run_football_pipeline(cfg)
    print("Football pipeline finished.")
    print("Metrics saved to:", cfg.metrics_dir / "football_metrics.csv")
    print(result["metrics_df"].head(20).to_string(index=False))
    confusion_df = result.get("confusion_df")
    if confusion_df is not None:
        print("\nBest Football confusion matrix:")
        print(confusion_df.to_string())


if __name__ == "__main__":
    main()
