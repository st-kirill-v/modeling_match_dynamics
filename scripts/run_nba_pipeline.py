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
from match_dynamics.pipeline import run_nba_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBA-only final-score pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Local data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--nba-matched-path",
        type=Path,
        default=None,
        help="Prepared NBA matched CSV. Default: data/processed/nba_matched_events_50.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        nba_matched_path=args.nba_matched_path,
    )
    result = run_nba_pipeline(cfg)
    print("NBA pipeline finished.")
    print("Regression metrics saved to:", cfg.metrics_dir / "nba_final_score_regression_metrics.csv")
    print("Classification metrics saved to:", cfg.metrics_dir / "nba_home_win_classification_metrics.csv")
    print(result["regression_metrics"].head(20).to_string(index=False))
    print(result["classification_metrics"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
