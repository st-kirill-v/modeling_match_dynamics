from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from match_dynamics.audit import AuditConfig, run_data_audit
from match_dynamics.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw/processed data quality audit tables for Football and NBA."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "audits" / "data_quality",
        help="Directory for generated audit CSV/Markdown files.",
    )
    parser.add_argument(
        "--nba-matched-path",
        type=Path,
        default=None,
        help="Prepared NBA matched CSV. Defaults to data/processed/nba_matched_events_200.csv.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="Number of rows to save in *_head.csv previews.",
    )
    parser.add_argument(
        "--movement-sample-games",
        type=int,
        default=1,
        help="Number of raw NBA movement JSON files to inspect for metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    cfg = ProjectConfig(nba_matched_path=args.nba_matched_path)
    audit_cfg = AuditConfig(
        output_dir=args.output_dir,
        sample_rows=args.sample_rows,
        movement_sample_games=args.movement_sample_games,
    )
    output_dir = run_data_audit(cfg, audit_cfg)
    print(f"Data audit finished. Reports saved to: {output_dir}")
    print(f"Open summary: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
