from __future__ import annotations

import pandas as pd

from match_dynamics.config import ProjectConfig
from match_dynamics.football_event_processing import (
    cleanup_minute_level_processed,
    save_minute_cleanup_reports,
    save_processed_target_reports,
)


def main() -> None:
    cfg = ProjectConfig()
    processed_path = cfg.data_dir / "football_merged_processed.csv"
    audit_dir = cfg.output_dir / "audits" / "data_quality"
    report_dir = cfg.output_dir / "reports"

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed football dataset was not found: {processed_path}")

    input_df = pd.read_csv(processed_path)
    output_df, cleanup_reports = cleanup_minute_level_processed(input_df)
    output_df.to_csv(processed_path, index=False)
    save_minute_cleanup_reports(cleanup_reports, audit_dir)
    target_reports = save_processed_target_reports(output_df, report_dir, audit_dir)

    print(f"Saved final minute-level dataset to: {processed_path}")
    print("\nDiagnostics:")
    print(target_reports["target_diagnostics"].to_string(index=False))
    print("\nTarget distribution:")
    print(target_reports["target_distribution"].to_string(index=False))
    print("\nTop feature-target correlations:")
    print(target_reports["feature_target_correlations"].head(20).to_string(index=False))
    print(f"\nCorrelation report: {report_dir / 'football_feature_target_correlations.csv'}")
    print(f"Audit tables: {audit_dir}")


if __name__ == "__main__":
    main()
