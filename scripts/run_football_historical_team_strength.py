from __future__ import annotations

import pandas as pd

from match_dynamics.config import ProjectConfig
from match_dynamics.football_feature_engineering import (
    add_historical_team_strength_features,
    save_feature_engineering_reports,
)


def main() -> None:
    cfg = ProjectConfig()
    input_path = cfg.data_dir / "football_merged_feature_engineering.csv"
    report_dir = cfg.output_dir / "reports"
    audit_dir = cfg.output_dir / "audits" / "data_quality"

    if not input_path.exists():
        raise FileNotFoundError(f"Feature engineering dataset was not found: {input_path}")

    input_df = pd.read_csv(input_path)
    output_df, reports = add_historical_team_strength_features(input_df)

    temp_path = input_path.with_suffix(".tmp.csv")
    output_df.to_csv(temp_path, index=False)
    temp_path.replace(input_path)
    save_feature_engineering_reports(reports, report_dir, audit_dir)

    print(f"Saved feature engineering dataset to: {input_path}")
    print("\nHistorical summary:")
    print(reports["historical_summary"].to_string(index=False))
    print("\nHistorical validation:")
    print(reports["historical_validation"].to_string(index=False))
    print("\nTarget distribution:")
    print(reports["target_distribution"].to_string(index=False))
    print("\nTop feature-target correlations:")
    print(reports["feature_target_correlations"].head(25).to_string(index=False))
    print("\nHISTORICAL FOOTBALL TEAM-STRENGTH FEATURES COMPLETE")


if __name__ == "__main__":
    main()
