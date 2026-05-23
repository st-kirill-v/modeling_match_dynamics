from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_event_processing import (
    cleanup_minute_level_processed,
    save_minute_cleanup_reports,
)


def main() -> None:
    cfg = ProjectConfig()
    input_path = cfg.data_dir / "football_merged_processed.csv"
    audit_dir = cfg.output_dir / "audits" / "data_quality"

    if not input_path.exists():
        raise FileNotFoundError(f"Processed football dataset was not found: {input_path}")

    import pandas as pd

    input_df = pd.read_csv(input_path)
    output_df, reports = cleanup_minute_level_processed(input_df)
    output_df.to_csv(input_path, index=False)
    save_minute_cleanup_reports(reports, audit_dir)

    final_diag = reports["final_diagnostics"]
    goal_semantics = reports["goal_semantics"]
    type_checks = reports["type_checks"]
    feature_nulls = reports["feature_null_counts"]
    roles = reports["column_roles"]

    print(f"Saved cleaned dataset to: {input_path}")
    print("\nFinal diagnostics:")
    print(final_diag.to_string(index=False))
    print("\nGoal semantics:")
    print(goal_semantics.to_string(index=False))
    print("\nType checks:")
    print(type_checks.to_string(index=False))
    print("\nFeature null columns:")
    print(
        feature_nulls.to_string(index=False)
        if not feature_nulls.empty
        else "No NaN in safe feature columns."
    )
    print("\nColumn roles:")
    print(roles.groupby("role").size().rename("columns").to_string())
    print(f"\nAudit tables saved to: {audit_dir}")


if __name__ == "__main__":
    main()
