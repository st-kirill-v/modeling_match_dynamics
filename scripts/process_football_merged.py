from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_event_processing import save_football_merged_processed_outputs


def main() -> None:
    cfg = ProjectConfig()
    audit_dir = cfg.output_dir / "audits" / "data_quality"
    input_path = cfg.data_dir / "football_merged.csv"
    output_path = cfg.data_dir / "football_merged_processed.csv"
    output_df, log = save_football_merged_processed_outputs(input_path, output_path, audit_dir)

    created = log.loc[log["action"].eq("created"), "column"].dropna().tolist()
    updated = log.loc[log["action"].eq("updated"), "column"].dropna().tolist()
    dropped = log.loc[log["action"].eq("dropped"), "column"].dropna().tolist()
    print(f"Football merged processed saved to: {output_path}")
    print(f"Input file: {input_path}")
    print(f"Output minute-level shape: {output_df.shape}")
    print(f"Created features ({len(created)}): {created}")
    print(f"Updated features ({len(updated)}): {updated}")
    print(f"Dropped columns ({len(dropped)}): {dropped}")
    print(f"Null values after transformation: {int(output_df.isna().sum().sum())}")
    print("\nFirst 5 rows of new features:")
    preview_cols = [col for col in [*created, *updated] if col in output_df.columns]
    print(output_df[preview_cols].head().to_string(index=False))
    print(f"\nAudit tables saved to: {audit_dir}")


if __name__ == "__main__":
    main()
