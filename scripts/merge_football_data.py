from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_merge import save_football_merge_outputs


def main() -> None:
    cfg = ProjectConfig()
    audit_dir = cfg.output_dir / "audits" / "data_quality"
    output_path = cfg.data_dir / "football_merged.csv"
    merged = save_football_merge_outputs(
        football_dir=cfg.football_dir,
        output_path=output_path,
        audit_dir=audit_dir,
    )
    print(f"Football merge saved to: {output_path}")
    print(f"Shape: {merged.shape}")
    print(f"Unique matches: {merged['id_odsp'].nunique(dropna=True)}")
    print(f"Audit tables saved to: {audit_dir}")


if __name__ == "__main__":
    main()
