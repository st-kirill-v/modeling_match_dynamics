from __future__ import annotations

import shutil

from match_dynamics.config import ProjectConfig


def main() -> None:
    cfg = ProjectConfig()
    source = cfg.data_dir / "football_merged_processed.csv"
    target = cfg.data_dir / "football_merged_feature_engineering.csv"

    if not source.exists():
        raise FileNotFoundError(f"Source dataset was not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Copied source dataset: {source}")
    print(f"Feature engineering dataset: {target}")


if __name__ == "__main__":
    main()
