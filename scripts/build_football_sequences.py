from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_sequence_dataset import (
    SequenceDatasetPaths,
    build_football_sequence_dataset,
)


def main() -> None:
    cfg = ProjectConfig()
    paths = SequenceDatasetPaths(
        train_csv=cfg.data_dir / "football_merged_train.csv",
        val_csv=cfg.data_dir / "football_merged_val.csv",
        test_csv=cfg.data_dir / "football_merged_test.csv",
        train_npz=cfg.data_dir / "football_train_sequences.npz",
        val_npz=cfg.data_dir / "football_val_sequences.npz",
        test_npz=cfg.data_dir / "football_test_sequences.npz",
        scaler=cfg.models_dir / "football_scaler.pkl",
        report_dir=cfg.output_dir / "reports",
    )
    reports = build_football_sequence_dataset(
        input_path=cfg.data_dir / "football_merged_feature_engineering.csv",
        paths=paths,
    )

    print("Football sequence dataset saved.")
    print("\nDiagnostics:")
    print(reports["diagnostics"].to_string(index=False))
    print("\nTarget distribution:")
    print(reports["target_distribution"].to_string(index=False))
    print("\nSequence length stats:")
    print(reports["sequence_length_stats"].to_string(index=False))
    print("\nTarget checks:")
    print(reports["target_checks"].to_string(index=False))
    print("\nOutput files:")
    for path in [
        paths.train_csv,
        paths.val_csv,
        paths.test_csv,
        paths.train_npz,
        paths.val_npz,
        paths.test_npz,
        paths.scaler,
    ]:
        print(path)
    print("\nFOOTBALL SEQUENCE DATASET READY FOR LSTM TRAINING")


if __name__ == "__main__":
    main()
