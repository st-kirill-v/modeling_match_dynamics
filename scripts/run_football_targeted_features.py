from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_sequence_dataset import SequenceDatasetPaths
from match_dynamics.football_targeted_features import (
    TargetedFeaturePaths,
    run_targeted_feature_build,
)
from match_dynamics.football_targeted_training import run_targeted_top50_training


def main() -> None:
    cfg = ProjectConfig()
    sequence_paths = SequenceDatasetPaths(
        train_csv=cfg.data_dir / "football_targeted_train.csv",
        val_csv=cfg.data_dir / "football_targeted_val.csv",
        test_csv=cfg.data_dir / "football_targeted_test.csv",
        train_npz=cfg.data_dir / "football_targeted_train_sequences.npz",
        val_npz=cfg.data_dir / "football_targeted_val_sequences.npz",
        test_npz=cfg.data_dir / "football_targeted_test_sequences.npz",
        scaler=cfg.models_dir / "football" / "targeted" / "football_targeted_scaler.pkl",
        report_dir=cfg.output_dir / "reports" / "football_targeted_sequences",
    )
    feature_paths = TargetedFeaturePaths(
        input_csv=cfg.data_dir / "football_merged_feature_engineering.csv",
        output_csv=cfg.data_dir / "football_merged_feature_engineering_targeted.csv",
        report_dir=cfg.output_dir / "reports" / "football_targeted",
        sequence_paths=sequence_paths,
    )

    feature_reports = run_targeted_feature_build(feature_paths)
    training_reports = run_targeted_top50_training(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football" / "targeted",
        metrics_dir=cfg.metrics_dir / "football" / "targeted",
        figures_dir=cfg.figures_dir / "football" / "targeted",
        baseline_metrics_path=cfg.metrics_dir
        / "football"
        / "top50_retrain_final_threshold_metrics.csv",
        epochs=25,
        batch_size=32,
    )

    print("\nTargeted feature build summary:")
    print(feature_reports["summary"].to_string(index=False))
    print("\nTrain-only thresholds:")
    print(feature_reports["thresholds"].to_string(index=False))
    print("\nNew feature validation:")
    print(feature_reports["validation"].to_string(index=False))
    print("\nTargeted feature correlations:")
    print(feature_reports["correlations"].to_string(index=False))
    print("\nTargeted sequence diagnostics:")
    print(feature_reports["sequence_diagnostics"].to_string(index=False))

    print("\nTargeted top-50 training summary:")
    print(training_reports["training_summary"].to_string(index=False))
    print("\nTargeted top-50 metrics with final thresholds:")
    print(
        training_reports["final_threshold_metrics"][
            [
                "split",
                "target",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier",
                "log_loss",
                "mae",
                "mse",
            ]
        ].to_string(index=False)
    )
    print("\nTargeted top-50 vs current baseline on test:")
    print(training_reports["comparison"].to_string(index=False))
    print("\nTARGETED FOOTBALL FEATURES COMPLETE")


if __name__ == "__main__":
    main()
