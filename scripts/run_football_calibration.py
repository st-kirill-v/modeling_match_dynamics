from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_calibration import run_football_probability_calibration


def main() -> None:
    cfg = ProjectConfig()
    reports = run_football_probability_calibration(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football" / "calibration",
        figures_dir=cfg.figures_dir / "football" / "calibration",
        calibrators_dir=cfg.models_dir / "football" / "calibration",
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        batch_size=32,
    )

    print("\nCalibration diagnostics:")
    print(reports["diagnostics"].to_string(index=False))
    print("\nTest metrics: raw vs calibrated probabilities")
    print(
        reports["metrics"][
            [
                "target",
                "calibration_method",
                "threshold",
                "brier",
                "log_loss",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )
    print("\nCalibration delta vs raw:")
    print(reports["comparison"].to_string(index=False))
    print("\nSaved calibrators:")
    print(reports["calibrators"].to_string(index=False))
    print("\nFOOTBALL LSTM CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
