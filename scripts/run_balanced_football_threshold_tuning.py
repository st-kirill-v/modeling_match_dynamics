from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_threshold_tuning import run_balanced_football_threshold_tuning


def main() -> None:
    cfg = ProjectConfig()
    reports = run_balanced_football_threshold_tuning(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football" / "threshold_tuning_balanced",
        figures_dir=cfg.figures_dir / "football" / "threshold_tuning_balanced",
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        batch_size=32,
    )

    print("\nPrevious F1-tuned thresholds selected on validation:")
    print(
        reports["previous_thresholds"][
            ["target", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )
    print("\nBalanced thresholds selected on validation:")
    print(
        reports["balanced_thresholds"][
            [
                "target",
                "threshold",
                "precision_constraint",
                "constraint_satisfied",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )
    print("\nValidation/test metrics: 0.5 vs previous tuned vs balanced:")
    print(
        reports["metrics"][
            [
                "split",
                "target",
                "threshold_mode",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "mae",
                "mse",
            ]
        ].to_string(index=False)
    )
    print("\nTest comparison:")
    print(reports["comparison"].to_string(index=False))
    print("\nBALANCED FOOTBALL THRESHOLD TUNING COMPLETE")


if __name__ == "__main__":
    main()
