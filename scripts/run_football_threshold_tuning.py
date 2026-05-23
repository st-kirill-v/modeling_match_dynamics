from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_threshold_tuning import run_football_threshold_tuning


def main() -> None:
    cfg = ProjectConfig()
    reports = run_football_threshold_tuning(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football" / "threshold_tuning",
        figures_dir=cfg.figures_dir / "football" / "threshold_tuning",
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        batch_size=32,
    )

    print("\nBest thresholds selected on validation set:")
    print(
        reports["best_thresholds"][
            ["target", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )
    print("\nValidation/test metrics with threshold=0.5 vs tuned:")
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
    print("\nTest precision/recall/f1 comparison:")
    print(reports["comparison"].to_string(index=False))
    print("\nFOOTBALL LSTM THRESHOLD TUNING COMPLETE")


if __name__ == "__main__":
    main()
