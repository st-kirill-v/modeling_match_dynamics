from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_threshold_tuning import run_final_football_threshold_evaluation


def main() -> None:
    cfg = ProjectConfig()
    reports = run_final_football_threshold_evaluation(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football" / "threshold_tuning_final",
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        batch_size=32,
    )

    print("\nFinal fixed thresholds:")
    print(reports["final_thresholds"].to_string(index=False))
    print("\nValidation/test metrics: 0.5 vs final fixed thresholds:")
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


if __name__ == "__main__":
    main()
