from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_lstm_ablation import refresh_top50_retrain_threshold_outputs


def main() -> None:
    cfg = ProjectConfig()
    reports = refresh_top50_retrain_threshold_outputs(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir,
        metrics_dir=cfg.metrics_dir,
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        batch_size=32,
        artifact_prefix="top50_retrain",
    )

    print("\nUpdated top-50 retrain final threshold metrics:")
    print(
        reports["final_threshold_metrics"][
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
                "mae",
                "mse",
            ]
        ].to_string(index=False)
    )
    print("\nUpdated top-50 retrain confusion matrices:")
    print(reports["confusion_matrices"].to_string(index=False))


if __name__ == "__main__":
    main()
