from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_lstm_ablation import run_top50_retrain


def main() -> None:
    cfg = ProjectConfig()
    reports = run_top50_retrain(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football",
        figures_dir=cfg.figures_dir / "football",
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        epochs=25,
        batch_size=32,
        artifact_prefix="top50_retrain",
    )

    print("\nTop-50 retrain metrics, threshold=0.5:")
    print(
        reports["metrics"][
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
    print("\nTop-50 retrain metrics, final fixed thresholds:")
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
    print("\nTraining summary:")
    print(reports["training_summary"].to_string(index=False))
    print("\nFOOTBALL TOP-50 LSTM RETRAIN COMPLETE")


if __name__ == "__main__":
    main()
