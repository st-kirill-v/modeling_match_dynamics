from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_lstm_ablation import run_football_feature_ablation


def main() -> None:
    cfg = ProjectConfig()
    reports = run_football_feature_ablation(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football",
        figures_dir=cfg.figures_dir / "football",
        epochs=25,
        batch_size=32,
        artifact_prefix="feature_ablation_fast",
    )

    print("\nFeature ablation comparison:")
    print(reports["comparison"].to_string(index=False))
    print("\nTraining summary:")
    print(reports["training_summary"].to_string(index=False))
    print("\nTop ranked features:")
    print(reports["ranking"].head(30).to_string(index=False))
    print("\nFOOTBALL LSTM FEATURE ABLATION COMPLETE")


if __name__ == "__main__":
    main()
