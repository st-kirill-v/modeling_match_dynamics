from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_error_analysis import run_football_error_analysis


def main() -> None:
    cfg = ProjectConfig()
    reports = run_football_error_analysis(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football",
        reports_dir=cfg.output_dir / "reports" / "football",
        selected_features_path=(
            cfg.metrics_dir / "football" / "feature_ablation_fast_top_50_selected_features.csv"
        ),
        batch_size=32,
    )

    print(reports["report"])
    print("\nFOOTBALL LSTM ERROR ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
