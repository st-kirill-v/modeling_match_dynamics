from __future__ import annotations

from match_dynamics.config import ProjectConfig
from match_dynamics.football_lstm_training import train_baseline_football_lstm


def main() -> None:
    cfg = ProjectConfig()
    reports = train_baseline_football_lstm(
        data_dir=cfg.data_dir,
        models_dir=cfg.models_dir / "football",
        metrics_dir=cfg.metrics_dir / "football",
        figures_dir=cfg.figures_dir / "football",
        epochs=25,
        batch_size=32,
    )

    print("\nShapes:")
    print(reports["shapes"].to_string(index=False))
    print("\nFinal metrics:")
    print(reports["metrics"].to_string(index=False))
    print("\nOverfitting analysis:")
    print(reports["overfitting"].to_string(index=False))
    print("\nArtifacts:")
    print(cfg.models_dir / "football" / "baseline_multioutput_lstm.keras")
    print(cfg.models_dir / "football" / "baseline_lstm_summary.txt")
    print(cfg.metrics_dir / "football" / "baseline_lstm_metrics.csv")
    print(cfg.metrics_dir / "football" / "baseline_lstm_history.csv")
    print(cfg.figures_dir / "football")
    print("\nBASELINE FOOTBALL LSTM TRAINING COMPLETE")


if __name__ == "__main__":
    main()
