from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


RANDOM_STATE = 42
EPOCHS = 10
TIME_STEPS = 10
WINDOW_EXPERIMENTS = [10, 20, 30, 40]
FOOTBALL_HALF_CUTOFF = 45
NBA_MAX_ARCHIVES = 30
NBA_MAX_GAMES = 200
NBA_MOMENT_STRIDE = 50

FOOTBALL_TARGETS = ["home_scores_next_half", "away_scores_next_half"]


BASE_FOOTBALL_FEATURES = [
    "time",
    "time_sin",
    "time_cos",
    "minutes_remaining",
    "home_score",
    "away_score",
    "score_diff",
    "abs_score_diff",
    "is_draw",
    "home_leading",
    "away_leading",
    "home_event_last_5min",
    "away_event_last_5min",
    "home_attempt_last_5min",
    "away_attempt_last_5min",
    "home_key_pass_last_5min",
    "away_key_pass_last_5min",
    "home_corner_last_5min",
    "away_corner_last_5min",
    "home_xg_last_5min",
    "away_xg_last_5min",
    "home_xg_last_10min",
    "away_xg_last_10min",
    "xg_diff_last_10min",
    "pressure_diff_last_5min",
    "event_activity_last_10min",
    "event_activity_momentum_5min",
    "home_event_share_last_5min",
    "away_event_share_last_5min",
    "is_yellow_last_10min",
    "is_red_last_10min",
    "is_fast_break_last_10min",
    "events_per_minute_last_5min",
    "events_per_minute_last_10min",
    "sort_order_velocity_last_5min",
    "sort_order_velocity_last_10min",
    "events_per_minute_momentum_5min",
]

TEAM_STRENGTH_FEATURES = [
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "team_attack_diff",
    "team_defense_diff",
]

TIME_FEATURE_SETS = {
    "raw_time": [c for c in BASE_FOOTBALL_FEATURES if c not in ["time_sin", "time_cos"]],
    "sin_cos": [c for c in BASE_FOOTBALL_FEATURES if c != "time"],
    "raw_plus_sincos": BASE_FOOTBALL_FEATURES,
    "no_time": [
        c
        for c in BASE_FOOTBALL_FEATURES
        if c not in ["time", "time_sin", "time_cos", "minutes_remaining"]
    ],
}


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs")
    football_path: Path | None = None
    nba_repo_dir: Path | None = None
    nba_extract_dir: Path | None = None
    nba_json_dir: Path | None = None
    nba_matched_path: Path | None = None
    epochs: int = EPOCHS
    main_window: int = 20
    skip_lstm: bool = False
    skip_nba_download: bool = False

    @property
    def football_dir(self) -> Path:
        return self.data_dir / "football"

    @property
    def football_events_path(self) -> Path:
        return self.football_dir / "events.csv"

    @property
    def default_nba_matched_path(self) -> Path:
        return self.data_dir / "processed" / "nba_matched_events_200.csv"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    def ensure_dirs(self) -> None:
        for path in [self.data_dir, self.output_dir, self.figures_dir, self.metrics_dir]:
            path.mkdir(parents=True, exist_ok=True)
