from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data_loading import ensure_football_events
from .football import preprocess_football_events
from .football_event_processing import save_football_merged_processed_outputs
from .football_merge import (
    build_football_event_match_merge,
    football_merge_summary,
    football_rows_per_match_stats,
)


@dataclass(frozen=True)
class AuditConfig:
    output_dir: Path
    sample_rows: int = 10
    movement_sample_games: int = 1

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        non_null = int(series.notna().sum())
        nunique = int(series.nunique(dropna=True))
        zero_count = int((series == 0).sum()) if pd.api.types.is_numeric_dtype(series) else np.nan
        zero_rate = (
            zero_count / total if total and pd.api.types.is_numeric_dtype(series) else np.nan
        )
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "missing": missing,
            "missing_rate": missing / total if total else np.nan,
            "n_unique": nunique,
            "zero_count": zero_count,
            "zero_rate": zero_rate,
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            row.update(
                {
                    "mean": desc.get("mean", np.nan),
                    "std": desc.get("std", np.nan),
                    "min": desc.get("min", np.nan),
                    "p01": desc.get("1%", np.nan),
                    "p05": desc.get("5%", np.nan),
                    "median": desc.get("50%", np.nan),
                    "p95": desc.get("95%", np.nan),
                    "p99": desc.get("99%", np.nan),
                    "max": desc.get("max", np.nan),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["missing_rate", "column"], ascending=[False, True])


def dataset_overview(name: str, df: pd.DataFrame) -> dict:
    numeric_cols = int(df.select_dtypes(include="number").shape[1])
    object_cols = int(df.select_dtypes(include=["object", "category"]).shape[1])
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    return {
        "dataset": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "object_columns": object_cols,
        "missing_cells": missing_cells,
        "missing_cell_rate": missing_cells / total_cells if total_cells else np.nan,
    }


def save_dataset_audit(name: str, df: pd.DataFrame, output_dir: Path, sample_rows: int) -> dict:
    safe_name = name.lower().replace(" ", "_")
    profile = column_profile(df)
    profile.to_csv(output_dir / f"{safe_name}_columns.csv", index=False)
    df.head(sample_rows).to_csv(output_dir / f"{safe_name}_head.csv", index=False)
    return dataset_overview(name, df)


def football_processing_changes(
    raw: pd.DataFrame, minute: pd.DataFrame, model_df: pd.DataFrame
) -> pd.DataFrame:
    raw_cols = set(raw.columns)
    minute_cols = set(minute.columns)
    model_cols = set(model_df.columns)
    return pd.DataFrame(
        [
            {
                "step": "raw_events_to_minute_level",
                "input_rows": len(raw),
                "output_rows": len(minute),
                "input_columns": len(raw_cols),
                "output_columns": len(minute_cols),
                "added_columns": ", ".join(sorted(minute_cols - raw_cols)),
                "removed_or_aggregated_columns": ", ".join(sorted(raw_cols - minute_cols)),
            },
            {
                "step": "minute_level_to_first_half_model_df",
                "input_rows": len(minute),
                "output_rows": len(model_df),
                "input_columns": len(minute_cols),
                "output_columns": len(model_cols),
                "added_columns": ", ".join(sorted(model_cols - minute_cols)),
                "removed_or_aggregated_columns": ", ".join(sorted(minute_cols - model_cols)),
            },
        ]
    )


def nba_join_quality(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(matched)
    checks = {
        "matched_to_play_by_play_event": "EVENTMSGTYPE",
        "has_score_text": "SCORE",
        "matched_to_shots_row": "SHOT_MADE_FLAG",
        "has_home_description": "HOMEDESCRIPTION",
        "has_visitor_description": "VISITORDESCRIPTION",
        "has_ball_coordinates": "ball_x",
        "has_shot_clock_start": "shot_clock_start",
        "has_shot_clock_end": "shot_clock_end",
    }
    for label, col in checks.items():
        if col not in matched.columns:
            rows.append(
                {
                    "check": label,
                    "column": col,
                    "available_rows": 0,
                    "missing_rows": total,
                    "available_rate": 0.0,
                }
            )
            continue
        available = int(matched[col].notna().sum())
        rows.append(
            {
                "check": label,
                "column": col,
                "available_rows": available,
                "missing_rows": total - available,
                "available_rate": available / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_nba_raw_files(
    base_dir: Path, output_dir: Path, movement_sample_games: int
) -> pd.DataFrame:
    rows = []
    locations = {
        "archives": base_dir / "archives",
        "events": base_dir / "events",
        "shots": base_dir / "shots",
        "movement_json": base_dir / "extracted_json",
    }
    for name, path in locations.items():
        files = sorted(path.glob("*")) if path.exists() else []
        rows.append(
            {
                "source": name,
                "path": str(path),
                "files": len([p for p in files if p.is_file()]),
                "total_size_mb": round(
                    sum(p.stat().st_size for p in files if p.is_file()) / (1024 * 1024), 2
                ),
            }
        )

    movement_files = sorted(locations["movement_json"].glob("*.json"))[:movement_sample_games]
    movement_rows = []
    for path in movement_files:
        with path.open(encoding="utf-8") as f:
            game = json.load(f)
        events = game.get("events", [])
        moment_counts = [len(event.get("moments", [])) for event in events]
        movement_rows.append(
            {
                "file": path.name,
                "gameid": game.get("gameid"),
                "events": len(events),
                "moments_total": int(sum(moment_counts)),
                "moments_min_per_event": int(min(moment_counts)) if moment_counts else 0,
                "moments_max_per_event": int(max(moment_counts)) if moment_counts else 0,
                "home_team_id": game.get("home", {}).get("teamid"),
                "visitor_team_id": game.get("visitor", {}).get("teamid"),
            }
        )
    pd.DataFrame(movement_rows).to_csv(output_dir / "nba_raw_movement_json_sample.csv", index=False)
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_dir: Path, overviews: list[dict], report_files: list[str]
) -> None:
    overview_df = pd.DataFrame(overviews)
    overview_lines = overview_df.to_csv(index=False).strip().splitlines()
    lines = [
        "# Data Audit Report",
        "",
        "Этот отчет сгенерирован без обучения моделей. Он нужен, чтобы проверить сырые данные, пропуски, результат предобработки и качество join-а.",
        "",
        "## Dataset Overview",
        "",
        "```csv",
        *overview_lines,
        "```",
        "",
        "## Generated Tables",
        "",
    ]
    lines[2] = (
        "Generated without model training. Use this report to inspect raw data, missing values, "
        "preprocessing output, and NBA join quality."
    )
    lines.extend(f"- `{name}`" for name in report_files)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_data_audit(cfg: ProjectConfig, audit_cfg: AuditConfig) -> Path:
    cfg.ensure_dirs()
    audit_cfg.ensure_dirs()

    overviews: list[dict] = []
    report_files: list[str] = []

    football_raw = ensure_football_events(cfg)
    ginf_path = cfg.football_dir / "ginf.csv"
    if ginf_path.exists():
        football_ginf = pd.read_csv(ginf_path)
        football_merged, football_merge_checks = build_football_event_match_merge(
            football_raw, football_ginf
        )
        football_merged_path = cfg.data_dir / "football_merged.csv"
        football_merged.to_csv(football_merged_path, index=False)
        football_merge_checks.to_csv(
            audit_cfg.output_dir / "football_merge_checks.csv", index=False
        )
        football_merge_summary(football_raw, football_ginf, football_merged).to_csv(
            audit_cfg.output_dir / "football_merge_summary.csv", index=False
        )
        football_rows_per_match_stats(football_merged).to_csv(
            audit_cfg.output_dir / "football_merge_rows_per_match_stats.csv", index=False
        )
        sample_cols = [
            c
            for c in [
                "id_odsp",
                "id_event",
                "time",
                "event_type",
                "side",
                "event_team",
                "opponent",
                "ht",
                "at",
                "fthg",
                "ftag",
                "final_score",
            ]
            if c in football_merged.columns
        ]
        football_merged[sample_cols].head(20).to_csv(
            audit_cfg.output_dir / "football_merged_head.csv", index=False
        )
        overviews.append(
            save_dataset_audit(
                "football_merged_event_match",
                football_merged,
                audit_cfg.output_dir,
                audit_cfg.sample_rows,
            )
        )
        report_files.extend(
            [
                "football_merge_checks.csv",
                "football_merge_summary.csv",
                "football_merge_rows_per_match_stats.csv",
                "football_merged_head.csv",
                "football_merged_event_match_columns.csv",
                "football_merged_event_match_head.csv",
            ]
        )
        football_processed_path = cfg.data_dir / "football_merged_processed.csv"
        football_processed, _ = save_football_merged_processed_outputs(
            input_path=football_merged_path,
            output_path=football_processed_path,
            audit_dir=audit_cfg.output_dir,
        )
        overviews.append(
            save_dataset_audit(
                "football_merged_processed",
                football_processed,
                audit_cfg.output_dir,
                audit_cfg.sample_rows,
            )
        )
        report_files.extend(
            [
                "football_merged_processed_binary_validation.csv",
                "football_merged_processed_columns.csv",
                "football_merged_processed_feature_log.csv",
                "football_merged_processed_head.csv",
                "football_merged_processed_new_features_head.csv",
                "football_merged_processed_summary.csv",
            ]
        )

    football_minute, football_model_df = preprocess_football_events(football_raw)
    for name, df in [
        ("football_raw_events", football_raw),
        ("football_minute_level_processed", football_minute),
        ("football_first_half_model_df", football_model_df),
    ]:
        overviews.append(save_dataset_audit(name, df, audit_cfg.output_dir, audit_cfg.sample_rows))
        report_files.extend([f"{name}_columns.csv", f"{name}_head.csv"])

    football_changes = football_processing_changes(football_raw, football_minute, football_model_df)
    football_changes.to_csv(audit_cfg.output_dir / "football_processing_changes.csv", index=False)
    report_files.append("football_processing_changes.csv")

    nba_matched_path = cfg.nba_matched_path or cfg.default_nba_matched_path
    if nba_matched_path.exists():
        nba_matched = pd.read_csv(nba_matched_path)
        overviews.append(
            save_dataset_audit(
                "nba_matched_processed", nba_matched, audit_cfg.output_dir, audit_cfg.sample_rows
            )
        )
        nba_join = nba_join_quality(nba_matched)
        nba_join.to_csv(audit_cfg.output_dir / "nba_join_quality.csv", index=False)
        report_files.extend(
            [
                "nba_matched_processed_columns.csv",
                "nba_matched_processed_head.csv",
                "nba_join_quality.csv",
            ]
        )

    checkpoint_path = cfg.data_dir / "processed" / "nba_final_score_checkpoint_5min.csv"
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        overviews.append(
            save_dataset_audit(
                "nba_final_score_checkpoint_5min",
                checkpoint,
                audit_cfg.output_dir,
                audit_cfg.sample_rows,
            )
        )
        report_files.extend(
            [
                "nba_final_score_checkpoint_5min_columns.csv",
                "nba_final_score_checkpoint_5min_head.csv",
            ]
        )

    nba_raw_base = cfg.data_dir / "nba_matched"
    if nba_raw_base.exists():
        raw_inventory = audit_nba_raw_files(
            nba_raw_base, audit_cfg.output_dir, audit_cfg.movement_sample_games
        )
        raw_inventory.to_csv(audit_cfg.output_dir / "nba_raw_file_inventory.csv", index=False)
        report_files.extend(["nba_raw_file_inventory.csv", "nba_raw_movement_json_sample.csv"])

        events_dir = nba_raw_base / "events"
        event_files = sorted(events_dir.glob("*.csv"))
        if event_files:
            raw_events_sample = pd.concat(
                [pd.read_csv(path).assign(source_file=path.name) for path in event_files[:5]],
                ignore_index=True,
            )
            overviews.append(
                save_dataset_audit(
                    "nba_raw_events_sample",
                    raw_events_sample,
                    audit_cfg.output_dir,
                    audit_cfg.sample_rows,
                )
            )
            report_files.extend(
                ["nba_raw_events_sample_columns.csv", "nba_raw_events_sample_head.csv"]
            )

        shots_path = nba_raw_base / "shots" / "shots.csv"
        if shots_path.exists():
            raw_shots_sample = pd.read_csv(shots_path, nrows=5000)
            overviews.append(
                save_dataset_audit(
                    "nba_raw_shots_sample",
                    raw_shots_sample,
                    audit_cfg.output_dir,
                    audit_cfg.sample_rows,
                )
            )
            report_files.extend(
                ["nba_raw_shots_sample_columns.csv", "nba_raw_shots_sample_head.csv"]
            )

    pd.DataFrame(overviews).to_csv(audit_cfg.output_dir / "dataset_overview.csv", index=False)
    report_files.append("dataset_overview.csv")
    write_markdown_summary(audit_cfg.output_dir, overviews, sorted(set(report_files)))
    return audit_cfg.output_dir
