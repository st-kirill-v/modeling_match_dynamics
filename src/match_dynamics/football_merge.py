from __future__ import annotations

from pathlib import Path

import pandas as pd


FOOTBALL_MERGED_COLUMNS_FOR_SAMPLE = [
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


def load_football_sources(football_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_path = football_dir / "events.csv"
    ginf_path = football_dir / "ginf.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Football events.csv was not found: {events_path}")
    if not ginf_path.exists():
        raise FileNotFoundError(f"Football ginf.csv was not found: {ginf_path}")
    return pd.read_csv(events_path), pd.read_csv(ginf_path)


def build_football_event_match_merge(
    events: pd.DataFrame,
    ginf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = football_merge_checks(events, ginf)
    duplicate_matches = int(
        checks.loc[checks["check"].eq("ginf_duplicate_id_odsp"), "value"].iloc[0]
    )
    if duplicate_matches:
        raise ValueError(
            "ginf.csv contains duplicated id_odsp values. "
            "A left merge would duplicate event rows unexpectedly."
        )

    # This is a one-to-many merge: one row in ginf.csv describes one match,
    # while events.csv has many event rows for the same id_odsp match id.
    merged = events.merge(ginf, on="id_odsp", how="left")

    # Repeating match-level columns such as ht/at/fthg/ftag on every event row is expected.
    # Event-level ML pipelines often need each event row to carry both event features
    # and the static match context for that event.
    if len(merged) != len(events):
        raise ValueError(
            f"Unexpected row count after merge: events={len(events)}, merged={len(merged)}"
        )

    merged["final_score"] = (
        merged["fthg"].astype("Int64").astype(str)
        + "-"
        + merged["ftag"].astype("Int64").astype(str)
    )
    return merged, checks


def football_merge_checks(events: pd.DataFrame, ginf: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "check": "events_shape_rows",
            "value": len(events),
            "details": f"events shape = {events.shape}",
        },
        {
            "check": "events_shape_columns",
            "value": events.shape[1],
            "details": ", ".join(events.columns),
        },
        {
            "check": "ginf_shape_rows",
            "value": len(ginf),
            "details": f"ginf shape = {ginf.shape}",
        },
        {
            "check": "ginf_shape_columns",
            "value": ginf.shape[1],
            "details": ", ".join(ginf.columns),
        },
        {
            "check": "events_unique_id_odsp",
            "value": events["id_odsp"].nunique(dropna=True),
            "details": "Unique match ids in events.csv",
        },
        {
            "check": "ginf_unique_id_odsp",
            "value": ginf["id_odsp"].nunique(dropna=True),
            "details": "Unique match ids in ginf.csv",
        },
        {
            "check": "events_null_id_odsp",
            "value": int(events["id_odsp"].isna().sum()),
            "details": "Null match ids in events.csv",
        },
        {
            "check": "ginf_null_id_odsp",
            "value": int(ginf["id_odsp"].isna().sum()),
            "details": "Null match ids in ginf.csv",
        },
        {
            "check": "ginf_duplicate_id_odsp",
            "value": int(ginf["id_odsp"].duplicated().sum()),
            "details": "Duplicates would break expected one-match-to-many-events merge",
        },
    ]
    return pd.DataFrame(rows)


def football_rows_per_match_stats(merged: pd.DataFrame) -> pd.DataFrame:
    rows_per_match = merged.groupby("id_odsp", dropna=False).size()
    stats = rows_per_match.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    return pd.DataFrame(
        {
            "metric": [
                "matches",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "median",
                "p95",
                "p99",
                "max",
            ],
            "value": [
                rows_per_match.size,
                stats.get("mean"),
                stats.get("std"),
                stats.get("min"),
                stats.get("1%"),
                stats.get("5%"),
                stats.get("50%"),
                stats.get("95%"),
                stats.get("99%"),
                stats.get("max"),
            ],
        }
    )


def football_merge_summary(
    events: pd.DataFrame,
    ginf: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    matched_rows = int(merged["ht"].notna().sum()) if "ht" in merged.columns else 0
    return pd.DataFrame(
        [
            {
                "dataset": "events",
                "rows": len(events),
                "columns": events.shape[1],
                "unique_matches": events["id_odsp"].nunique(dropna=True),
                "null_id_odsp": int(events["id_odsp"].isna().sum()),
            },
            {
                "dataset": "ginf",
                "rows": len(ginf),
                "columns": ginf.shape[1],
                "unique_matches": ginf["id_odsp"].nunique(dropna=True),
                "null_id_odsp": int(ginf["id_odsp"].isna().sum()),
            },
            {
                "dataset": "merged",
                "rows": len(merged),
                "columns": merged.shape[1],
                "unique_matches": merged["id_odsp"].nunique(dropna=True),
                "null_id_odsp": int(merged["id_odsp"].isna().sum()),
                "matched_rows": matched_rows,
                "matched_rate": matched_rows / len(merged) if len(merged) else pd.NA,
            },
        ]
    )


def save_football_merge_outputs(
    football_dir: Path,
    output_path: Path,
    audit_dir: Path | None = None,
) -> pd.DataFrame:
    events, ginf = load_football_sources(football_dir)
    merged, checks = build_football_event_match_merge(events, ginf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        checks.to_csv(audit_dir / "football_merge_checks.csv", index=False)
        football_merge_summary(events, ginf, merged).to_csv(
            audit_dir / "football_merge_summary.csv", index=False
        )
        football_rows_per_match_stats(merged).to_csv(
            audit_dir / "football_merge_rows_per_match_stats.csv", index=False
        )
        sample_cols = [c for c in FOOTBALL_MERGED_COLUMNS_FOR_SAMPLE if c in merged.columns]
        merged[sample_cols].head(20).to_csv(audit_dir / "football_merged_head.csv", index=False)
    return merged
