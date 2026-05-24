from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import py7zr

from .nba import parse_nba_event_level_movement


REPO_API_DATA_URL = "https://api.github.com/repos/sealneaward/nba-movement-data/contents/data"
REPO_API_EVENTS_URL = (
    "https://api.github.com/repos/sealneaward/nba-movement-data/contents/data/events"
)
RAW_BASE_URL = "https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data"


@dataclass(frozen=True)
class NbaMergePaths:
    data_dir: Path
    movement_dir: Path
    events_dir: Path
    shots_dir: Path
    merged_csv: Path
    audit_dir: Path
    report_dir: Path

    @property
    def movement_json_dir(self) -> Path:
        return self.data_dir / "movement_json"


def github_raw_url(relative_path: str) -> str:
    return f"{RAW_BASE_URL}/{quote(relative_path, safe='/')}"


def read_github_directory(api_url: str) -> list[dict]:
    with urllib.request.urlopen(api_url, timeout=90) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected GitHub API response from {api_url}: {payload}")
    return payload


def list_repo_files() -> tuple[list[str], list[str]]:
    data_items = read_github_directory(REPO_API_DATA_URL)
    event_items = read_github_directory(REPO_API_EVENTS_URL)
    archive_names = sorted(item["name"] for item in data_items if item["name"].endswith(".7z"))
    event_names = sorted(item["name"] for item in event_items if item["name"].endswith(".csv"))
    return archive_names, event_names


def download_file(relative_path: str, output_path: Path, skipped: list[dict]) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        skipped.append(
            {
                "file": str(output_path),
                "source": relative_path,
                "reason": "already_exists",
            }
        )
        return False
    try:
        with urllib.request.urlopen(github_raw_url(relative_path), timeout=240) as response:
            with output_path.open("wb") as f:
                shutil.copyfileobj(response, f)
        return True
    except Exception as exc:
        skipped.append(
            {
                "file": str(output_path),
                "source": relative_path,
                "reason": f"download_failed: {exc}",
            }
        )
        return False


def download_nba_sources(
    paths: NbaMergePaths,
    max_archives: int = 500,
    max_events: int = 500,
) -> pd.DataFrame:
    skipped: list[dict] = []
    archive_names, event_names = list_repo_files()

    downloaded_rows = []
    for name in archive_names[:max_archives]:
        output_path = paths.movement_dir / name
        downloaded = download_file(name, output_path, skipped)
        downloaded_rows.append(
            {
                "source_type": "movement_archive",
                "relative_path": name,
                "local_path": str(output_path),
                "downloaded_now": downloaded,
                "exists": output_path.exists(),
                "size_mb": output_path.stat().st_size / 1024 / 1024 if output_path.exists() else 0,
            }
        )

    for name in event_names[:max_events]:
        relative_path = f"events/{name}"
        output_path = paths.events_dir / name
        downloaded = download_file(relative_path, output_path, skipped)
        downloaded_rows.append(
            {
                "source_type": "events_csv",
                "relative_path": relative_path,
                "local_path": str(output_path),
                "downloaded_now": downloaded,
                "exists": output_path.exists(),
                "size_mb": output_path.stat().st_size / 1024 / 1024 if output_path.exists() else 0,
            }
        )

    shots_path = paths.shots_dir / "shots_fixed.csv"
    downloaded = download_file("shots/shots_fixed.csv", shots_path, skipped)
    downloaded_rows.append(
        {
            "source_type": "shots_fixed",
            "relative_path": "shots/shots_fixed.csv",
            "local_path": str(shots_path),
            "downloaded_now": downloaded,
            "exists": shots_path.exists(),
            "size_mb": shots_path.stat().st_size / 1024 / 1024 if shots_path.exists() else 0,
        }
    )

    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(downloaded_rows)
    inventory.to_csv(paths.audit_dir / "nba_merge_download_inventory.csv", index=False)
    pd.DataFrame(skipped).to_csv(paths.audit_dir / "nba_merge_skipped_files.csv", index=False)
    return inventory


def inspect_movement_archive(movement_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive_rows = []
    head_rows = []
    for archive_path in sorted(movement_dir.glob("*.7z")):
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                names = archive.getnames()
                archive_rows.append(
                    {
                        "archive_name": archive_path.name,
                        "archive_path": str(archive_path),
                        "files_inside": ", ".join(names),
                        "readable": True,
                        "warning": "",
                    }
                )
                json_names = [name for name in names if name.lower().endswith(".json")]
                if json_names:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        archive.extract(path=tmpdir, targets=[json_names[0]])
                        extracted = Path(tmpdir) / json_names[0]
                        with extracted.open(encoding="utf-8") as f:
                            payload = json.load(f)
                    events = payload.get("events", [])
                    sample_event = events[0] if events else {}
                    moments = (
                        sample_event.get("moments", []) if isinstance(sample_event, dict) else []
                    )
                    head_rows.append(
                        {
                            "archive_name": archive_path.name,
                            "file_inside": json_names[0],
                            "gameid": payload.get("gameid"),
                            "events_count": len(events),
                            "sample_event_id": sample_event.get("eventId")
                            if isinstance(sample_event, dict)
                            else None,
                            "sample_moments_count": len(moments),
                            "sample_first_moment": str(moments[0])[:500] if moments else "",
                        }
                    )
                break
        except Exception as exc:
            archive_rows.append(
                {
                    "archive_name": archive_path.name,
                    "archive_path": str(archive_path),
                    "files_inside": "",
                    "readable": False,
                    "warning": str(exc),
                }
            )
            continue
    return pd.DataFrame(archive_rows), pd.DataFrame(head_rows)


def read_events(events_dir: Path, limit: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    frames = []
    for path in sorted(events_dir.glob("*.csv"))[:limit]:
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)
            rows.append(
                {
                    "file": path.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "readable": True,
                    "warning": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": path.name,
                    "rows": 0,
                    "columns": 0,
                    "readable": False,
                    "warning": str(exc),
                }
            )
    events = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return events, pd.DataFrame(rows)


def read_events_for_game_ids(
    events_dir: Path, game_ids: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    frames = []
    for game_id in sorted(game_ids):
        path = events_dir / f"{game_id:010d}.csv"
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)
            rows.append(
                {
                    "file": path.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "readable": True,
                    "warning": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": path.name,
                    "rows": 0,
                    "columns": 0,
                    "readable": False,
                    "warning": str(exc),
                }
            )
    events = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return events, pd.DataFrame(rows)


def extract_first_movement_jsons(
    movement_dir: Path,
    output_dir: Path,
    limit: int,
) -> tuple[list[Path], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files: list[Path] = []
    rows = []
    for archive_path in sorted(movement_dir.glob("*.7z")):
        if len(json_files) >= limit:
            break
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                names = [name for name in archive.getnames() if name.lower().endswith(".json")]
                if not names:
                    rows.append(
                        {
                            "archive_name": archive_path.name,
                            "game_id": pd.NA,
                            "matched_requested_game": False,
                            "json_path": "",
                            "warning": "no_json_inside",
                        }
                    )
                    continue
                json_name = names[0]
                output_path = output_dir / Path(json_name).name
                if not output_path.exists() or output_path.stat().st_size == 0:
                    archive.extract(path=output_dir, targets=[json_name])
                    extracted = output_dir / json_name
                    if extracted != output_path:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        extracted.replace(output_path)
                with output_path.open(encoding="utf-8") as f:
                    payload = json.load(f)
                game_id = int(payload.get("gameid", output_path.stem))
                json_files.append(output_path)
                rows.append(
                    {
                        "archive_name": archive_path.name,
                        "game_id": game_id,
                        "matched_requested_game": True,
                        "json_path": str(output_path),
                        "warning": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "archive_name": archive_path.name,
                    "game_id": pd.NA,
                    "matched_requested_game": False,
                    "json_path": "",
                    "warning": str(exc),
                }
            )
            continue
    return json_files, pd.DataFrame(rows)


def extract_movement_jsons_for_games(
    movement_dir: Path,
    output_dir: Path,
    game_ids: set[int],
) -> tuple[list[Path], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    found: dict[int, Path] = {}
    rows = []
    for archive_path in sorted(movement_dir.glob("*.7z")):
        if len(found) >= len(game_ids):
            break
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                names = [name for name in archive.getnames() if name.lower().endswith(".json")]
                if not names:
                    rows.append(
                        {
                            "archive_name": archive_path.name,
                            "game_id": pd.NA,
                            "matched_requested_game": False,
                            "json_path": "",
                            "warning": "no_json_inside",
                        }
                    )
                    continue
                json_name = names[0]
                output_path = output_dir / Path(json_name).name
                if not output_path.exists() or output_path.stat().st_size == 0:
                    archive.extract(path=output_dir, targets=[json_name])
                    extracted = output_dir / json_name
                    if extracted != output_path:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        extracted.replace(output_path)
                with output_path.open(encoding="utf-8") as f:
                    payload = json.load(f)
                game_id = int(payload.get("gameid", output_path.stem))
                matched = game_id in game_ids
                if matched:
                    found[game_id] = output_path
                rows.append(
                    {
                        "archive_name": archive_path.name,
                        "game_id": game_id,
                        "matched_requested_game": matched,
                        "json_path": str(output_path),
                        "warning": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "archive_name": archive_path.name,
                    "game_id": pd.NA,
                    "matched_requested_game": False,
                    "json_path": "",
                    "warning": str(exc),
                }
            )
            continue
    return list(found.values()), pd.DataFrame(rows)


def build_event_level_movement(
    paths: NbaMergePaths,
    game_ids: set[int],
    moment_stride: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    json_files, extraction_report = extract_movement_jsons_for_games(
        paths.movement_dir,
        paths.movement_json_dir,
        game_ids,
    )
    movement = parse_nba_event_level_movement(
        json_files,
        max_games=len(json_files),
        moment_stride=moment_stride,
    )
    if movement.empty:
        return movement, extraction_report
    movement = movement.rename(columns={"game_id_int": "GAME_ID", "event_id": "EVENTNUM"})
    return movement, extraction_report


def build_first_event_level_movement(
    paths: NbaMergePaths,
    limit: int,
    moment_stride: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    json_files, extraction_report = extract_first_movement_jsons(
        paths.movement_dir,
        paths.movement_json_dir,
        limit=limit,
    )
    movement = parse_nba_event_level_movement(
        json_files,
        max_games=len(json_files),
        moment_stride=moment_stride,
    )
    if movement.empty:
        return movement, extraction_report
    movement = movement.rename(columns={"game_id_int": "GAME_ID", "event_id": "EVENTNUM"})
    return movement, extraction_report


def column_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": null_count,
                "null_percent": null_count / len(df) if len(df) else 0,
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("null_percent", ascending=False)


def select_merge_keys(
    events_df: pd.DataFrame, shots_df: pd.DataFrame
) -> tuple[list[str], list[str], pd.DataFrame]:
    events_cols = set(events_df.columns)
    shots_cols = set(shots_df.columns)
    if {"GAME_ID", "EVENT_ID"}.issubset(events_cols) and {"GAME_ID", "EVENT_ID"}.issubset(
        shots_cols
    ):
        left_keys = ["GAME_ID", "EVENT_ID"]
        right_keys = ["GAME_ID", "EVENT_ID"]
        reason = "GAME_ID and EVENT_ID are present in both datasets."
    elif {"GAME_ID", "EVENTNUM"}.issubset(events_cols) and {
        "GAME_ID",
        "GAME_EVENT_ID",
    }.issubset(shots_cols):
        left_keys = ["GAME_ID", "EVENTNUM"]
        right_keys = ["GAME_ID", "GAME_EVENT_ID"]
        reason = "Using play-by-play EVENTNUM matched to shot chart GAME_EVENT_ID."
    elif "GAME_ID" in events_cols and "GAME_ID" in shots_cols:
        left_keys = ["GAME_ID"]
        right_keys = ["GAME_ID"]
        reason = (
            "EVENT_ID/EVENTNUM keys are absent or named differently; merged only by GAME_ID. "
            "This can create a large many-to-many join."
        )
    else:
        raise ValueError("Cannot merge NBA events and shots: GAME_ID is missing.")
    diagnostics = pd.DataFrame(
        [
            {
                "left_keys": " + ".join(left_keys),
                "right_keys": " + ".join(right_keys),
                "reason": reason,
                "events_columns": ", ".join(events_df.columns.astype(str)),
                "shots_columns": ", ".join(shots_df.columns.astype(str)),
            }
        ]
    )
    return left_keys, right_keys, diagnostics


def build_nba_events_shots_merge(
    paths: NbaMergePaths,
    max_events: int = 500,
    include_movement: bool = False,
    moment_stride: int = 50,
) -> dict[str, pd.DataFrame]:
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    movement_inventory, movement_head = inspect_movement_archive(paths.movement_dir)
    movement_df = pd.DataFrame()
    movement_extraction = pd.DataFrame()
    if include_movement:
        movement_df, movement_extraction = build_first_event_level_movement(
            paths,
            limit=max_events,
            moment_stride=moment_stride,
        )
        movement_game_ids = (
            set(movement_df["GAME_ID"].dropna().astype(int).unique().tolist())
            if "GAME_ID" in movement_df
            else set()
        )
        events_df, event_file_report = read_events_for_game_ids(paths.events_dir, movement_game_ids)
    else:
        events_df, event_file_report = read_events(paths.events_dir, limit=max_events)
    shots_path = paths.shots_dir / "shots_fixed.csv"
    shots_df = pd.read_csv(shots_path) if shots_path.exists() else pd.DataFrame()

    if events_df.empty:
        raise ValueError("No readable NBA events CSV files were found.")
    if shots_df.empty:
        raise ValueError(f"shots_fixed.csv is missing or empty: {shots_path}")

    left_keys, right_keys, merge_diagnostics = select_merge_keys(events_df, shots_df)
    merged = events_df.merge(
        shots_df,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffixes=("_event", "_shot"),
    )

    if include_movement:
        if not movement_df.empty:
            movement_feature_cols = [
                col for col in movement_df.columns if col not in {"game_id", "GAME_ID", "EVENTNUM"}
            ]
            movement_small = movement_df[["GAME_ID", "EVENTNUM", *movement_feature_cols]].copy()
            merged = merged.merge(
                movement_small,
                on=["GAME_ID", "EVENTNUM"],
                how="left",
                suffixes=("", "_movement"),
            )

    paths.merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(paths.merged_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "movement_archives_local",
                "value": len(list(paths.movement_dir.glob("*.7z"))),
            },
            {"metric": "events_files_readable", "value": int(event_file_report["readable"].sum())},
            {"metric": "events_shape", "value": str(events_df.shape)},
            {"metric": "shots_fixed_shape", "value": str(shots_df.shape)},
            {"metric": "movement_shape", "value": str(movement_df.shape)},
            {
                "metric": "movement_games_matched",
                "value": int(movement_df["GAME_ID"].nunique()) if "GAME_ID" in movement_df else 0,
            },
            {"metric": "merged_shape", "value": str(merged.shape)},
            {
                "metric": "merged_unique_GAME_ID",
                "value": int(merged["GAME_ID"].nunique()) if "GAME_ID" in merged else 0,
            },
            {
                "metric": "merge_keys",
                "value": f"{' + '.join(left_keys)} -> {' + '.join(right_keys)}",
            },
            {"metric": "merged_output", "value": str(paths.merged_csv)},
        ]
    )
    quality = column_quality(merged)
    missing = quality.sort_values("null_percent", ascending=False).head(50)

    outputs = {
        "movement_inventory": movement_inventory,
        "movement_head": movement_head,
        "events_head": events_df.head(50),
        "shots_fixed_head": shots_df.head(50),
        "event_file_report": event_file_report,
        "merge_diagnostics": merge_diagnostics,
        "movement_extraction": movement_extraction,
        "movement_features_head": movement_df.head(50),
        "merged_summary": summary,
        "merged_head": merged.head(50),
        "merged_column_quality": quality,
        "merged_top_missing_columns": missing,
    }
    for name, table in outputs.items():
        table.to_csv(paths.audit_dir / f"nba_merge_{name}.csv", index=False)
        table.to_csv(paths.report_dir / f"nba_merge_{name}.csv", index=False)
    return outputs


def run_nba_merge_pipeline(
    paths: NbaMergePaths,
    max_archives: int = 500,
    max_events: int = 500,
) -> dict[str, pd.DataFrame]:
    inventory = download_nba_sources(paths, max_archives=max_archives, max_events=max_events)
    reports = build_nba_events_shots_merge(paths, max_events=max_events)
    reports["download_inventory"] = inventory
    return reports
