from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import py7zr

from match_dynamics.nba import parse_nba_event_level_movement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_REPORT = (
    PROJECT_ROOT / "outputs" / "reports" / "nba_merge" / "nba_movement_archives_for_merged_50.csv"
)
JSON_REPORT = (
    PROJECT_ROOT / "outputs" / "reports" / "nba_merge" / "nba_movement_json_for_merged_50.csv"
)
MOVEMENT_FEATURES = PROJECT_ROOT / "data" / "nba" / "nba_movement_event_features_merged_50.csv"
OUTPUT_MERGED = PROJECT_ROOT / "data" / "nba" / "nba_events_shots_movement_merged_50.csv"


def extract_jsons() -> list[Path]:
    start = time.perf_counter()
    report = pd.read_csv(ARCHIVE_REPORT)
    names = report.loc[
        report["status"].isin(["already_exists", "downloaded"]), "archive_name"
    ].dropna()
    output_dir = PROJECT_ROOT / "data" / "nba" / "movement_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    json_paths: list[Path] = []
    print(f"{time.perf_counter() - start:.1f}s extracting archives={len(names)}", flush=True)
    for idx, name in enumerate(names, 1):
        archive_path = PROJECT_ROOT / "data" / "nba" / "movement" / str(name)
        try:
            with py7zr.SevenZipFile(archive_path, "r") as archive:
                json_names = [n for n in archive.getnames() if n.lower().endswith(".json")]
                if not json_names:
                    raise RuntimeError("no json inside")
                json_name = json_names[0]
                target = output_dir / Path(json_name).name
                if not target.exists() or target.stat().st_size == 0:
                    archive.extract(path=output_dir, targets=[json_name])
                    extracted = output_dir / json_name
                    if extracted != target:
                        extracted.replace(target)
            with target.open(encoding="utf-8") as f:
                payload = json.load(f)
            game_id = int(payload.get("gameid", target.stem))
            rows.append(
                {
                    "archive_name": name,
                    "GAME_ID": game_id,
                    "json_path": str(target),
                    "status": "ok",
                }
            )
            json_paths.append(target)
            print(
                f"{time.perf_counter() - start:.1f}s [{idx}/{len(names)}] {name} -> {target.name}",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "archive_name": name,
                    "GAME_ID": pd.NA,
                    "json_path": "",
                    "status": f"failed:{exc}",
                }
            )
            print(
                f"{time.perf_counter() - start:.1f}s [{idx}/{len(names)}] failed {name}: {exc}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(JSON_REPORT, index=False)
    print(
        f"{time.perf_counter() - start:.1f}s extraction complete jsons={len(json_paths)}",
        flush=True,
    )
    return json_paths


def main() -> None:
    start = time.perf_counter()
    json_paths = extract_jsons()
    print(f"{time.perf_counter() - start:.1f}s parsing movement features stride=200", flush=True)
    movement = parse_nba_event_level_movement(
        json_paths, max_games=len(json_paths), moment_stride=200
    )
    movement = movement.rename(columns={"game_id_int": "GAME_ID", "event_id": "EVENTNUM"})
    movement.to_csv(MOVEMENT_FEATURES, index=False)
    print(f"{time.perf_counter() - start:.1f}s movement shape={movement.shape}", flush=True)

    merged = pd.read_csv(PROJECT_ROOT / "data" / "nba" / "nba_events_shots_merged_50.csv")
    movement_cols = [c for c in movement.columns if c not in {"game_id", "GAME_ID", "EVENTNUM"}]
    final = merged.merge(
        movement[["GAME_ID", "EVENTNUM", *movement_cols]],
        on=["GAME_ID", "EVENTNUM"],
        how="left",
        suffixes=("", "_movement"),
    )
    final.to_csv(OUTPUT_MERGED, index=False)
    print(f"{time.perf_counter() - start:.1f}s final shape={final.shape}", flush=True)
    print(f"saved {OUTPUT_MERGED}", flush=True)


if __name__ == "__main__":
    main()
