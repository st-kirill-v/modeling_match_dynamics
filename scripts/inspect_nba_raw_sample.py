from __future__ import annotations

import argparse
import json
import shutil
import textwrap
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import py7zr


RAW_BASE = "https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data"

ARCHIVE_FILES = [
    "01.01.2016.CHA.at.TOR.7z",
    "01.01.2016.DAL.at.MIA.7z",
    "01.01.2016.NYK.at.CHI.7z",
    "01.01.2016.ORL.at.WAS.7z",
    "01.01.2016.PHI.at.LAL.7z",
]

EVENT_FILES = [
    "0021500490.csv",
    "0021500491.csv",
    "0021500492.csv",
    "0021500493.csv",
    "0021500494.csv",
]

SHOT_FILES = ["shots.csv", "shots_fixed.csv"]


def raw_url(relative_path: str) -> str:
    return f"{RAW_BASE}/{quote(relative_path, safe='/')}"


def download_file(relative_path: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path}")
        return

    url = raw_url(relative_path)
    print(f"[download] {relative_path}")
    with urllib.request.urlopen(url, timeout=120) as response, output_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    print(f"          saved {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def dataframe_summary(path: Path, max_rows: int = 5) -> str:
    df = pd.read_csv(path)
    head = df.head(max_rows).to_string(index=False)
    nulls = df.isna().sum().sort_values(ascending=False).head(10)
    return textwrap.dedent(
        f"""
        FILE: {path}
        SHAPE: {df.shape}
        COLUMNS ({len(df.columns)}):
        {list(df.columns)}

        HEAD:
        {head}

        TOP NULL COUNTS:
        {nulls.to_string()}
        """
    ).strip()


def inspect_archive(path: Path, extract_dir: Path) -> str:
    lines = [f"ARCHIVE: {path}", f"SIZE_MB: {path.stat().st_size / 1024 / 1024:.2f}"]
    extract_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(path, mode="r") as archive:
        names = archive.getnames()
    lines.append(f"FILES_INSIDE ({len(names)}): {names[:20]}")

    json_names = [name for name in names if name.lower().endswith(".json")]
    if not json_names:
        lines.append("No JSON files found inside archive.")
        return "\n".join(lines)

    sample_name = json_names[0]
    sample_output = extract_dir / sample_name
    if not sample_output.exists():
        with py7zr.SevenZipFile(path, mode="r") as archive:
            archive.extract(path=extract_dir, targets=[sample_name])

    with sample_output.open(encoding="utf-8") as f:
        sample = json.load(f)

    events = sample.get("events", [])
    lines.append(f"SAMPLE_JSON: {sample_name}")
    lines.append(f"TOP_LEVEL_KEYS: {list(sample.keys())}")
    lines.append(f"GAME_ID: {sample.get('gameid')}")
    lines.append(f"EVENTS_COUNT: {len(events)}")
    if events:
        first_event = events[0]
        moments = first_event.get("moments", [])
        lines.append(f"FIRST_EVENT_KEYS: {list(first_event.keys())}")
        lines.append(f"FIRST_EVENT_ID: {first_event.get('eventId')}")
        lines.append(f"FIRST_EVENT_MOMENTS_COUNT: {len(moments)}")
        if moments:
            first_moment = moments[0]
            players = first_moment[5] if len(first_moment) > 5 else []
            lines.append(
                "FIRST_MOMENT_SCHEMA_HINT: "
                "[quarter, unknown, game_clock, shot_clock, unknown, players]"
            )
            lines.append(f"FIRST_MOMENT_RAW_PREFIX: {first_moment[:5]}")
            lines.append(f"PLAYERS_IN_FIRST_MOMENT: {len(players)}")
            lines.append(f"FIRST_PLAYER_OR_BALL_ROW: {players[0] if players else None}")
    return "\n".join(lines)


def run(output_dir: Path, report_path: Path) -> None:
    archives_dir = output_dir / "archives"
    events_dir = output_dir / "events"
    shots_dir = output_dir / "shots"
    extract_dir = output_dir / "extracted_sample"

    for name in ARCHIVE_FILES:
        download_file(name, archives_dir / name)
    for name in EVENT_FILES:
        download_file(f"events/{name}", events_dir / name)
    for name in SHOT_FILES:
        download_file(f"shots/{name}", shots_dir / name)

    sections = ["# NBA raw sample inspection"]

    sections.append("\n## Downloaded archives")
    for name in ARCHIVE_FILES:
        sections.append(inspect_archive(archives_dir / name, extract_dir))

    sections.append("\n## Play-by-play events CSV files")
    for name in EVENT_FILES:
        sections.append(dataframe_summary(events_dir / name))

    sections.append("\n## Shot CSV files")
    for name in SHOT_FILES:
        sections.append(dataframe_summary(shots_dir / name))

    report = "\n\n" + "\n\n---\n\n".join(sections)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and inspect a small raw sample from nba-movement-data."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/nba_sample"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/metrics/nba_raw_sample_report.txt"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir, args.report_path)
