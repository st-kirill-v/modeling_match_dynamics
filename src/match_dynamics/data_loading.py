from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from .config import NBA_MAX_ARCHIVES, ProjectConfig


LOCAL_WINDOWS_FOOTBALL_ZIP = Path(r"D:\Учеба\Глубокое обучение (DL)\Football Events.zip")
GOOGLE_DRIVE_FOOTBALL_ZIP = Path("/content/drive/MyDrive/Football Events.zip")


def _extract_or_place_football_file(src: Path, cfg: ProjectConfig) -> Path:
    cfg.football_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".csv":
        shutil.copy(src, cfg.football_events_path)
        return cfg.football_events_path
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as archive:
            archive.extractall(cfg.football_dir)
        csv_candidates = sorted(cfg.football_dir.rglob("events.csv")) or sorted(
            cfg.football_dir.rglob("*.csv")
        )
        if not csv_candidates:
            raise FileNotFoundError("No CSV file was found inside Football Events archive.")
        if csv_candidates[0].resolve() != cfg.football_events_path.resolve():
            shutil.copy(csv_candidates[0], cfg.football_events_path)
        return cfg.football_events_path
    raise ValueError(f"Unsupported Football Events file format: {src}")


def ensure_football_events(cfg: ProjectConfig) -> pd.DataFrame:
    cfg.ensure_dirs()
    if cfg.football_events_path.exists():
        return pd.read_csv(cfg.football_events_path)

    candidates = [
        cfg.football_path,
        Path("events.csv"),
        Path("Football Events.zip"),
        Path("/content/events.csv"),
        Path("/content/Football Events.zip"),
        GOOGLE_DRIVE_FOOTBALL_ZIP,
        LOCAL_WINDOWS_FOOTBALL_ZIP,
    ]
    for candidate in [c for c in candidates if c is not None]:
        if candidate.exists():
            prepared = _extract_or_place_football_file(candidate, cfg)
            return pd.read_csv(prepared)

    try:
        from google.colab import drive, files  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "events.csv was not found. In Colab upload Football Events.zip or use Google Drive. "
            "The local D: drive is available only in local Jupyter/VS Code, not in cloud Colab."
        ) from exc

    drive.mount("/content/drive", force_remount=False)
    if GOOGLE_DRIVE_FOOTBALL_ZIP.exists():
        prepared = _extract_or_place_football_file(GOOGLE_DRIVE_FOOTBALL_ZIP, cfg)
        return pd.read_csv(prepared)

    uploaded = files.upload()
    uploaded_names = list(uploaded.keys())
    data_files = [name for name in uploaded_names if name.lower().endswith((".zip", ".csv"))]
    if data_files:
        prepared = _extract_or_place_football_file(Path(data_files[0]), cfg)
        return pd.read_csv(prepared)

    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("Football Events.zip/events.csv/kaggle.json was not provided.")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    shutil.move("kaggle.json", kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "secareanualin/football-events",
            "--unzip",
            "-p",
            str(cfg.football_dir),
        ],
        check=True,
    )
    return pd.read_csv(cfg.football_events_path)


def ensure_nba_files(cfg: ProjectConfig, max_archives: int = NBA_MAX_ARCHIVES) -> list[Path]:
    if cfg.nba_json_dir is not None:
        json_files = sorted(cfg.nba_json_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No NBA JSON files found in {cfg.nba_json_dir}.")
        return json_files

    if cfg.skip_nba_download:
        return []

    repo_dir = cfg.nba_repo_dir or cfg.data_dir / "nba-movement-data"
    extract_dir = cfg.nba_extract_dir or cfg.data_dir / "nba_extracted"
    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/sealneaward/nba-movement-data.git",
                str(repo_dir),
            ],
            check=True,
        )

    data_dir = repo_dir / "data"
    archives = sorted(data_dir.glob("*.7z"))[:max_archives]
    if not archives:
        raise FileNotFoundError(f"No .7z NBA Movement Data archives found in {data_dir}.")

    extract_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(extract_dir.glob("*.json"))
    if json_files:
        return json_files

    if shutil.which("7z") is None:
        # TODO: On Windows, document a reliable 7-Zip installation path if 7z is not on PATH.
        raise RuntimeError("7z is required to extract NBA Movement Data archives.")

    for archive in archives:
        subprocess.run(["7z", "x", str(archive), f"-o{extract_dir}", "-y"], check=True)

    json_files = sorted(extract_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No NBA JSON files were found after extraction.")
    return json_files


def load_json_sample(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)
