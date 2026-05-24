from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audits" / "data_quality"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def csv_path(name: str) -> Path:
    return AUDIT_DIR / name


def file_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_data(show_spinner=False)
def read_csv(path: str, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    return pd.read_csv(fpath)


@st.cache_data(show_spinner=False)
def read_csv_head(path: str, nrows: int, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    return pd.read_csv(fpath, nrows=nrows)


@st.cache_data(show_spinner=False)
def read_football_match_preview(path: str, nrows: int, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    cols = [
        "id_odsp",
        "date",
        "league",
        "season",
        "country",
        "ht",
        "at",
        "fthg",
        "ftag",
        "final_score",
    ]
    df = pd.read_csv(fpath, usecols=lambda col: col in cols)
    rows_per_match = df.groupby("id_odsp", dropna=False).size().rename("event_rows")
    matches = df.drop_duplicates("id_odsp").merge(rows_per_match, on="id_odsp", how="left")
    return matches.head(nrows)


@st.cache_data(show_spinner=False)
def read_event_dataset_summary(path: str, dataset_name: str, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(fpath, usecols=lambda col: col in ["id_odsp"])
    columns = len(pd.read_csv(fpath, nrows=0).columns)
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "rows": len(df),
                "columns": columns,
                "unique_matches": df["id_odsp"].nunique(dropna=True)
                if "id_odsp" in df.columns
                else pd.NA,
                "null_id_odsp": int(df["id_odsp"].isna().sum())
                if "id_odsp" in df.columns
                else pd.NA,
            }
        ]
    )


def compare_column_profiles(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty or after.empty:
        return pd.DataFrame()
    before_cols = before.set_index("column")
    after_cols = after.set_index("column")
    all_cols = sorted(set(before_cols.index) | set(after_cols.index))
    rows = []
    for col in all_cols:
        before_exists = col in before_cols.index
        after_exists = col in after_cols.index
        rows.append(
            {
                "column": col,
                "status": "created"
                if after_exists and not before_exists
                else "dropped"
                if before_exists and not after_exists
                else "kept",
                "dtype_before": before_cols.at[col, "dtype"] if before_exists else pd.NA,
                "dtype_after": after_cols.at[col, "dtype"] if after_exists else pd.NA,
                "missing_rate_before": before_cols.at[col, "missing_rate"]
                if before_exists
                else pd.NA,
                "missing_rate_after": after_cols.at[col, "missing_rate"] if after_exists else pd.NA,
                "n_unique_before": before_cols.at[col, "n_unique"] if before_exists else pd.NA,
                "n_unique_after": after_cols.at[col, "n_unique"] if after_exists else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def load_table(name: str) -> pd.DataFrame:
    path = csv_path(name)
    return read_csv(str(path), file_signature(path))


def load_metric_table(name: str) -> pd.DataFrame:
    path = METRICS_DIR / name
    return read_csv(str(path), file_signature(path))


def load_football_metric_table(name: str) -> pd.DataFrame:
    path = METRICS_DIR / "football" / name
    return read_csv(str(path), file_signature(path))


def load_report_table(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return read_csv(str(path), file_signature(path))


@st.cache_data(show_spinner=False)
def read_direct_csv_head(path: str, nrows: int, stamp: str) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    return pd.read_csv(fpath, nrows=nrows)


@st.cache_data(show_spinner=False)
def direct_nba_file_inventory(stamp: str) -> pd.DataFrame:
    nba_dir = PROJECT_ROOT / "data" / "nba"
    rows = []
    for source, folder, pattern in [
        ("movement", nba_dir / "movement", "*.7z"),
        ("events", nba_dir / "events", "*.csv"),
        ("shots", nba_dir / "shots", "*.csv"),
    ]:
        files = sorted(folder.glob(pattern)) if folder.exists() else []
        rows.append(
            {
                "source": source,
                "folder": str(folder),
                "files": len(files),
                "total_size_mb": sum(path.stat().st_size for path in files) / 1024 / 1024,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def inspect_direct_movement_archive(stamp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    movement_dir = PROJECT_ROOT / "data" / "nba" / "movement"
    archive_files = sorted(movement_dir.glob("*.7z")) if movement_dir.exists() else []
    if not archive_files:
        return pd.DataFrame(), pd.DataFrame()
    archive_path = archive_files[0]
    try:
        import json
        import tempfile

        import py7zr

        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            names = archive.getnames()
            json_names = [name for name in names if name.lower().endswith(".json")]
            inventory = pd.DataFrame(
                [
                    {
                        "archive_name": archive_path.name,
                        "archive_path": str(archive_path),
                        "files_inside": ", ".join(names),
                        "readable": True,
                        "warning": "",
                    }
                ]
            )
            if not json_names:
                return inventory, pd.DataFrame()
            with tempfile.TemporaryDirectory() as tmpdir:
                archive.extract(path=tmpdir, targets=[json_names[0]])
                extracted = Path(tmpdir) / json_names[0]
                with extracted.open(encoding="utf-8") as f:
                    payload = json.load(f)
            events = payload.get("events", [])
            sample_event = events[0] if events else {}
            moments = sample_event.get("moments", []) if isinstance(sample_event, dict) else []
            head = pd.DataFrame(
                [
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
                ]
            )
            return inventory, head
    except Exception as exc:
        inventory = pd.DataFrame(
            [
                {
                    "archive_name": archive_path.name,
                    "archive_path": str(archive_path),
                    "files_inside": "",
                    "readable": False,
                    "warning": str(exc),
                }
            ]
        )
        return inventory, pd.DataFrame()


def direct_column_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
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


def show_missing_bar(profile: pd.DataFrame, title: str, limit: int = 30) -> None:
    if profile.empty or "missing_rate" not in profile.columns:
        st.info("Нет данных для графика пропусков.")
        return
    plot_df = (
        profile.sort_values("missing_rate", ascending=False)
        .head(limit)
        .sort_values("missing_rate", ascending=True)
    )
    fig = px.bar(
        plot_df,
        x="missing_rate",
        y="column",
        orientation="h",
        title=title,
        hover_data=[c for c in ["missing", "non_null", "dtype", "n_unique"] if c in plot_df],
    )
    fig.update_layout(height=max(420, 24 * len(plot_df)), xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def show_dtype_bar(profile: pd.DataFrame, title: str) -> None:
    if profile.empty or "dtype" not in profile.columns:
        return
    counts = profile["dtype"].value_counts().reset_index()
    counts.columns = ["dtype", "columns"]
    fig = px.bar(counts, x="dtype", y="columns", title=title, text="columns")
    st.plotly_chart(fig, use_container_width=True)


def show_profile_table(profile: pd.DataFrame) -> None:
    if profile.empty:
        st.warning("Таблица не найдена. Сначала запусти аудит данных.")
        return
    search = st.text_input("Фильтр по названию колонки", "")
    view = profile
    if search:
        view = view[view["column"].astype(str).str.contains(search, case=False, na=False)]
    st.dataframe(view, use_container_width=True, height=420)


def show_head_table(name: str, title: str) -> None:
    df = load_table(name)
    st.subheader(title)
    if df.empty:
        st.warning(f"Файл {name} не найден.")
        return
    st.dataframe(df, use_container_width=True, height=360)


def show_overview() -> None:
    st.header("Анализ и моделирование динамики спортивных событий")
    st.markdown(
        """
        ### Цель работы

        Применить методы глубокого обучения для анализа данных о спортивных событиях,
        моделирования динамики матчей и прогнозирования ключевых событий на основе
        временных рядов.

        В рамках текущего проекта основной фокус сделан на задаче:

        **Моделирование динамики матчей**
        Прогнозировать изменение счета или ключевые события на основе последовательности
        событий внутри матча.

        Этапы работы:
        1. Подготовить данные из Football Events и NBA Tracking как временные ряды событий.
        2. Построить LSTM-модель для анализа последовательностей.
        3. Обучить модель на исторических данных.
        4. Оценить качество прогноза.
        """
    )

    st.subheader("Football Events")
    st.markdown(
        """
        **Источник:** [Kaggle: Football Events](https://www.kaggle.com/datasets/secareanualin/football-events)

        В проекте используются два основных файла:
        - `events.csv` — event-level данные: события матча, минута, сторона, тип события,
          игроки, удары, передачи, карточки и другие игровые действия.
        - `ginf.csv` — match-level данные: команды, лига, сезон, страна, дата матча,
          финальный счет и betting odds.

        Что было сделано с football data:
        - выполнен корректный merge `events.csv` + `ginf.csv` по `id_odsp`;
        - event-level данные преобразованы в minute-level формат;
        - одна строка теперь соответствует одной минуте одного матча;
        - добавлены признаки первого тайма, rolling/momentum/context features;
        - добавлены historical team-strength features без leakage;
        - построены targets второго тайма:
          `home_scores_next_half` и `away_scores_next_half`;
        - для LSTM используется только первый тайм: `time <= 45`.

        Текущая football задача:

        **по событиям первого тайма предсказать, забьют ли хозяева и гости во втором тайме.**
        """
    )

    st.subheader("NBA Player Tracking")
    st.markdown(
        """
        **Источник:** [GitHub: nba-movement-data](https://github.com/sealneaward/nba-movement-data)

        Раздел NBA оставлен как отдельная часть проекта. Здесь будут использоваться
        tracking данные игроков и мяча, play-by-play события и shot данные.
        """
    )

    st.subheader("Быстрая сводка по локальным датасетам")
    overview = load_table("dataset_overview.csv")
    if overview.empty:
        st.warning("Аудит еще не сгенерирован. Нажми кнопку обновления в sidebar.")
        return

    cols = st.columns(4)
    cols[0].metric("Datasets", len(overview))
    cols[1].metric("Total rows", f"{int(overview['rows'].sum()):,}".replace(",", " "))
    cols[2].metric("Total columns", int(overview["columns"].sum()))
    cols[3].metric("Avg missing rate", f"{overview['missing_cell_rate'].mean():.1%}")

    st.dataframe(overview, use_container_width=True)

    fig = px.bar(
        overview.sort_values("missing_cell_rate"),
        x="missing_cell_rate",
        y="dataset",
        orientation="h",
        title="Missing Cell Rate By Dataset",
        text=overview.sort_values("missing_cell_rate")["missing_cell_rate"].map(
            lambda x: f"{x:.1%}"
        ),
    )
    fig.update_layout(height=430, xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    size_df = overview.melt(
        id_vars="dataset",
        value_vars=["rows", "columns"],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        size_df, x="dataset", y="value", color="metric", barmode="group", title="Rows / Columns"
    )
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)


def show_conclusion() -> None:
    st.header("Conclusion")

    st.subheader("Футбол")
    st.markdown(
        """
        ### Итоговый вывод по football LSTM pipeline

        Был построен полноценный pipeline прогнозирования событий второго тайма по событиям
        первого тайма футбольного матча.

        Pipeline включает:
        - merge и preprocessing event data;
        - leakage-safe temporal split;
        - advanced feature engineering;
        - historical team-strength features;
        - rolling/momentum/context features;
        - sequence construction для LSTM;
        - feature ablation;
        - threshold tuning;
        - calibration;
        - error analysis.

        Финальная модель:
        - multi-output LSTM;
        - top-50 features;
        - sequence length = 45 минут;
        - prediction targets:
          `home_scores_next_half`, `away_scores_next_half`.

        Финальные thresholds:
        - HOME = 0.47
        - AWAY = 0.49

        Главный результат:
        модель научилась извлекать реальный signal из match dynamics и historical team context.

        Наиболее информативными оказались:
        - team strength;
        - team form;
        - first-half pressure;
        - momentum/intensity features;
        - contextual targeted features.

        Финальные test результаты:

        **HOME**
        - ROC-AUC ≈ 0.65
        - PR-AUC ≈ 0.72
        - MAE = 0.4678
        - MSE = 0.2313
        - RMSE = 0.4810

        **AWAY**
        - ROC-AUC ≈ 0.58-0.59
        - PR-AUC ≈ 0.57-0.60
        - MAE = 0.4893
        - MSE = 0.2466
        - RMSE = 0.4966

        Важно:
        так как задача бинарная, MSE по вероятностям эквивалентен Brier score.
        Чем ниже значение, тем лучше калиброваны вероятности модели.

        По сравнению с baseline:
        - HOME улучшился: MSE 0.2338 -> 0.2313
        - AWAY немного ухудшился: MSE 0.2455 -> 0.2466

        Home-модель показала стабильное качество:
        - высокий recall;
        - устойчивые probabilities;
        - meaningful signal из historical strength и pressure dynamics.

        Away-модель оказалась значительно сложнее:
        - away second-half goals имеют высокий уровень случайности и variance;
        - модель склонна к false positives;
        - targeted features улучшили recall, но не решили полностью проблему precision.

        Error analysis показал, что модель ведет себя логично:
        - переоценивает сильные домашние команды;
        - хуже работает в low-tempo матчах и при счете 0:0 после первого тайма;
        - корректно использует pressure и historical strength signals.

        Calibration не дал устойчивого улучшения, поэтому финальная версия использует
        raw probabilities.

        Главный вывод:
        получен методологически корректный и честный sports forecasting pipeline без leakage
        и с realistic temporal evaluation.

        Результаты нельзя считать “идеальными”, но они являются реалистичными для noisy
        football forecasting задачи без:
        - xG;
        - player tracking;
        - live odds;
        - lineup quality;
        - betting market features.

        Pipeline можно считать успешным baseline-решением для дальнейших исследований
        sports analytics и sequence modeling.
        """
    )

    st.subheader("NBA")
    st.markdown("Раздел NBA будет дополнен отдельно после финализации basketball pipeline.")


def show_football_merge() -> None:
    st.header("Football Merge: events.csv + ginf.csv")
    st.caption(
        "One match row from ginf.csv is left-joined to many event rows from events.csv by id_odsp. "
        "Duplicated match-level values per event are expected for event-level ML pipelines."
    )

    summary = load_table("football_merge_summary.csv")
    profile = load_table("football_merged_event_match_columns.csv")

    if summary.empty:
        st.warning("Merge tables not found. Click `Refresh audit tables` in the sidebar.")
        return

    st.subheader("Merge summary")
    st.dataframe(summary, use_container_width=True)

    merged_row = summary[summary["dataset"].eq("merged")]
    if not merged_row.empty:
        cols = st.columns(4)
        cols[0].metric("Merged rows", f"{int(merged_row['rows'].iloc[0]):,}".replace(",", " "))
        cols[1].metric("Merged columns", int(merged_row["columns"].iloc[0]))
        cols[2].metric("Unique matches", int(merged_row["unique_matches"].iloc[0]))
        if "matched_rate" in merged_row:
            cols[3].metric("Matched rows", f"{float(merged_row['matched_rate'].iloc[0]):.1%}")

    st.subheader("Merged event-level head() with all columns")
    head_rows = st.slider("Rows to show from data/football_merged.csv", 5, 200, 30)
    merged_path = PROJECT_ROOT / "data" / "football_merged.csv"
    merged_head = read_csv_head(str(merged_path), head_rows, file_signature(merged_path))
    if merged_head.empty:
        st.warning(
            "data/football_merged.csv не найден. Сначала запусти merge или Refresh audit tables."
        )
    else:
        st.caption(
            "Это первые строки полного event-level merge: все колонки events.csv + match-level колонки ginf.csv."
        )
        st.dataframe(merged_head, use_container_width=True, height=560)

    st.subheader("Compact merged preview: matches")
    match_rows = st.slider(
        "Rows to show from match-level preview",
        30,
        200,
        50,
        key="football_merge_match_preview_rows",
    )
    match_preview = read_football_match_preview(
        str(merged_path), match_rows, file_signature(merged_path)
    )
    if match_preview.empty:
        st.warning(
            "Match preview is unavailable. Refresh audit tables or rebuild football_merged.csv."
        )
    else:
        st.caption(
            "Каждая строка здесь - один матч, а `event_rows` показывает число событий в матче."
        )
        st.dataframe(match_preview, use_container_width=True, height=460)

    st.subheader("Merged columns: data types and quality")
    if profile.empty:
        st.warning("Профиль колонок merge не найден.")
    else:
        dtype_cols = [
            c
            for c in [
                "column",
                "dtype",
                "non_null",
                "missing",
                "missing_rate",
                "n_unique",
                "zero_count",
                "zero_rate",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
            if c in profile.columns
        ]
        st.dataframe(profile[dtype_cols], use_container_width=True, height=520)

    show_missing_bar(profile, "Football Merged: Top Missing Columns")
    show_dtype_bar(profile, "Football Merged: Column Types")


def show_football_merged_processed() -> None:
    st.header("Football Merged Processed: football_merged_processed.csv")
    st.caption(
        "Same view structure as Football Merge, but using the processed minute-level dataset."
    )

    processed_path = PROJECT_ROOT / "data" / "football_merged_processed.csv"
    processed_stamp = file_signature(processed_path)
    summary = read_event_dataset_summary(
        str(processed_path), "football_merged_processed", processed_stamp
    )
    profile = load_table("football_merged_processed_columns.csv")

    if summary.empty:
        st.warning(
            "data/football_merged_processed.csv not found. Run processing or click `Refresh audit tables`."
        )
        return

    st.subheader("Processed summary")
    st.dataframe(summary, use_container_width=True)

    processed_row = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Processed rows", f"{int(processed_row['rows']):,}".replace(",", " "))
    cols[1].metric("Processed columns", int(processed_row["columns"]))
    cols[2].metric("Unique matches", int(processed_row["unique_matches"]))
    cols[3].metric("Null match ids", int(processed_row["null_id_odsp"]))

    st.subheader("Processed minute-level head() with all columns")
    head_rows = st.slider(
        "Rows to show from data/football_merged_processed.csv",
        5,
        200,
        30,
        key="football_merged_processed_head_rows",
    )
    processed_head = read_csv_head(str(processed_path), head_rows, processed_stamp)
    if processed_head.empty:
        st.warning("data/football_merged_processed.csv not found. Run processing or refresh audit.")
    else:
        st.caption("First rows of the full processed minute-level dataset with all columns.")
        st.dataframe(processed_head, use_container_width=True, height=560)

    st.subheader("Compact processed preview: matches")
    match_rows = st.slider(
        "Rows to show from processed match-level preview",
        30,
        200,
        50,
        key="football_processed_match_preview_rows",
    )
    match_preview = read_football_match_preview(str(processed_path), match_rows, processed_stamp)
    if match_preview.empty:
        st.warning(
            "Processed match preview is unavailable. Refresh audit tables or rebuild processed file."
        )
    else:
        st.caption("Each row is one match; `event_rows` is the number of first-half minute rows.")
        st.dataframe(match_preview, use_container_width=True, height=460)

    st.subheader("Processed columns: data types and quality")
    if profile.empty:
        st.warning("Processed column profile not found.")
    else:
        dtype_cols = [
            c
            for c in [
                "column",
                "dtype",
                "non_null",
                "missing",
                "missing_rate",
                "n_unique",
                "zero_count",
                "zero_rate",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
            if c in profile.columns
        ]
        st.dataframe(profile[dtype_cols], use_container_width=True, height=520)

    show_missing_bar(profile, "Football Merged Processed: Top Missing Columns")
    show_dtype_bar(profile, "Football Merged Processed: Column Types")

    st.subheader("Target distribution")
    target_dist = load_report_table("football_target_distribution.csv")
    if target_dist.empty:
        st.info("Target distribution report is not ready. Run target analysis first.")
    else:
        st.dataframe(target_dist, use_container_width=True)
        fig = px.bar(
            target_dist,
            x="value",
            y="matches",
            color="target",
            barmode="group",
            title="Target Distribution By Matches",
            text="matches",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature-target correlations")
    corr = load_report_table("football_feature_target_correlations.csv")
    if corr.empty:
        st.info("Correlation report is not ready. Run target analysis first.")
    else:
        st.dataframe(corr, use_container_width=True, height=420)
        for target in ["home_scores_next_half", "away_scores_next_half"]:
            plot_df = (
                corr[corr["target"].eq(target)]
                .sort_values("abs_correlation", ascending=False)
                .head(20)
                .sort_values("abs_correlation", ascending=True)
            )
            if plot_df.empty:
                continue
            fig = px.bar(
                plot_df,
                x="correlation",
                y="feature",
                orientation="h",
                title=f"Top 20 Features vs {target}",
                hover_data=["abs_correlation"],
            )
            fig.update_layout(height=max(460, 24 * len(plot_df)))
            st.plotly_chart(fig, use_container_width=True)


def show_football_merged_feature_engineering() -> None:
    st.header("Football Merged Feature Engineering: football_merged_feature_engineering.csv")
    st.caption(
        "Working copy for feature engineering. From this point, feature changes should happen here, "
        "while football_merged_processed.csv stays as the clean processed baseline."
    )

    feature_path = PROJECT_ROOT / "data" / "football_merged_feature_engineering.csv"
    feature_stamp = file_signature(feature_path)
    summary = read_event_dataset_summary(
        str(feature_path), "football_merged_feature_engineering", feature_stamp
    )

    if summary.empty:
        st.warning(
            "data/football_merged_feature_engineering.csv not found. Create it from the processed dataset first."
        )
        return

    st.subheader("Feature engineering dataset summary")
    st.dataframe(summary, use_container_width=True)

    row = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Rows", f"{int(row['rows']):,}".replace(",", " "))
    cols[1].metric("Columns", int(row["columns"]))
    cols[2].metric("Unique matches", int(row["unique_matches"]))
    cols[3].metric("Null match ids", int(row["null_id_odsp"]))

    st.subheader("head() with all columns")
    head_rows = st.slider(
        "Rows to show from data/football_merged_feature_engineering.csv",
        5,
        200,
        30,
        key="football_merged_feature_engineering_head_rows",
    )
    head = read_csv_head(str(feature_path), head_rows, feature_stamp)
    st.dataframe(head, use_container_width=True, height=620)

    st.subheader("Feature engineering reports")
    summary = load_report_table("football_merged_feature_engineering_historical_summary.csv")
    validation = load_report_table("football_merged_feature_engineering_historical_validation.csv")
    created = load_report_table(
        "football_merged_feature_engineering_historical_created_features.csv"
    )
    if not summary.empty:
        st.markdown("**Historical team-strength summary**")
        st.dataframe(summary, use_container_width=True)
    if not validation.empty:
        st.markdown("**Historical leakage validation**")
        st.dataframe(validation, use_container_width=True)
    if not created.empty:
        with st.expander("Historical features created"):
            st.dataframe(created, use_container_width=True, height=360)

    st.subheader("Target distribution")
    target_dist = load_report_table("football_merged_feature_engineering_target_distribution.csv")
    if target_dist.empty:
        st.info("Target distribution is not ready. Run football feature engineering first.")
    else:
        st.dataframe(target_dist, use_container_width=True)
        fig = px.bar(
            target_dist,
            x="value",
            y="matches",
            color="target",
            barmode="group",
            title="Target Distribution By Matches",
            text="matches",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance proxy: correlation with targets")
    corr = load_report_table("football_merged_feature_engineering_feature_target_correlations.csv")
    if corr.empty:
        st.info("Feature-target correlation report is not ready.")
    else:
        st.dataframe(corr, use_container_width=True, height=420)
        for top_n in [20, 30, 40]:
            st.markdown(f"**Top {top_n} absolute correlations**")
            for target in ["home_scores_next_half", "away_scores_next_half"]:
                plot_df = (
                    corr[corr["target"].eq(target)]
                    .sort_values("abs_correlation", ascending=False)
                    .head(top_n)
                    .sort_values("abs_correlation", ascending=True)
                )
                if plot_df.empty:
                    continue
                fig = px.bar(
                    plot_df,
                    x="correlation",
                    y="feature",
                    orientation="h",
                    title=f"Top {top_n} Features vs {target}",
                    hover_data=["abs_correlation"],
                )
                fig.update_layout(height=max(520, 22 * len(plot_df)))
                st.plotly_chart(fig, use_container_width=True)


def show_nba_raw() -> None:
    st.header("NBA Raw")
    inventory = load_table("nba_raw_file_inventory.csv")
    movement = load_table("nba_raw_movement_json_sample.csv")

    st.subheader("Raw file inventory")
    st.dataframe(inventory, use_container_width=True)
    if not inventory.empty:
        fig = px.bar(
            inventory, x="source", y="files", title="NBA Raw Files By Source", text="files"
        )
        st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(
            inventory,
            x="source",
            y="total_size_mb",
            title="NBA Raw Size By Source, MB",
            text="total_size_mb",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Movement JSON sample")
    st.dataframe(movement, use_container_width=True)

    raw_kind = st.radio("Raw table", ["events", "shots"], horizontal=True)
    if raw_kind == "events":
        show_head_table("nba_raw_events_sample_head.csv", "NBA raw events head()")
        profile = load_table("nba_raw_events_sample_columns.csv")
        show_missing_bar(profile, "NBA Raw Events: Top Missing Columns")
    else:
        show_head_table("nba_raw_shots_sample_head.csv", "NBA raw shots head()")
        profile = load_table("nba_raw_shots_sample_columns.csv")
        show_missing_bar(profile, "NBA Raw Shots: Top Missing Columns")
    show_dtype_bar(profile, f"NBA Raw {raw_kind}: Column Types")
    st.subheader("Column profile")
    show_profile_table(profile)


def show_nba_merge() -> None:
    st.header("NBA Merge")
    st.caption(
        "NBA raw inspection page. Merge is intentionally paused for now; movement archives are "
        "downloaded and inspected only, and events/shots_fixed are shown separately."
    )

    download_inventory = load_table("nba_merge_download_inventory.csv")
    skipped = load_table("nba_merge_skipped_files.csv")
    movement_inventory = load_table("nba_merge_movement_inventory.csv")
    movement_head = load_table("nba_merge_movement_head.csv")
    events_head = load_table("nba_merge_events_head.csv")
    shots_head = load_table("nba_merge_shots_fixed_head.csv")
    merge_diagnostics = load_table("nba_merge_merge_diagnostics.csv")
    merged_summary = load_table("nba_merge_merged_summary.csv")
    merged_head = load_table("nba_merge_merged_head.csv")
    quality = load_table("nba_merge_merged_column_quality.csv")
    top_missing = load_table("nba_merge_merged_top_missing_columns.csv")
    event_file_report = load_table("nba_merge_event_file_report.csv")

    if merged_summary.empty:
        st.info(
            "Merge is intentionally not built yet. Showing raw downloaded NBA sources from `data/nba`."
        )
        raw_stamp = file_signature(PROJECT_ROOT / "data" / "nba" / "shots" / "shots_fixed.csv")
        direct_inventory = direct_nba_file_inventory(raw_stamp)
        st.subheader("Downloaded raw NBA files")
        st.dataframe(direct_inventory, use_container_width=True, height=180)
        if not direct_inventory.empty:
            fig = px.bar(
                direct_inventory,
                x="files",
                y="source",
                orientation="h",
                text="files",
                title="NBA raw files currently available locally",
            )
            st.plotly_chart(fig, use_container_width=True)

        movement_inventory, movement_head = inspect_direct_movement_archive(raw_stamp)
        st.subheader("Movement sample head()")
        st.markdown("Movement пока не мержим: это тяжелый tracking source, сейчас только sample.")
        st.dataframe(movement_inventory, use_container_width=True, height=220)
        st.dataframe(movement_head, use_container_width=True, height=260)

        events_dir = PROJECT_ROOT / "data" / "nba" / "events"
        event_files = sorted(events_dir.glob("*.csv")) if events_dir.exists() else []
        event_path = event_files[0] if event_files else None
        events_df = (
            read_direct_csv_head(str(event_path), 50, file_signature(event_path))
            if event_path is not None
            else pd.DataFrame()
        )
        st.subheader("Events head()")
        if event_path is not None:
            st.caption(f"Sample file: `{event_path.name}`")
        st.dataframe(events_df, use_container_width=True, height=320)
        st.subheader("Events column quality")
        st.dataframe(direct_column_quality(events_df), use_container_width=True, height=320)

        shots_path = PROJECT_ROOT / "data" / "nba" / "shots" / "shots_fixed.csv"
        shots_df = read_direct_csv_head(str(shots_path), 50, file_signature(shots_path))
        st.subheader("Shots fixed head()")
        st.dataframe(shots_df, use_container_width=True, height=320)
        st.subheader("Shots fixed column quality")
        st.dataframe(direct_column_quality(shots_df), use_container_width=True, height=320)

        st.subheader("Future merge keys")
        st.markdown(
            """
            Для будущего корректного merge нельзя соединять только по `GAME_ID`, потому что это
            создаёт huge many-to-many join.

            Правильная логика:
            - `events.GAME_ID` = `shots_fixed.GAME_ID`
            - `events.EVENTNUM` = `shots_fixed.GAME_EVENT_ID`
            """
        )
        return

    st.subheader("Download inventory")
    st.dataframe(download_inventory, use_container_width=True, height=260)
    if not download_inventory.empty:
        fig = px.bar(
            download_inventory.groupby("source_type", as_index=False)
            .agg(files=("local_path", "count"), size_mb=("size_mb", "sum"))
            .sort_values("files"),
            x="files",
            y="source_type",
            orientation="h",
            text="files",
            title="Downloaded / cached NBA source files",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not skipped.empty:
        with st.expander("Skipped / cached / failed files"):
            st.dataframe(skipped, use_container_width=True, height=260)

    st.subheader("Movement sample head()")
    st.markdown("Movement пока не входит в основной merge, потому что tracking data тяжелые.")
    st.dataframe(movement_inventory, use_container_width=True, height=220)
    st.dataframe(movement_head, use_container_width=True, height=260)

    st.subheader("Events head()")
    st.dataframe(events_head, use_container_width=True, height=320)
    with st.expander("Events file read diagnostics"):
        st.dataframe(event_file_report, use_container_width=True, height=320)

    st.subheader("Shots fixed head()")
    st.dataframe(shots_head, use_container_width=True, height=320)

    st.subheader("Merge diagnostics")
    st.dataframe(merge_diagnostics, use_container_width=True, height=220)
    st.dataframe(merged_summary, use_container_width=True, height=260)

    st.subheader("Merged head()")
    st.dataframe(merged_head, use_container_width=True, height=360)

    st.subheader("Merged columns: data types and quality")
    st.dataframe(quality, use_container_width=True, height=420)

    st.subheader("Top missing columns")
    st.dataframe(top_missing, use_container_width=True, height=320)
    if not top_missing.empty:
        plot_df = top_missing.sort_values("null_percent", ascending=True).tail(30)
        fig = px.bar(
            plot_df,
            x="null_percent",
            y="column",
            orientation="h",
            title="NBA merged top missing columns",
            text=plot_df["null_percent"].map(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=max(520, 20 * len(plot_df)), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def show_nba_processed() -> None:
    st.header("NBA Processed")
    table = st.radio(
        "Processed table", ["matched_processed", "final_score_checkpoint_5min"], horizontal=True
    )
    if table == "matched_processed":
        show_head_table("nba_matched_processed_head.csv", "NBA matched dataset head()")
        profile = load_table("nba_matched_processed_columns.csv")
        show_missing_bar(profile, "NBA Matched: Top Missing Columns")
    else:
        show_head_table("nba_final_score_checkpoint_5min_head.csv", "NBA checkpoint dataset head()")
        profile = load_table("nba_final_score_checkpoint_5min_columns.csv")
        show_missing_bar(profile, "NBA Checkpoint: Top Missing Columns")

    show_dtype_bar(profile, f"NBA {table}: Column Types")

    numeric_options = profile.loc[
        profile["dtype"].astype(str).str.contains("int|float", case=False, regex=True), "column"
    ].tolist()
    if numeric_options:
        selected = st.selectbox("Numeric column distribution", numeric_options)
        head_file = (
            "nba_matched_processed_head.csv"
            if table == "matched_processed"
            else "nba_final_score_checkpoint_5min_head.csv"
        )
        sample = load_table(head_file)
        if selected in sample.columns:
            fig = px.histogram(
                sample, x=selected, title=f"Distribution in saved sample: {selected}"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Column profile")
    show_profile_table(profile)


def show_join_quality() -> None:
    st.header("NBA Join Quality")
    join = load_table("nba_join_quality.csv")
    if join.empty:
        st.warning("nba_join_quality.csv не найден.")
        return
    st.dataframe(join, use_container_width=True)
    fig = px.bar(
        join.sort_values("available_rate"),
        x="available_rate",
        y="check",
        orientation="h",
        title="NBA Join / Availability Rate",
        text=join.sort_values("available_rate")["available_rate"].map(lambda x: f"{x:.1%}"),
        hover_data=["column", "available_rows", "missing_rows"],
    )
    fig.update_layout(height=440, xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


def show_feature_engineering() -> None:
    st.header("Feature Engineering")
    football_report = load_metric_table("football_feature_report.csv")
    nba_sequence = load_metric_table("nba_sequence_feature_report.csv")
    nba_checkpoint = load_metric_table("nba_checkpoint_feature_report.csv")

    dataset = st.radio(
        "Feature report", ["Football", "NBA sequence", "NBA checkpoint"], horizontal=True
    )
    if dataset == "Football":
        report = football_report
        importance_col = "mean_rf_importance"
        corr_col = "max_abs_corr"
    elif dataset == "NBA sequence":
        report = nba_sequence
        importance_col = "rf_importance"
        corr_col = "abs_corr_target"
    else:
        report = nba_checkpoint
        importance_col = "rf_importance"
        corr_col = "abs_corr_target"

    if report.empty:
        st.warning(
            "Feature report не найден. Запусти pipeline/feature-selection, чтобы его создать."
        )
        return

    st.dataframe(report, use_container_width=True, height=420)

    top_n = st.slider("Top N features", 5, min(50, len(report)), min(25, len(report)))
    top = (
        report.sort_values(importance_col, ascending=False).head(top_n).sort_values(importance_col)
    )
    fig = px.bar(
        top, x=importance_col, y="feature", orientation="h", title=f"{dataset}: Feature Importance"
    )
    fig.update_layout(height=max(420, 24 * len(top)))
    st.plotly_chart(fig, use_container_width=True)

    if corr_col in report.columns:
        corr_top = report.sort_values(corr_col, ascending=False).head(top_n).sort_values(corr_col)
        fig = px.bar(
            corr_top,
            x=corr_col,
            y="feature",
            orientation="h",
            title=f"{dataset}: Absolute Correlation",
        )
        fig.update_layout(height=max(420, 24 * len(corr_top)))
        st.plotly_chart(fig, use_container_width=True)


def show_correlations() -> None:
    st.header("Correlations")
    football = load_metric_table("football_feature_report.csv")
    nba = load_metric_table("nba_sequence_feature_report.csv")

    dataset = st.radio("Correlation table", ["Football", "NBA sequence"], horizontal=True)
    report = football if dataset == "Football" else nba
    if report.empty:
        st.warning("Correlation report не найден.")
        return

    corr_cols = [c for c in report.columns if "corr" in c]
    if not corr_cols:
        st.info("В этом отчете нет correlation columns.")
        return

    top_n = st.slider("Top correlated features", 10, min(60, len(report)), min(30, len(report)))
    sort_col = st.selectbox("Sort by", corr_cols)
    view = report.sort_values(sort_col, key=lambda s: s.abs(), ascending=False).head(top_n)
    st.dataframe(view[["feature", *corr_cols]], use_container_width=True, height=420)

    heat = view.set_index("feature")[corr_cols]
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(title=f"{dataset}: Correlation Heatmap", height=max(500, 24 * len(heat)))
    st.plotly_chart(fig, use_container_width=True)


def show_model_metrics() -> None:
    st.header("Model Metrics")
    football = load_metric_table("football_feature_selection_metrics.csv")
    nba = load_metric_table("nba_regression_metrics.csv")

    if not football.empty:
        st.subheader("Football feature-selection metrics")
        st.dataframe(football, use_container_width=True, height=360)
        metric = st.selectbox(
            "Football metric",
            [c for c in ["pr_auc", "roc_auc", "f1", "mae", "mse", "brier"] if c in football],
        )
        fig = px.bar(
            football.sort_values(metric, ascending=False),
            x=metric,
            y="model",
            color="feature_set" if "feature_set" in football else None,
            orientation="h",
            title=f"Football models by {metric}",
        )
        fig.update_layout(height=max(520, 24 * len(football)))
        st.plotly_chart(fig, use_container_width=True)

    if not nba.empty:
        st.subheader("NBA regression metrics")
        st.dataframe(nba, use_container_width=True)
        metric = st.selectbox("NBA metric", [c for c in ["mae", "mse", "rmse", "r2"] if c in nba])
        fig = px.bar(
            nba.sort_values(metric),
            x=metric,
            y="model",
            orientation="h",
            title=f"NBA models by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)


def show_football_metrics() -> None:
    st.header("Football Metrics")
    st.caption("Sequence dataset diagnostics and baseline multi-output LSTM metrics.")

    diagnostics = load_report_table("football_sequence_diagnostics.csv")
    target_dist = load_report_table("football_sequence_target_distribution.csv")
    seq_stats = load_report_table("football_sequence_sequence_length_stats.csv")
    features = load_report_table("football_sequence_feature_columns.csv")
    excluded = load_report_table("football_sequence_excluded_columns.csv")
    target_checks = load_report_table("football_sequence_target_checks.csv")
    lstm_metrics = load_football_metric_table("baseline_lstm_metrics.csv")
    lstm_history = load_football_metric_table("baseline_lstm_history.csv")
    lstm_shapes = load_football_metric_table("baseline_lstm_shapes.csv")
    overfit = load_football_metric_table("baseline_lstm_overfitting_report.csv")
    confusion = load_football_metric_table("baseline_lstm_confusion_matrices.csv")
    top50_retrain_confusion = load_football_metric_table("top50_retrain_confusion_matrices.csv")
    ablation_comparison = load_football_metric_table("feature_ablation_fast_comparison.csv")
    ablation_summary = load_football_metric_table("feature_ablation_fast_training_summary.csv")
    ablation_ranking = load_football_metric_table("feature_ablation_fast_feature_ranking.csv")
    if ablation_comparison.empty:
        ablation_comparison = load_football_metric_table("feature_ablation_comparison.csv")
    if ablation_summary.empty:
        ablation_summary = load_football_metric_table("feature_ablation_training_summary.csv")
    if ablation_ranking.empty:
        ablation_ranking = load_football_metric_table("feature_ablation_feature_ranking.csv")

    if not lstm_metrics.empty:
        st.subheader("Baseline multi-output LSTM metrics")
        st.dataframe(lstm_metrics, use_container_width=True, height=420)
        metric = st.selectbox(
            "Metric",
            [
                c
                for c in ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "mae", "mse"]
                if c in lstm_metrics
            ],
            key="football_lstm_metric_select",
        )
        fig = px.bar(
            lstm_metrics.sort_values(metric, ascending=False),
            x=metric,
            y="target",
            color="split",
            barmode="group",
            orientation="h",
            title=f"Baseline Football LSTM by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not lstm_history.empty:
        st.subheader("Training history")
        st.dataframe(lstm_history, use_container_width=True, height=320)
        loss_cols = [
            c
            for c in ["loss", "val_loss", "home_output_loss", "away_output_loss"]
            if c in lstm_history
        ]
        hist_long = lstm_history.melt(
            id_vars="epoch", value_vars=loss_cols, var_name="curve", value_name="value"
        )
        fig = px.line(
            hist_long, x="epoch", y="value", color="curve", title="Train / Validation Loss"
        )
        st.plotly_chart(fig, use_container_width=True)

    display_confusion = top50_retrain_confusion if not top50_retrain_confusion.empty else confusion
    if not display_confusion.empty:
        st.subheader("Confusion matrices")
        st.caption(
            "Showing top-50 retrain with final fixed thresholds when available; otherwise baseline."
        )
        st.dataframe(display_confusion, use_container_width=True, height=260)
        split_values = sorted(display_confusion["split"].unique().tolist())
        split = st.selectbox(
            "Confusion matrix split",
            split_values,
            index=split_values.index("test") if "test" in split_values else 0,
            key="football_confusion_split",
        )
        cols = st.columns(2)
        for idx, target in enumerate(["home_scores_next_half", "away_scores_next_half"]):
            matrix_df = display_confusion[
                display_confusion["split"].eq(split) & display_confusion["target"].eq(target)
            ].pivot(index="true_label", columns="predicted_label", values="count")
            matrix_df = matrix_df.reindex(index=[0, 1], columns=[0, 1]).fillna(0).astype(int)
            with cols[idx]:
                st.markdown(f"**{split}: {target}**")
                st.dataframe(matrix_df, use_container_width=True)
                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "True", "color": "Count"},
                    title=f"Confusion matrix: {target}",
                )
                st.plotly_chart(fig, use_container_width=True)

    if not ablation_comparison.empty:
        st.subheader("Feature-count ablation")
        st.dataframe(ablation_comparison, use_container_width=True, height=420)
        metric = st.selectbox(
            "Ablation metric",
            [c for c in ["pr_auc", "roc_auc", "f1", "mae", "mse"] if c in ablation_comparison],
            key="football_ablation_metric_select",
        )
        fig = px.bar(
            ablation_comparison.sort_values(metric, ascending=False),
            x=metric,
            y="feature_set",
            color="target",
            barmode="group",
            orientation="h",
            title=f"Feature-count ablation by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not ablation_summary.empty:
        st.subheader("Feature-count ablation training summary")
        st.dataframe(ablation_summary, use_container_width=True)

    if not ablation_ranking.empty:
        with st.expander("Feature ranking by aggregate absolute correlation"):
            st.dataframe(ablation_ranking, use_container_width=True, height=520)

    top50_retrain_metrics = load_football_metric_table("top50_retrain_metrics.csv")
    top50_retrain_final = load_football_metric_table("top50_retrain_final_threshold_metrics.csv")
    top50_retrain_summary = load_football_metric_table("top50_retrain_training_summary.csv")
    top50_retrain_history = load_football_metric_table("top50_retrain_top_50_history.csv")
    if not top50_retrain_metrics.empty:
        st.subheader("Top-50 LSTM retrain")
        st.caption("Fresh retrain of the selected top-50 feature model.")
        st.dataframe(top50_retrain_metrics, use_container_width=True, height=320)
    if not top50_retrain_final.empty:
        st.markdown("**Top-50 retrain with final fixed thresholds**")
        st.dataframe(top50_retrain_final, use_container_width=True, height=320)
    if not top50_retrain_summary.empty:
        st.dataframe(top50_retrain_summary, use_container_width=True)
    if not top50_retrain_history.empty:
        loss_cols = [c for c in ["loss", "val_loss"] if c in top50_retrain_history]
        if loss_cols:
            hist_long = top50_retrain_history.melt(
                id_vars="epoch", value_vars=loss_cols, var_name="curve", value_name="value"
            )
            fig = px.line(
                hist_long,
                x="epoch",
                y="value",
                color="curve",
                title="Top-50 retrain loss",
            )
            st.plotly_chart(fig, use_container_width=True)
    if not top50_retrain_confusion.empty:
        st.markdown("**Top-50 retrain confusion matrices with final fixed thresholds**")
        split_values = sorted(top50_retrain_confusion["split"].unique().tolist())
        split = st.selectbox(
            "Top-50 retrain confusion split",
            split_values,
            index=split_values.index("test") if "test" in split_values else 0,
            key="top50_retrain_confusion_split",
        )
        cols = st.columns(2)
        for idx, target in enumerate(["home_scores_next_half", "away_scores_next_half"]):
            matrix_df = top50_retrain_confusion[
                top50_retrain_confusion["split"].eq(split)
                & top50_retrain_confusion["target"].eq(target)
            ].pivot(index="true_label", columns="predicted_label", values="count")
            matrix_df = matrix_df.reindex(index=[0, 1], columns=[0, 1]).fillna(0).astype(int)
            with cols[idx]:
                st.markdown(f"**{split}: {target}**")
                st.dataframe(matrix_df, use_container_width=True)
                fig = px.imshow(
                    matrix_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "True", "color": "Count"},
                    title=f"Top-50 retrain confusion matrix: {target}",
                )
                st.plotly_chart(fig, use_container_width=True)

    threshold_best = load_football_metric_table("threshold_tuning/best_thresholds.csv")
    threshold_metrics = load_football_metric_table("threshold_tuning/threshold_metrics.csv")
    threshold_comparison = load_football_metric_table(
        "threshold_tuning/threshold_0_5_vs_tuned_comparison.csv"
    )
    if not threshold_best.empty:
        st.subheader("Top-50 threshold tuning")
        st.caption("Best thresholds are selected on validation by F1 and then applied to test.")
        st.dataframe(threshold_best, use_container_width=True)
    if not threshold_metrics.empty:
        st.dataframe(threshold_metrics, use_container_width=True, height=320)
    if not threshold_comparison.empty:
        st.dataframe(threshold_comparison, use_container_width=True, height=240)
        fig = px.bar(
            threshold_comparison,
            x="delta",
            y="metric",
            color="target",
            barmode="group",
            orientation="h",
            title="Test metric delta: tuned threshold minus 0.5",
        )
        st.plotly_chart(fig, use_container_width=True)
    threshold_figures_dir = PROJECT_ROOT / "outputs" / "figures" / "football" / "threshold_tuning"
    threshold_figures = sorted(threshold_figures_dir.glob("threshold_curves_*.png"))
    if threshold_figures:
        cols = st.columns(2)
        for idx, fig_path in enumerate(threshold_figures):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    balanced_best = load_football_metric_table("threshold_tuning_balanced/balanced_thresholds.csv")
    balanced_metrics = load_football_metric_table(
        "threshold_tuning_balanced/balanced_threshold_metrics.csv"
    )
    balanced_comparison = load_football_metric_table(
        "threshold_tuning_balanced/balanced_threshold_comparison.csv"
    )
    if not balanced_best.empty:
        st.subheader("Balanced top-50 threshold tuning")
        st.caption(
            "Validation F1 is optimized under precision constraints: home >= 0.60, away >= 0.55."
        )
        st.dataframe(balanced_best, use_container_width=True)
    if not balanced_metrics.empty:
        st.dataframe(balanced_metrics, use_container_width=True, height=320)
    if not balanced_comparison.empty:
        st.dataframe(balanced_comparison, use_container_width=True, height=260)
        fig = px.bar(
            balanced_comparison,
            x="balanced_minus_0_5",
            y="metric",
            color="target",
            barmode="group",
            orientation="h",
            title="Balanced threshold test delta vs 0.5",
        )
        st.plotly_chart(fig, use_container_width=True)
    balanced_figures_dir = (
        PROJECT_ROOT / "outputs" / "figures" / "football" / "threshold_tuning_balanced"
    )
    balanced_figures = sorted(balanced_figures_dir.glob("balanced_threshold_curves_*.png"))
    if balanced_figures:
        cols = st.columns(2)
        for idx, fig_path in enumerate(balanced_figures):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    final_thresholds = load_football_metric_table(
        "threshold_tuning_final/final_fixed_thresholds.csv"
    )
    final_metrics = load_football_metric_table(
        "threshold_tuning_final/final_fixed_threshold_metrics.csv"
    )
    final_comparison = load_football_metric_table(
        "threshold_tuning_final/final_fixed_threshold_comparison.csv"
    )
    calibration_metrics = load_football_metric_table("calibration/calibration_metrics.csv")
    calibration_comparison = load_football_metric_table("calibration/calibration_comparison.csv")
    calibration_diagnostics = load_football_metric_table("calibration/calibration_diagnostics.csv")
    if not final_thresholds.empty:
        st.subheader("Final fixed thresholds")
        st.dataframe(final_thresholds, use_container_width=True)
    if not final_metrics.empty:
        st.dataframe(final_metrics, use_container_width=True, height=280)
    if not final_comparison.empty:
        st.dataframe(final_comparison, use_container_width=True, height=260)

    if not calibration_metrics.empty:
        st.subheader("Top-50 probability calibration")
        st.caption(
            "Calibrators are fitted on validation predictions only; test is used only for final evaluation."
        )
        if not calibration_diagnostics.empty:
            st.dataframe(calibration_diagnostics, use_container_width=True)
        st.dataframe(calibration_metrics, use_container_width=True, height=320)
        metric = st.selectbox(
            "Calibration metric",
            [
                c
                for c in [
                    "brier",
                    "log_loss",
                    "roc_auc",
                    "pr_auc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ]
                if c in calibration_metrics
            ],
            key="football_calibration_metric_select",
        )
        fig = px.bar(
            calibration_metrics,
            x="calibration_method",
            y=metric,
            color="target",
            barmode="group",
            title=f"Raw vs calibrated probabilities by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)
    if not calibration_comparison.empty:
        st.markdown("**Calibration delta vs raw**")
        st.dataframe(calibration_comparison, use_container_width=True, height=320)
        main_calib = calibration_comparison[
            calibration_comparison["metric"].isin(["brier", "log_loss"])
        ].copy()
        if not main_calib.empty:
            fig = px.bar(
                main_calib,
                x="delta_calibrated_minus_raw",
                y="metric",
                color="calibrated_method",
                facet_col="target",
                barmode="group",
                orientation="h",
                title="Calibration delta for Brier/log_loss. Lower is better, so negative is improvement.",
            )
            st.plotly_chart(fig, use_container_width=True)
    calibration_figures_dir = PROJECT_ROOT / "outputs" / "figures" / "football" / "calibration"
    calibration_figures = sorted(calibration_figures_dir.glob("calibration_curve_*.png"))
    if calibration_figures:
        cols = st.columns(2)
        for idx, fig_path in enumerate(calibration_figures):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "football"
    loss_fig = figures_dir / "baseline_lstm_loss_curves.png"
    if loss_fig.exists():
        st.subheader("Saved figures")
        st.image(str(loss_fig), caption="Train/validation loss curves", use_container_width=True)
        figure_files = sorted(figures_dir.glob("test_*_*.png"))
        cols = st.columns(2)
        for idx, fig_path in enumerate(figure_files):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    if not lstm_shapes.empty:
        st.subheader("LSTM tensor shapes")
        st.dataframe(lstm_shapes, use_container_width=True)

    if not overfit.empty:
        st.subheader("Overfitting analysis")
        st.dataframe(overfit, use_container_width=True)

    if diagnostics.empty:
        st.info(
            "Football sequence reports are not ready. Run `uv run python scripts/build_football_sequences.py`."
        )
        return

    st.subheader("Sequence diagnostics")
    st.dataframe(diagnostics, use_container_width=True)

    st.subheader("Target checks")
    st.dataframe(target_checks, use_container_width=True)

    st.subheader("Target distribution")
    st.dataframe(target_dist, use_container_width=True)
    if not target_dist.empty:
        fig = px.bar(
            target_dist,
            x="value",
            y="matches",
            color="target",
            facet_col="split",
            barmode="group",
            title="Football Target Distribution By Split",
            text="matches",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sequence length stats")
    st.dataframe(seq_stats, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Feature columns")
        st.dataframe(features, use_container_width=True, height=520)
    with col_right:
        st.subheader("Excluded leakage / metadata columns")
        st.dataframe(excluded, use_container_width=True, height=520)


def refresh_audit() -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "audit_data_quality.py")]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        st.cache_data.clear()
        st.success("Аудит обновлен.")
        if completed.stdout:
            st.code(completed.stdout)
    else:
        st.error("Не удалось обновить аудит.")
        st.code(completed.stderr or completed.stdout)


def main() -> None:
    st.set_page_config(page_title="Match Dynamics Data Audit", layout="wide")
    st.title("Match Dynamics Data Audit")

    with st.sidebar:
        st.caption(f"Project: `{PROJECT_ROOT}`")
        st.caption(f"Audit files: `{AUDIT_DIR}`")
        if st.button("Refresh audit tables"):
            refresh_audit()
        st.divider()
        page = st.radio(
            "Page",
            [
                "Overview",
                "Football Merge",
                "Football Merged Processed",
                "Football Merged Feature Engineering",
                "NBA Raw",
                "NBA Merge",
                "NBA Processed",
                "NBA Join Quality",
                "Feature Engineering",
                "Correlations",
                "Football Metrics",
                "Conclusion",
                "Model Metrics",
            ],
        )

    if page == "Overview":
        show_overview()
    elif page == "Football Merge":
        show_football_merge()
    elif page == "Football Merged Processed":
        show_football_merged_processed()
    elif page == "Football Merged Feature Engineering":
        show_football_merged_feature_engineering()
    elif page == "NBA Raw":
        show_nba_raw()
    elif page == "NBA Merge":
        show_nba_merge()
    elif page == "NBA Processed":
        show_nba_processed()
    elif page == "NBA Join Quality":
        show_join_quality()
    elif page == "Feature Engineering":
        show_feature_engineering()
    elif page == "Correlations":
        show_correlations()
    elif page == "Football Metrics":
        show_football_metrics()
    elif page == "Conclusion":
        show_conclusion()
    elif page == "Model Metrics":
        show_model_metrics()


if __name__ == "__main__":
    main()
