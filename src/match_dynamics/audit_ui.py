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


def load_report_table(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return read_csv(str(path), file_signature(path))


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
    st.header("Overview")
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


def show_football_raw() -> None:
    st.header("Football Raw")
    profile = load_table("football_raw_events_columns.csv")
    show_head_table("football_raw_events_head.csv", "Raw events head()")
    show_missing_bar(profile, "Football Raw: Top Missing Columns")
    show_dtype_bar(profile, "Football Raw: Column Types")
    st.subheader("Column profile")
    show_profile_table(profile)


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


def show_football_processed() -> None:
    st.header("Football Processed")
    level = st.radio(
        "Processed table",
        ["minute_level_processed", "first_half_model_df"],
        horizontal=True,
    )
    prefix = f"football_{level}"
    show_head_table(f"{prefix}_head.csv", f"{level} head()")
    profile = load_table(f"{prefix}_columns.csv")
    show_missing_bar(profile, f"Football {level}: Top Missing Columns")
    show_dtype_bar(profile, f"Football {level}: Column Types")

    changes = load_table("football_processing_changes.csv")
    st.subheader("Processing changes")
    st.dataframe(changes, use_container_width=True, height=260)

    head = load_table(f"{prefix}_head.csv")
    target_cols = [
        c for c in ["home_scores_next_half", "away_scores_next_half"] if c in head.columns
    ]
    if target_cols:
        target_rows = []
        for col in target_cols:
            counts = head[col].value_counts(dropna=False).reset_index()
            counts.columns = ["value", "count"]
            counts["target"] = col
            target_rows.append(counts)
        target_df = pd.concat(target_rows, ignore_index=True)
        fig = px.bar(
            target_df,
            x="value",
            y="count",
            color="target",
            barmode="group",
            title="Target Values In Saved head()",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Column profile")
    show_profile_table(profile)


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
                "Football Raw",
                "Football Merge",
                "Football Merged Processed",
                "Football Merged Feature Engineering",
                "Football Processed",
                "NBA Raw",
                "NBA Processed",
                "NBA Join Quality",
                "Feature Engineering",
                "Correlations",
                "Model Metrics",
            ],
        )

    if page == "Overview":
        show_overview()
    elif page == "Football Raw":
        show_football_raw()
    elif page == "Football Merge":
        show_football_merge()
    elif page == "Football Merged Processed":
        show_football_merged_processed()
    elif page == "Football Merged Feature Engineering":
        show_football_merged_feature_engineering()
    elif page == "Football Processed":
        show_football_processed()
    elif page == "NBA Raw":
        show_nba_raw()
    elif page == "NBA Processed":
        show_nba_processed()
    elif page == "NBA Join Quality":
        show_join_quality()
    elif page == "Feature Engineering":
        show_feature_engineering()
    elif page == "Correlations":
        show_correlations()
    elif page == "Model Metrics":
        show_model_metrics()


if __name__ == "__main__":
    main()
