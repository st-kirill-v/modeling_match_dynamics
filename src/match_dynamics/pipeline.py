from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BASE_FOOTBALL_FEATURES,
    FOOTBALL_TARGETS,
    NBA_MATCHED_FEATURES,
    NBA_POSSESSION_FEATURES,
    NBA_TARGET,
    NBA_TARGETS,
    RANDOM_STATE,
    TEAM_STRENGTH_FEATURES,
    TIME_FEATURE_SETS,
    WINDOW_EXPERIMENTS,
    ProjectConfig,
)
from .data_loading import ensure_football_events
from .evaluation import (
    calibration_table,
    compute_weights,
    confusion_frame,
    evaluate_binary,
    evaluate_regression,
)
from .football import (
    add_team_strength,
    build_team_strength,
    football_feature_importance,
    preprocess_football_events,
)
from .models import build_football_tabular_baseline, build_lstm_binary, build_nba_baselines
from .nba import build_nba_final_score_checkpoint_dataset, nba_feature_importance
from .sequences import make_sequences, scale_split, split_match_ids
from .visualization import (
    save_calibration_curve,
    save_confusion_matrix,
    save_correlation_heatmap,
    save_feature_importance,
    save_football_training_curves,
    save_pr_curve,
)


def set_global_seed(include_tensorflow: bool = True) -> None:
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    if not include_tensorflow:
        return
    try:
        import tensorflow as tf

        tf.random.set_seed(RANDOM_STATE)
    except Exception:
        pass


def split_football_with_team_strength(
    football_model_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f_train_ids, f_val_ids, f_test_ids = split_match_ids(football_model_df)
    football_train = football_model_df[football_model_df["match_id"].isin(f_train_ids)].copy()
    football_val = football_model_df[football_model_df["match_id"].isin(f_val_ids)].copy()
    football_test = football_model_df[football_model_df["match_id"].isin(f_test_ids)].copy()

    team_stats, global_attack, global_defense = build_team_strength(football_train)
    football_train = add_team_strength(football_train, team_stats, global_attack, global_defense)
    football_val = add_team_strength(football_val, team_stats, global_attack, global_defense)
    football_test = add_team_strength(football_test, team_stats, global_attack, global_defense)
    return football_train, football_val, football_test


def build_sequence_data(
    football_train: pd.DataFrame,
    football_val: pd.DataFrame,
    football_test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    football_train_s, football_val_s, football_test_s, _ = scale_split(
        football_train, football_val, football_test, feature_cols
    )
    sequence_data = {}
    for window in WINDOW_EXPERIMENTS:
        sequence_data[window] = {}
        for target in FOOTBALL_TARGETS:
            sequence_data[window][target] = {
                "train": make_sequences(football_train_s, feature_cols, target, "time", window),
                "val": make_sequences(football_val_s, feature_cols, target, "time", window),
                "test": make_sequences(football_test_s, feature_cols, target, "time", window),
            }
    return sequence_data, (football_train_s, football_val_s, football_test_s)


def train_football_lstm(
    sequence_data: dict, feature_cols: list[str], cfg: ProjectConfig
) -> tuple[dict, dict]:
    models, histories = {}, {}
    if cfg.skip_lstm:
        return models, histories

    for target in FOOTBALL_TARGETS:
        X_train, y_train = sequence_data[cfg.main_window][target]["train"]
        X_val, y_val = sequence_data[cfg.main_window][target]["val"]
        model = build_lstm_binary((cfg.main_window, len(feature_cols)), f"football_{target}_w{cfg.main_window}")
        history = model.fit(
            X_train,
            y_train,
            epochs=cfg.epochs,
            batch_size=128,
            validation_data=(X_val, y_val),
            class_weight=compute_weights(y_train),
            verbose=1,
        )
        models[target] = model
        histories[target] = history
    return models, histories


def train_football_tabular_baselines(
    football_train: pd.DataFrame,
) -> dict:
    tabular_models = {}
    for target in FOOTBALL_TARGETS:
        tabular_models[target] = {}
        for mode, cols in TIME_FEATURE_SETS.items():
            cols = cols + TEAM_STRENGTH_FEATURES
            model = build_football_tabular_baseline()
            model.fit(football_train[cols], football_train[target].astype(int))
            tabular_models[target][mode] = (model, cols)
    return tabular_models


def train_nba_baselines(nba_train: pd.DataFrame) -> dict:
    nba_baselines = {}
    for target in NBA_TARGETS:
        nba_baselines[target] = {}
        for name, model in build_nba_baselines().items():
            model.fit(nba_train[NBA_MATCHED_FEATURES], nba_train[target].astype(int))
            nba_baselines[target][name] = model
    return nba_baselines


def evaluate_all(
    football_models: dict,
    tabular_models: dict,
    sequence_data: dict,
    football_test: pd.DataFrame,
    nba_baselines: dict | None = None,
    nba_test: pd.DataFrame | None = None,
    cfg: ProjectConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    main_window = cfg.main_window if cfg is not None else 20
    metric_rows, prob_store = [], {}

    for target in FOOTBALL_TARGETS:
        if target in football_models:
            X_test, y_test = sequence_data[main_window][target]["test"]
            prob = football_models[target].predict(X_test, verbose=0).ravel()
            name = f"LSTM_{target}_w{main_window}"
            metric_rows.append(evaluate_binary(y_test.astype(int), prob, name))
            prob_store[name] = (y_test.astype(int), prob)

        for mode, (model, cols) in tabular_models[target].items():
            prob = model.predict_proba(football_test[cols])[:, 1]
            name = f"HGB_{mode}_{target}"
            metric_rows.append(evaluate_binary(football_test[target].astype(int), prob, name))
            prob_store[name] = (football_test[target].astype(int).to_numpy(), prob)

    if nba_baselines and nba_test is not None and not nba_test.empty:
        for target, models_by_name in nba_baselines.items():
            for name, model in models_by_name.items():
                prob = model.predict_proba(nba_test[NBA_MATCHED_FEATURES])[:, 1]
                model_name = f"NBA_{name}_{target}"
                metric_rows.append(evaluate_binary(nba_test[target].astype(int), prob, model_name))
                prob_store[model_name] = (nba_test[target].astype(int).to_numpy(), prob)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    return metrics_df, prob_store


def run_pipeline(cfg: ProjectConfig) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    set_global_seed()
    cfg.ensure_dirs()

    print("[1/9] Loading Football Events...")
    df_events = ensure_football_events(cfg)
    print(f"      Football Events shape: {df_events.shape}")

    print("[2/9] Preprocessing Football: proxy-xG, minute-level table, rolling features...")
    football_minute, football_model_df = preprocess_football_events(df_events)
    print(f"      football_model_df shape: {football_model_df.shape}")

    print("[3/9] Splitting Football and adding train-only team strength...")
    football_train, football_val, football_test = split_football_with_team_strength(football_model_df)

    football_features = TIME_FEATURE_SETS["raw_plus_sincos"] + TEAM_STRENGTH_FEATURES
    print("[4/9] Building Football LSTM sequences...")
    sequence_data, _ = build_sequence_data(football_train, football_val, football_test, football_features)

    print("[5/9] Training Football LSTM models..." if not cfg.skip_lstm else "[5/9] Skipping Football LSTM models...")
    football_models, football_histories = train_football_lstm(sequence_data, football_features, cfg)

    print("[6/9] Training Football tabular baselines...")
    tabular_models = train_football_tabular_baselines(football_train)

    print("[7/9] Saving Football correlation and feature-importance plots...")
    target_for_analysis = "home_scores_next_half"
    corr = football_model_df[BASE_FOOTBALL_FEATURES + [target_for_analysis]].corr(numeric_only=True)[
        target_for_analysis
    ].drop(target_for_analysis)
    top_cols = corr.abs().sort_values(ascending=False).head(20).index.tolist() + [target_for_analysis]
    save_correlation_heatmap(
        football_model_df,
        top_cols,
        "Football top correlations with home_scores_next_half",
        cfg.figures_dir / "football_correlations.png",
    )
    football_importance = football_feature_importance(
        football_model_df, BASE_FOOTBALL_FEATURES, target_for_analysis
    )
    football_importance.to_csv(cfg.metrics_dir / "football_feature_importance.csv", index=False)
    save_feature_importance(
        football_importance,
        "Football feature importance",
        cfg.figures_dir / "football_feature_importance.png",
    )

    nba_possession_df = pd.DataFrame()
    nba_train = nba_test = pd.DataFrame()
    nba_baselines = {}
    print("[8/9] Loading prepared NBA matched dataset...")
    nba_path = cfg.nba_matched_path or cfg.default_nba_matched_path
    if nba_path.exists():
        nba_possession_df = pd.read_csv(nba_path)
        nba_possession_df = nba_possession_df.dropna(subset=NBA_MATCHED_FEATURES + NBA_TARGETS)
        print(f"      NBA matched dataset: {nba_path}")
        print(f"      shape: {nba_possession_df.shape}, games: {nba_possession_df['game_id'].nunique()}")
        n_train_ids, _, n_test_ids = split_match_ids(nba_possession_df, match_col="game_id")
        nba_train = nba_possession_df[nba_possession_df["game_id"].isin(n_train_ids)].copy()
        nba_test = nba_possession_df[nba_possession_df["game_id"].isin(n_test_ids)].copy()
        nba_baselines = train_nba_baselines(nba_train)
        save_correlation_heatmap(
            nba_possession_df,
            NBA_MATCHED_FEATURES + [NBA_TARGET],
            "NBA matched movement correlations with shot_made",
            cfg.figures_dir / "nba_correlations.png",
        )
        nba_importance = nba_feature_importance(
            nba_possession_df, NBA_MATCHED_FEATURES, NBA_TARGET
        )
        nba_importance.to_csv(cfg.metrics_dir / "nba_feature_importance.csv", index=False)
        save_feature_importance(
            nba_importance,
            "NBA matched movement feature importance",
            cfg.figures_dir / "nba_feature_importance.png",
        )
    else:
        print(f"      NBA skipped: prepared dataset not found at {nba_path}")
        print("      Build it with: python scripts\\build_nba_matched_dataset.py --max-games 50")

    print("[9/9] Evaluating models and saving final plots/metrics...")
    metrics_df, prob_store = evaluate_all(
        football_models,
        tabular_models,
        sequence_data,
        football_test,
        nba_baselines,
        nba_test,
        cfg,
    )
    metrics_df.to_csv(cfg.metrics_dir / "metrics.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    y_best, p_best = prob_store[best_name]
    confusion_frame(y_best, p_best).to_csv(cfg.metrics_dir / "best_confusion_matrix.csv")
    calib = calibration_table(y_best, p_best)
    calib.to_csv(cfg.metrics_dir / "best_calibration.csv")

    if football_histories:
        save_football_training_curves(
            football_histories,
            FOOTBALL_TARGETS,
            cfg.figures_dir / "football_lstm_training_curves.png",
        )
    save_confusion_matrix(
        y_best,
        p_best,
        f"Confusion matrix: {best_name}",
        cfg.figures_dir / "best_confusion_matrix.png",
    )
    save_pr_curve(y_best, p_best, f"PR curve: {best_name}", cfg.figures_dir / "best_pr_curve.png")
    save_calibration_curve(
        calib,
        f"Calibration: {best_name}",
        cfg.figures_dir / "best_calibration_curve.png",
    )

    return {
        "football_minute": football_minute,
        "football_model_df": football_model_df,
        "nba_possession_df": nba_possession_df,
        "metrics_df": metrics_df,
        "best_model": best_name,
    }


def run_football_pipeline(cfg: ProjectConfig) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    set_global_seed()
    cfg.ensure_dirs()

    print("[1/7] Loading Football Events...")
    df_events = ensure_football_events(cfg)
    print(f"      Football Events shape: {df_events.shape}")

    print("[2/7] Preprocessing Football...")
    football_minute, football_model_df = preprocess_football_events(df_events)
    print(f"      football_model_df shape: {football_model_df.shape}")

    print("[3/7] Splitting Football and adding train-only team strength...")
    football_train, football_val, football_test = split_football_with_team_strength(football_model_df)

    football_features = TIME_FEATURE_SETS["raw_plus_sincos"] + TEAM_STRENGTH_FEATURES
    print("[4/7] Building Football LSTM sequences...")
    sequence_data, _ = build_sequence_data(football_train, football_val, football_test, football_features)

    print("[5/7] Training Football LSTM models..." if not cfg.skip_lstm else "[5/7] Skipping Football LSTM models...")
    football_models, football_histories = train_football_lstm(sequence_data, football_features, cfg)

    print("[6/7] Training Football tabular baselines and plots...")
    tabular_models = train_football_tabular_baselines(football_train)
    target_for_analysis = "home_scores_next_half"
    corr = football_model_df[BASE_FOOTBALL_FEATURES + [target_for_analysis]].corr(numeric_only=True)[
        target_for_analysis
    ].drop(target_for_analysis)
    top_cols = corr.abs().sort_values(ascending=False).head(20).index.tolist() + [target_for_analysis]
    save_correlation_heatmap(
        football_model_df,
        top_cols,
        "Football top correlations with home_scores_next_half",
        cfg.figures_dir / "football_correlations.png",
    )
    football_importance = football_feature_importance(
        football_model_df, BASE_FOOTBALL_FEATURES, target_for_analysis
    )
    football_importance.to_csv(cfg.metrics_dir / "football_feature_importance.csv", index=False)
    save_feature_importance(
        football_importance,
        "Football feature importance",
        cfg.figures_dir / "football_feature_importance.png",
    )

    print("[7/7] Evaluating Football models...")
    metrics_df, prob_store = evaluate_all(
        football_models,
        tabular_models,
        sequence_data,
        football_test,
        cfg=cfg,
    )
    metrics_df.to_csv(cfg.metrics_dir / "football_metrics.csv", index=False)
    if not metrics_df.empty:
        best_name = metrics_df.iloc[0]["model"]
        y_best, p_best = prob_store[best_name]
        confusion_frame(y_best, p_best).to_csv(cfg.metrics_dir / "football_best_confusion_matrix.csv")
        calib = calibration_table(y_best, p_best)
        calib.to_csv(cfg.metrics_dir / "football_best_calibration.csv")
        if football_histories:
            save_football_training_curves(
                football_histories,
                FOOTBALL_TARGETS,
                cfg.figures_dir / "football_lstm_training_curves.png",
            )
        save_confusion_matrix(
            y_best,
            p_best,
            f"Confusion matrix: {best_name}",
            cfg.figures_dir / "football_best_confusion_matrix.png",
        )
        save_pr_curve(
            y_best,
            p_best,
            f"PR curve: {best_name}",
            cfg.figures_dir / "football_best_pr_curve.png",
        )
        save_calibration_curve(
            calib,
            f"Calibration: {best_name}",
            cfg.figures_dir / "football_best_calibration_curve.png",
        )
    return {"metrics_df": metrics_df, "football_model_df": football_model_df}


NBA_FINAL_SCORE_FEATURES = [
    "checkpoint_seconds_remaining",
    "current_home_score",
    "current_visitor_score",
    "current_score_diff_home",
    "current_total_score",
    "shot_attempt_before_checkpoint",
    "shot_made_before_checkpoint",
    "shot_missed_before_checkpoint",
    "free_throw_before_checkpoint",
    "turnover_before_checkpoint",
    "foul_before_checkpoint",
    "ball_hoop_dist_at_checkpoint",
    "min_player_hoop_dist_at_checkpoint",
    "players_near_hoop_at_checkpoint",
    "intensity_at_checkpoint",
    "avg_distance_mean_before_checkpoint",
    "std_distance_mean_before_checkpoint",
    "spread_x_mean_before_checkpoint",
    "spread_y_mean_before_checkpoint",
]


def _nba_regressors() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gbdt": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
    }


def _nba_classifiers() -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "hist_gbdt": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def run_nba_pipeline(cfg: ProjectConfig) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    set_global_seed(include_tensorflow=False)
    cfg.ensure_dirs()

    nba_path = cfg.nba_matched_path or cfg.default_nba_matched_path
    print("[1/5] Loading prepared NBA matched dataset...")
    if not nba_path.exists():
        raise FileNotFoundError(
            f"Prepared NBA matched dataset not found: {nba_path}. "
            "Build it with: python scripts\\build_nba_matched_dataset.py --max-games 50"
        )
    matched = pd.read_csv(nba_path)
    print(f"      matched shape: {matched.shape}, games: {matched['game_id'].nunique()}")

    print("[2/5] Building 5-minute final-score checkpoint dataset...")
    checkpoint_df = build_nba_final_score_checkpoint_dataset(matched, checkpoint_seconds=300.0)
    checkpoint_df = checkpoint_df.dropna(subset=NBA_FINAL_SCORE_FEATURES + ["home_win"])
    checkpoint_path = cfg.data_dir / "processed" / "nba_final_score_checkpoint_5min.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_df.to_csv(checkpoint_path, index=False)
    print(f"      checkpoint shape: {checkpoint_df.shape}")
    print(f"      saved: {checkpoint_path}")

    train_ids, _, test_ids = split_match_ids(checkpoint_df, match_col="game_id")
    train = checkpoint_df[checkpoint_df["game_id"].isin(train_ids)].copy()
    test = checkpoint_df[checkpoint_df["game_id"].isin(test_ids)].copy()
    print(f"[3/5] Split by game_id: train={len(train)}, test={len(test)}")

    X_train, X_test = train[NBA_FINAL_SCORE_FEATURES], test[NBA_FINAL_SCORE_FEATURES]
    reg_targets = ["final_home_score", "final_visitor_score", "final_score_diff_home", "score_diff_change_after_checkpoint"]

    print("[4/5] Training NBA final-score regressors...")
    regression_rows = []
    for target in reg_targets:
        for name, model in _nba_regressors().items():
            model.fit(X_train, train[target])
            pred = model.predict(X_test)
            regression_rows.append(evaluate_regression(test[target].to_numpy(), pred, f"NBA_{name}_{target}"))
    regression_metrics = pd.DataFrame(regression_rows).sort_values(["mae", "rmse"])
    regression_metrics.to_csv(cfg.metrics_dir / "nba_final_score_regression_metrics.csv", index=False)

    print("[5/5] Training NBA winner classifier...")
    classification_rows, prob_store = [], {}
    for name, model in _nba_classifiers().items():
        model.fit(X_train, train["home_win"].astype(int))
        prob = model.predict_proba(X_test)[:, 1]
        model_name = f"NBA_{name}_home_win_5min"
        classification_rows.append(evaluate_binary(test["home_win"].astype(int), prob, model_name))
        prob_store[model_name] = (test["home_win"].astype(int).to_numpy(), prob)
    classification_metrics = pd.DataFrame(classification_rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    classification_metrics.to_csv(cfg.metrics_dir / "nba_home_win_classification_metrics.csv", index=False)

    print("\nNBA final-score regression metrics:")
    print(regression_metrics.to_string(index=False))
    print("\nNBA home-win classification metrics:")
    print(classification_metrics.to_string(index=False))

    return {
        "checkpoint_df": checkpoint_df,
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
    }
