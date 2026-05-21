from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from .config import (
    BASE_FOOTBALL_FEATURES,
    FOOTBALL_TARGETS,
    NBA_MAIN_SEQUENCE_WINDOW,
    NBA_SEQUENCE_WINDOWS,
    RANDOM_STATE,
    TEAM_STRENGTH_FEATURES,
    TIME_FEATURE_SETS,
    WINDOW_EXPERIMENTS,
    ProjectConfig,
)
from .data_loading import ensure_football_events
from .evaluation import (
    calibrate_probabilities,
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
from .models import (
    build_lstm_multilabel,
    build_lstm_regression,
)
from .nba import add_score_columns, build_nba_final_score_checkpoint_dataset
from .sequences import make_sequences, scale_split, split_match_ids
from .visualization import (
    save_calibration_curve,
    save_confusion_matrix,
    save_correlation_heatmap,
    save_feature_importance,
    save_football_error_curves,
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

    windows = WINDOW_EXPERIMENTS if cfg.compare_windows else [cfg.main_window]
    for window in windows:
        X_train, y_train_home = sequence_data[window][FOOTBALL_TARGETS[0]]["train"]
        X_val, y_val_home = sequence_data[window][FOOTBALL_TARGETS[0]]["val"]
        _, y_train_away = sequence_data[window][FOOTBALL_TARGETS[1]]["train"]
        _, y_val_away = sequence_data[window][FOOTBALL_TARGETS[1]]["val"]
        y_train = np.column_stack([y_train_home, y_train_away]).astype(int)
        y_val = np.column_stack([y_val_home, y_val_away]).astype(int)

        weights_by_target = [compute_weights(y_train[:, idx]) for idx in range(y_train.shape[1])]
        sample_weight = np.mean(
            [
                np.array([weights_by_target[idx][int(value)] for value in y_train[:, idx]])
                for idx in range(y_train.shape[1])
            ],
            axis=0,
        )

        model = build_lstm_multilabel(
            (window, len(feature_cols)),
            len(FOOTBALL_TARGETS),
            f"football_multilabel_w{window}",
        )
        print(f"      Training Football LSTM window={window}")
        history = model.fit(
            X_train,
            y_train,
            epochs=cfg.epochs,
            batch_size=128,
            validation_data=(X_val, y_val),
            sample_weight=sample_weight,
            verbose=1,
        )
        models[f"football_multilabel_w{window}"] = model
        histories[f"football_multilabel_w{window}"] = history
        if window == cfg.main_window:
            for target in FOOTBALL_TARGETS:
                histories[target] = history
    return models, histories


def evaluate_all(
    football_models: dict,
    sequence_data: dict,
    cfg: ProjectConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    metric_rows, prob_store = [], {}

    def add_fixed_threshold_row(y_test: np.ndarray, prob: np.ndarray, name: str) -> None:
        threshold = 0.5
        metric_rows.append(evaluate_binary(y_test.astype(int), prob, name, threshold))
        prob_store[name] = (y_test.astype(int), prob, threshold)

    multilabel_models = {
        int(name.rsplit("_w", 1)[1]): model
        for name, model in football_models.items()
        if name.startswith("football_multilabel_w")
    }
    if multilabel_models:
        for window, model in sorted(multilabel_models.items()):
            X_val, _ = sequence_data[window][FOOTBALL_TARGETS[0]]["val"]
            X_test, _ = sequence_data[window][FOOTBALL_TARGETS[0]]["test"]
            val_probs = model.predict(X_val, verbose=0)
            probs = model.predict(X_test, verbose=0)
            for idx, target in enumerate(FOOTBALL_TARGETS):
                _, y_val = sequence_data[window][target]["val"]
                _, y_test = sequence_data[window][target]["test"]
                val_prob = val_probs[:, idx]
                prob = probs[:, idx]
                name = f"LSTM_multilabel_{target}_w{window}"
                add_fixed_threshold_row(y_test, prob, name)

                calibrated_prob = calibrate_probabilities(
                    y_val.astype(int),
                    val_prob,
                    prob,
                )
                calibrated_name = f"{name}_calibrated"
                add_fixed_threshold_row(y_test, calibrated_prob, calibrated_name)
    else:
        main_window = cfg.main_window if cfg is not None else 20
        for target in FOOTBALL_TARGETS:
            if target in football_models:
                X_val, y_val = sequence_data[main_window][target]["val"]
                X_test, y_test = sequence_data[main_window][target]["test"]
                val_prob = football_models[target].predict(X_val, verbose=0).ravel()
                prob = football_models[target].predict(X_test, verbose=0).ravel()
                name = f"LSTM_{target}_w{main_window}"
                add_fixed_threshold_row(y_test, prob, name)

                calibrated_prob = calibrate_probabilities(y_val.astype(int), val_prob, prob)
                calibrated_name = f"{name}_calibrated"
                add_fixed_threshold_row(y_test, calibrated_prob, calibrated_name)

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
    football_train, football_val, football_test = split_football_with_team_strength(
        football_model_df
    )

    football_features = TIME_FEATURE_SETS["raw_plus_sincos"] + TEAM_STRENGTH_FEATURES
    print("[4/9] Building Football LSTM sequences...")
    sequence_data, _ = build_sequence_data(
        football_train, football_val, football_test, football_features
    )

    print(
        "[5/9] Training Football LSTM models..."
        if not cfg.skip_lstm
        else "[5/9] Skipping Football LSTM models..."
    )
    football_models, football_histories = train_football_lstm(sequence_data, football_features, cfg)

    print("[6/9] Football baselines skipped: active pipeline uses LSTM only.")

    print("[7/9] Saving Football correlation and feature-importance plots...")
    target_for_analysis = "home_scores_next_half"
    corr = (
        football_model_df[BASE_FOOTBALL_FEATURES + [target_for_analysis]]
        .corr(numeric_only=True)[target_for_analysis]
        .drop(target_for_analysis)
    )
    top_cols = corr.abs().sort_values(ascending=False).head(20).index.tolist() + [
        target_for_analysis
    ]
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
    print("[8/9] Loading prepared NBA matched dataset...")
    nba_path = cfg.nba_matched_path or cfg.default_nba_matched_path
    if nba_path.exists():
        nba_possession_df = pd.read_csv(nba_path)
        print(f"      NBA matched dataset: {nba_path}")
        print(
            f"      shape: {nba_possession_df.shape}, games: {nba_possession_df['game_id'].nunique()}"
        )
        print(
            "      NBA baselines skipped: use scripts\\run_nba_pipeline.py for the NBA LSTM task."
        )
    else:
        print(f"      NBA skipped: prepared dataset not found at {nba_path}")
        print("      Build it with: python scripts\\build_nba_matched_dataset.py --max-games 200")

    print("[9/9] Evaluating models and saving final plots/metrics...")
    metrics_df, prob_store = evaluate_all(
        football_models,
        sequence_data,
        cfg,
    )
    metrics_df.to_csv(cfg.metrics_dir / "metrics.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    y_best, p_best, best_threshold = prob_store[best_name]
    best_confusion_df = confusion_frame(y_best, p_best, best_threshold)
    best_confusion_df.to_csv(cfg.metrics_dir / "best_confusion_matrix.csv")
    calib = calibration_table(y_best, p_best)
    calib.to_csv(cfg.metrics_dir / "best_calibration.csv")

    if football_histories:
        save_football_training_curves(
            football_histories,
            FOOTBALL_TARGETS,
            cfg.figures_dir / "football_lstm_training_curves.png",
        )
        save_football_error_curves(
            football_histories,
            FOOTBALL_TARGETS,
            cfg.figures_dir / "football_lstm_mse_mae_curves.png",
        )
    save_confusion_matrix(
        y_best,
        p_best,
        f"Confusion matrix: {best_name}",
        cfg.figures_dir / "best_confusion_matrix.png",
        best_threshold,
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
        "confusion_df": best_confusion_df,
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
    football_train, football_val, football_test = split_football_with_team_strength(
        football_model_df
    )

    football_features = TIME_FEATURE_SETS["raw_plus_sincos"] + TEAM_STRENGTH_FEATURES
    print("[4/7] Building Football LSTM sequences...")
    sequence_data, _ = build_sequence_data(
        football_train, football_val, football_test, football_features
    )

    print(
        "[5/7] Training Football LSTM models..."
        if not cfg.skip_lstm
        else "[5/7] Skipping Football LSTM models..."
    )
    football_models, football_histories = train_football_lstm(sequence_data, football_features, cfg)

    print("[6/7] Saving Football analysis plots...")
    target_for_analysis = "home_scores_next_half"
    corr = (
        football_model_df[BASE_FOOTBALL_FEATURES + [target_for_analysis]]
        .corr(numeric_only=True)[target_for_analysis]
        .drop(target_for_analysis)
    )
    top_cols = corr.abs().sort_values(ascending=False).head(20).index.tolist() + [
        target_for_analysis
    ]
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
        sequence_data,
        cfg=cfg,
    )
    metrics_df.to_csv(cfg.metrics_dir / "football_metrics.csv", index=False)
    best_confusion_df = pd.DataFrame()
    if not metrics_df.empty:
        best_name = metrics_df.iloc[0]["model"]
        y_best, p_best, best_threshold = prob_store[best_name]
        best_confusion_df = confusion_frame(y_best, p_best, best_threshold)
        best_confusion_df.to_csv(cfg.metrics_dir / "football_best_confusion_matrix.csv")
        calib = calibration_table(y_best, p_best)
        calib.to_csv(cfg.metrics_dir / "football_best_calibration.csv")
        if football_histories:
            save_football_training_curves(
                football_histories,
                FOOTBALL_TARGETS,
                cfg.figures_dir / "football_lstm_training_curves.png",
            )
            save_football_error_curves(
                football_histories,
                FOOTBALL_TARGETS,
                cfg.figures_dir / "football_lstm_mse_mae_curves.png",
            )
        save_confusion_matrix(
            y_best,
            p_best,
            f"Confusion matrix: {best_name}",
            cfg.figures_dir / "football_best_confusion_matrix.png",
            best_threshold,
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
    return {
        "metrics_df": metrics_df,
        "football_model_df": football_model_df,
        "confusion_df": best_confusion_df,
    }


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

NBA_SEQUENCE_FEATURES = [
    "period",
    "game_clock_start",
    "game_clock_end",
    "shot_clock_start",
    "shot_clock_end",
    "home_score_ffill",
    "visitor_score_ffill",
    "score_diff_home",
    "avg_distance",
    "std_distance",
    "spread_x",
    "spread_y",
    "ball_x",
    "ball_y",
    "ball_hoop_dist",
    "min_player_hoop_dist",
    "players_near_hoop",
    "low_shot_clock",
    "intensity",
    "shot_attempt",
    "shot_made",
    "shot_missed",
    "free_throw",
    "turnover",
    "foul",
    "home_shot_attempt",
    "away_shot_attempt",
    "home_shot_made",
    "away_shot_made",
    "home_turnover",
    "away_turnover",
    "home_foul",
    "away_foul",
    "home_scoring_event",
    "away_scoring_event",
]


def prepare_nba_lstm_source(matched: pd.DataFrame) -> pd.DataFrame:
    out = add_score_columns(matched)
    home_event = out["HOMEDESCRIPTION"].notna()
    away_event = out["VISITORDESCRIPTION"].notna()
    for col in ["shot_attempt", "shot_made", "shot_missed", "free_throw", "turnover", "foul"]:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col not in out.columns:
            out[home_col] = (out[col].eq(1) & home_event).astype(int)
        if away_col not in out.columns:
            out[away_col] = (out[col].eq(1) & away_event).astype(int)
    if "home_scoring_event" not in out.columns:
        out["home_scoring_event"] = (
            out["home_shot_made"].eq(1) | out["home_free_throw"].eq(1)
        ).astype(int)
    if "away_scoring_event" not in out.columns:
        out["away_scoring_event"] = (
            out["away_shot_made"].eq(1) | out["away_free_throw"].eq(1)
        ).astype(int)
    return out


def build_nba_lstm_sequences(
    matched_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "score_diff_change_after_checkpoint",
    time_steps: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, game_ids = [], [], []
    matched = matched_df.sort_values(["game_id", "period", "event_id"]).copy()
    checkpoint_lookup = checkpoint_df.set_index("game_id")
    for game_id, group in matched.groupby("game_id"):
        if game_id not in checkpoint_lookup.index:
            continue
        checkpoint = checkpoint_lookup.loc[game_id]
        history = group[
            (group["period"].lt(4))
            | ((group["period"].eq(4)) & (group["event_id"].le(checkpoint["checkpoint_event_id"])))
        ]
        history = history.dropna(subset=feature_cols)
        if history.empty:
            continue
        values = history[feature_cols].to_numpy(dtype=np.float32)
        if len(values) >= time_steps:
            seq = values[-time_steps:]
        else:
            pad = np.zeros((time_steps - len(values), len(feature_cols)), dtype=np.float32)
            seq = np.vstack([pad, values])
        X.append(seq)
        y.append(float(checkpoint[target_col]))
        game_ids.append(game_id)
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(game_ids),
    )


def run_nba_pipeline(cfg: ProjectConfig) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    set_global_seed(include_tensorflow=True)
    cfg.ensure_dirs()

    nba_path = cfg.nba_matched_path or cfg.default_nba_matched_path
    print("[1/4] Loading prepared NBA matched dataset...")
    if not nba_path.exists():
        raise FileNotFoundError(
            f"Prepared NBA matched dataset not found: {nba_path}. "
            "Build it with: python scripts\\build_nba_matched_dataset.py --max-games 200"
        )
    matched = pd.read_csv(nba_path)
    print(f"      matched shape: {matched.shape}, games: {matched['game_id'].nunique()}")

    print("[2/4] Building 5-minute final-score checkpoint dataset...")
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
    print(f"[3/4] Split by game_id: train={len(train)}, test={len(test)}")

    print("[4/4] Training NBA LSTM on event sequences before 5-minute checkpoint...")
    sequence_source = prepare_nba_lstm_source(matched).dropna(subset=NBA_SEQUENCE_FEATURES)
    nba_windows = NBA_SEQUENCE_WINDOWS if cfg.compare_windows else [NBA_MAIN_SEQUENCE_WINDOW]
    train_game_ids = set(train["game_id"])
    test_game_ids = set(test["game_id"])

    metric_rows = []
    for time_steps in nba_windows:
        print(f"      Training NBA LSTM time_steps={time_steps}")
        X_seq, y_seq, seq_game_ids = build_nba_lstm_sequences(
            sequence_source,
            checkpoint_df,
            NBA_SEQUENCE_FEATURES,
            target_col="score_diff_change_after_checkpoint",
            time_steps=time_steps,
        )
        train_mask = np.array([gid in train_game_ids for gid in seq_game_ids])
        test_mask = np.array([gid in test_game_ids for gid in seq_game_ids])
        X_train_seq, y_train_seq = X_seq[train_mask], y_seq[train_mask]
        X_test_seq, y_test_seq = X_seq[test_mask], y_seq[test_mask]

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        seq_scaler = StandardScaler()
        flat_train = X_train_seq.reshape(-1, X_train_seq.shape[-1])
        flat_test = X_test_seq.reshape(-1, X_test_seq.shape[-1])
        X_train_seq = seq_scaler.fit_transform(flat_train).reshape(X_train_seq.shape)
        X_test_seq = seq_scaler.transform(flat_test).reshape(X_test_seq.shape)

        baseline_pred = np.full_like(y_test_seq, fill_value=float(y_train_seq.mean()))
        metric_rows.append(
            evaluate_regression(
                y_test_seq,
                baseline_pred,
                f"NBA_mean_baseline_score_diff_change_5min_w{time_steps}",
            )
        )

        lstm = build_lstm_regression(
            (X_train_seq.shape[1], X_train_seq.shape[2]),
            f"nba_lstm_score_diff_change_5min_w{time_steps}",
        )
        import tensorflow as tf

        lstm.fit(
            X_train_seq,
            y_train_seq,
            epochs=cfg.epochs,
            batch_size=8,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            ],
            verbose=1,
        )
        lstm_pred = lstm.predict(X_test_seq, verbose=0).ravel()
        metric_rows.append(
            evaluate_regression(
                y_test_seq,
                lstm_pred,
                f"NBA_LSTM_score_diff_change_5min_w{time_steps}",
            )
        )

    lstm_metrics = pd.DataFrame(metric_rows)
    if not lstm_metrics.empty:
        lstm_metrics = lstm_metrics.sort_values("mae")
        lstm_metrics.to_csv(cfg.metrics_dir / "nba_regression_metrics.csv", index=False)
        lstm_metrics.to_csv(cfg.metrics_dir / "nba_lstm_final_score_metrics.csv", index=False)

    if not lstm_metrics.empty:
        print("\nNBA score-diff-change regression comparison:")
        print(lstm_metrics.to_string(index=False))

    return {
        "checkpoint_df": checkpoint_df,
        "regression_metrics": lstm_metrics,
        "classification_metrics": pd.DataFrame(),
        "lstm_metrics": lstm_metrics,
    }
