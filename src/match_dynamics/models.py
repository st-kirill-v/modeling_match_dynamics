from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_lstm_binary(input_shape: tuple[int, int], name: str):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(48, return_sequences=True, kernel_regularizer=l2(1e-4)),
            Dropout(0.25),
            LSTM(24, kernel_regularizer=l2(1e-4)),
            Dropout(0.25),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name=name,
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_football_tabular_baseline() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=42)


def build_nba_baselines() -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
