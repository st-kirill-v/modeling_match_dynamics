from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import RANDOM_STATE


def split_match_ids(
    df: pd.DataFrame, match_col: str = "match_id", test_size: float = 0.2, val_size: float = 0.1
) -> tuple[set, set, set]:
    ids = np.array(sorted(df[match_col].unique()))
    train_val_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=RANDOM_STATE)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE
    )
    return set(train_ids), set(val_ids), set(test_ids)


def scale_split(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_s, val_s, test_s = train_df.copy(), val_df.copy(), test_df.copy()
    train_s[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_s[feature_cols] = scaler.transform(val_df[feature_cols])
    test_s[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_s, val_s, test_s, scaler


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_col: str,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, group in df.sort_values(["match_id", time_col]).groupby("match_id"):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        target = group[target_col].to_numpy(dtype=np.float32)
        for idx in range(time_steps - 1, len(group)):
            X.append(values[idx - time_steps + 1 : idx + 1])
            y.append(target[idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
