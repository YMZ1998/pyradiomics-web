# -- coding: utf-8 --
"""Shared helpers for legacy scripts in Classification."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_df(df: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by row order into train/test, preserving original behavior."""
    cut_idx = int(round(ratio * df.shape[0]))
    print(cut_idx)
    train_data, test_data = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return train_data, test_data


def load_labeled_data(
    mcn_csv: str,
    scn_csv: str,
    random_state=None,
    drop_string_columns: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load two class csv files and prepend binary labels (MCN=1, SCN=0)."""
    mcn_data = pd.read_csv(mcn_csv)
    scn_data = pd.read_csv(scn_csv)

    if "label" in mcn_data.columns:
        mcn_data = mcn_data.drop(columns=["label"])
    if "label" in scn_data.columns:
        scn_data = scn_data.drop(columns=["label"])

    mcn_data.insert(0, "label", 1)
    scn_data.insert(0, "label", 0)

    mcn_data = mcn_data.sample(frac=1.0, random_state=random_state)
    scn_data = scn_data.sample(frac=1.0, random_state=random_state)

    if drop_string_columns:
        mcn_data = drop_non_numeric_columns(mcn_data)
        scn_data = drop_non_numeric_columns(scn_data)

    return mcn_data, scn_data


def drop_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose values are strings (legacy filtering rule)."""
    cols = [name for i, name in enumerate(df.columns) if isinstance(df.iat[1, i], str)]
    return df.drop(cols, axis=1)


def merge_and_shuffle(mcn_data: pd.DataFrame, scn_data: pd.DataFrame, random_state=None) -> pd.DataFrame:
    data = pd.concat([mcn_data, scn_data])
    return data.sample(frac=1.0, random_state=random_state)


def build_features_and_labels(
    data: pd.DataFrame,
    feature_start_col: int,
    label_col: str = "label",
    scale: bool = False,
):
    x = data[data.columns[feature_start_col:]]
    y = data[label_col]
    if not scale:
        return x, y

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y
