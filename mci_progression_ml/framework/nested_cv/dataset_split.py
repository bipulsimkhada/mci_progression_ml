import numpy as np
import pandas as pd
from typing import Any

from mci_progression_ml.config import RANDOM_STATE, cog, dem, fs_col
from mci_progression_ml.framework.kfold import RepeatedStratifiedGroupKFold, StratifiedGroupKFold


def create_train_test_split(df, ocv_n_splits=5, ocv_n_repeats=5, icv_n_splits=5):
    outer_cv = RepeatedStratifiedGroupKFold(
        n_splits=ocv_n_splits,
        n_repeats=ocv_n_repeats,
        random_state=RANDOM_STATE
    )

    cv = StratifiedGroupKFold(
        n_splits=icv_n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    feature_cols = cog + dem + fs_col
    df_X = df[feature_cols].copy()
    y = df["DIAGNOSIS_24"].to_numpy(dtype=int)
    groups = df["RID"].to_numpy(dtype=int)
    diag_change = df["DIAGNOSIS_CHANGE_MONTH"].to_numpy()

    train_idx, test_idx = next(cv.split(df_X, y, groups))

    X_train = df_X.iloc[train_idx]
    X_test = df_X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    diag_change_train = diag_change[train_idx]
    diag_change_test = diag_change[test_idx]

    return (
        outer_cv,
        cv,
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        diag_change_train,
        diag_change_test,
    )
