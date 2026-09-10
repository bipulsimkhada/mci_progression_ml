import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from sklearn.metrics import (f1_score, make_scorer, 
                            recall_score, matthews_corrcoef)

from mci_progression_ml.config import RANDOM_STATE, FINAL_N_ITER, cog, dem, fs_col
from mci_progression_ml.framework.threshold_tuning import tune_threshold_balanced
from mci_progression_ml.framework.calibration import make_group_calibrated_model
from mci_progression_ml.framework.pipeline import make_pipeline
from mci_progression_ml.framework.kfold import RepeatedStratifiedGroupKFold
from mci_progression_ml.framework.search_config import *


def f1_sMCI(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=0)

def f1_pMCI(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)

def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

scoring = {
    "f1_macro": "f1_macro",
    "f1_sMCI": make_scorer(f1_sMCI),
    "f1_pMCI": make_scorer(f1_pMCI),
    "sensitivity": make_scorer(sensitivity_score),
    "specificity": make_scorer(specificity_score),
    "bal_acc": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "mcc": make_scorer(matthews_corrcoef),
}


def final_deployment_model(X_train, y_train, groups, model, sampling, features, name, n_splits=5, n_repeats=5):
    inner_cv = RepeatedStratifiedGroupKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    model_name = model.__class__.__name__.lower()

    print(f"Running {name}")

    if "logisticregression" in model_name and sampling == "none":
        params = lr_param_dist_none
    elif "logisticregression" in model_name and sampling != "none":
        params = lr_param_dist_smote
    elif "svc" in model_name and sampling == "none":
        params = svc_param_dist_none
    elif "svc" in model_name and sampling != "none":
        params = svc_param_dist_smote
    elif "randomforest" in model_name and sampling == "none":
        params = rf_param_dist_none
    elif "randomforest" in model_name and sampling != "none":
        params = rf_param_dist_smote
    elif ("xgb" in model_name or "xgboost" in model_name) and sampling == "none":
        params = xgb_param_dist_none
    elif ("xgb" in model_name or "xgboost" in model_name) and sampling != "none":
        params = xgb_param_dist_smote

    
    if len(features) > len(cog + dem):
        params = params | select_param_common

    cat_cols = [f for f in features if f in ["PTGENDER", "PTHAND"]]
    num_cols = [f for f in features if f not in cat_cols]
    
    
    pipe = make_pipeline(
        model,
        feature_select_model=select_base if len(features) > len(cog + dem) else None,
        cat_cols=cat_cols,
        num_cols=num_cols,
        drop_low_variance=True,
        drop_high_corr=True,
        sampling_method=sampling,
        use_sampling_for_trees=True
    )

    X_train = X_train[features]

    # Use training labels for the imbalance ratio
    label_count = pd.Series(y_train).value_counts()
    scale_pos_weight = label_count[0] / label_count[1]

    if ("xgb" in model_name or "xgboost" in model_name) and sampling == "none":
        params = params | {"clf__scale_pos_weight": uniform(scale_pos_weight * 0.5, scale_pos_weight)}
    elif ("xgb" in model_name or "xgboost" in model_name) and sampling != "none":
        params = params | {"clf__scale_pos_weight": [1.0]}

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=FINAL_N_ITER,
        scoring=scoring,
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit="bal_acc",
        return_train_score=False,
        error_score="raise",
    )

    search.fit(X_train, y_train, groups=groups)
    best_model = search.best_estimator_

    # ---------------------------------------------------------
    # Strict OOF threshold tuning inside the outer training fold
    # ---------------------------------------------------------

    oof_scores = np.zeros(len(y_train), dtype=float)

    for inner_train_idx, inner_valid_idx in inner_cv.split(
        X_train,
        y_train,
        groups
    ):
        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        groups_inner_train = groups[inner_train_idx]
        
        X_inner_valid = X_train.iloc[inner_valid_idx]

        inner_model = clone(best_model)
        

        calibrated_inner_model = make_group_calibrated_model(
            base_model=inner_model,
            X=X_inner_train,
            y=y_inner_train,
            groups=groups_inner_train,
            n_splits=5,
            method="sigmoid",
            random_state=RANDOM_STATE,
        )

        oof_scores[inner_valid_idx] = calibrated_inner_model.predict_proba(
            X_inner_valid
        )[:, 1]

    best_threshold, inner_threshold_f1_macro = tune_threshold_balanced(
        y_train,
        oof_scores
    )

    # ---------------------------------------------------------
    # Final calibrated model trained only on outer training fold
    # ---------------------------------------------------------

    final_model = make_group_calibrated_model(
        base_model=clone(best_model),
        X=X_train,
        y=y_train,
        groups=groups,
        n_splits=5,
        method="sigmoid",
        random_state=RANDOM_STATE,
    )

    final_model.threshold = best_threshold



    return best_model, final_model, best_threshold, search.best_params_


