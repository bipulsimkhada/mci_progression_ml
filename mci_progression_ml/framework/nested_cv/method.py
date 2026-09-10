import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score

from mci_progression_ml.config import RANDOM_STATE, NESTED_CV_N_ITER, cog, dem, fs_col
from mci_progression_ml.framework.utils import get_selected_feature_names, inspect_dropped_features
from mci_progression_ml.framework.threshold_tuning import tune_threshold_balanced
from mci_progression_ml.framework.calibration import make_group_calibrated_model
from mci_progression_ml.framework.xai.permutation_importance import compute_permutation_importance
from mci_progression_ml.framework.xai.shap import compute_shap_global_importance
from mci_progression_ml.framework.pipeline import make_pipeline
from mci_progression_ml.framework.nested_cv.metrics import evaluate_outer_fold

from mci_progression_ml.framework.search_config import *

def nested_cv(X_train, y_train, groups, inner_cv, outer_cv, model, sampling, modality, diag_change_period, xai=False):
    model_name = model.__class__.__name__.lower()

    print(f"Running model={model_name} sampling={sampling} modality={modality}")

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

    features = []
    if "cog" in modality:
        features += cog
    
    if "mri" in modality:
        features += fs_col
        params = params | select_param_common

    if "dem" in modality:
        features +=dem

    cat_cols = [f for f in features if f in ["PTGENDER", "PTHAND"]]
    num_cols = [f for f in features if f not in cat_cols]
    
    
    pipe = make_pipeline(
        model,
        feature_select_model=select_base if "mri" in modality else None,
        cat_cols=cat_cols,
        num_cols=num_cols,
        drop_low_variance=True,
        drop_high_corr=True,
        sampling_method=sampling,
        use_sampling_for_trees=True
    )

    
    results = []
    for fold, (train_index, test_index) in enumerate(
        outer_cv.split(X_train, y_train, groups), start=1
    ):
        # print(f"Running fold {fold}")
        df_train_fold = X_train.iloc[train_index]
        df_test_fold  = X_train.iloc[test_index]
        y_train_fold  = y_train[train_index]
        y_test_fold   = y_train[test_index]

        groups_fold = groups[train_index]
        diag_change_test = diag_change_period[test_index]
        
        # Use training labels for the imbalance ratio
        label_count = pd.Series(y_train_fold).value_counts()
        scale_pos_weight = label_count[0] / label_count[1]

        if ("xgb" in model_name or "xgboost" in model_name) and sampling == "none":
            params = params | {"clf__scale_pos_weight": uniform(scale_pos_weight * 0.5, scale_pos_weight)}
        elif ("xgb" in model_name or "xgboost" in model_name) and sampling != "none":
            params = params | {"clf__scale_pos_weight": [1.0]}

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=NESTED_CV_N_ITER,
            scoring={"pr_auc": "average_precision", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit="balanced_accuracy",
            return_train_score=False,
            error_score="raise",
        )

        search.fit(df_train_fold, y_train_fold, groups=groups_fold)
        best_model = search.best_estimator_

        # ---------------------------------------------------------
        # Strict OOF threshold tuning inside the outer training fold
        # ---------------------------------------------------------

        oof_scores = np.zeros(len(y_train_fold), dtype=float)

        for inner_train_idx, inner_valid_idx in inner_cv.split(
            df_train_fold,
            y_train_fold,
            groups_fold
        ):
            X_inner_train = df_train_fold.iloc[inner_train_idx]
            y_inner_train = y_train_fold[inner_train_idx]
            groups_inner_train = groups_fold[inner_train_idx]
            
            X_inner_valid = df_train_fold.iloc[inner_valid_idx]

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
            y_train_fold,
            oof_scores
        )

        # ---------------------------------------------------------
        # Final calibrated model trained only on outer training fold
        # ---------------------------------------------------------

        final_model = make_group_calibrated_model(
            base_model=clone(best_model),
            X=df_train_fold,
            y=y_train_fold,
            groups=groups_fold,
            n_splits=5,
            method="sigmoid",
            random_state=RANDOM_STATE,
        )


        outer_eval = evaluate_outer_fold(final_model, df_test_fold, y_test_fold, best_threshold, diag_change_test)

        result = {
            "fold": fold,
            "outer_f1_macro": outer_eval["f1_macro"],
            "outer_pr_auc": outer_eval["pr_auc"],
            "outer_roc": outer_eval["roc_auc"],
            "outer_balanced_acc": outer_eval["balanced_acc"],
            "outer_sensitivity": outer_eval["sensitivity"],
            "outer_specificity": outer_eval["specificity"],
            "best_params": search.best_params_,

            "best_threshold": best_threshold,
            "best_score_threshold": inner_threshold_f1_macro,

            # Calibrated diagnostics
            "calibrated_brier_score": outer_eval["brier_score"],
            "calibrated_mean_predicted_risk": outer_eval["mean_predicted_risk"],
            "calibrated_observed_event_rate": outer_eval["observed_event_rate"],
            "calibrated_calibration_bias": outer_eval["calibration_bias"],
            "calibrated_calibration_direction": outer_eval["calibration_direction"],

            "outer_y_true": outer_eval["y_true"],
            "outer_y_pred": outer_eval["y_pred"],
            "outer_y_score": outer_eval["y_score"],
            "outer_pr_curve": outer_eval["pr_curve"],
            "outer_roc_curve": outer_eval["roc_curve"],
            "sensitivity_by_period": outer_eval["sensitivity_by_period"],
            "selected_features": get_selected_feature_names(best_model),
        }

        if xai:
            print("Running xai")
            info = inspect_dropped_features(best_model, num_cols)

            def f1_macro_with_threshold(estimator, X, y_true):
                proba = estimator.predict_proba(X)[:, 1]
                y_pred = (proba >= best_threshold).astype(int)
                return f1_score(y_true, y_pred, average="macro")
            
            def balanced_acc_at_threshold(estimator, X, y):
                probs = estimator.predict_proba(X)[:, 1]
                preds = (probs >= best_threshold).astype(int)
                return balanced_accuracy_score(y, preds)
            
            def pr_auc_scorer(estimator, X, y):
                probs = estimator.predict_proba(X)[:, 1]
                return average_precision_score(y, probs)
            
            scores = {
                "f1_score": f1_macro_with_threshold,
                "balanced_acc": balanced_acc_at_threshold,
                "pr_auc": pr_auc_scorer
            }

            perm_importance_calibrated = compute_permutation_importance(
                fitted_model=final_model,
                X=df_test_fold,
                y=y_test_fold,
                scoring=scores,
                n_repeats=50,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            result.update({
                "variance_dropped_features": info["variance_dropped"],
                "correlation_dropped_features": info["correlation_dropped"],
                "perm_importance": perm_importance_calibrated,
            })

            shap_importance_calibrated = compute_shap_global_importance(
                final_model,
                df_train_fold,
                df_test_fold,
                y_test_fold,
                max_samples=100,
                random_state=RANDOM_STATE,
                nsamples="auto",
            )

            result.update({
                "shap_importance": shap_importance_calibrated,
            })

        results.append(result)
    
    return results