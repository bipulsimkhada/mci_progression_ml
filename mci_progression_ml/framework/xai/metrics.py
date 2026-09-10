import numpy as np
import pandas as pd
from collections import Counter


def xai_summary(results_by_fold, features, feature_map=None):
    """
    Aggregate feature selection, feature dropping, SHAP importance,
    and permutation importance across folds.

    Rules
    -----
    1. `selected_features` determines whether a feature contributes
       to SHAP / permutation importance.

    2. If a feature is NOT selected in a fold:
           SHAP importance = 0
           permutation importance = 0

       Even if SHAP / permutation importance contains a non-zero
       value for that feature.

    3. *_all_folds:
       Average across all folds, with non-selected folds = 0.

    4. *_selected_only:
       Average only across folds where the feature was selected.

    5. Feature dropping:
       Counts and fractions for variance and correlation filtering.

    Returns
    -------
    pd.DataFrame
        Consolidated XAI summary.
    """

    n_folds = len(results_by_fold)

    # ==========================================================
    # Accumulators
    # ==========================================================

    shap_class0_sum = {f: 0.0 for f in features}
    shap_class1_sum = {f: 0.0 for f in features}

    permutation_f1_sum = {f: 0.0 for f in features}
    permutation_balanced_acc_sum = {f: 0.0 for f in features}
    permutation_pr_auc_sum = {f: 0.0 for f in features}

    # Number of folds where feature was selected
    selected_count = {f: 0 for f in features}

    # ==========================================================
    # Feature-drop counters
    # ==========================================================

    variance_drop_counter = Counter()
    correlation_drop_counter = Counter()

    # ==========================================================
    # Loop through folds
    # ==========================================================

    for fold_result in results_by_fold:

        # ======================================================
        # Selected features
        # ======================================================

        selected_features = set(
            fold_result.get("selected_features", [])
        )

        selected_features = selected_features.intersection(features)

        for feature in selected_features:
            selected_count[feature] += 1

        # ======================================================
        # Variance dropped features
        # ======================================================

        variance_dropped_features = fold_result.get(
            "variance_dropped_features",
            []
        )

        variance_drop_counter.update(
            variance_dropped_features
        )

        # ======================================================
        # Correlation dropped features
        # ======================================================

        correlation_dropped_features = fold_result.get(
            "correlation_dropped_features",
            []
        )

        correlation_drop_counter.update(
            correlation_dropped_features
        )

        # ======================================================
        # SHAP
        # ======================================================

        shap_importance = fold_result.get(
            "shap_importance"
        )

        if shap_importance is not None:

            fold_features = list(
                shap_importance["feature_names"]
            )

            shap_values_class1 = np.asarray(
                shap_importance["shap_values_class1"]
            )

            y_test = np.asarray(
                shap_importance["y_test"]
            )

            # ----------------------------------------------
            # Split SHAP values by class
            # ----------------------------------------------

            shap_class0_values = shap_values_class1[
                y_test == 0
            ]

            shap_class1_values = shap_values_class1[
                y_test == 1
            ]

            # ----------------------------------------------
            # Mean absolute SHAP
            # ----------------------------------------------

            if shap_class0_values.shape[0] > 0:
                mean_shap_class0 = np.abs(
                    shap_class0_values
                ).mean(axis=0)
            else:
                mean_shap_class0 = np.zeros(
                    shap_values_class1.shape[1]
                )

            if shap_class1_values.shape[0] > 0:
                mean_shap_class1 = np.abs(
                    shap_class1_values
                ).mean(axis=0)
            else:
                mean_shap_class1 = np.zeros(
                    shap_values_class1.shape[1]
                )

            # ----------------------------------------------
            # Create feature -> importance lookup
            # ----------------------------------------------

            shap_class0_lookup = dict(
                zip(
                    fold_features,
                    mean_shap_class0
                )
            )

            shap_class1_lookup = dict(
                zip(
                    fold_features,
                    mean_shap_class1
                )
            )

            # ----------------------------------------------
            # ONLY selected features contribute
            # ----------------------------------------------

            for feature in selected_features:

                shap_class0_sum[feature] += (
                    shap_class0_lookup.get(
                        feature,
                        0.0
                    )
                )

                shap_class1_sum[feature] += (
                    shap_class1_lookup.get(
                        feature,
                        0.0
                    )
                )

        # ======================================================
        # Permutation importance
        # ======================================================

        permutation_importance = fold_result.get(
            "perm_importance"
        )

        if permutation_importance is not None:

            fold_features = list(
                permutation_importance["feature_names"]
            )

            permutation_f1 = np.asarray(
                permutation_importance[
                    "importance"
                ][
                    "f1_score"
                ][
                    "importances_mean"
                ]
            )

            permutation_balanced_acc = np.asarray(
                permutation_importance[
                    "importance"
                ][
                    "balanced_acc"
                ][
                    "importances_mean"
                ]
            )

            permutation_pr_auc = np.asarray(
                permutation_importance[
                    "importance"
                ][
                    "pr_auc"
                ][
                    "importances_mean"
                ]
            )

            # ----------------------------------------------
            # Create lookup dictionaries
            # ----------------------------------------------

            permutation_f1_lookup = dict(
                zip(
                    fold_features,
                    permutation_f1
                )
            )

            permutation_balanced_acc_lookup = dict(
                zip(
                    fold_features,
                    permutation_balanced_acc
                )
            )

            permutation_pr_auc_lookup = dict(
                zip(
                    fold_features,
                    permutation_pr_auc
                )
            )

            # ----------------------------------------------
            # ONLY selected features contribute
            # ----------------------------------------------

            for feature in selected_features:

                permutation_f1_sum[feature] += (
                    permutation_f1_lookup.get(
                        feature,
                        0.0
                    )
                )

                permutation_balanced_acc_sum[feature] += (
                    permutation_balanced_acc_lookup.get(
                        feature,
                        0.0
                    )
                )

                permutation_pr_auc_sum[feature] += (
                    permutation_pr_auc_lookup.get(
                        feature,
                        0.0
                    )
                )

    # ==========================================================
    # Helper functions
    # ==========================================================

    def average_all_folds(values):
        """
        Average across all folds.
        Non-selected folds contribute zero.
        """

        if n_folds == 0:
            return {
                feature: 0.0
                for feature in features
            }

        return {
            feature: values[feature] / n_folds
            for feature in features
        }

    def average_selected_only(values):
        """
        Average only across folds where the feature
        was selected.
        """

        return {
            feature: (
                values[feature] / selected_count[feature]
                if selected_count[feature] > 0
                else 0.0
            )
            for feature in features
        }

    # ==========================================================
    # SHAP averages
    # ==========================================================

    shap_class0_all_folds = average_all_folds(
        shap_class0_sum
    )

    shap_class1_all_folds = average_all_folds(
        shap_class1_sum
    )

    shap_class0_selected_only = average_selected_only(
        shap_class0_sum
    )

    shap_class1_selected_only = average_selected_only(
        shap_class1_sum
    )

    # ==========================================================
    # Permutation averages
    # ==========================================================

    permutation_f1_all_folds = average_all_folds(
        permutation_f1_sum
    )

    permutation_balanced_acc_all_folds = average_all_folds(
        permutation_balanced_acc_sum
    )

    permutation_pr_auc_all_folds = average_all_folds(
        permutation_pr_auc_sum
    )

    permutation_f1_selected_only = average_selected_only(
        permutation_f1_sum
    )

    permutation_balanced_acc_selected_only = (
        average_selected_only(
            permutation_balanced_acc_sum
        )
    )

    permutation_pr_auc_selected_only = (
        average_selected_only(
            permutation_pr_auc_sum
        )
    )

    # ==========================================================
    # Drop statistics
    # ==========================================================

    variance_dropped_count = {
        feature: variance_drop_counter[feature]
        for feature in features
    }

    correlation_dropped_count = {
        feature: correlation_drop_counter[feature]
        for feature in features
    }

    total_dropped_count = {
        feature: (
            variance_dropped_count[feature]
            + correlation_dropped_count[feature]
        )
        for feature in features
    }

    if n_folds > 0:

        variance_dropped_fraction = {
            feature:
                variance_dropped_count[feature] / n_folds
            for feature in features
        }

        correlation_dropped_fraction = {
            feature:
                correlation_dropped_count[feature] / n_folds
            for feature in features
        }

        total_dropped_fraction = {
            feature:
                total_dropped_count[feature] / n_folds
            for feature in features
        }

    else:

        variance_dropped_fraction = {
            feature: 0.0
            for feature in features
        }

        correlation_dropped_fraction = {
            feature: 0.0
            for feature in features
        }

        total_dropped_fraction = {
            feature: 0.0
            for feature in features
        }

    # ==========================================================
    # Feature display names
    # ==========================================================

    if feature_map is None:
        display_names = {
            feature: feature
            for feature in features
        }
    else:
        display_names = {
            feature: feature_map.get(
                feature,
                feature
            )
            for feature in features
        }

    # ==========================================================
    # Consolidated XAI summary
    # ==========================================================

    xai_summary = pd.DataFrame({

        # ------------------------------------------------------
        # Feature
        # ------------------------------------------------------

        "feature": [
            display_names[feature]
            for feature in features
        ],

        # ------------------------------------------------------
        # Selection
        # ------------------------------------------------------

        "selected_count": [
            selected_count[feature]
            for feature in features
        ],

        "selection_frequency": [
            (
                selected_count[feature] / n_folds
                if n_folds > 0
                else 0.0
            )
            for feature in features
        ],

        # ------------------------------------------------------
        # Dropping
        # ------------------------------------------------------

        "variance_dropped_count": [
            variance_dropped_count[feature]
            for feature in features
        ],

        "variance_dropped_fraction": [
            variance_dropped_fraction[feature]
            for feature in features
        ],

        "correlation_dropped_count": [
            correlation_dropped_count[feature]
            for feature in features
        ],

        "correlation_dropped_fraction": [
            correlation_dropped_fraction[feature]
            for feature in features
        ],

        "total_dropped_count": [
            total_dropped_count[feature]
            for feature in features
        ],

        "total_dropped_fraction": [
            total_dropped_fraction[feature]
            for feature in features
        ],

        # ------------------------------------------------------
        # SHAP Class 0
        # ------------------------------------------------------

        "shap_class0_all_folds": [
            shap_class0_all_folds[feature]
            for feature in features
        ],

        "shap_class0_selected_only": [
            shap_class0_selected_only[feature]
            for feature in features
        ],

        # ------------------------------------------------------
        # SHAP Class 1
        # ------------------------------------------------------

        "shap_class1_all_folds": [
            shap_class1_all_folds[feature]
            for feature in features
        ],

        "shap_class1_selected_only": [
            shap_class1_selected_only[feature]
            for feature in features
        ],

        # ------------------------------------------------------
        # SHAP balanced
        # ------------------------------------------------------

        "shap_balanced_all_folds": [
            (
                shap_class0_all_folds[feature]
                + shap_class1_all_folds[feature]
            ) / 2
            for feature in features
        ],

        "shap_balanced_selected_only": [
            (
                shap_class0_selected_only[feature]
                + shap_class1_selected_only[feature]
            ) / 2
            for feature in features
        ],

        # ------------------------------------------------------
        # Permutation F1
        # ------------------------------------------------------

        "permutation_f1_all_folds": [
            permutation_f1_all_folds[feature]
            for feature in features
        ],

        "permutation_f1_selected_only": [
            permutation_f1_selected_only[feature]
            for feature in features
        ],

        # ------------------------------------------------------
        # Permutation Balanced Accuracy
        # ------------------------------------------------------

        "permutation_balanced_acc_all_folds": [
            permutation_balanced_acc_all_folds[feature]
            for feature in features
        ],

        "permutation_balanced_acc_selected_only": [
            permutation_balanced_acc_selected_only[feature]
            for feature in features
        ],

        # ------------------------------------------------------
        # Permutation PR-AUC
        # ------------------------------------------------------

        "permutation_pr_auc_all_folds": [
            permutation_pr_auc_all_folds[feature]
            for feature in features
        ],

        "permutation_pr_auc_selected_only": [
            permutation_pr_auc_selected_only[feature]
            for feature in features
        ],
    })

    # ==========================================================
    # Sort by overall SHAP balanced importance
    # ==========================================================

    xai_summary = (
        xai_summary
        .sort_values(
            "shap_balanced_all_folds",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return xai_summary
