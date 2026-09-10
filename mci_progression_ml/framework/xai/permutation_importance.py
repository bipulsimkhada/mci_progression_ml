from sklearn.inspection import permutation_importance

from mci_progression_ml.config import RANDOM_STATE

def compute_permutation_importance(
    fitted_model,
    X,
    y,
    scoring,
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
):
    perm = permutation_importance(
        estimator=fitted_model,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    feature_names = list(X.columns)

    return {
        "feature_names": feature_names,
        "importance": perm
    }