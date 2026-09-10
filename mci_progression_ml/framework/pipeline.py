from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import TomekLinks

from mci_progression_ml.config import RANDOM_STATE
from mci_progression_ml.framework.smote import CustomSMOTE
from mci_progression_ml.framework.correlation import CorrelationFilter
from mci_progression_ml.framework.knn_imputer import KNNImputerWithScaling

def make_pipeline(
    model,
    feature_select_model=None,      # <- e.g. SelectFromModel(...)
    use_sampling_for_trees=False,
    sampling_strategy="minority",
    random_state=RANDOM_STATE,
    k_neighbors=5,
    cat_cols=None,
    num_cols=None,
    drop_low_variance=False,
    var_threshold=0.0,
    drop_high_corr=False,
    corr_threshold=0.95,
    sampling_method="smote_tomek",   # "none", "smote", "smote_tomek"
):
    cat_cols = cat_cols or []
    num_cols = num_cols or []
    n_cat = len(cat_cols)

    name = model.__class__.__name__.lower()

    is_linear = any(s in name for s in ["logisticregression", "svc", "linearsvc"])
    is_tree   = any(s in name for s in ["randomforest", "xgb", "xgboost", "lgbm", "catboost"])

    use_scaler = is_linear
    use_sampling = is_linear or (use_sampling_for_trees and is_tree)

    # --- Preprocess ---
    num_steps = [("imputer", KNNImputerWithScaling(
                n_neighbors=5,
                return_scaled=use_scaler
            ))]

    if drop_low_variance:
        num_steps.append(("var", VarianceThreshold(threshold=var_threshold)))

    if drop_high_corr:
        num_steps.append(("corr", CorrelationFilter(threshold=corr_threshold)))
    
    # if use_scaler:
    #     num_steps.append(("scaler", StandardScaler()))

    preprocess = ColumnTransformer(
        transformers=[
            ("num", ImbPipeline(num_steps), num_cols),
            ("cat", ImbPipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps = [("preprocess", preprocess)]

    if use_sampling:
        if sampling_method in {"smote", "smote_tomek"}:
            steps.append((
                "smote",
                CustomSMOTE(
                    n_cat=n_cat,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors,
                )
            ))

        if sampling_method == "smote_tomek":
            steps.append(("tomek", TomekLinks()))

    # IMPORTANT: keep selection AFTER SMOTENC so cat_index stays valid
    if feature_select_model is not None:
        steps.append(("select", feature_select_model))

    steps.append(("clf", model))

    return ImbPipeline(steps=steps)
