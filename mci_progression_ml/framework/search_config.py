from functools import partial
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

from scipy.stats import loguniform
from scipy.stats import uniform
from scipy.stats import randint

from mci_progression_ml.config import RANDOM_STATE

select_base = SelectKBest(score_func=f_classif, k=100)
mi_score = partial(mutual_info_classif, random_state=RANDOM_STATE)

select_param_common = {
    "select__k": [20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150],
    "select__score_func": [f_classif, mi_score],
}

prune_param_common = {
    "preprocess__num__var__threshold": [0.0, 1e-6, 1e-5, 1e-4],
    "preprocess__num__corr__threshold": [0.90, 0.95, 0.98],
}

lr_param_dist_none = {
    **prune_param_common,
    "clf__C": loguniform(1e-3, 1e2),
    "clf__solver": ["lbfgs"],
    "clf__class_weight": [None, "balanced"]
}

lr_param_dist_smote = {
    **prune_param_common,
    "smote__k_neighbors": randint(3, 8),   # 3..7
    "clf__C": loguniform(1e-3, 1e2),
    "clf__solver": ["lbfgs"],
    "clf__class_weight": [None]
}

svc_param_dist_none = {
    **prune_param_common,
    "clf__kernel": ["rbf"],
    "clf__C": loguniform(1e-3, 1e3),
    "clf__gamma": loguniform(1e-4, 1e1),
    "clf__class_weight": [None, "balanced"],
}

svc_param_dist_smote = {
    **prune_param_common,
    "smote__k_neighbors": randint(3, 8),   # 3..7
    "clf__kernel": ["rbf"],
    "clf__C": loguniform(1e-3, 1e3),
    "clf__gamma": loguniform(1e-4, 1e1),
    "clf__class_weight": [None],
}


rf_param_dist_none = {
    **prune_param_common,
    "clf__n_estimators": randint(100, 1001),      # 100..1200
    "clf__max_depth": [None] + list(range(3, 31)),
    "clf__min_samples_split": randint(2, 21),     # 2..20
    "clf__min_samples_leaf": randint(1, 11),      # 1..10
    "clf__max_features": ["sqrt", "log2", None],
    "clf__bootstrap": [True, False],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}

rf_param_dist_smote = {
    **prune_param_common,
    "smote__k_neighbors": randint(3, 8),          # 3..7
    "clf__n_estimators": randint(100, 1001),
    "clf__max_depth": [None] + list(range(3, 31)),
    "clf__min_samples_split": randint(2, 21),
    "clf__min_samples_leaf": randint(1, 11),
    "clf__max_features": ["sqrt", "log2", None],
    "clf__bootstrap": [True, False],
    "clf__class_weight": [None],
}



xgb_param_dist_none = {
    **prune_param_common,
    "clf__max_depth": randint(3, 8),                  # 3..7
    "clf__min_child_weight": randint(1, 8),           # 1..7
    "clf__learning_rate": loguniform(1e-3, 3e-1),     # 0.001..0.3
    "clf__n_estimators": randint(100, 1001),          # 100..1000
    "clf__subsample": uniform(0.6, 0.4),              # 0.6..1.0
    "clf__colsample_bytree": uniform(0.6, 0.4),       # 0.6..1.0
    "clf__gamma": uniform(0.0, 2.0),                  # 0..2
    "clf__reg_lambda": loguniform(1e-2, 1e1),         # 0.01..10
    "clf__reg_alpha": loguniform(1e-4, 1e0),          # 0.0001..1
    
}


xgb_param_dist_smote = {
    **prune_param_common,
    "smote__k_neighbors": randint(3, 8),             # 3..7
    "clf__max_depth": randint(3, 8),
    "clf__min_child_weight": randint(1, 8),
    "clf__learning_rate": loguniform(1e-3, 3e-1),
    "clf__n_estimators": randint(100, 1001),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__gamma": uniform(0.0, 2.0),
    "clf__reg_lambda": loguniform(1e-2, 1e1),
    "clf__reg_alpha": loguniform(1e-4, 1e0),
}
