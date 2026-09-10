from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from mci_progression_ml.config import RANDOM_STATE

lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    n_jobs=1
)

svc = SVC(
    class_weight="balanced",   # SVM equivalent of RF balancing
    random_state=RANDOM_STATE,            # only affects probability estimation & shuffling
    probability=False,
)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",      # better monitoring for imbalance
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=1
)
