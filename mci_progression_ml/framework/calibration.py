from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

from mci_progression_ml.config import RANDOM_STATE


def make_group_calibrated_model(
    base_model,
    X,
    y,
    groups,
    n_splits=5,
    method="sigmoid",
    random_state=RANDOM_STATE,
):
    calibration_cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    calibration_splits = list(
        calibration_cv.split(X, y, groups)
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=clone(base_model),
        method=method,
        cv=calibration_splits,
    )

    calibrated_model.fit(X, y)

    return calibrated_model