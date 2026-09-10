from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.base import BaseSampler
import numpy as np

from mci_progression_ml.config import RANDOM_STATE


class CustomSMOTE(BaseSampler):
    _sampling_type = "over-sampling"

    _parameter_constraints = {
        "n_cat": [int],
        "sampling_strategy": [object],
        "random_state": [object],
        "k_neighbors": [int],
    }

    def __init__(
        self,
        n_cat: int,
        sampling_strategy="minority",
        random_state=RANDOM_STATE,
        k_neighbors=5,
    ):
        self.n_cat = int(n_cat)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _fit_resample(self, X, y):
        X = np.asarray(X)
        n_features = X.shape[1]

        if self.n_cat <=0:
            sm = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
            )

            return sm.fit_resample(X, y)
        
        if self.n_cat >= n_features:
            raise ValueError(
                f"n_cat={self.n_cat} must be < n_features={n_features}. "
                "Check preprocessing output."
            )
        
        cat_idx = list(range(n_features - self.n_cat, n_features))

        smnc = SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )
        return smnc.fit_resample(X, y)
