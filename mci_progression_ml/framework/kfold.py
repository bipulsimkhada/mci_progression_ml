from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits

class RepeatedStratifiedGroupKFold(_RepeatedSplits):

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            StratifiedGroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    def split(self, X, y, groups):
        return super().split(X, y, groups=groups)