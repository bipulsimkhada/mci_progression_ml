import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class KNNImputerWithScaling(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, return_scaled=False):
        self.n_neighbors = n_neighbors
        self.return_scaled = return_scaled

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.imputer_ = KNNImputer(
            n_neighbors=self.n_neighbors
        )
        self.imputer_.fit(X_scaled)

        return self

    def transform(self, X):
        X_scaled = self.scaler_.transform(X)
        X_imputed = self.imputer_.transform(X_scaled)

        if self.return_scaled:
            return X_imputed

        return self.scaler_.inverse_transform(X_imputed)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)
