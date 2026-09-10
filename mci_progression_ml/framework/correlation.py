# Drop highly corrlated feature
from sklearn.base import clone, BaseEstimator, TransformerMixin
import numpy as np


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.keep_mask_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.shape[1] <=1:
            self.keep_mask_ = np.ones(X.shape[1], dtype=bool)
            return self
        
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        
        corr = np.nan_to_num(corr, nan=0.0)

        n = corr.shape[0]
        keep = np.ones(n, dtype=bool)

        for j in range(1,n):
            if not keep[j]:
                continue
            for i in range(j):
                if keep[i] and abs(corr[i,j]) > self.threshold:
                    keep[j] = False
                    break

        self.keep_mask_ = keep
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_mask_]
    
    def get_feature_names_out(self, input_features=None):
        input_features = np.asarray(input_features, dtype=object)
        return input_features[self.keep_mask_]