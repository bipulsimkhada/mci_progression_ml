import json
import numpy as np

def get_binary_scores(estimator, X):
    """Continuous scores for PR-AUC (binary): proba[:,1] if available else decision_function."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None

def convert(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)  # fallback (for safety)
