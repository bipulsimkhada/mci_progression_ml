import shap
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mci_progression_ml.config import RANDOM_STATE

def compute_shap_global_importance(
    model,
    X_train,
    X_test,
    y_test,
    max_samples=300,
    random_state=RANDOM_STATE,
    nsamples="auto",
):
    
    selected_features = X_train.columns.tolist()

    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()

    # Use a sampled background, not full X_train
    X_bg = shap.sample(X_train, min(max_samples, len(X_train)), random_state=random_state)

    # explainer = shap.KernelExplainer(f_t, X_bg)
    # shap_values = explainer.shap_values(X_test_t, nsamples=nsamples)

    def normalize_shap_output(shap_values):
        if isinstance(shap_values, list):
            return np.asarray(shap_values[1])  # class 1
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values


    def _shap_chunk(X_chunk):
        # print("Chunk type check:", type(X_chunk), type(X_bg))
        explainer_local = shap.KernelExplainer(model.predict_proba, X_bg, keep_index=True)
        vals = explainer_local.shap_values(X_chunk, nsamples=nsamples)
        return normalize_shap_output(vals)


    # split into chunks
    chunk_size = 12
    chunks = [
        X_test.iloc[i:i + chunk_size]
        for i in range(0, len(X_test), chunk_size)
    ]

    # parallel execution
    chunk_results = Parallel(n_jobs=30, backend="loky")(
        delayed(_shap_chunk)(chunk) for chunk in chunks
    )

    shap_values = np.vstack(chunk_results)

    

    # Normalize output shape
    if isinstance(shap_values, list):
        # Binary classification: [class0, class1]
        if len(shap_values) == 2:
            shap_class1 = np.asarray(shap_values[1])
        else:
            raise ValueError(f"Unexpected number of class outputs: {len(shap_values)}")
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes)
            shap_class1 = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            # already class-specific
            shap_class1 = shap_values
        else:
            raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

    

    if len(selected_features) != shap_class1.shape[1]:
        raise ValueError(
            f"Feature name mismatch: {len(selected_features)} names vs "
            f"{shap_class1.shape[1]} SHAP columns"
        )


    return {
        "feature_names": selected_features,
        "y_test": y_test,
        "shap_values_class1": shap_class1,
    }