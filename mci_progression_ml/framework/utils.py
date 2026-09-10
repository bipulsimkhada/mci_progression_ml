

def inspect_dropped_features(pipe, num_cols):
    num_pipe = pipe.named_steps["preprocess"].named_transformers_["num"]

    current_features = list(num_cols)

    result = {
        "variance_dropped": [],
        "correlation_dropped": [],
        "final_numeric_features": current_features,
    }

    # 1. VarianceThreshold
    if "var" in num_pipe.named_steps:
        var_step = num_pipe.named_steps["var"]
        var_mask = var_step.get_support()

        variance_dropped = [f for f, keep in zip(current_features, var_mask) if not keep]
        current_features = [f for f, keep in zip(current_features, var_mask) if keep]

        result["variance_dropped"] = variance_dropped

    # 2. CorrelationFilter
    if "corr" in num_pipe.named_steps:
        corr_step = num_pipe.named_steps["corr"]
        corr_mask = corr_step.keep_mask_

        correlation_dropped = [f for f, keep in zip(current_features, corr_mask) if not keep]
        current_features = [f for f, keep in zip(current_features, corr_mask) if keep]

        result["correlation_dropped"] = correlation_dropped

    result["final_numeric_features"] = current_features
    return result

import numpy as np

def get_preprocessed_feature_names(pipe):
    """
    Returns feature names output by ColumnTransformer.
    Works when preprocess.verbose_feature_names_out=False (your case).
    """
    pre = pipe.named_steps["preprocess"]
    return np.asarray(pre.get_feature_names_out(), dtype=object)


def get_selected_feature_names(pipe):
    """
    Returns the final feature names reaching the classifier.
    If no selector step exists, returns all preprocessed names.
    """
    names = get_preprocessed_feature_names(pipe)

    sel = pipe.named_steps.get("select", None)
    if sel is None:
        return names

    if hasattr(sel, "get_support"):
        mask = sel.get_support()
        return names[mask]

    # Fallback for selectors that expose selected indices
    if hasattr(sel, "support_"):
        return names[sel.support_]

    raise TypeError(f"Selector of type {type(sel)} doesn't expose get_support/support_.")
