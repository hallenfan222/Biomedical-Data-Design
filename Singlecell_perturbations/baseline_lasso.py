"""Direct Lasso baseline: one-hot cell/drug inputs predict each gene independently.

Fits training data only. L1 sparsity affects coefficients, not necessarily predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder


def _as_frame(values):
    """Normalize pandas or array-like inputs to a DataFrame."""
    return values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame(values)


def fit_predict(X_train, y_train, X_predict, alpha=0.1, max_iter=10000, tol=1e-4):
    """Fit independent gene-level Lasso regressions and predict gene-level DE values.

    X inputs must include categorical columns cell_type and sm_name.
    Only training targets enter model fitting.
    """
    X_train = _as_frame(X_train)
    X_predict = _as_frame(X_predict)
    y_train = _as_frame(y_train)
    required = ["cell_type", "sm_name"]

    for name, frame in [("X_train", X_train), ("X_predict", X_predict)]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame[required].isna().any().any():
            raise ValueError(f"{name} contains missing category values.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")

    y_values = y_train.to_numpy(dtype=float)
    if not np.isfinite(y_values).all():
        raise ValueError("y_train contains missing or infinite values.")

    # Categories are learned from training data only. Unknown categories in
    # prediction data are safely represented by all-zero indicator columns.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # Compatibility with older scikit-learn releases.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_train_encoded = encoder.fit_transform(X_train[required].astype(str))
    X_predict_encoded = encoder.transform(X_predict[required].astype(str))

    lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    lasso.fit(X_train_encoded, y_values)
    predictions = lasso.predict(X_predict_encoded)

    index = X_predict.index if isinstance(X_predict, pd.DataFrame) else None
    return pd.DataFrame(predictions, columns=y_train.columns, index=index)

