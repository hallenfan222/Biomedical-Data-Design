"""PCA + Ridge baseline for cell-type/drug differential-expression prediction.

Inputs are categorical cell_type and sm_name identifiers.
Targets are numeric gene-level DE values. PCA is fitted to training targets
only, Ridge predicts PCA scores from one-hot input features, and predictions
are transformed back to the original gene space.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def _as_frame(values):
    """Normalize pandas or array-like inputs to a DataFrame."""
    return values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame(values)


def fit_predict(X_train, y_train, X_predict, n_components=50, alpha=10.0):
    """Fit PCA + multi-output Ridge and predict gene-level DE values.

    X inputs must include categorical columns cell_type and sm_name.
    PCA sees y_train only; validation targets never enter model fitting.
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
    if len(X_train) < 2:
        raise ValueError("At least two training rows are required for PCA.")
    if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")

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

    # Reduce the many gene targets to a smaller set of training-only PCs.
    actual_components = min(n_components, len(X_train) - 1, y_values.shape[1])
    pca = PCA(n_components=actual_components, svd_solver="randomized", random_state=42)
    y_train_pcs = pca.fit_transform(y_values)

    # Ridge predicts each PC score from cell/drug identity features.
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_encoded, y_train_pcs)
    predicted_pcs = ridge.predict(X_predict_encoded)

    # Reconstruct a prediction for every original gene.
    predictions = pca.inverse_transform(predicted_pcs)
    index = X_predict.index if isinstance(X_predict, pd.DataFrame) else None
    return pd.DataFrame(predictions, columns=y_train.columns, index=index)


def mrrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)).mean())
