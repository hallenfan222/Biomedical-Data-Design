"""Ridge baseline: predict gene DE scores directly, without PCA."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def fit_predict(X_train, y_train, X_predict, alpha=4.5):
    """Fit on training DataFrames and return predicted DE scores for each gene."""
    feature_cols = ["cell_type", "sm_name"]

    # Learn categories from training data only, as in the PCA + Ridge baseline.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # Support older scikit-learn versions.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_train_encoded = encoder.fit_transform(X_train[feature_cols].astype(str))
    X_predict_encoded = encoder.transform(X_predict[feature_cols].astype(str))

    # Predict the original gene targets directly.
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_encoded, y_train.to_numpy(dtype=float))
    predictions = ridge.predict(X_predict_encoded)

    return pd.DataFrame(
        predictions, columns=y_train.columns, index=X_predict.index
    )

