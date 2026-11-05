"""Train-only PCA with gene-space MRRMSE and L1 regularization."""
from baseline_MRRMSE_L1 import _fit_predict


def fit_predict(X_train, y_train, X_predict, n_components=160, alpha=0.01,
                max_iter=300, tol=1e-7):
    return _fit_predict(X_train, y_train, X_predict, alpha, "L1", n_components,
                        max_iter, tol)
