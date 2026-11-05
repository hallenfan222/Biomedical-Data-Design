"""One-hot linear DE prediction with row-wise RMSE loss and L2 regularization."""
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits


def fit_predict(X_train, y_train, X_predict, alpha=0.01, max_iter=150, tol=1e-7):
    """Minimize mean row RMSE + alpha * sum(W**2) / (2 * number of genes).

    The intercept is unpenalized. All targets stay in gene space.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive.")
    cols = ["cell_type", "sm_name"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    x = encoder.fit_transform(X_train[cols]).tocsr()
    x_test = encoder.transform(X_predict[cols])
    y = y_train.to_numpy(dtype=float)
    n, genes = y.shape
    intercept = y.mean(axis=0)
    residual = intercept - y
    row_rmse = np.sqrt(np.mean(residual**2, axis=1))
    objective = row_rmse.mean()

    # Reweighted least squares: each iteration minimizes a quadratic
    # approximation to MRRMSE, with weights determined by training errors.
    with threadpool_limits(limits=2, user_api="blas"):
        for _ in range(max_iter):
            weights = 1 / np.maximum(row_rmse, 1e-10)
            weighted_x = x.multiply(weights[:, None])
            feature_sum = np.asarray(weighted_x.sum(axis=0)).ravel()
            gram = (x.T @ weighted_x).toarray()
            # Center under the current weights to fit an unpenalized intercept.
            x_mean = feature_sum / weights.sum()
            y_mean = np.average(y, axis=0, weights=weights)
            gram -= np.outer(feature_sum, x_mean)
            gram.flat[::gram.shape[0] + 1] += n * alpha
            rhs = weighted_x.T @ y - feature_sum[:, None] * y_mean
            coef = solve(gram, rhs, assume_a="pos")
            intercept = y_mean - x_mean @ coef
            residual = x @ coef + intercept - y
            row_rmse = np.sqrt(np.mean(residual**2, axis=1))
            new_objective = row_rmse.mean() + alpha * np.sum(coef**2) / (2 * genes)
            if abs(objective - new_objective) <= tol * max(1.0, objective):
                break
            objective = new_objective
        else:
            warnings.warn("MRRMSE fit reached max_iter before convergence.", RuntimeWarning)

    return pd.DataFrame(x_test @ coef + intercept, index=X_predict.index,
                        columns=y_train.columns)
