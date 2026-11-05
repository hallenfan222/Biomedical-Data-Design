"""Linear MRRMSE regression. Each training row is one cell-drug pair."""
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=2, user_api="blas")
def _fit_predict(X_train, y_train, X_predict, alpha, penalty, n_components,
                 max_iter, tol):
    """MRRMSE + alpha/G times the coefficient penalty (G = number of genes).

    L1 uses sum(abs(W)); L2 uses sum(W**2)/2. Intercepts are unpenalized.
    With PCA, W predicts PC scores and row errors include discarded gene variance.
    """
    if not np.isfinite(alpha) or alpha <= 0 or max_iter < 1 or tol <= 0:
        raise ValueError("alpha, max_iter and tol must be positive.")
    y = y_train.to_numpy(dtype=float)
    n, genes = y.shape
    train_blocks = []
    test_blocks = []
    for col in ["cell_type", "sm_name"]:
        encoder = OneHotEncoder(handle_unknown="ignore")
        train_blocks.append(encoder.fit_transform(X_train[[col]]).tocsr())
        test_blocks.append(encoder.transform(X_predict[[col]]).tocsr())

    pca = None
    targets = y
    discarded_sq = np.zeros(n)
    if n_components is not None:
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        pca = PCA(n_components=min(n_components, n - 1, genes),
                  svd_solver="randomized", random_state=42)
        pca.fit(y)
        targets = pca.transform(y)
        discarded_sq = np.sum((y - pca.inverse_transform(targets)) ** 2, axis=1)

    intercept = targets.mean(axis=0)
    coef = [np.zeros((block.shape[1], targets.shape[1])) for block in train_blocks]
    residual = intercept - targets

    def row_errors():
        # Orthogonal PCA decomposition: this is RMSE in original gene space.
        return np.sqrt((np.sum(residual ** 2, axis=1) + discarded_sq) / genes)

    errors = row_errors()
    objective = errors.mean()
    for _ in range(max_iter):
        weights = 1 / np.maximum(errors, 1e-10)
        shift = -np.average(residual, axis=0, weights=weights)
        intercept += shift
        residual += shift
        penalty_sum = 0.0
        # Each one-hot block has disjoint columns. Update all its coefficients
        # together using weighted least squares (L2) or soft thresholding (L1).
        for i, block in enumerate(train_blocks):
            residual -= block @ coef[i]
            numerator = -(block.T @ (weights[:, None] * residual)) / n
            counts = np.asarray(block.T @ weights)[:, None] / n
            if penalty == "L1":
                coef[i] = np.sign(numerator) * np.maximum(abs(numerator) - alpha, 0) / counts
                penalty_sum += np.abs(coef[i]).sum()
            else:
                coef[i] = numerator / (counts + alpha)
                penalty_sum += 0.5 * np.sum(coef[i] ** 2)
            residual += block @ coef[i]
        errors = row_errors()
        updated = errors.mean() + alpha * penalty_sum / genes
        if abs(updated - objective) <= tol * max(1.0, objective):
            break
        objective = updated
    else:
        warnings.warn("MRRMSE fit reached max_iter before convergence.", RuntimeWarning)

    prediction = intercept + test_blocks[0] @ coef[0] + test_blocks[1] @ coef[1]
    if pca is not None:
        prediction = pca.inverse_transform(prediction)
    return pd.DataFrame(prediction, index=X_predict.index, columns=y_train.columns)


def fit_predict(X_train, y_train, X_predict, alpha=0.01, max_iter=300, tol=1e-7):
    return _fit_predict(X_train, y_train, X_predict, alpha, "L1", None, max_iter, tol)
