import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)

    y = np.asarray(y, dtype=float)

    if X.ndim == 1:

        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    w = np.zeros(n_features)

    b = 0.0

    eps = 1e-15

    for _ in range(steps):

        z = X @ w + b

        p = _sigmoid(z)


        p_safe = np.clip(p, eps, 1 - eps)

        loss = -np.mean(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))

        grad_w = (X.T @ (p - y)) / n_samples

        grad_b = np.mean(p - y)

        w -= lr * grad_w

        b -= lr * grad_b

    return w, b