# COMP 3105 Assignment 3

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.special import logsumexp


def minMulDev(X, Y):
    """Fit a linear multiclass classifier by minimizing multinomial deviance.

    Args:
        X: (n, d) feature matrix, already augmented with a column of ones.
        Y: (n, k) one-hot class labels.

    Returns:
        W: (d, k) matrix of fitted classifier weights.
    """
    n, d = X.shape
    k = Y.shape[1]

    def loss_and_gradient(flat_W):
        W = flat_W.reshape(d, k)
        scores = X @ W

        # Log-sum-exp keeps the cross-entropy numerically stable.
        normalizers = logsumexp(scores, axis=1, keepdims=True)
        loss = np.mean(normalizers[:, 0] - np.sum(Y * scores, axis=1))

        # Softmax probabilities and the analytical cross-entropy gradient.
        probabilities = np.exp(scores - normalizers)
        gradient = X.T @ (probabilities - Y) / n
        return loss, gradient.ravel()

    result = minimize(
        loss_and_gradient,
        np.zeros(d * k),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-11},
    )
    return result.x.reshape(d, k)


def classify(Xtest, W):
    """Predict one-hot class labels using learned linear model weights.

    Args:
        Xtest: (m, d) augmented test-feature matrix.
        W: (d, k) fitted weight matrix.

    Returns:
        Yhat: (m, k) one-hot prediction matrix.
    """
    scores = Xtest @ W  # One score per class for each test example.
    predicted_classes = np.argmax(scores, axis=1)
    return np.eye(W.shape[1], dtype=float)[predicted_classes]


def calculateAcc(Yhat, Y):
    """Return the fraction of correctly predicted one-hot class labels.

    Args:
        Yhat: (m, k) one-hot prediction matrix.
        Y: (m, k) one-hot ground-truth label matrix.

    Returns:
        acc: scalar classification accuracy in the range [0, 1].
    """
    predicted_classes = np.argmax(Yhat, axis=1)
    true_classes = np.argmax(Y, axis=1)
    return float(np.mean(predicted_classes == true_classes))


def PCA(X, k):
    """Find the top-k principal component directions of unaugmented data.

    Args:
        X: (n, d) data matrix.
        k: number of principal components, between 1 and d.

    Returns:
        U: (k, d) array with one principal direction per row.
        mu: (d, 1) column vector containing the training-feature means.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # PCA must use data centered by the training mean.
    mu = X.mean(axis=0).reshape(d, 1)
    X_centered = X - mu.T

    # Directions with greatest variance are the leading eigenvectors.
    scatter = X_centered.T @ X_centered
    eigenvalues, eigenvectors = eigh(scatter, subset_by_index=[d - k, d - 1])

    # eigh returns eigenvalues in ascending order and eigenvectors in columns.
    # Reverse them and transpose so U has the top direction in its first row.
    U = eigenvectors[:, ::-1].T
    return U, mu


def projPCA(Xtest, mu, U):
    """Project test data onto PCA directions learned from training data.

    Args:
        Xtest: (m, d) unaugmented data matrix.
        mu: (d, 1) training-feature mean returned by PCA.
        U: (k, d) principal directions returned by PCA.

    Returns:
        Xproj: (m, k) projected feature matrix.
    """
    # Use the training mean, not a new mean computed from the test samples.
    X_centered = np.asarray(Xtest, dtype=float) - mu.T
    return X_centered @ U.T


def kernelPCA(X, k, kernel_func):
    """Find the top-k normalized kernel PCA coefficients.

    Args:
        X: (n, d) unaugmented training data.
        k: number of components, between 1 and n - 1.
        kernel_func: callable that returns an (n, n) kernel matrix.

    Returns:
        A: (k, n) matrix of kernel PCA coefficients, one per row.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # Center Gram matrix in feature space using training row/column means.
    K = np.asarray(kernel_func(X, X), dtype=float)
    K_centered = (
        K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    )

    # Compensate for tiny floating-point asymmetries before eigendecomposition.
    K_centered = (K_centered + K_centered.T) / 2
    eigenvalues, eigenvectors = eigh(K_centered, subset_by_index=[n - k, n - 1])

    # eigh returns ascending eigenvalues. Reverse to largest first.
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Normalize alpha so alpha.T @ K_centered @ alpha == 1.
    # Near-zero eigenvalues correspond to zero-variance directions.
    scale = max(1.0, np.max(np.abs(K_centered)))
    tol = np.finfo(float).eps * n * scale
    inv_sqrt = np.zeros_like(eigenvalues)
    valid = eigenvalues > tol
    inv_sqrt[valid] = 1.0 / np.sqrt(eigenvalues[valid])
    A = (eigenvectors * inv_sqrt).T
    return A


def projKernelPCA(Xtest, Xtrain, kernel_func, A):
    """Project unseen data onto kernel PCA directions learned from Xtrain."""
    Xtest = np.asarray(Xtest, dtype=float)
    Xtrain = np.asarray(Xtrain, dtype=float)

    # Center test/train similarities using the original training distribution.
    K_train = np.asarray(kernel_func(Xtrain, Xtrain), dtype=float)
    K_test = np.asarray(kernel_func(Xtest, Xtrain), dtype=float)
    K_test_centered = (
        K_test
        - K_train.mean(axis=0, keepdims=True)
        - K_test.mean(axis=1, keepdims=True)
        + K_train.mean()
    )
    return K_test_centered @ A.T


def synClsExperimentsPCA():
    """Compare linear classification after PCA on two synthetic data models.

    Returns:
        train_acc: (2, 2) average training accuracies. Rows correspond to
            PCA dimensions 1 and 2; columns correspond to generative models 1, 2.
        test_acc: (2, 2) average testing accuracies, same row/column ordering.
    """
    from A3helpers import generateData, augmentX

    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    train_acc = np.zeros((len(dim_list), len(gen_model_list), n_runs))
    test_acc = np.zeros_like(train_acc)

    np.random.seed(33)

    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)

                # Learn PCA directions from training data ONLY.
                U, mu = PCA(Xtrain, k)
                Xtrain_proj = projPCA(Xtrain, mu, U)
                Xtest_proj = projPCA(Xtest, mu, U)

                # The linear classifier needs a constant bias feature.
                Xtrain_proj = augmentX(Xtrain_proj)
                Xtest_proj = augmentX(Xtest_proj)

                W = minMulDev(Xtrain_proj, Ytrain)
                train_predictions = classify(Xtrain_proj, W)
                test_predictions = classify(Xtest_proj, W)
                train_acc[i, j, r] = calculateAcc(train_predictions, Ytrain)
                test_acc[i, j, r] = calculateAcc(test_predictions, Ytest)

    return train_acc.mean(axis=2), test_acc.mean(axis=2)


def kmeans(X, k, max_iter=1000):
    """Run K-means clustering using alternating nearest-center assignments.

    Args:
        X: (n, d) matrix of observations.
        k: number of clusters, with 1 < k < n.
        max_iter: maximum number of assignment/center-update iterations.

    Returns:
        Y: (n, k) one-hot cluster assignment matrix.
        U: (k, d) matrix of cluster centers.
        obj_val: value of ||X - Y @ U||_F**2 / (2 * n).
    """
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert max_iter > 0 and 1 < k < n

    # Choose distinct training examples as initial centers; do not set a seed.
    U = X[np.random.choice(n, size=k, replace=False)].copy()

    for _ in range(max_iter):
        # Squared Euclidean distances: D[i, j] = ||X[i] - U[j]||^2.
        D = cdist(X, U, metric="sqeuclidean")

        # Each sample belongs to exactly one nearest cluster.
        cluster_ids = np.argmin(D, axis=1)
        Y = np.eye(k, dtype=float)[cluster_ids]

        old_U = U.copy()
        # Pseudoinverse gives the per-cluster means (zeros if a cluster is empty).
        U = np.linalg.pinv(Y) @ X
        if np.allclose(old_U, U):
            break

    # Compute loss from the final returned assignments AND centers.
    obj_val = float(0.5 / n * np.sum((X - Y @ U) ** 2))
    return Y, U, obj_val


def repeatKmeans(X, k, n_runs=100):
    """Run K-means from multiple random initializations and keep the best result.

    Args:
        X: (n, d) data matrix.
        k: number of clusters, with 1 < k < n.
        n_runs: number of independent K-means restarts.

    Returns:
        Y: (n, k) one-hot assignment matrix from the best run.
        U: (k, d) center matrix from the best run.
        obj_val: smallest objective value across all restarts.
    """
    best_obj_val = float("inf")
    best_Y = None
    best_U = None

    for _ in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        if obj_val < best_obj_val:
            best_Y = Y.copy()
            best_U = U.copy()
            best_obj_val = obj_val

    return best_Y, best_U, best_obj_val


def chooseK(X, k_candidates=[2, 3, 4, 5, 6, 7, 8, 9]):
    """Return the best K-means objective for each candidate cluster count.

    Args:
        X: (n, d) matrix of observations.
        k_candidates: list of candidate numbers of clusters.

    Returns:
        obj_val_list: Python list of best objective values, in the same order
            as k_candidates. Each value is obtained with 100 K-means restarts.
    """
    obj_val_list = []
    for k in k_candidates:
        _, _, obj_val = repeatKmeans(X, k)
        obj_val_list.append(float(obj_val))
    return obj_val_list


def kernelKmeans(X, kernel_func, k, init_Y, max_iter=1000):
    """Cluster observations using squared distances in a kernel feature space.

    Args:
        X: (n, d) input data matrix.
        kernel_func: callable producing an (n, n) Gram matrix for X.
        k: number of clusters, with 1 < k < n.
        init_Y: (n, k) initial one-hot cluster assignments.
        max_iter: maximum number of reassignment iterations.

    Returns:
        Y: (n, k) final one-hot cluster assignments.
        obj_val: last iteration's kernel K-means objective (scalar).
    """
    # Convert before invoking the kernel: integer polynomial kernels can overflow.
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert max_iter > 0 and 1 < k < n

    K = np.asarray(kernel_func(X, X), dtype=float)
    Y = np.asarray(init_Y, dtype=float).copy()
    kernel_diagonal = np.diag(K)[:, None]

    for _ in range(max_iter):
        # Each row of Y_plus defines a cluster's feature-space centroid.
        # Pseudoinverse also handles clusters with no assigned points.
        Y_plus = np.linalg.pinv(Y)

        # D[i, j] is the squared distance from point i to cluster j's
        # centroid in the implicit kernel feature space.
        D = kernel_diagonal + np.diag(Y_plus @ K @ Y_plus.T)[None, :] - 2 * K @ Y_plus.T

        # Reassign each observation to its nearest kernel-space centroid.
        cluster_ids = np.argmin(D, axis=1)
        new_Y = np.eye(k, dtype=float)[cluster_ids]

        if np.allclose(Y, new_Y):
            Y = new_Y
            break
        Y = new_Y

    # Recompute the centers for the FINAL returned membership matrix. This
    # matters when the iteration cap is reached before convergence.
    Y_plus = np.linalg.pinv(Y)
    D_final = (
        kernel_diagonal + np.diag(Y_plus @ K @ Y_plus.T)[None, :] - 2 * K @ Y_plus.T
    )
    obj_val = float(0.5 / n * np.sum(Y * D_final))
    # Very small negative values can arise from floating-point cancellation.
    return Y, max(0.0, obj_val)
