# COMP 3105 Assignment 2

import numpy as np
from scipy.optimize import minimize


# Question 1(a): minSqHinge — Squared Hinge Loss (Primal Form)
def minSqHinge(X, y, lamb):
    """Fit a linear classifier with squared hinge loss and L2 regularization.

    Parameters
    ----------
    X : (n, d) array of input features
    y : (n, 1) array of labels in {-1, +1}
    lamb : positive float, L2 regularization strength

    Returns
    -------
    w : (d, 1) learned weights
    w0 : scalar intercept
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape

    def objective(parameters):
        w = parameters[:d]
        w0 = parameters[d]

        margin = y * (X @ w + w0)
        violations = np.maximum(0.0, 1.0 - margin)
        loss = np.sum(violations**2) + (lamb / 2.0) * (w @ w)

        grad_w = -2.0 * (X.T @ (y * violations)) + lamb * w
        grad_w0 = -2.0 * np.sum(y * violations)
        gradient = np.append(grad_w, grad_w0)
        return loss, gradient

    initial = np.zeros(d + 1)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"minSqHinge optimization failed: {result.message}")

    w = result.x[:d].reshape(d, 1)
    w0 = float(result.x[d])
    return w, w0


# Question 1(b): minHinge — Hinge Loss (Primal Form)
def minHinge(X, y, lamb, stabilizer=1e-5):
    """Fit a linear classifier using hinge loss and L2 regularization (QP).

    Minimize sum_i xi_i + (lamb/2)*||w||^2 subject to
    xi_i >= 0 and y_i*(x_i @ w + w0) >= 1 - xi_i.
    The intercept w0 is not regularized (except for QP stabilization).
    """
    from cvxopt import matrix, solvers

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape

    # Decision variables z = [w (d), w0 (1), slack variables (n)].
    n_variables = d + 1 + n

    # 1/2 z.T @ P @ z + q.T @ z
    P = np.zeros((n_variables, n_variables))
    P[:d, :d] = lamb * np.eye(d)
    P += stabilizer * np.eye(n_variables)
    q = np.r_[np.zeros(d + 1), np.ones(n)]

    # G @ z <= h encodes xi >= 0 and y * (X @ w + w0) >= 1 - xi.
    slack_constraints = np.hstack((np.zeros((n, d + 1)), -np.eye(n)))
    margin_constraints = np.hstack((-(y[:, None] * X), -y[:, None], -np.eye(n)))
    G = np.vstack((slack_constraints, margin_constraints))
    h = np.r_[np.zeros(n), -np.ones(n)]

    solution = solvers.qp(
        matrix(P),
        matrix(q),
        matrix(G),
        matrix(h),
        options={"show_progress": False},
    )
    if solution["status"] != "optimal":
        raise RuntimeError(f"minHinge QP failed: {solution['status']}")

    parameters = np.asarray(solution["x"]).reshape(-1)
    w = parameters[:d].reshape(d, 1)
    w0 = float(parameters[d])
    return w, w0


# Question 1(c): classify — Primal Classification
def classify(Xtest, w, w0):
    """Predict {-1, +1} labels for test inputs using a linear classifier.

    A score of exactly zero is classified as +1.
    Returns a column vector of shape (m, 1).
    """
    Xtest = np.asarray(Xtest, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1, 1)

    scores = Xtest @ w + w0
    return np.where(scores >= 0, 1, -1)


# Question 1(d): synExperimentsRegularize — Regularization Experiments
def synExperimentsRegularize():
    """Compare squared hinge and hinge loss under four L2 strengths.

    For each of 100 runs, train and evaluate both losses on the same
    generated train/test data for each lambda and generative model.

    Returns
    -------
    train_acc, test_acc : (4, 6) ndarrays
        Rows: lambda = [0.001, 0.01, 0.1, 1.0].
        Columns: [SqHinge model 1, 2, 3, Hinge model 1, 2, 3].
    """
    from A2helpers import generateData

    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.0]
    gen_model_list = [1, 2, 3]

    shape = (len(lamb_list), len(gen_model_list), n_runs)
    train_acc_sq_hinge = np.zeros(shape)
    test_acc_sq_hinge = np.zeros(shape)
    train_acc_hinge = np.zeros(shape)
    test_acc_hinge = np.zeros(shape)

    # Group ID: A2-104
    np.random.seed(104)

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                w, w0 = minSqHinge(Xtrain, ytrain, lamb)
                train_acc_sq_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)
                test_acc_sq_hinge[i, j, r] = np.mean(classify(Xtest, w, w0) == ytest)

                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)
                test_acc_hinge[i, j, r] = np.mean(classify(Xtest, w, w0) == ytest)

    train_acc = np.hstack(
        (train_acc_sq_hinge.mean(axis=2), train_acc_hinge.mean(axis=2))
    )
    test_acc = np.hstack((test_acc_sq_hinge.mean(axis=2), test_acc_hinge.mean(axis=2)))
    return train_acc, test_acc


# Question 2(a): adjSqHinge — Squared Hinge Loss (Adjoint Form)
def adjSqHinge(X, y, lamb, kernel_func):
    """Train an adjoint/kernel classifier using squared hinge loss and L2 penalty.

    Minimize sum(max(0, 1 - y*(K @ a + a0))**2)
             + (lamb/2)*a.T @ K @ a,
    with K = kernel_func(X, X).

    Parameters
    ----------
    X : (n, d) training data
    y : (n, 1) labels in {-1, +1}
    lamb : positive float regularization parameter
    kernel_func : callable taking two feature matrices

    Returns
    -------
    a : (n, 1) adjoint weights
    a0 : float intercept
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = X.shape[0]
    K = np.asarray(kernel_func(X, X), dtype=float)
    K = 0.5 * (K + K.T)  # Symmetric Gram matrix for a valid kernel.

    def objective(parameters):
        a = parameters[:n]
        a0 = parameters[n]

        margins = y * (K @ a + a0)
        violations = np.maximum(0.0, 1.0 - margins)
        loss = np.sum(violations**2) + (lamb / 2.0) * (a @ K @ a)

        gradient_a = -2.0 * (K @ (y * violations)) + lamb * (K @ a)
        gradient_a0 = -2.0 * np.sum(y * violations)
        return float(loss), np.r_[gradient_a, gradient_a0]

    initial = np.zeros(n + 1)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        # Full BFGS is a more reliable fallback for ill-conditioned kernels.
        result = minimize(
            objective, result.x, jac=True, method="BFGS", options={"maxiter": 3000}
        )
    if not result.success and np.max(np.abs(result.jac)) > 1e-3:
        raise RuntimeError(f"adjSqHinge optimization failed: {result.message}")

    a = result.x[:n].reshape(n, 1)
    a0 = float(result.x[n])
    return a, a0


# Question 2(b): adjHinge — Hinge Loss (Adjoint Form)
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    """Fit a kernel classifier with hinge loss and L2 regularization.

    Solve the adjoint quadratic program using variables [a, a0, xi]:
      minimize sum(xi) + (lamb/2) * a.T @ K @ a
      subject to xi >= 0 and y*(K @ a + a0) >= 1 - xi.
    Here K = kernel_func(X, X), and a0 is the unregularized intercept.

    Returns
    -------
    a : (n, 1) array of adjoint coefficients
    a0 : float intercept
    """
    from cvxopt import matrix, solvers

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = X.shape[0]
    K = np.asarray(kernel_func(X, X), dtype=float)
    K = 0.5 * (K + K.T)  # Gram matrix is symmetric for valid kernels.

    # Optimization variables: z = [a (n), a0 (1), xi (n)].
    n_variables = 2 * n + 1

    # Objective (1/2) z.T @ P @ z + q.T @ z.
    P = np.zeros((n_variables, n_variables))
    P[:n, :n] = lamb * K
    P += stabilizer * np.eye(n_variables)
    q = np.r_[np.zeros(n + 1), np.ones(n)]

    # Constraints G @ z <= h.
    # First: -xi <= 0.
    slack_constraints = np.hstack((np.zeros((n, n + 1)), -np.eye(n)))
    # Second: -y*(K @ a + a0) - xi <= -1.
    margin_constraints = np.hstack((-(y[:, None] * K), -y[:, None], -np.eye(n)))
    G = np.vstack((slack_constraints, margin_constraints))
    h = np.r_[np.zeros(n), -np.ones(n)]

    solution = solvers.qp(
        matrix(P),
        matrix(q),
        matrix(G),
        matrix(h),
        options={"show_progress": False},
    )

    if solution["status"] == "optimal":
        parameters = np.asarray(solution["x"]).reshape(-1)
        if np.all(np.isfinite(parameters)):
            a = parameters[:n].reshape(n, 1)
            a0 = float(parameters[n])
            return a, a0

    # A rank-deficient/ill-conditioned kernel matrix can make CVXOPT return
    # status "unknown" even though this hinge-loss problem is well-defined.
    # Solve the equivalent SVM dual (Question 3(a)), which has only n variables
    # rather than 2*n+1. The representer theorem / KKT conditions give:
    #       a = (y * alpha_dual) / lamb,  a0 = b_dual.
    # This produces the same kernel decision function at the optimum.
    dual_a, dual_b = dualHinge(
        X, y.reshape(-1, 1), lamb, kernel_func, stabilizer=stabilizer
    )
    a = (y.reshape(-1, 1) * dual_a) / lamb
    a0 = float(dual_b)
    return a, a0


# Question 2(c): adjClassify — Classification (Adjoint Form)
def adjClassify(Xtest, a, a0, X, kernel_func):
    """Predict {-1, +1} labels using a kernel classifier.

    Parameters
    ----------
    Xtest : (m, d) test inputs
    a : (n, 1) adjoint coefficients from training
    a0 : scalar intercept
    X : (n, d) original training inputs
    kernel_func : callable that returns an (m, n) kernel matrix

    Returns
    -------
    yhat : (m, 1) predictions, using +1 when the score is exactly zero
    """
    Xtest = np.asarray(Xtest, dtype=float)
    X = np.asarray(X, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)

    scores = np.asarray(kernel_func(Xtest, X), dtype=float) @ a + a0
    return np.where(scores >= 0, 1, -1)


# Question 2(d): synExperimentsKernel — Kernel Experiments (Adjoint Form)
def synExperimentsKernel():
    """Compare squared hinge and hinge classifiers using five kernels.

    Each of 10 runs generates independent train/test datasets for each
    kernel and generative model, and averages their classification accuracy.

    Returns
    -------
    train_acc, test_acc : (5, 6) ndarrays
        Rows: linear, polynomial degree 2, polynomial degree 3,
              Gaussian width 1.0, Gaussian width 0.5.
        Columns: [SqHinge model 1, 2, 3, Hinge model 1, 2, 3].
    """
    from A2helpers import generateData, linearKernel, polyKernel, gaussKernel

    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [
        linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5),
    ]
    gen_model_list = [1, 2, 3]

    shape = (len(kernel_list), len(gen_model_list), n_runs)
    train_acc_sq_hinge = np.zeros(shape)
    test_acc_sq_hinge = np.zeros(shape)
    train_acc_hinge = np.zeros(shape)
    test_acc_hinge = np.zeros(shape)

    # Group ID: A2-104
    np.random.seed(104)

    for r in range(n_runs):
        for i, kernel_func in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                a, a0 = adjSqHinge(Xtrain, ytrain, lamb, kernel_func)
                train_acc_sq_hinge[i, j, r] = np.mean(
                    adjClassify(Xtrain, a, a0, Xtrain, kernel_func) == ytrain
                )
                test_acc_sq_hinge[i, j, r] = np.mean(
                    adjClassify(Xtest, a, a0, Xtrain, kernel_func) == ytest
                )

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_func)
                train_acc_hinge[i, j, r] = np.mean(
                    adjClassify(Xtrain, a, a0, Xtrain, kernel_func) == ytrain
                )
                test_acc_hinge[i, j, r] = np.mean(
                    adjClassify(Xtest, a, a0, Xtrain, kernel_func) == ytest
                )

    train_acc = np.hstack(
        (train_acc_sq_hinge.mean(axis=2), train_acc_hinge.mean(axis=2))
    )
    test_acc = np.hstack((test_acc_sq_hinge.mean(axis=2), test_acc_hinge.mean(axis=2)))
    return train_acc, test_acc


# Question 3(a): dualHinge — Hinge Loss (SVM Dual Form)
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    """Train a kernel SVM using the dual quadratic program.

    Maximize 1.T @ a - (1 / (2*lamb)) * a.T @ diag(y) @ K @ diag(y) @ a
    subject to 0 <= a <= 1 and y.T @ a = 0.

    Returns
    -------
    a : (n, 1) array of dual coefficients (not multiplied by labels)
    b : float intercept in (K @ (y*a))/lamb + b
    """
    from cvxopt import matrix, solvers

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = X.shape[0]
    K = np.asarray(kernel_func(X, X), dtype=float)
    K = 0.5 * (K + K.T)

    # cvxopt minimizes (1/2)*a.T @ P @ a + q.T @ a.
    P = (y[:, None] * K * y[None, :]) / lamb
    P += stabilizer * np.eye(n)
    q = -np.ones(n)

    # G @ a <= h: -a <= 0 and a <= 1.
    G = np.vstack((-np.eye(n), np.eye(n)))
    h = np.r_[np.zeros(n), np.ones(n)]

    # A @ a = b_eq: sum_i y_i*a_i = 0.
    A = y.reshape(1, n)
    b_eq = np.array([0.0])

    solution = solvers.qp(
        matrix(P),
        matrix(q),
        matrix(G),
        matrix(h),
        matrix(A),
        matrix(b_eq),
        options={"show_progress": False},
    )
    if solution["status"] != "optimal":
        raise RuntimeError(f"dualHinge QP failed: {solution['status']}")

    a_flat = np.clip(np.asarray(solution["x"]).reshape(-1), 0.0, 1.0)

    # Recover the intercept from a point strictly inside the box constraints.
    # Choosing alpha closest to 0.5 avoids points numerically near 0 or 1.
    unshifted_scores = (K @ (y * a_flat)) / lamb
    tol = 1e-6
    interior = np.flatnonzero((a_flat > tol) & (a_flat < 1.0 - tol))
    if interior.size:
        index = interior[np.argmin(np.abs(a_flat[interior] - 0.5))]
        b = float(y[index] - unshifted_scores[index])
    else:
        # Degenerate case: no coefficient strictly between 0 and 1.
        # KKT conditions bound the valid intercept from above and below.
        bounds = y - unshifted_scores
        lower = np.where(
            ((a_flat <= tol) & (y > 0)) | ((a_flat >= 1.0 - tol) & (y < 0))
        )[0]
        upper = np.where(
            ((a_flat <= tol) & (y < 0)) | ((a_flat >= 1.0 - tol) & (y > 0))
        )[0]
        if lower.size and upper.size:
            b = float((np.max(bounds[lower]) + np.min(bounds[upper])) / 2.0)
        else:
            index = int(np.argmin(np.abs(a_flat - 0.5)))
            b = float(bounds[index])

    return a_flat.reshape(n, 1), b


# Question 3(b): dualClassify — Classification (SVM Dual Form)
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    """Predict {-1, +1} labels using the trained dual-form SVM.

    For each test point, compute:
        score = kernel_func(Xtest, X) @ (y * a) / lamb + b

    A score of exactly zero is classified as +1.

    Parameters
    ----------
    Xtest : (m, d) array of test features
    a : (n, 1) dual coefficients returned by dualHinge
    b : scalar intercept returned by dualHinge
    X : (n, d) training features
    y : (n, 1) training labels in {-1, +1}
    lamb : positive regularization parameter
    kernel_func : callable returning an (m, n) kernel matrix

    Returns
    -------
    yhat : (m, 1) integer array with values in {-1, +1}
    """
    Xtest = np.asarray(Xtest, dtype=float)
    X = np.asarray(X, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    scores = np.asarray(kernel_func(Xtest, X), dtype=float) @ (y * a) / lamb + b
    return np.where(scores >= 0, 1, -1)


# Question 3(c): cvMnist — Cross-Validation for Handwritten Digit Classification
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    """Select the best dual-form SVM regularization parameter and kernel.

    Uses k-fold cross-validation to classify handwritten 4s (label -1)
    versus 9s (label +1). Every candidate is evaluated on the same
    shuffled folds, and accuracy is averaged across all k validation sets.

    Parameters
    ----------
    dataset_folder : str or path-like
        Directory containing the file A2train.csv.
    lamb_list : list of positive floats
        Candidate regularization strengths.
    kernel_list : list of callables
        Candidate kernel functions taking (X1, X2).
    k : int, default 5
        Number of cross-validation folds.

    Returns
    -------
    cv_acc : (len(lamb_list), len(kernel_list)) ndarray
        Mean validation accuracies over the k folds.
    best_lamb : float
        Candidate lambda with the highest mean validation accuracy.
    best_kernel : callable
        Candidate kernel function with the highest mean validation accuracy.
    """
    import pandas as pd

    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.0
    y = train_data[:, 0].reshape(-1, 1).copy()
    y[y == 4] = -1
    y[y == 9] = 1

    # Same folds are used for every lambda/kernel combination.
    # Group ID: A2-104
    np.random.seed(104)
    shuffled_indices = np.random.permutation(X.shape[0])
    folds = np.array_split(shuffled_indices, k)

    # Fold-first dimension makes it easy to average the fold accuracies.
    cv_acc = np.zeros((k, len(lamb_list), len(kernel_list)))

    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for fold in range(k):
                validation_indices = folds[fold]
                training_indices = np.concatenate(folds[:fold] + folds[fold + 1 :])

                Xtrain = X[training_indices]
                ytrain = y[training_indices]
                Xval = X[validation_indices]
                yval = y[validation_indices]

                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                cv_acc[fold, i, j] = np.mean(yhat == yval)

    # Average over validation folds, keeping lambda/kernel axes intact.
    mean_cv_acc = cv_acc.mean(axis=0)
    best_i, best_j = np.unravel_index(np.argmax(mean_cv_acc), mean_cv_acc.shape)
    best_lamb = lamb_list[best_i]
    best_kernel = kernel_list[best_j]

    return mean_cv_acc, best_lamb, best_kernel
