import os

import numpy as np
import pandas as pd
import cvxopt

from scipy.optimize import minimize
from scipy.special import expit


def minimizeL2(X, y):
    """
    Compute the L2 regression solution using the analytic solution.

    Returns:
        d x 1 weight vector.
    """
    w = np.linalg.solve(X.T @ X, X.T @ y)

    return w


def minimizeL1(X, y):
    """
    Compute the L1 regression solution using linear programming.

    Returns:
        d x 1 weight vector.
    """
    n, d = X.shape

    # Optimization variables are [w, delta].
    c = np.concatenate((np.zeros(d), np.ones(n)))

    # delta >= 0
    G1 = np.concatenate((np.zeros((n, d)), -np.eye(n)), axis=1)

    # Xw - y <= delta
    G2 = np.concatenate((X, -np.eye(n)), axis=1)

    # y - Xw <= delta
    G3 = np.concatenate((-X, -np.eye(n)), axis=1)

    G = np.concatenate((G1, G2, G3), axis=0)

    h = np.concatenate((np.zeros(n), y.flatten(), -y.flatten()))

    # Convert to CVXOPT matrices.
    c = cvxopt.matrix(c)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    cvxopt.solvers.options["show_progress"] = False

    result = cvxopt.solvers.lp(c, G, h)

    # First d optimization variables correspond to w.
    w = np.array(result["x"][:d]).reshape(d, 1)

    return w


def minimizeLinf(X, y):
    """
    Compute the L-infinity regression solution using linear programming.

    Returns:
        d x 1 weight vector.
    """
    n, d = X.shape

    # Optimization variables are [w, delta].
    # Objective is to minimize delta.
    c = np.concatenate((np.zeros(d), np.ones(1)))

    # delta >= 0
    G1 = np.concatenate((np.zeros((1, d)), -np.ones((1, 1))), axis=1)

    h1 = np.zeros(1)

    # Xw - y <= delta
    G2 = np.concatenate((X, -np.ones((n, 1))), axis=1)

    h2 = y.flatten()

    # y - Xw <= delta
    G3 = np.concatenate((-X, -np.ones((n, 1))), axis=1)

    h3 = -y.flatten()

    G = np.concatenate((G1, G2, G3), axis=0)

    h = np.concatenate((h1, h2, h3))

    # Convert to CVXOPT matrices.
    c = cvxopt.matrix(c)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    cvxopt.solvers.options["show_progress"] = False

    result = cvxopt.solvers.lp(c, G, h)

    # First d optimization variables correspond to w.
    w = np.array(result["x"][:d]).reshape(d, 1)

    return w


def synRegExperiments():
    """
    Run the synthetic regression experiments.

    Returns:
        train_loss: 3 x 3 average training-loss matrix.
        test_loss:  3 x 3 average test-loss matrix.
    """

    def genData(n_points, is_training=False):
        X = np.random.randn(n_points, d)

        # Add intercept column.
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1)

        y = X @ w_true + np.random.randn(n_points, 1) * noise

        # Introduce an outlier into the training data.
        if is_training:
            y[0] *= -0.1

        return X, y

    def compute_losses(X_data, y_data, w):
        error = X_data @ w - y_data

        # Average L2 loss.
        l2_loss = np.mean(0.5 * error**2)

        # Average L1 loss.
        l1_loss = np.mean(np.abs(error))

        # L-infinity loss.
        linf_loss = np.max(np.abs(error))

        return [l2_loss, l1_loss, linf_loss]

    n_runs = 50
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2

    train_loss = np.zeros((n_runs, 3, 3))

    test_loss = np.zeros((n_runs, 3, 3))

    np.random.seed(101331588)

    for r in range(n_runs):

        w_true = np.random.randn(d + 1, 1)

        Xtrain, ytrain = genData(n_train, is_training=True)

        Xtest, ytest = genData(n_test, is_training=False)

        # Train the three regression models.
        w_L2 = minimizeL2(Xtrain, ytrain)

        w_L1 = minimizeL1(Xtrain, ytrain)

        w_Linf = minimizeLinf(Xtrain, ytrain)

        models = [w_L2, w_L1, w_Linf]

        # Evaluate every model using all three loss metrics.
        for i, w in enumerate(models):

            train_loss[r, i, :] = compute_losses(Xtrain, ytrain, w)

            test_loss[r, i, :] = compute_losses(Xtest, ytest, w)

    # Average results over the 50 runs.
    train_loss = np.mean(train_loss, axis=0)

    test_loss = np.mean(test_loss, axis=0)

    return train_loss, test_loss


def preprocessCCS(dataset_folder):
    """
    Load and preprocess the Concrete Compressive Strength dataset.

    Returns:
        X: n x d raw feature matrix.
        y: n x 1 target vector.
    """
    file_path = os.path.join(dataset_folder, "Concrete Data.xls")

    # UCI currently distributes the file using an underscore.
    if not os.path.exists(file_path):
        file_path = os.path.join(dataset_folder, "Concrete_Data.xls")

    data = pd.read_excel(file_path)

    # All columns except the final column are features.
    X = data.iloc[:, :-1].to_numpy(dtype=float)

    # Final column is concrete compressive strength.
    y = data.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

    return X, y


def runCCS(dataset_folder):
    """
    Run the regression models on the Concrete Compressive Strength
    dataset over 50 random train/test splits.

    Returns:
        train_loss: 3 x 3 average training-loss matrix.
        test_loss:  3 x 3 average test-loss matrix.
    """
    X, y = preprocessCCS(dataset_folder)

    n, d = X.shape

    # Add intercept column.
    X = np.concatenate((np.ones((n, 1)), X), axis=1)

    def compute_losses(X_data, y_data, w):
        error = X_data @ w - y_data

        l2_loss = np.mean(0.5 * error**2)

        l1_loss = np.mean(np.abs(error))

        linf_loss = np.max(np.abs(error))

        return [l2_loss, l1_loss, linf_loss]

    n_runs = 50

    train_loss = np.zeros((n_runs, 3, 3))

    test_loss = np.zeros((n_runs, 3, 3))

    np.random.seed(101331588)

    for r in range(n_runs):

        # Randomly shuffle dataset indices.
        indices = np.random.permutation(n)

        # 50/50 train-test split.
        split = n // 2

        train_indices = indices[:split]
        test_indices = indices[split:]

        Xtrain = X[train_indices]
        ytrain = y[train_indices]

        Xtest = X[test_indices]
        ytest = y[test_indices]

        # Train the three regression models.
        w_L2 = minimizeL2(Xtrain, ytrain)

        w_L1 = minimizeL1(Xtrain, ytrain)

        w_Linf = minimizeLinf(Xtrain, ytrain)

        models = [w_L2, w_L1, w_Linf]

        # Evaluate all models using all three metrics.
        for i, w in enumerate(models):

            train_loss[r, i, :] = compute_losses(Xtrain, ytrain, w)

            test_loss[r, i, :] = compute_losses(Xtest, ytest, w)

    train_loss = np.mean(train_loss, axis=0)

    test_loss = np.mean(test_loss, axis=0)

    return train_loss, test_loss


def linearRegL2Obj(w, X, y):
    """
    Compute the L2 linear regression objective value.

    Returns:
        Scalar objective value.
    """
    n = X.shape[0]

    error = X @ w - y

    obj_val = (error.T @ error / (2 * n)).item()

    return obj_val


def linearRegL2Grad(w, X, y):
    """
    Compute the analytic gradient of the L2 regression objective.

    Returns:
        d x 1 gradient vector.
    """
    n = X.shape[0]

    error = X @ w - y

    gradient = (X.T @ error) / n

    return gradient


def find_opt(obj_func, grad_func, X, y):
    """
    Find an optimal parameter vector using scipy.optimize.minimize.

    Returns:
        d x 1 optimal parameter vector.
    """
    d = X.shape[1]

    # scipy.optimize.minimize requires a 1-D parameter vector.
    w_0 = np.random.randn(d)

    def func(w):
        # Convert the 1-D scipy vector to a column vector.
        w_col = w[:, None]

        return obj_func(w_col, X, y)

    def gd(w):
        # Convert to a column vector for our gradient function.
        w_col = w[:, None]

        # scipy expects the returned gradient to be 1-D.
        return grad_func(w_col, X, y).flatten()

    return minimize(func, w_0, jac=gd)["x"][:, None]


def logisticRegObj(w, X, y):
    """
    Compute the logistic regression cross-entropy objective.

    Returns:
        Scalar objective value.
    """
    z = X @ w

    # Numerically stable versions of the two log terms.
    positive_loss = np.logaddexp(0, -z)

    negative_loss = np.logaddexp(z, 0)

    obj_val = np.mean(y * positive_loss + (1 - y) * negative_loss)

    return float(obj_val)


def logisticRegGrad(w, X, y):
    """
    Compute the analytic gradient of logistic regression.

    Returns:
        d x 1 gradient vector.
    """
    n = X.shape[0]

    z = X @ w

    # Numerically stable sigmoid.
    predictions = expit(z)

    gradient = (X.T @ (predictions - y)) / n

    return gradient


def synClsExperiments():
    """
    Run synthetic binary classification experiments.

    Returns:
        train_acc: 4 x 3 average training-accuracy matrix.
        test_acc:  4 x 3 average test-accuracy matrix.
    """

    def genData(n_points, dim1, dim2):
        """
        Generate synthetic binary classification data.
        """
        c0 = np.ones((1, dim1))

        c1 = -np.ones((1, dim1))

        # Class 0.
        X0 = np.random.randn(n_points, dim1 + dim2)

        X0[:, :dim1] += c0

        # Class 1.
        X1 = np.random.randn(n_points, dim1 + dim2)

        X1[:, :dim1] += c1

        X = np.concatenate((X0, X1), axis=0)

        # Add intercept column.
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1)

        y = np.concatenate((np.zeros((n_points, 1)), np.ones((n_points, 1))), axis=0)

        return X, y

    def runClsExp(m=100, dim1=2, dim2=2):
        """
        Run one synthetic classification experiment.
        """
        n_test = 1000

        Xtrain, ytrain = genData(m, dim1, dim2)

        Xtest, ytest = genData(n_test, dim1, dim2)

        w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # Training predictions.
        ytrain_hat = (expit(Xtrain @ w_logit) >= 0.5).astype(float)

        train_acc = np.mean(ytrain_hat == ytrain)

        # Test predictions.
        ytest_hat = (expit(Xtest @ w_logit) >= 0.5).astype(float)

        test_acc = np.mean(ytest_hat == ytest)

        return train_acc, test_acc

    n_runs = 50

    train_acc = np.zeros((n_runs, 4, 3))

    test_acc = np.zeros((n_runs, 4, 3))

    np.random.seed(101331588)

    for r in range(n_runs):

        # Column 0: vary training-set size.
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)

        # Column 1: vary informative dimensions.
        for i, dim1 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)

        # Column 2: vary uninformative dimensions.
        for i, dim2 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)

    train_acc = np.mean(train_acc, axis=0)

    test_acc = np.mean(test_acc, axis=0)

    return train_acc, test_acc


def preprocessBCW(dataset_folder):
    """
    Load and preprocess the Breast Cancer Wisconsin dataset.

    Returns:
        X: n x d feature matrix.
        y: n x 1 label vector with B = 0 and M = 1.
    """
    file_path = os.path.join(dataset_folder, "wdbc.data")

    data = pd.read_csv(file_path, header=None)

    # Column 1 is the diagnosis.
    y = data.iloc[:, 1].map({"B": 0, "M": 1}).to_numpy(dtype=float).reshape(-1, 1)

    # Columns after ID and diagnosis are the input features.
    X = data.iloc[:, 2:].to_numpy(dtype=float)

    return X, y


def runBCW(dataset_folder):
    """
    Run logistic regression on the Breast Cancer Wisconsin dataset.

    Returns:
        train_acc: average training accuracy over 50 runs.
        test_acc: average test accuracy over 50 runs.
    """
    X, y = preprocessBCW(dataset_folder)

    n, d = X.shape

    # Add intercept column.
    X = np.concatenate((np.ones((n, 1)), X), axis=1)

    n_runs = 50

    train_acc = np.zeros(n_runs)

    test_acc = np.zeros(n_runs)

    np.random.seed(101331588)

    for r in range(n_runs):

        # Randomly shuffle dataset indices.
        indices = np.random.permutation(n)

        # Split as evenly as possible.
        split = n // 2

        train_indices = indices[:split]
        test_indices = indices[split:]

        Xtrain = X[train_indices]
        ytrain = y[train_indices]

        Xtest = X[test_indices]
        ytest = y[test_indices]

        # Train logistic regression.
        w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # Training predictions.
        ytrain_hat = (expit(Xtrain @ w) >= 0.5).astype(float)

        train_acc[r] = np.mean(ytrain_hat == ytrain)

        # Test predictions.
        ytest_hat = (expit(Xtest @ w) >= 0.5).astype(float)

        test_acc[r] = np.mean(ytest_hat == ytest)

    train_acc = np.mean(train_acc)

    test_acc = np.mean(test_acc)

    return train_acc, test_acc
