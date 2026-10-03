"""
Q1 test script for COMP 3105 Assignment 1.

Tests:
1. Q1(a-c): L2, L1, and L-infinity regression on the professor's toy data.
2. Q1(d): Synthetic regression experiment.
3. Q1(e): Concrete Compressive Strength preprocessing and experiment.

This script is NOT part of the assignment submission.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Allow this script to work in either layout:
#
# Original:
#   A1codes.py
#   test_q1.py
#
# Organized GitHub layout:
#   submission/A1codes.py
#   tests/test_q1.py
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent

candidate_roots = [
    HERE,
    HERE.parent,
]

for root in candidate_roots:
    if (root / "A1codes.py").exists():
        sys.path.insert(0, str(root))
        break
    if (root / "submission" / "A1codes.py").exists():
        sys.path.insert(0, str(root / "submission"))
        break

import A1codes


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def find_file(*relative_candidates):
    """
    Return the first existing candidate file.
    """
    search_roots = [
        HERE,
        HERE.parent,
    ]

    for root in search_roots:
        for rel in relative_candidates:
            path = root / rel
            if path.exists():
                return path

    return None


def augment(X):
    """
    Add an intercept column of ones.
    """
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def compute_losses(X, y, w):
    """
    Return (L2, L1, Linf) losses.
    """
    error = X @ w - y

    l2 = np.mean(0.5 * error**2)
    l1 = np.mean(np.abs(error))
    linf = np.max(np.abs(error))

    return np.array([l2, l1, linf], dtype=float)


def print_matrix(title, matrix):
    print(title)
    print(np.array2string(np.asarray(matrix), precision=8, suppress_small=False))
    print()


# ------------------------------------------------------------
# Q1(a-c): Toy regression tests
# ------------------------------------------------------------
def test_toy_regression():
    print("=" * 60)
    print("Q1(a-c): TOY REGRESSION TEST")
    print("=" * 60)

    train_path = find_file(
        "regression_train.csv",
        "data/toy/regression_train.csv",
    )
    test_path = find_file(
        "regression_test.csv",
        "data/toy/regression_test.csv",
    )

    if train_path is None or test_path is None:
        print("SKIPPED: regression_train.csv / regression_test.csv not found.")
        print()
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    Xtrain = augment(train.iloc[:, :-1].to_numpy(dtype=float))
    ytrain = train.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

    Xtest = augment(test.iloc[:, :-1].to_numpy(dtype=float))
    ytest = test.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

    w_L2 = A1codes.minimizeL2(Xtrain, ytrain)
    w_L1 = A1codes.minimizeL1(Xtrain, ytrain)
    w_Linf = A1codes.minimizeLinf(Xtrain, ytrain)

    expected_w_L2 = np.array([
        [-26.72481802],
        [-1.16639040],
    ])

    expected_w_L1 = np.array([
        [-26.88576639],
        [-1.16315391],
    ])

    expected_w_Linf = np.array([
        [-25.33653980],
        [-1.39556938],
    ])

    print("L2 weights:")
    print(w_L2)
    print()

    print("L1 weights:")
    print(w_L1)
    print()

    print("Linf weights:")
    print(w_Linf)
    print()

    assert np.allclose(w_L2, expected_w_L2, atol=1e-6), "L2 weights do not match expected values."
    assert np.allclose(w_L1, expected_w_L1, atol=1e-5), "L1 weights do not match expected values."
    assert np.allclose(w_Linf, expected_w_Linf, atol=1e-5), "Linf weights do not match expected values."

    models = [w_L2, w_L1, w_Linf]

    train_losses = np.vstack([
        compute_losses(Xtrain, ytrain, w)
        for w in models
    ])

    test_losses = np.vstack([
        compute_losses(Xtest, ytest, w)
        for w in models
    ])

    expected_train = np.array([
        [2.00305568, 1.56314076, 7.31711154],
        [2.01278477, 1.55829245, 7.46827750],
        [2.36347873, 1.72061687, 6.62153570],
    ])

    expected_test = np.array([
        [1.66071948, 1.42055149, 5.58874693],
        [1.62892803, 1.40563309, 5.45534197],
        [1.98684780, 1.58480749, 5.02664782],
    ])

    print_matrix("Training losses [rows=L2,L1,Linf models; cols=L2,L1,Linf]:", train_losses)
    print_matrix("Test losses [rows=L2,L1,Linf models; cols=L2,L1,Linf]:", test_losses)

    assert np.allclose(train_losses, expected_train, atol=1e-6), "Toy training losses do not match expected values."
    assert np.allclose(test_losses, expected_test, atol=1e-6), "Toy test losses do not match expected values."

    print("PASS: Toy Q1 regression results match the professor's expected values.")
    print()


# ------------------------------------------------------------
# Q1(d): Synthetic regression
# ------------------------------------------------------------
def test_synthetic_regression():
    print("=" * 60)
    print("Q1(d): SYNTHETIC REGRESSION TEST")
    print("=" * 60)

    train_loss, test_loss = A1codes.synRegExperiments()

    print_matrix("Average training loss:", train_loss)
    print_matrix("Average test loss:", test_loss)

    assert train_loss.shape == (3, 3), f"Expected train_loss shape (3, 3), got {train_loss.shape}"
    assert test_loss.shape == (3, 3), f"Expected test_loss shape (3, 3), got {test_loss.shape}"
    assert np.all(np.isfinite(train_loss)), "train_loss contains NaN or infinity."
    assert np.all(np.isfinite(test_loss)), "test_loss contains NaN or infinity."

    # Each model should minimize its own training metric.
    assert train_loss[0, 0] <= train_loss[1, 0] and train_loss[0, 0] <= train_loss[2, 0], \
        "L2 model should have the lowest training L2 loss."

    assert train_loss[1, 1] <= train_loss[0, 1] and train_loss[1, 1] <= train_loss[2, 1], \
        "L1 model should have the lowest training L1 loss."

    assert train_loss[2, 2] <= train_loss[0, 2] and train_loss[2, 2] <= train_loss[1, 2], \
        "Linf model should have the lowest training Linf loss."

    print("PASS: Synthetic regression shapes and training-loss behavior look correct.")
    print()


# ------------------------------------------------------------
# Q1(e): CCS
# ------------------------------------------------------------
def test_ccs():
    print("=" * 60)
    print("Q1(e): CCS TEST")
    print("=" * 60)

    ccs_path = find_file(
        "Concrete_Data.xls",
        "Concrete Data.xls",
        "data/concrete/Concrete_Data.xls",
        "data/concrete/Concrete Data.xls",
    )

    if ccs_path is None:
        print("SKIPPED: Concrete_Data.xls / Concrete Data.xls not found.")
        print()
        return

    dataset_folder = str(ccs_path.parent.resolve())

    X, y = A1codes.preprocessCCS(dataset_folder)

    print("CCS shapes:")
    print("X:", X.shape)
    print("y:", y.shape)
    print()

    assert X.shape == (1030, 8), f"Expected X shape (1030, 8), got {X.shape}"
    assert y.shape == (1030, 1), f"Expected y shape (1030, 1), got {y.shape}"
    assert np.all(np.isfinite(X)), "X contains NaN or infinity."
    assert np.all(np.isfinite(y)), "y contains NaN or infinity."

    train_loss, test_loss = A1codes.runCCS(dataset_folder)

    print_matrix("CCS average training loss:", train_loss)
    print_matrix("CCS average test loss:", test_loss)

    assert train_loss.shape == (3, 3), f"Expected train_loss shape (3, 3), got {train_loss.shape}"
    assert test_loss.shape == (3, 3), f"Expected test_loss shape (3, 3), got {test_loss.shape}"
    assert np.all(np.isfinite(train_loss)), "CCS train_loss contains NaN or infinity."
    assert np.all(np.isfinite(test_loss)), "CCS test_loss contains NaN or infinity."

    # Sanity check: each optimizer should minimize its own training metric.
    assert train_loss[0, 0] <= train_loss[1, 0] and train_loss[0, 0] <= train_loss[2, 0], \
        "L2 model should have the lowest CCS training L2 loss."

    assert train_loss[1, 1] <= train_loss[0, 1] and train_loss[1, 1] <= train_loss[2, 1], \
        "L1 model should have the lowest CCS training L1 loss."

    assert train_loss[2, 2] <= train_loss[0, 2] and train_loss[2, 2] <= train_loss[1, 2], \
        "Linf model should have the lowest CCS training Linf loss."

    print("PASS: CCS preprocessing and experiment outputs look correct.")
    print()


if __name__ == "__main__":
    test_toy_regression()
    test_synthetic_regression()
    test_ccs()

    print("=" * 60)
    print("ALL AVAILABLE Q1 TESTS PASSED")
    print("=" * 60)
