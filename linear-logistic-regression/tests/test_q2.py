import numpy as np
import pandas as pd

import A1codes

# --------------------------------------------------
# TEST 1: Q2(a) find_opt on linear regression
# --------------------------------------------------

print("\n========== Q2(a): FIND_OPT TEST ==========")

train = pd.read_csv("regression_train.csv")

Xtrain = train.iloc[:, :-1].to_numpy(dtype=float)
ytrain = train.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

# Add intercept
Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)

w_analytic = A1codes.minimizeL2(Xtrain, ytrain)

w_scipy = A1codes.find_opt(
    A1codes.linearRegL2Obj, A1codes.linearRegL2Grad, Xtrain, ytrain
)

print("\nAnalytic L2 weights:")
print(w_analytic)

print("\nSciPy weights:")
print(w_scipy)

print("\nAbsolute difference:")
print(np.abs(w_analytic - w_scipy))


# --------------------------------------------------
# TEST 2: Logistic regression toy dataset
# --------------------------------------------------

print("\n========== Q2(b/c): LOGISTIC TEST ==========")

train = pd.read_csv("classification_train.csv")
test = pd.read_csv("classification_test.csv")

Xtrain = train.iloc[:, :-1].to_numpy(dtype=float)
ytrain = train.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

Xtest = test.iloc[:, :-1].to_numpy(dtype=float)
ytest = test.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

# Add intercept
Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)

Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w_logit = A1codes.find_opt(
    A1codes.logisticRegObj, A1codes.logisticRegGrad, Xtrain, ytrain
)

print("\nLogistic weights:")
print(w_logit)

# Scores >= 0 correspond to sigmoid >= 0.5
ytrain_hat = (Xtrain @ w_logit >= 0).astype(float)
ytest_hat = (Xtest @ w_logit >= 0).astype(float)

train_acc = np.mean(ytrain_hat == ytrain)
test_acc = np.mean(ytest_hat == ytest)

print("\nTraining accuracy:")
print(train_acc)

print("\nTest accuracy:")
print(test_acc)


# --------------------------------------------------
# TEST 3: Synthetic classification experiments
# --------------------------------------------------

print("\n========== Q2(c): SYNTHETIC CLASSIFICATION ==========")

train_acc_syn, test_acc_syn = A1codes.synClsExperiments()

print("\nAverage training accuracies:")
print(train_acc_syn)

print("\nAverage test accuracies:")
print(test_acc_syn)

print("\nShapes:")
print("Train:", train_acc_syn.shape)
print("Test: ", test_acc_syn.shape)

# --------------------------------------------------
# TEST 4: Breast Cancer Wisconsin dataset
# --------------------------------------------------

import os

print("\n========== Q2(d): BCW TEST ==========")

folder = os.path.abspath(".")

# Test preprocessing
X_bcw, y_bcw = A1codes.preprocessBCW(folder)

print("\nBCW shapes:")
print("X:", X_bcw.shape)
print("y:", y_bcw.shape)

print("\nUnique labels:")
print(np.unique(y_bcw))

# Run the full 50-run experiment
train_acc_bcw, test_acc_bcw = A1codes.runBCW(folder)

print("\nAverage BCW training accuracy:")
print(train_acc_bcw)

print("\nAverage BCW test accuracy:")
print(test_acc_bcw)
