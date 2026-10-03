# Linear & Logistic Regression from Scratch

A from-scratch implementation of core supervised machine learning algorithms using **NumPy, SciPy, CVXOPT, and pandas**.

This project focuses on understanding the optimization behind regression and binary classification rather than relying on pre-built machine learning libraries.

## Features

- **L2 Linear Regression**
  - Closed-form analytic solution

- **L1 Linear Regression**
  - Reformulated as a linear programming problem
  - Solved with `cvxopt.solvers.lp`

- **L∞ Linear Regression**
  - Reformulated as a linear program
  - Minimizes the maximum absolute residual

- **Logistic Regression**
  - Cross-entropy objective implemented from scratch
  - Analytic gradient
  - Numerically stable loss using `numpy.logaddexp`
  - Optimized with `scipy.optimize.minimize`

- **Synthetic Experiments**
  - Comparison of L2, L1, and L∞ regression
  - Classification experiments varying:
    - training-set size
    - informative dimensions
    - noise dimensions

- **Real-World Datasets**
  - Concrete Compressive Strength dataset
  - Breast Cancer Wisconsin Diagnostic dataset

## Project Structure

```text
linear-logistic-regression-from-scratch/
├── README.md
├── submission/
│   ├── A1codes.py
│   ├── A1report.pdf
│   └── requirements.txt
├── tests/
│   ├── A1testbed.py
│   ├── test_q1.py
│   └── test_q2.py
└── data/
    ├── toy/
    │   ├── classification_train.csv
    │   ├── classification_test.csv
    │   ├── regression_train.csv
    │   └── regression_test.csv
    ├── concrete/
    │   └── Concrete_Data.xls
    └── breast_cancer/
        └── wdbc.data
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r submission/requirements.txt
```

## Running the Tests

Run the regression tests:

```bash
python tests/test_q1.py
```

Run the classification tests:

```bash
python tests/test_q2.py
```

The included tests check the implementations against expected toy-data results and also run the synthetic and real-world experiments.

## Core Algorithms

### L2 Regression

The L2 regression solution is computed analytically:

```text
w = (X^T X)^(-1) X^T y
```

The implementation uses `numpy.linalg.solve` instead of explicitly computing the inverse.

### L1 Regression

L1 regression minimizes the sum of absolute residuals. The problem is converted into a linear program using auxiliary variables representing the absolute errors.

### L∞ Regression

L∞ regression minimizes the largest absolute residual, producing a model that focuses on controlling the worst-case prediction error.

### Logistic Regression

Binary classification is implemented using the sigmoid function and cross-entropy loss. The objective is optimized using SciPy with an analytic gradient.

## Tech Stack

- Python 3.10
- NumPy
- SciPy
- CVXOPT
- pandas
- matplotlib

## What I Learned

This project strengthened my understanding of:

- linear regression under different loss functions
- convex optimization
- linear programming formulations
- gradient-based optimization
- numerical stability in logistic regression
- model generalization and overfitting
- evaluating models across different training conditions

## Notes

This repository implements the algorithms directly rather than using machine learning libraries such as scikit-learn.
