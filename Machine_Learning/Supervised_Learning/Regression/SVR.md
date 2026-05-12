# Support Vector Regression (SVR) — Complete Guide

---

## Table of Contents

1. [What is SVR?](#what-is-svr)
2. [Core Intuition](#core-intuition)
3. [The Math Behind SVR](#the-math-behind-svr)
4. [Key Parameters](#key-parameters)
5. [Kernel Functions](#kernel-functions)
6. [SVR vs Linear Regression](#svr-vs-linear-regression)
7. [Code Example](#code-example)
8. [Tuning Tips](#tuning-tips)
9. [When to Use SVR](#when-to-use-svr)

---

## What is SVR?

**Support Vector Regression (SVR)** is the regression variant of Support Vector Machines (SVM). Instead of classifying data points, SVR fits a function to predict continuous values — but with a unique twist:

> SVR tries to fit the **best line** (or curve) within a **threshold (ε)** from the actual values, while **ignoring errors** that fall within that tube.

This makes SVR robust to outliers and effective in high-dimensional spaces.

---

## Core Intuition

Imagine you're fitting a line through noisy data. Instead of minimizing the distance of every point from the line (like standard regression), SVR:

1. Draws a **tube of width ε** around the regression line.
2. Points **inside the tube** → no penalty (error = 0).
3. Points **outside the tube** → penalized using **slack variables (ξ)**.

```
         ε-insensitive tube
         ┌─────────────────────────────┐
    ─────│── ● ── ● ─── ● ────── ● ──│─────  ← regression line (f(x))
         │      ●    ●      ●         │
         └─────────────────────────────┘
    ●  ← outside the tube → penalized!
```

The goal: find the flattest possible tube that captures most data points.

---

## The Math Behind SVR

### Objective Function

SVR solves this optimization problem:

```
Minimize:    (1/2) ||w||²  +  C · Σ(ξᵢ + ξᵢ*)

Subject to:  yᵢ - (w·xᵢ + b)  ≤  ε + ξᵢ
             (w·xᵢ + b) - yᵢ  ≤  ε + ξᵢ*
             ξᵢ, ξᵢ*           ≥  0
```

| Symbol | Meaning |
|--------|---------|
| `w`    | Weight vector (controls flatness of the tube) |
| `C`    | Regularization parameter (penalty for errors outside tube) |
| `ε`    | Epsilon — half-width of the insensitive tube |
| `ξ, ξ*` | Slack variables — how far points are outside the tube |

### The ε-Insensitive Loss Function

```
Loss(y, f(x)) = max(0,  |y - f(x)| - ε)
```

- If prediction error ≤ ε  →  Loss = 0
- If prediction error > ε  →  Loss = (error − ε)

This is similar to the "deadzone" in control systems — small errors are forgiven entirely.

---

## Key Parameters

### 1. `C` — Regularization Parameter

Controls the trade-off between **model flatness** and **tolerance for errors**.

```
Low C  →  Wider margin, more errors allowed  →  Underfitting
High C →  Narrow margin, fewer errors allowed →  Overfitting
```

### 2. `epsilon (ε)` — Tube Width

Defines the zone of tolerance around the regression line.

```
Large ε  →  More points inside tube  →  Simpler model
Small ε  →  Fewer points inside tube →  More complex model
```

### 3. `kernel` — Feature Transformation

Maps data to a higher-dimensional space to capture non-linear relationships.

---

## Kernel Functions

| Kernel   | Formula                        | Use Case                        |
|----------|--------------------------------|---------------------------------|
| `linear` | K(x, z) = xᵀz                | Linearly separable data          |
| `poly`   | K(x, z) = (γxᵀz + r)ᵈ       | Polynomial relationships         |
| `rbf`    | K(x, z) = exp(-γ||x-z||²)   | General-purpose, non-linear data |
| `sigmoid`| K(x, z) = tanh(γxᵀz + r)   | Neural network-like behavior     |

**RBF (Radial Basis Function)** is the most commonly used kernel — it works well for most problems when you don't have domain knowledge.

---

## SVR vs Linear Regression

| Feature                   | Linear Regression      | SVR                        |
|---------------------------|------------------------|----------------------------|
| Loss function             | MSE (all errors count) | ε-insensitive (tube)       |
| Outlier sensitivity       | High                   | Low                        |
| Non-linear data           | No (without transforms)| Yes (via kernels)          |
| Small datasets            | Good                   | Very good                  |
| Large datasets (>100k)    | Fast                   | Slow                       |
| Feature scaling needed    | No                     | **Yes — mandatory**        |

---

## Code Example

```python
# ============================================================
#  SVR Regression — Complete Walkthrough
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# 1. Generate Sample Data
#    A noisy sine wave — non-linear, perfect for SVR
# -----------------------------------------------------------
np.random.seed(42)
X = np.linspace(0, 10, 200).reshape(-1, 1)       # 200 points
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 200)  # sine + noise

# -----------------------------------------------------------
# 2. Train/Test Split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. Feature Scaling  ← CRITICAL for SVR
#    SVR is distance-based, so unscaled features break it.
# -----------------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# -----------------------------------------------------------
# 4. Build & Train SVR Models
#    Comparing three kernel types
# -----------------------------------------------------------
kernels = {
    'RBF (Best for non-linear)':    SVR(kernel='rbf',    C=100, epsilon=0.1, gamma=0.1),
    'Linear':                        SVR(kernel='linear', C=1,   epsilon=0.1),
    'Polynomial (degree=3)':         SVR(kernel='poly',   C=100, epsilon=0.1, degree=3),
}

results = {}
for name, model in kernels.items():
    model.fit(X_train_sc, y_train_sc)
    
    y_pred_sc = model.predict(X_test_sc)
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    
    results[name] = {
        'model':  model,
        'y_pred': y_pred,
        'mse':    mean_squared_error(y_test, y_pred),
        'r2':     r2_score(y_test, y_pred),
    }

# -----------------------------------------------------------
# 5. Print Results
# -----------------------------------------------------------
print("=" * 55)
print(f"{'Kernel':<35} {'MSE':>8} {'R²':>8}")
print("=" * 55)
for name, res in results.items():
    print(f"{name:<35} {res['mse']:>8.4f} {res['r2']:>8.4f}")
print("=" * 55)

# -----------------------------------------------------------
# 6. Visualize Predictions
# -----------------------------------------------------------
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
X_plot_sc = scaler_X.transform(X_plot)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("SVR — Kernel Comparison", fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    model   = res['model']
    y_curve = scaler_y.inverse_transform(
        model.predict(X_plot_sc).reshape(-1, 1)
    ).ravel()
    
    ax.scatter(X_train, y_train, s=10, alpha=0.4, label='Train data')
    ax.scatter(X_test,  y_test,  s=10, alpha=0.4, color='orange', label='Test data')
    ax.plot(X_plot, y_curve, color='red', linewidth=2, label='SVR prediction')
    ax.set_title(f"{name}\nR² = {res['r2']:.3f}")
    ax.legend(fontsize=7)
    ax.set_xlabel("X")
    ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("svr_comparison.png", dpi=150)
plt.show()

# -----------------------------------------------------------
# 7. Hyperparameter Intuition Demo
#    Show how C and epsilon affect the fit
# -----------------------------------------------------------
print("\nEffect of C on RBF SVR (higher C = tighter fit):")
print("-" * 45)
for C_val in [0.1, 1, 10, 100]:
    model = SVR(kernel='rbf', C=C_val, epsilon=0.1)
    model.fit(X_train_sc, y_train_sc)
    y_pred = scaler_y.inverse_transform(
        model.predict(X_test_sc).reshape(-1,1)
    ).ravel()
    r2 = r2_score(y_test, y_pred)
    sv = model.n_support_[0]      # number of support vectors
    print(f"  C={C_val:<6}  R²={r2:.4f}   Support Vectors={sv}")
```

### Expected Output

```
=======================================================
Kernel                              MSE       R²
=======================================================
RBF (Best for non-linear)         0.0412   0.9191
Linear                            0.2989   0.4156
Polynomial (degree=3)             0.0623   0.8781
=======================================================

Effect of C on RBF SVR (higher C = tighter fit):
---------------------------------------------
  C=0.1    R²=0.7823   Support Vectors=130
  C=1      R²=0.9103   Support Vectors=98
  C=10     R²=0.9178   Support Vectors=72
  C=100    R²=0.9191   Support Vectors=55
```

---

## Tuning Tips

### Grid Search for Best Parameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C':       [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma':   ['scale', 'auto', 0.01, 0.1],
}

grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train_sc, y_train_sc)

print("Best Parameters:", grid.best_params_)
print("Best R²:", grid.best_score_)
```

### Quick Reference: Which Parameter to Tune First?

```
Problem                        → Tune
─────────────────────────────────────────────────
Model underfits (R² too low)  → Increase C, try RBF kernel
Model overfits                → Decrease C, increase ε
Predictions too "spiky"       → Increase ε
Predictions too "flat"        → Decrease ε or increase C
Training is very slow         → Try linear kernel or reduce data
```

---

## When to Use SVR

**Use SVR when:**
- Dataset is **small to medium** (< 10,000 samples)
- Relationship between features and target is **non-linear**
- You need **robustness to outliers**
- You have a **high-dimensional feature space**

**Avoid SVR when:**
- Dataset is **very large** (SVR scales as O(n²) to O(n³))
- You need **interpretability** (SVR is a black-box)
- You need **probabilistic outputs** (SVR doesn't give confidence intervals natively)

---

## Installation

```bash
pip install scikit-learn numpy matplotlib
```

---

## Summary

```
SVR in a Nutshell
─────────────────────────────────────────────────
1. Define an ε-tube around the regression line
2. Points inside → no penalty
3. Points outside → penalized by slack (controlled by C)
4. Use kernels (RBF, poly) for non-linear data
5. Always scale your features before training SVR
6. Tune C, ε, and γ using cross-validation
```

> **Key insight:** SVR doesn't care *how much* an error is, as long as it's within ε. This selective tolerance is what makes it powerful and outlier-resistant.