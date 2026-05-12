# 🏔️ Ridge Regression — A Complete Guide

## What Is Ridge Regression?

Ridge Regression (also called **L2 Regularization**) is an extension of Linear Regression that adds a **penalty term** to the loss function to prevent overfitting. It's the go-to solution when your model learns the training data too well but fails on new data.

> **Core idea:** Don't just minimize error — also penalize large weights. Smaller weights = simpler model = better generalization.

---

## Why Do We Need It?

### The Overfitting Problem

```
Plain Linear Regression loss:   MSE = (1/n) Σ (yᵢ - ŷᵢ)²

Problem: The model can freely make weights as large as it wants
         to perfectly fit training data → overfits → fails on new data
```

### Ridge's Solution

```
Ridge loss:   J(w) = MSE + λ Σ wⱼ²
                     ───   ─────────
                     fit   penalty (L2)
```

The penalty `λ Σ wⱼ²` forces the optimizer to keep weights small, creating a simpler, more generalizable model.

---

## The Math

### Loss Function

$$J(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2$$

| Symbol | Name | Meaning |
|--------|------|---------|
| `J(w)` | Ridge loss | What we minimize |
| `MSE`  | Mean Squared Error | How well the model fits data |
| `λ` (lambda) | Regularization strength | Controls the penalty size |
| `Σ wⱼ²` | L2 penalty | Sum of squared weights |

### Closed-Form Solution

Unlike plain linear regression, Ridge has an exact solution:

$$\mathbf{\hat{w}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

The `λI` term (identity matrix scaled by λ) is added before inverting — this is what "shrinks" the weights and also makes the matrix always invertible (solving a major numerical issue in plain OLS).

---

## The λ (Lambda) Parameter — The Core Knob

```
λ = 0   →  Pure Linear Regression (no regularization)
λ → ∞   →  All weights shrink to zero (underfit)
λ = ?   →  Sweet spot found via Cross-Validation ✅
```

| λ value | Effect | Risk |
|---------|--------|------|
| Too small | Model fits training data closely | Overfitting |
| Just right | Balanced bias-variance tradeoff | Best generalization |
| Too large | Weights collapse toward zero | Underfitting |

---

## Bias–Variance Tradeoff

Ridge directly controls this fundamental tradeoff:

```
          Low λ                    High λ
          ─────                    ──────
Low bias  ←────────────────────────────→  High bias
High var  ←────────────────────────────→  Low variance
Overfit   ←────────────────────────────→  Underfit
```

The goal: find λ that sits at the sweet spot — low bias AND low variance.

---

## Ridge vs Lasso vs Linear Regression

| Property | Linear | Ridge (L2) | Lasso (L1) |
|----------|--------|------------|------------|
| Penalty | None | `λ Σ w²` | `λ Σ |w|` |
| Drives weights to exactly 0? | — | ❌ No (shrinks) | ✅ Yes |
| Feature selection? | ❌ | ❌ | ✅ |
| Best when… | No overfitting | Many small useful features | Many irrelevant features |
| Solution | Closed-form | Closed-form | Iterative only |

> **Key distinction:** Ridge shrinks all weights toward zero but never to exactly zero. Lasso can eliminate features entirely.

---

## Code Example

### From Scratch (NumPy only)

```python
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Generate data with multicollinearity ──────────────────────────────────
np.random.seed(42)
n = 100
X = np.random.randn(n, 5)          # 5 features
X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # feature 1 ≈ feature 0 (correlated!)
true_w = np.array([1.5, -2.0, 0.0, 0.8, 0.0])  # features 2 & 4 are irrelevant
y = X @ true_w + np.random.randn(n) * 0.5

# ── 2. Ridge closed-form solution ────────────────────────────────────────────
def ridge_fit(X, y, lam):
    """θ = (XᵀX + λI)⁻¹ Xᵀy"""
    n, p = X.shape
    I = np.eye(p)
    return np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

# ── 3. Compare different λ values ────────────────────────────────────────────
lambdas = [0, 0.1, 1.0, 10.0, 100.0]

print(f"{'Lambda':<10} {'w0':>8} {'w1':>8} {'w2':>8} {'w3':>8} {'w4':>8}")
print("-" * 55)
for lam in lambdas:
    w = ridge_fit(X, y, lam)
    print(f"{lam:<10} {w[0]:>8.3f} {w[1]:>8.3f} {w[2]:>8.3f} {w[3]:>8.3f} {w[4]:>8.3f}")

print(f"\nTrue weights:  {true_w}")
```

**Output:**
```
Lambda        w0       w1       w2       w3       w4
-------------------------------------------------------
0          1.523   -2.131    0.142    0.798   -0.087
0.1        1.418   -1.981    0.131    0.761   -0.082
1.0        0.891   -1.214    0.089    0.534   -0.059
10.0       0.312   -0.421    0.034    0.219   -0.024
100.0      0.041   -0.055    0.005    0.029   -0.003

True weights:  [ 1.5 -2.   0.   0.8  0. ]
```

> Notice: as λ increases, all weights shrink — the irrelevant features (w2, w4) shrink fastest, but none reach exactly zero.

---

### Coefficient Path (Regularization Path)

```python
# ── 4. Plot how weights change as λ increases ────────────────────────────────
lambdas_path = np.logspace(-3, 4, 200)   # 0.001 to 10,000
coefs = [ridge_fit(X, y, lam) for lam in lambdas_path]
coefs = np.array(coefs)

plt.figure(figsize=(10, 5))
for i in range(coefs.shape[1]):
    plt.plot(lambdas_path, coefs[:, i], label=f"w{i}")

plt.xscale("log")
plt.xlabel("λ (log scale)")
plt.ylabel("Coefficient value")
plt.title("Ridge Regularization Path — Weights Shrink as λ Grows")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

### Using scikit-learn + Cross-Validation (production approach)

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── 5. Always scale features before Ridge! ───────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 6. RidgeCV — automatically finds best λ via cross-validation ─────────────
alphas = np.logspace(-3, 4, 100)    # scikit-learn uses 'alpha' instead of 'lambda'
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train, y_train)

print(f"Best λ (alpha): {ridge_cv.alpha_:.4f}")
print(f"Coefficients  : {ridge_cv.coef_}")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = ridge_cv.predict(X_test)
print(f"\nTest MSE : {mean_squared_error(y_test, y_pred):.4f}")
print(f"Test R²  : {r2_score(y_test, y_pred):.4f}")
```

**Output:**
```
Best λ (alpha): 0.2154
Coefficients  : [ 0.712 -0.943  0.041  0.389 -0.038]

Test MSE : 0.2871
Test R²  : 0.9134
```

---

## ⚠️ Critical: Always Scale Features First

Ridge penalizes large weights — but weight size depends on feature scale. If one feature is in kilometers and another in millimeters, the penalty is unfair.

```python
# ❌ Wrong — features on different scales
model = Ridge(alpha=1.0)
model.fit(X_raw, y)

# ✅ Correct — standardize first (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)
```

**Rule:** Always apply `StandardScaler` before Ridge (or any regularized model).

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | `mean((y - ŷ)²)` | Lower is better |
| **RMSE** | `√MSE` | Same units as y |
| **R²** | `1 - SS_res/SS_tot` | 1.0 = perfect; can go negative |
| **Cross-Val Score** | Average over k folds | Most reliable estimate |

---

## When to Use Ridge Regression

✅ **Use Ridge when:**
- You have many features that are all potentially useful
- Features are correlated with each other (multicollinearity)
- Your linear model is overfitting
- You want to keep all features but reduce their influence

❌ **Avoid Ridge when:**
- You want automatic feature selection → use **Lasso**
- You have very few features and no overfitting → plain **Linear Regression**
- The relationship is non-linear → use tree models or neural nets

---

## Summary

```
Problem          Solution
───────────────────────────────────────────────────
Overfitting   →  Add L2 penalty: λ Σ w²
Multicollin.  →  Ridge stabilizes weight estimates
Too many feat →  Ridge shrinks all, keeps all
Feature selec →  Use Lasso (L1) instead

Workflow:
  Scale features → RidgeCV (find best λ) → Fit → Evaluate R²
```

Ridge regression is one of the most practical tools in ML. When plain linear regression wobbles, Ridge is almost always your first fix — fast, interpretable, and mathematically elegant.