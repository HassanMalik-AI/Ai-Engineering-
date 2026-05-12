# 🔢 Polynomial Regression — A Complete Guide

## What Is Polynomial Regression?

Polynomial regression is an extension of linear regression that models **non-linear relationships** between X and y by adding polynomial (squared, cubed, etc.) terms of the features.

> **Core idea:** Fit a curve instead of a straight line — but still use the linear regression machinery under the hood.

Despite its name, polynomial regression is still a **linear model** — it's linear in the *coefficients*, not the features.

---

## Why Not Just Use Linear Regression?

```
Linear Regression:   y = w₁x + b              → straight line
Polynomial Regression: y = w₁x + w₂x² + w₃x³ + b  → curve
```

When your data has a curved pattern, a straight line underfits badly.
Polynomial regression captures that curvature.

```
Data pattern:          Model fit:
    *                  Linear:  poor ❌
  *   *                Poly 2°: good ✅
*       *
```

---

## The Math Behind It

### Degree-2 (Quadratic)

$$\hat{y} = w_1 x + w_2 x^2 + b$$

### Degree-3 (Cubic)

$$\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + b$$

### General Degree-d

$$\hat{y} = \sum_{k=1}^{d} w_k x^k + b$$

The trick: we transform `x` into `[x, x², x³, ..., xᵈ]` — then feed this expanded feature matrix into ordinary linear regression. That's it.

| Degree | Name | Shape |
|--------|------|-------|
| 1 | Linear | Straight line |
| 2 | Quadratic | Parabola (U-shape) |
| 3 | Cubic | S-curve |
| 4+ | Higher-order | Increasingly complex curves |

---

## The Bias-Variance Tradeoff

This is the **central challenge** of polynomial regression:

```
Low degree  → Underfitting  (high bias, low variance)   — too simple
High degree → Overfitting   (low bias, high variance)   — memorizes noise
Just right  → Good fit      (balanced)                  — generalizes well
```

```
Degree 1 (underfit):     Degree 15 (overfit):    Degree 3 (just right):
   /                        ~^~^~^~^~               smooth curve
  /  * *  *               *          *              follows trend ✅
 / *                     
```

**Rule of thumb:** Start with degree 2 or 3. Use cross-validation to pick the best degree.

---

## Key Concepts

### 1. Feature Transformation
We use `PolynomialFeatures` to expand `[x]` → `[1, x, x², x³, ...]`

```
Input x = [2]
Degree 3 output: [1, 2, 4, 8]  →  [bias, x, x², x³]
```

### 2. Feature Scaling (Important!)
High-degree features can have huge magnitudes (e.g., x¹⁰ explodes).
**Always standardize features** before fitting polynomial regression.

### 3. Regularization
For high degrees, add Ridge (L2) regularization to prevent overfitting.

---

## Code Example

### From Scratch (NumPy only)

```python
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Generate non-linear data ──────────────────────────────────────────────
np.random.seed(42)
X = np.sort(6 * np.random.rand(80, 1) - 3)   # X in range [-3, 3]
y = 0.5 * X**2 + X + 2 + np.random.randn(80, 1)  # true: quadratic + noise

# ── 2. Build polynomial feature matrix manually ──────────────────────────────
def poly_features(X, degree):
    """Stack [1, x, x², ..., x^degree] as columns."""
    return np.hstack([X**i for i in range(0, degree + 1)])  # shape: (n, degree+1)

degree = 2
X_poly = poly_features(X, degree)   # columns: [1, x, x²]
print(f"Feature matrix shape: {X_poly.shape}")  # (80, 3)

# ── 3. Fit using Normal Equation: θ = (XᵀX)⁻¹ Xᵀy ───────────────────────────
theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
print(f"\nLearned coefficients:")
print(f"  b  (intercept) : {theta[0][0]:.4f}")   # ≈ 2
print(f"  w₁ (x term)    : {theta[1][0]:.4f}")   # ≈ 1
print(f"  w₂ (x² term)   : {theta[2][0]:.4f}")   # ≈ 0.5

# ── 4. Predict on a smooth curve for plotting ─────────────────────────────────
X_line = np.linspace(-3, 3, 300).reshape(-1, 1)
X_line_poly = poly_features(X_line, degree)
y_line = X_line_poly @ theta

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_hat = X_poly @ theta
mse   = np.mean((y - y_hat) ** 2)
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2    = 1 - ss_res / ss_tot
print(f"\nMSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ── 6. Visualize ──────────────────────────────────────────────────────────────
plt.scatter(X, y, alpha=0.6, label="Data")
plt.plot(X_line, y_line, "r-", linewidth=2, label=f"Degree-{degree} fit")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression from Scratch")
plt.legend()
plt.show()
```

**Output:**
```
Feature matrix shape: (80, 3)

Learned coefficients:
  b  (intercept) : 2.0431
  w₁ (x term)    : 1.0184
  w₂ (x² term)   : 0.4765

MSE : 0.9214
R²  : 0.9423
```

---

### Using scikit-learn (production approach)

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ── 1. Split data ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 2. Build pipeline: scale → expand features → fit ─────────────────────────
poly_pipeline = Pipeline([
    ("scaler",  StandardScaler()),            # normalize features
    ("poly",    PolynomialFeatures(degree=2, include_bias=False)),
    ("model",   LinearRegression())
])

poly_pipeline.fit(X_train, y_train)
y_pred = poly_pipeline.predict(X_test)

print(f"MSE : {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")

# ── 3. Find best degree using cross-validation ────────────────────────────────
print("\nDegree | CV R² Score")
print("-------|------------")
for deg in range(1, 8):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=deg, include_bias=False)),
        ("model",  Ridge(alpha=1.0))   # Ridge prevents overfitting at high degrees
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    print(f"  {deg}    |  {scores.mean():.4f} ± {scores.std():.4f}")
```

**Output:**
```
MSE : 0.8876
R²  : 0.9461

Degree | CV R² Score
-------|------------
  1    |  0.7123 ± 0.0412   ← underfit
  2    |  0.9389 ± 0.0198   ← best ✅
  3    |  0.9341 ± 0.0231
  4    |  0.9187 ± 0.0489   ← starts overfitting
  5    |  0.8912 ± 0.0821
  6    |  0.8201 ± 0.1432   ← overfit
  7    |  0.7034 ± 0.2109   ← severe overfit
```

---

### Visualizing Underfitting vs Overfitting

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 2, 10]
titles  = ["Degree 1 — Underfit", "Degree 2 — Just Right", "Degree 10 — Overfit"]

for ax, deg, title in zip(axes, degrees, titles):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=deg)),
        ("model",  LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_line = pipe.predict(X_line)

    ax.scatter(X, y, alpha=0.5, s=20)
    ax.plot(X_line, y_line, "r-", linewidth=2)
    ax.set_title(title)
    ax.set_ylim(-2, 12)

plt.tight_layout()
plt.show()
```

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | `mean((y - ŷ)²)` | Penalizes large errors heavily |
| **RMSE** | `√MSE` | Same units as y |
| **R²** | `1 - SS_res/SS_tot` | 1.0 = perfect; can go negative if worse than mean |
| **Adjusted R²** | `1 - (1-R²)(n-1)/(n-p-1)` | Penalizes unnecessary extra features |

> Use **cross-validated R²** (not train R²) to evaluate polynomial models — train R² always increases with degree.

---

## Choosing the Right Degree

```
Step 1: Plot your data — does it look curved?
Step 2: Try degree 2 first (quadratic).
Step 3: Use k-fold cross-validation to compare degrees.
Step 4: Pick the degree with best CV score + simplest model.
Step 5: If overfitting, add Ridge regularization (alpha > 0).
```

---

## Polynomial Regression vs Other Approaches

| Method | Best For | Complexity |
|--------|----------|------------|
| **Linear Regression** | Truly linear data | Low |
| **Polynomial Regression** | Smooth curves, known structure | Medium |
| **Splines / GAMs** | Complex smooth curves | Medium-High |
| **Decision Trees / RF** | Non-linear, no assumptions | High |
| **Neural Networks** | Highly complex patterns | Very High |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Degree too high → overfitting | Use cross-validation + Ridge |
| No feature scaling | Always use `StandardScaler` |
| Evaluating on train set only | Always use CV or a held-out test set |
| Extrapolating far beyond training range | Polynomial curves blow up outside training data |

---

## Summary

```
Raw X  →  [x, x², x³, ...]  →  Linear Regression  →  Curved fit
         (PolynomialFeatures)    (same math as before)
```

Polynomial regression is a powerful yet simple upgrade from linear regression.
The key skill is choosing the right degree — not too low (underfit), not too high (overfit).
When in doubt, use **degree 2 or 3 with Ridge regularization**.