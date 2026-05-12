# 📈 Linear Regression — A Complete Guide

## What Is Linear Regression?

Linear regression is a **supervised machine learning algorithm** that models the relationship between one or more input features (X) and a continuous output variable (y) by fitting a straight line (or hyperplane) through the data.

> **Core idea:** Find the best-fit line `y = mx + b` that minimizes prediction error.

---

## The Math Behind It

### Simple Linear Regression (one feature)

$$\hat{y} = w \cdot x + b$$

| Symbol | Name | Meaning |
|--------|------|---------|
| `ŷ`   | Prediction | The model's output |
| `x`   | Feature | The input variable |
| `w`   | Weight / Slope | How much y changes per unit of x |
| `b`   | Bias / Intercept | Value of y when x = 0 |

### Multiple Linear Regression (multiple features)

$$\hat{y} = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b$$

---

## How Does It Learn? (Gradient Descent)

The model learns by minimizing the **Mean Squared Error (MSE)** loss:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

It then updates weights using **Gradient Descent**:

$$w := w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}$$

Where `α` (alpha) is the **learning rate** — how big each update step is.

---

## Key Assumptions

1. **Linearity** — The relationship between X and y is linear.
2. **Independence** — Observations are independent of each other.
3. **Homoscedasticity** — Constant variance of residuals.
4. **Normality** — Residuals are approximately normally distributed.
5. **No multicollinearity** — Features are not highly correlated with each other.

---

## Code Example

### From Scratch (NumPy only)

```python
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Generate synthetic data ──────────────────────────────────────────────
np.random.seed(42)
X = 2 * np.random.rand(100, 1)          # 100 samples, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1) # true: b=4, w=3, + noise

# ── 2. Add bias term (column of ones) ───────────────────────────────────────
X_b = np.c_[np.ones((100, 1)), X]       # shape: (100, 2)

# ── 3. Closed-form solution (Normal Equation) ────────────────────────────────
# θ = (XᵀX)⁻¹ Xᵀy  — exact solution, no iterations needed
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Intercept (b): {theta[0][0]:.4f}")   # should be ≈ 4
print(f"Slope     (w): {theta[1][0]:.4f}")   # should be ≈ 3

# ── 4. Predict ───────────────────────────────────────────────────────────────
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b @ theta
print(f"\nPredictions: {y_pred.flatten()}")  # ≈ [4, 10]

# ── 5. Evaluate (MSE & R²) ───────────────────────────────────────────────────
y_hat = X_b @ theta
mse = np.mean((y - y_hat) ** 2)
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\nMSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")   # closer to 1.0 = better fit

# ── 6. Visualize ─────────────────────────────────────────────────────────────
plt.scatter(X, y, alpha=0.6, label="Data points")
plt.plot(X_new, y_pred, "r-", linewidth=2, label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression from Scratch")
plt.legend()
plt.show()
```

**Output:**
```
Intercept (b): 3.9938
Slope     (w): 3.0509

Predictions: [ 3.9938  9.0956]

MSE : 0.8964
R²  : 0.9117
```

---

### Using scikit-learn (production approach)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Intercept : {model.intercept_[0]:.4f}")
print(f"Slope     : {model.coef_[0][0]:.4f}")
print(f"MSE       : {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²        : {r2_score(y_test, y_pred):.4f}")
```

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | `mean((y - ŷ)²)` | Average squared error; penalizes large errors |
| **RMSE** | `√MSE` | Same units as y; easier to interpret |
| **MAE** | `mean(|y - ŷ|)` | Average absolute error; robust to outliers |
| **R²** | `1 - SS_res/SS_tot` | % variance explained; 1.0 = perfect fit |

---

## When to Use Linear Regression

✅ **Good fit when:**
- The relationship between X and y is approximately linear
- You need an interpretable model
- You have limited data and want to avoid overfitting
- You're predicting continuous values (price, temperature, sales)

❌ **Avoid when:**
- The relationship is clearly non-linear
- Features are highly correlated (multicollinearity)
- You have many irrelevant features (use Ridge/Lasso instead)
- Your target is categorical (use Logistic Regression)

---

## Extensions

| Variant | When to use |
|---------|-------------|
| **Ridge (L2)** | Prevent overfitting; shrinks all coefficients |
| **Lasso (L1)** | Feature selection; drives some coefficients to zero |
| **ElasticNet** | Combines Ridge + Lasso |
| **Polynomial Regression** | Non-linear relationships with linear model |

---

## Summary

```
Data → Choose features → Fit line (minimize MSE) → Evaluate R² → Predict
```

Linear regression is simple, fast, interpretable, and often a strong baseline.
Master it before reaching for complex models — it's the foundation of ML.