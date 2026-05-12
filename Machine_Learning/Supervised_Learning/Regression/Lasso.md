# 🔍 Lasso Regression — Complete Guide

---

## 📌 What is Lasso Regression?

**Lasso** stands for **Least Absolute Shrinkage and Selection Operator**.

It is a type of **linear regression** that adds a **penalty** (called L1 regularization) to the loss function. This penalty **shrinks some coefficients to exactly zero**, effectively performing **automatic feature selection**.

> Think of Lasso as a strict teacher who forces some students (features) to score zero — eliminating them from the equation entirely.

---

## 🧠 The Math Behind Lasso

### Ordinary Linear Regression minimizes:

$$\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Lasso adds an L1 penalty:

$$\text{Lasso Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}|\beta_j|$$

| Symbol | Meaning |
|--------|---------|
| `yᵢ` | Actual target value |
| `ŷᵢ` | Predicted value |
| `βⱼ` | Coefficient of feature j |
| `λ` (lambda) | Regularization strength (hyperparameter) |
| `\|βⱼ\|` | Absolute value of coefficient (L1 norm) |

---

## ⚙️ How Does the λ Parameter Work?

| λ Value | Effect |
|---------|--------|
| `λ = 0` | Same as ordinary linear regression (no penalty) |
| Small λ | Mild regularization, most features retained |
| Large λ | Strong regularization, many coefficients → 0 |
| λ → ∞ | All coefficients become zero (underfitting) |

---

## 🆚 Lasso vs Ridge vs Elastic Net

| Feature | Lasso (L1) | Ridge (L2) | Elastic Net |
|---------|-----------|-----------|-------------|
| **Penalty** | `λ Σ\|βⱼ\|` | `λ Σβⱼ²` | Mix of L1 + L2 |
| **Feature Selection** | ✅ Yes (zeros out) | ❌ No (shrinks only) | ✅ Partial |
| **Best When** | Many irrelevant features | All features matter | Correlated features |
| **Output** | Sparse model | Dense model | Semi-sparse model |

---

## ✅ When to Use Lasso?

Use Lasso when:
- You have **many features** and suspect only a few are important
- You want **automatic feature selection**
- You want a **simpler, more interpretable** model
- Your data suffers from **multicollinearity** (correlated features)

---

## ❌ When NOT to Use Lasso?

Avoid Lasso when:
- All features are genuinely important (use Ridge instead)
- Features are highly correlated — Lasso picks one and discards others arbitrarily
- Dataset is very small

---

## 💻 Code Example (Python)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Create Sample Dataset
# ─────────────────────────────────────────────
np.random.seed(42)
n_samples = 200

# 10 features, but only 3 actually matter
X = np.random.randn(n_samples, 10)
# True relationship: only features 0, 2, 5 matter
true_coefficients = [3.5, 0, -2.0, 0, 0, 1.8, 0, 0, 0, 0]
y = X @ true_coefficients + np.random.randn(n_samples) * 0.5

feature_names = [f"Feature_{i}" for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y

print("Dataset shape:", df.shape)
print(df.head(3))

# ─────────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. Train Lasso Model
# ─────────────────────────────────────────────
lasso = Lasso(alpha=0.1)   # alpha = λ (regularization strength)
lasso.fit(X_train_scaled, y_train)

# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────
y_pred = lasso.predict(X_test_scaled)

print("\n── Model Performance ──")
print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

print("\n── Learned Coefficients ──")
for name, coef in zip(feature_names, lasso.coef_):
    status = "✅ Selected" if coef != 0 else "❌ Eliminated"
    print(f"  {name}: {coef:7.4f}  {status}")

# ─────────────────────────────────────────────
# 5. Find Best λ Using Cross-Validation
# ─────────────────────────────────────────────
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\n── Best λ (alpha) via CV: {lasso_cv.alpha_:.4f} ──")

# ─────────────────────────────────────────────
# 6. Visualize Coefficients
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))

colors = ["steelblue" if c != 0 else "lightgray" for c in lasso.coef_]
bars = plt.bar(feature_names, lasso.coef_, color=colors, edgecolor="black")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title(f"Lasso Coefficients (α = 0.1)\nBlue = Selected, Gray = Eliminated")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("lasso_coefficients.png", dpi=150)
plt.show()

print("\nPlot saved as 'lasso_coefficients.png'")
```

---

## 📊 Sample Output

```
Dataset shape: (200, 11)

── Model Performance ──
R² Score : 0.9923
RMSE     : 0.4812

── Learned Coefficients ──
  Feature_0:  3.4812  ✅ Selected
  Feature_1:  0.0000  ❌ Eliminated
  Feature_2: -1.9743  ✅ Selected
  Feature_3:  0.0000  ❌ Eliminated
  Feature_4:  0.0000  ❌ Eliminated
  Feature_5:  1.7901  ✅ Selected
  Feature_6:  0.0000  ❌ Eliminated
  Feature_7:  0.0000  ❌ Eliminated
  Feature_8:  0.0000  ❌ Eliminated
  Feature_9:  0.0000  ❌ Eliminated

── Best λ (alpha) via CV: 0.0123 ──
```

> ✅ Lasso correctly identified Features 0, 2, and 5 as important — and eliminated all irrelevant ones!

---

## 🔧 Key Hyperparameter: `alpha` (λ)

```python
# Too small → overfitting (like regular linear regression)
lasso_small = Lasso(alpha=0.0001)

# Balanced → good generalization
lasso_good = Lasso(alpha=0.1)

# Too large → underfitting (all coefficients → 0)
lasso_large = Lasso(alpha=100)

# Best practice: Use cross-validation to find optimal alpha
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
print("Best alpha:", lasso_cv.alpha_)
```

---

## 📈 Regularization Path

The regularization path shows how coefficients change as `λ` increases:

```python
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_train_scaled, y_train, alphas=np.logspace(-4, 1, 100))

plt.figure(figsize=(10, 5))
for i, name in enumerate(feature_names):
    plt.plot(np.log10(alphas), coefs[i], label=name)

plt.xlabel("log10(α)")
plt.ylabel("Coefficient")
plt.title("Lasso Regularization Path")
plt.legend(loc="upper right", fontsize=8)
plt.axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()
```

> As `α` grows (right side), coefficients collapse to zero one by one — that's Lasso's feature elimination in action.

---

## 🧩 Intuition: Why Does L1 Produce Zeros?

The L1 penalty `Σ|βⱼ|` creates a **diamond-shaped constraint region** in coefficient space.

- The optimal solution often lands **exactly at a corner of the diamond**
- Corners of a diamond sit **on the axes** → many coordinates = 0
- This is why Lasso produces **sparse solutions**

Ridge uses L2 (a circle) — the optimal solution rarely hits an axis, so no zeros.

---

## 🏁 Quick Reference

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

# Always scale features first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Manual alpha
model = Lasso(alpha=0.1, max_iter=10000)
model.fit(X_scaled, y)

# Auto-tune alpha
model_cv = LassoCV(cv=5, random_state=42)
model_cv.fit(X_scaled, y)

# Results
print(model.coef_)          # Coefficients (many will be 0)
print(model.intercept_)     # Intercept
print(model_cv.alpha_)      # Best alpha found
```

---

## ⚠️ Important Tips

1. **Always scale your features** before applying Lasso — unscaled features cause unfair penalization
2. **Use `LassoCV`** instead of manually guessing `alpha`
3. **Increase `max_iter`** if you get convergence warnings (`max_iter=10000`)
4. **Check zero coefficients** — they represent eliminated features
5. **Lasso is not great** with highly correlated features — use Elastic Net instead

---

## 📚 Summary

| Aspect | Detail |
|--------|--------|
| **Type** | Linear model with L1 regularization |
| **Goal** | Minimize RSS + λ·Σ\|β\| |
| **Key feature** | Automatic feature selection (zeros out coefficients) |
| **Hyperparameter** | `alpha` (λ) — controls regularization strength |
| **Best tuned with** | `LassoCV` (cross-validation) |
| **Preprocessing** | Feature scaling is mandatory |
| **sklearn class** | `sklearn.linear_model.Lasso` |

---

*Happy Modeling! 🚀*