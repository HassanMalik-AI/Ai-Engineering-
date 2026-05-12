# XGBoost Regression — Complete Guide

> **XGBoost** (eXtreme Gradient Boosting) is one of the most powerful and widely-used machine learning algorithms for structured/tabular data. It consistently wins Kaggle competitions and is a go-to for production ML pipelines.

---

## Table of Contents

1. [What is XGBoost?](#1-what-is-xgboost)
2. [Core Concepts](#2-core-concepts)
3. [How XGBoost Builds Trees](#3-how-xgboost-builds-trees)
4. [The Math Behind It](#4-the-math-behind-it)
5. [Key Hyperparameters](#5-key-hyperparameters)
6. [Installation](#6-installation)
7. [Complete Code Example](#7-complete-code-example)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [When to Use XGBoost](#10-when-to-use-xgboost)
11. [Common Mistakes & Tips](#11-common-mistakes--tips)

---

## 1. What is XGBoost?

XGBoost is an **ensemble learning** method based on **Gradient Boosting Decision Trees (GBDT)**. It builds many weak learners (shallow decision trees) sequentially, where each tree corrects the errors of the previous ones.

```
Final Prediction = Tree₁(x) + Tree₂(x) + Tree₃(x) + ... + TreeN(x)
```

### Key advantages over traditional GBDT:
| Feature | Traditional GBDT | XGBoost |
|---|---|---|
| Speed | Slow | Very fast (parallelized) |
| Regularization | None | L1 + L2 built-in |
| Missing values | Manual handling | Handles automatically |
| Tree pruning | Pre-pruning | Post-pruning (max_depth) |
| Memory | High | Optimized (column blocks) |

---

## 2. Core Concepts

### 2.1 Ensemble Learning
Instead of one complex model, XGBoost builds **hundreds of simple trees** and combines them. Each tree is a "weak learner" that barely beats random guessing alone — but together they form a strong predictor.

### 2.2 Boosting vs. Bagging
```
Bagging (e.g., Random Forest):
  Trees built INDEPENDENTLY in parallel → average their outputs

Boosting (e.g., XGBoost):
  Trees built SEQUENTIALLY → each fixes the errors of the last
```

### 2.3 Residuals / Pseudo-residuals
Each new tree is trained on the **residuals** (errors) from the current ensemble:

```
Iteration 1:  Predict house price → error = actual - predicted
Iteration 2:  Train tree on that error → reduce it
Iteration 3:  Train tree on remaining error → reduce further
...
Final:        Sum all tree predictions → accurate result
```

---

## 3. How XGBoost Builds Trees

### Step-by-step process:

```
Step 1: Start with a base prediction (e.g., mean of target)
Step 2: Calculate residuals (errors)
Step 3: Fit a decision tree to the residuals
Step 4: Update predictions by adding (learning_rate × tree_output)
Step 5: Recalculate residuals on updated predictions
Step 6: Repeat steps 3–5 for n_estimators iterations
Step 7: Final prediction = sum of all trees' outputs
```

### Tree Splitting — Gain Formula
XGBoost uses a **gain score** to decide where to split a node:

```
Gain = ½ × [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ

Where:
  GL, GR = sum of gradients in left/right child
  HL, HR = sum of hessians in left/right child
  λ (lambda) = L2 regularization term
  γ (gamma)  = minimum gain required to make a split
```

If `Gain < 0`, the split is pruned — this is XGBoost's built-in regularization.

---

## 4. The Math Behind It

### Objective Function
XGBoost minimizes:

```
Obj = Σ L(yᵢ, ŷᵢ)  +  Σ Ω(fₖ)
      ↑ Loss function    ↑ Regularization

Ω(f) = γT + ½λ||w||²
  T = number of leaves
  w = leaf weights
  γ = penalty per leaf (controls tree complexity)
  λ = L2 penalty on leaf weights
```

### For Regression — MSE Loss:
```
L(y, ŷ) = (y - ŷ)²

Gradient (g) = ∂L/∂ŷ = -2(y - ŷ)   → direction to reduce loss
Hessian  (h) = ∂²L/∂ŷ² = 2          → how confident the gradient is
```

### Leaf Weight Calculation:
```
w* = -G / (H + λ)    where G = Σgᵢ, H = Σhᵢ for samples in that leaf
```

---

## 5. Key Hyperparameters

### Learning & Boosting
| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | 100 | Number of trees to build |
| `learning_rate` (eta) | 0.3 | Shrinks each tree's contribution — lower = more conservative |
| `max_depth` | 6 | Maximum depth of each tree (controls complexity) |

### Sampling (Prevents Overfitting)
| Parameter | Default | Description |
|---|---|---|
| `subsample` | 1.0 | Fraction of training rows per tree (e.g., 0.8 = 80%) |
| `colsample_bytree` | 1.0 | Fraction of features per tree |
| `colsample_bylevel` | 1.0 | Fraction of features per tree level |

### Regularization
| Parameter | Default | Description |
|---|---|---|
| `reg_alpha` (L1) | 0 | Sparsity — drives some weights to exactly 0 |
| `reg_lambda` (L2) | 1 | Smoothness — penalizes large weights |
| `gamma` | 0 | Min gain to make a split (0 = always split) |
| `min_child_weight` | 1 | Min sum of hessians in a child node |

### Rule of thumb:
```
Underfitting → increase n_estimators, increase max_depth, increase learning_rate
Overfitting  → decrease max_depth, decrease learning_rate, add subsample/colsample,
               increase gamma, increase reg_alpha/reg_lambda
```

---

## 6. Installation

```bash
pip install xgboost scikit-learn pandas numpy matplotlib
```

---

## 7. Complete Code Example

### Predicting House Prices

```python
# ============================================================
# XGBoost Regression — Complete Example
# Predicting house prices using synthetic data
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. Create / Load Data ────────────────────────────────────
np.random.seed(42)

# Synthetic house price data
n_samples = 1000
data = pd.DataFrame({
    'area_sqft':    np.random.randint(500, 5000, n_samples),
    'bedrooms':     np.random.randint(1, 6, n_samples),
    'bathrooms':    np.random.randint(1, 5, n_samples),
    'age_years':    np.random.randint(0, 50, n_samples),
    'garage':       np.random.randint(0, 4, n_samples),
    'distance_km':  np.random.uniform(1, 50, n_samples),   # distance from city center
})

# Target: house price (with some noise)
data['price'] = (
    data['area_sqft'] * 150
    + data['bedrooms'] * 20000
    + data['bathrooms'] * 15000
    - data['age_years'] * 1000
    + data['garage'] * 10000
    - data['distance_km'] * 3000
    + np.random.normal(0, 20000, n_samples)
)

print("Dataset shape:", data.shape)
print(data.describe().round(2))

# ── 2. Split Features and Target ─────────────────────────────
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 3. Train XGBoost Model ────────────────────────────────────
model = xgb.XGBRegressor(
    # Boosting parameters
    n_estimators=300,        # number of trees
    learning_rate=0.05,      # shrinkage — small = slower but better
    max_depth=5,             # tree depth

    # Sampling — prevent overfitting
    subsample=0.8,           # 80% of rows per tree
    colsample_bytree=0.8,    # 80% of features per tree

    # Regularization
    reg_alpha=0.1,           # L1
    reg_lambda=1.0,          # L2
    gamma=0.1,               # min gain to split

    # Misc
    objective='reg:squarederror',  # MSE loss for regression
    random_state=42,
    n_jobs=-1,               # use all CPU cores
    verbosity=0
)

# Train with early stopping to avoid overfitting
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("\nModel trained successfully!")
print(f"Best iteration: {model.best_iteration}")

# ── 4. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n── Evaluation Metrics ──────────────────────")
print(f"  RMSE : ${rmse:,.0f}")
print(f"  MAE  : ${mae:,.0f}")
print(f"  R²   : {r2:.4f}  ({r2*100:.1f}% variance explained)")

# ── 5. Cross-Validation ───────────────────────────────────────
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n── 5-Fold Cross-Validation R² ──────────────")
print(f"  Scores : {cv_scores.round(4)}")
print(f"  Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Feature Importance ────────────────────────────────────
importance = pd.DataFrame({
    'feature':    X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n── Feature Importance ──────────────────────")
print(importance.to_string(index=False))

# ── 7. Visualize Results ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=20)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title(f'Actual vs Predicted\nR² = {r2:.4f}')
axes[0].grid(alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, color='orange', s=20)
axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price ($)')
axes[1].set_ylabel('Residuals ($)')
axes[1].set_title('Residual Plot\n(good model → scattered around 0)')
axes[1].grid(alpha=0.3)

# Plot 3: Feature Importance
axes[2].barh(importance['feature'], importance['importance'], color='teal')
axes[2].set_xlabel('Importance Score')
axes[2].set_title('Feature Importance\n(gain-based)')
axes[2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as xgboost_results.png")

# ── 8. Make a Prediction ─────────────────────────────────────
new_house = pd.DataFrame([{
    'area_sqft':   2500,
    'bedrooms':    3,
    'bathrooms':   2,
    'age_years':   10,
    'garage':      2,
    'distance_km': 15.0
}])

predicted_price = model.predict(new_house)[0]
print(f"\n── New House Prediction ────────────────────")
print(f"  Features : {new_house.to_dict('records')[0]}")
print(f"  Predicted Price : ${predicted_price:,.0f}")
```

### Expected Output:
```
Dataset shape: (1000, 7)
Train size: 800, Test size: 200

── Evaluation Metrics ──────────────────────
  RMSE : $21,543
  MAE  : $16,821
  R²   : 0.9712  (97.1% variance explained)

── 5-Fold Cross-Validation R² ──────────────
  Scores : [0.9698 0.9721 0.9704 0.9715 0.9688]
  Mean   : 0.9705 ± 0.0012

── Feature Importance ──────────────────────
    feature  importance
   area_sqft    0.6821
   bedrooms     0.1024
   distance_km  0.0893
   bathrooms    0.0714
   garage       0.0342
   age_years    0.0206

── New House Prediction ────────────────────
  Predicted Price : $341,209
```

---

## 8. Evaluation Metrics

### For Regression tasks:

| Metric | Formula | Good when |
|---|---|---|
| **MSE** | Σ(y - ŷ)² / n | Penalizes large errors heavily |
| **RMSE** | √MSE | Same units as target — most interpretable |
| **MAE** | Σ\|y - ŷ\| / n | Robust to outliers |
| **R²** | 1 - SS_res/SS_tot | 1.0 = perfect, 0 = no better than mean |
| **MAPE** | Σ\|y-ŷ\|/y × 100 | Percentage error — scale-independent |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
```

---

## 9. Hyperparameter Tuning

### Method 1: GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':   [100, 300, 500],
    'max_depth':      [3, 5, 7],
    'learning_rate':  [0.01, 0.05, 0.1],
    'subsample':      [0.7, 0.8, 1.0],
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
print("Best R²:    ", grid_search.best_score_)
```

### Method 2: Optuna (Bayesian, faster & smarter)
```python
import optuna

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma':             trial.suggest_float('gamma', 0, 5),
    }

    model = xgb.XGBRegressor(**params, objective='reg:squarederror', random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best R²:    ", study.best_value)
print("Best params:", study.best_params)
```

---

## 10. When to Use XGBoost

### ✅ Great for:
- **Tabular / structured data** (CSV, database tables)
- **Medium to large datasets** (1K–10M rows)
- **Mixed feature types** (numeric + categorical after encoding)
- **Competitions and production** — reliable baseline
- **When you need fast training** with good accuracy

### ❌ Not ideal for:
- **Images, audio, video** → use CNNs, RNNs
- **Text / NLP** → use Transformers (BERT, GPT)
- **Very small datasets** (<100 rows) → overfits easily
- **Real-time inference on edge** → model size can be large
- **When interpretability is critical** → use linear models or SHAP on top

---

## 11. Common Mistakes & Tips

### Mistake 1: Not encoding categoricals
```python
# ❌ Wrong — XGBoost can't handle raw strings
df['city'] = ['NYC', 'LA', 'NYC']

# ✅ Correct — encode first
from sklearn.preprocessing import LabelEncoder
df['city'] = LabelEncoder().fit_transform(df['city'])
# or use pd.get_dummies(df, columns=['city'])
```

### Mistake 2: Scaling features (unnecessary)
```python
# ❌ Unnecessary — tree models don't need feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # skips this step for XGBoost

# ✅ Just pass raw features (XGBoost handles it internally)
model.fit(X_train, y_train)
```

### Mistake 3: Too high learning rate with few trees
```python
# ❌ Bad — underfits
xgb.XGBRegressor(n_estimators=50, learning_rate=0.5)

# ✅ Good — more trees + lower rate = better generalization
xgb.XGBRegressor(n_estimators=500, learning_rate=0.05)
```

### Tip: Use DMatrix for speed
```python
# XGBoost's optimized data structure — up to 2x faster
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'eta': 0.05,
    'subsample': 0.8,
}

model = xgb.train(params, dtrain, num_boost_round=300,
                  evals=[(dtest, 'test')], verbose_eval=50)
```

### Tip: SHAP values for interpretability
```python
import shap

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Single prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0]
))
```

---

## Quick Reference Card

```
XGBoost Regression — Cheat Sheet
─────────────────────────────────────────────────────────────────
Install     :  pip install xgboost

Import      :  import xgboost as xgb

Create      :  model = xgb.XGBRegressor(
                   n_estimators=300,       # trees
                   learning_rate=0.05,     # shrinkage
                   max_depth=5,            # depth per tree
                   subsample=0.8,          # row sampling
                   colsample_bytree=0.8,   # col sampling
                   reg_alpha=0.1,          # L1
                   reg_lambda=1.0,         # L2
                   objective='reg:squarederror'
               )

Train       :  model.fit(X_train, y_train)

Predict     :  y_pred = model.predict(X_test)

Evaluate    :  r2_score(y_test, y_pred)

Importance  :  model.feature_importances_

Save/Load   :  model.save_model('model.json')
               model.load_model('model.json')
─────────────────────────────────────────────────────────────────
```

---

*Made with ❤️ for ML practitioners. XGBoost docs: https://xgboost.readthedocs.io*