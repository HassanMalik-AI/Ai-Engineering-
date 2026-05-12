# 🌳 Decision Tree Regression — Complete Guide

---

## 📌 What is Decision Tree Regression?

A **Decision Tree Regression** is a supervised machine learning algorithm that predicts a **continuous numerical value** by learning simple **if-else decision rules** from the training data.

Unlike classification trees that predict categories, regression trees predict **real numbers** (e.g., house prices, temperature, salary).

---

## 🧠 Core Intuition

Imagine you want to predict a person's salary based on their years of experience:

```
Is experience > 5 years?
    ├── YES → Is experience > 10 years?
    │           ├── YES → Predict: $120,000
    │           └── NO  → Predict: $85,000
    └── NO  → Predict: $45,000
```

The tree keeps splitting the data into smaller groups until each group is "pure enough" (similar values), then predicts the **average** of values in that group.

---

## 🔑 Key Terminology

| Term | Meaning |
|------|---------|
| **Root Node** | The very first split (top of the tree) |
| **Internal Node** | A decision/split point based on a feature |
| **Leaf Node** | Terminal node — outputs the predicted value |
| **Depth** | Number of levels from root to leaf |
| **Split** | Dividing data based on a feature threshold |
| **Impurity** | How mixed/varied the values are in a node |

---

## ⚙️ How It Works — Step by Step

### Step 1: Start with all data at the root

All training samples begin at the root node.

### Step 2: Find the best split

For each feature and each possible threshold, calculate the **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(yᵢ − ȳ)²
```

The split that **minimizes the weighted MSE** of the two child nodes is chosen:

```
MSE_split = (n_left/n) × MSE_left + (n_right/n) × MSE_right
```

### Step 3: Recurse

Repeat Step 2 on each child node independently.

### Step 4: Stop splitting (stopping criteria)

- Maximum depth reached (`max_depth`)
- Node has too few samples (`min_samples_split`)
- MSE improvement is too small
- Only one sample remains

### Step 5: Predict

For a new data point, traverse the tree and output the **mean of training values** in the reached leaf node.

---

## 📐 Mathematical Foundation

### Prediction at a Leaf

```
ŷ_leaf = (1 / |R_m|) × Σ yᵢ    for all i in region R_m
```

Where `R_m` is the set of training samples in that leaf.

### Splitting Criterion (MSE Reduction)

```
Gain = MSE_parent − [(n_L/n) × MSE_L + (n_R/n) × MSE_R]
```

The algorithm picks the feature and threshold that **maximizes Gain**.

---

## 🌿 Tree Structure Diagram

```
                    [Root: All Data]
                   MSE = 1200, n=100
                         |
              Feature: Experience > 5?
                    /           \
               [Left]          [Right]
            exp ≤ 5 yrs      exp > 5 yrs
            n=60, ȳ=45k       n=40, ȳ=95k
                /                    \
     Degree = Bachelor?         exp > 10?
         /        \              /       \
      [Leaf]    [Leaf]       [Leaf]    [Leaf]
      ȳ=38k    ȳ=52k        ȳ=85k    ȳ=120k
```

---

## 💻 Short Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Create sample dataset ──────────────────────────────────────────
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)   # Feature: 1 column
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Target

# ── 2. Train/Test Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Build the Model ────────────────────────────────────────────────
model = DecisionTreeRegressor(
    max_depth=4,          # Limit tree depth to avoid overfitting
    min_samples_split=5,  # Min samples needed to split a node
    min_samples_leaf=2,   # Min samples required in a leaf node
    random_state=42
)
model.fit(X_train, y_train)

# ── 4. Predict & Evaluate ─────────────────────────────────────────────
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ── 5. Visualize Predictions ──────────────────────────────────────────
X_plot = np.linspace(0, 5, 500).reshape(-1, 1)
y_plot = model.predict(X_plot)

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color="steelblue", label="Train data", s=20)
plt.scatter(X_test,  y_test,  color="orange",    label="Test data",  s=20)
plt.plot(X_plot, y_plot, color="red", linewidth=2, label="Tree prediction")
plt.title("Decision Tree Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# ── 6. Visualize the Tree ─────────────────────────────────────────────
plt.figure(figsize=(20, 8))
plot_tree(model, filled=True, feature_names=["X"], rounded=True, fontsize=9)
plt.title("Decision Tree Structure")
plt.tight_layout()
plt.show()
```

### 📊 Sample Output

```
MSE  : 0.0182
RMSE : 0.1349
R²   : 0.9623
```

---

## 🎛️ Key Hyperparameters

| Parameter | What it Controls | Effect if Too High | Effect if Too Low |
|-----------|-----------------|-------------------|-------------------|
| `max_depth` | Maximum depth of the tree | **Overfitting** | **Underfitting** |
| `min_samples_split` | Min samples to split a node | Underfitting | Overfitting |
| `min_samples_leaf` | Min samples in each leaf | Underfitting | Overfitting |
| `max_features` | Features considered per split | Underfitting | Overfitting |
| `max_leaf_nodes` | Max number of leaf nodes | Underfitting | Overfitting |

---

## 📈 Overfitting vs Underfitting

```
Depth=1 (Underfit)       Depth=4 (Good Fit)       Depth=20 (Overfit)
─────────────────        ──────────────────        ─────────────────
Prediction: flat line    Follows the curve         Memorizes every point
High Bias                Low Bias + Low Variance   High Variance
```

**Rule of thumb:** Use `max_depth` between **3 and 10** and tune with cross-validation.

---

## ✅ Advantages

- **Interpretable** — easy to visualize and explain
- **No feature scaling needed** — works with raw data
- **Handles non-linear relationships** automatically
- **Fast training and prediction**
- **Works with mixed feature types** (numerical + categorical)

---

## ❌ Disadvantages

- **Prone to overfitting** — especially with deep trees
- **Unstable** — small data changes can produce very different trees
- **Greedy algorithm** — doesn't find the globally optimal tree
- **Poor extrapolation** — can't predict beyond the range of training values

---

## 🚀 When to Use

| ✅ Good Choice | ❌ Not Ideal |
|---------------|-------------|
| Need interpretability | Need smooth predictions |
| Non-linear data | Data has strong extrapolation needs |
| Mixed feature types | Very high-dimensional data |
| Quick baseline model | Need highest possible accuracy |

---

## 🔗 Extensions & Related Algorithms

| Algorithm | How It Improves on a Single Tree |
|-----------|----------------------------------|
| **Random Forest** | Averages many trees → reduces variance |
| **Gradient Boosting** | Builds trees sequentially → reduces bias |
| **XGBoost / LightGBM** | Optimized gradient boosting with regularization |
| **Extra Trees** | Randomizes splits further → even lower variance |

---

## 📦 Quick Reference

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

*Generated with ❤️ for learning ML fundamentals*