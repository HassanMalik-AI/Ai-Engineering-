# 🤖 K-Nearest Neighbors (KNN) — Complete Guide

> **Learn KNN in simple words with real Python code examples**

---

## 📌 Table of Contents

1. [What is KNN?](#what-is-knn)
2. [How Does KNN Work?](#how-does-knn-work)
3. [Key Concepts](#key-concepts)
4. [Distance Metrics](#distance-metrics)
5. [Choosing the Right K](#choosing-the-right-k)
6. [When to Use KNN?](#when-to-use-knn)
7. [Code Examples](#code-examples)
8. [Pros & Cons](#pros--cons)
9. [Summary](#summary)

---

## What is KNN?

**KNN (K-Nearest Neighbors)** is one of the **simplest** supervised learning algorithms used for:
- **Classification** — deciding which category something belongs to
- **Regression** — predicting a number

### 🏘️ Simple Analogy

Imagine you move to a **new neighborhood** and want to know if an area is "safe" or "unsafe".

You ask your **K nearest neighbors** (e.g., 5 closest houses). If 4 out of 5 say "safe" → you conclude it's safe!

**KNN does the exact same thing with data points.**

```
      🏠 safe   🏠 safe
           🏠 safe
    🏠 unsafe        🏠 safe
           ❓ new point
    → Ask 5 nearest neighbors
    → 4 say safe, 1 says unsafe
    → Decision: SAFE ✅
```

---

## How Does KNN Work?

### Step-by-Step

```
Step 1: Store ALL training data (KNN is lazy — no training phase!)
Step 2: New data point arrives
Step 3: Calculate distance from new point to ALL training points
Step 4: Pick K closest points (K nearest neighbors)
Step 5: Majority vote (classification) OR average (regression)
Step 6: Assign the result to the new point
```

### Visual Example

```
         ● ●                       K=3 nearest neighbors
       ●   ●   ← Class A (Blue)    of the new point ❓:
                                     2 Blue ●
              ❓  ← New Point        1 Red  ▲
                                   → Classified as Blue ●
       ▲   ▲
         ▲     ← Class B (Red)
```

---

## Key Concepts

### 1. 🔢 K Value
The number of neighbors to look at.
- **K=1** → Look at only the 1 nearest neighbor (very sensitive)
- **K=5** → Look at 5 nearest neighbors (more stable)
- **K=N** → Look at all points (always predicts the majority class)

```
Small K  →  Complex boundary  →  Overfitting   ❌
Large K  →  Smooth boundary   →  Underfitting  ❌
Right K  →  Balanced boundary →  Best results  ✅
```

### 2. 📏 Distance
KNN uses distance to find "closeness". The most common is **Euclidean distance**:

```
Distance = √((x2-x1)² + (y2-y1)²)
```

Think of it as the straight-line distance between two points on a map.

### 3. 🗳️ Voting (for Classification)
Once K neighbors are found, they **vote**:
- Class with the most votes → wins
- Ties are broken by distance (closer neighbor wins)

### 4. 📊 Averaging (for Regression)
For predicting numbers, KNN averages the K neighbors' values:
```
Prediction = (value1 + value2 + ... + valueK) / K
```

---

## Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean** | `√(Σ(xi - yi)²)` | Continuous numerical data (most common) |
| **Manhattan** | `Σ\|xi - yi\|` | Grid-like data, robust to outliers |
| **Minkowski** | Generalization of both | Flexible, controlled by parameter p |
| **Hamming** | Count of differing positions | Categorical / text data |
| **Cosine** | Angle between vectors | Text, high-dimensional data |

### Euclidean vs Manhattan — Visual

```
  A ────────────────── B
  │                  ↗
  │    Euclidean   ↗   ← Straight line (shortest)
  │              ↗
  │ Manhattan  ↗
  │ (L-shape) ↗
  └──────────────────→
  
  Euclidean = diagonal straight line
  Manhattan = only horizontal + vertical moves (like city blocks)
```

---

## Choosing the Right K

### The Elbow Method

Run KNN for different K values and plot the error rate:

```
Error
  |
  |  ●
  |    ●
  |      ●
  |        ●──────────────  ← error flattens
  |            (elbow here = best K)
  └──────────────────────→ K
         1  3  5  7  9 ...
```

### Rules of Thumb

```
✅ K should be ODD (avoids ties in binary classification)
✅ K = √N  (where N = number of training samples) is a good starting point
✅ Use cross-validation to find the best K
❌ Avoid K=1 (too sensitive to noise)
❌ Avoid very large K (too much smoothing)
```

---

## When to Use KNN?

### ✅ Use KNN When:
- Dataset is **small to medium** in size
- Data has **clear clusters** or neighborhoods
- You need a **quick baseline** model
- You want a **non-linear** decision boundary
- You have **no assumptions** about data distribution

### ❌ Avoid KNN When:
- Dataset is **very large** (slow predictions)
- Data has **many features** (curse of dimensionality)
- Features have **very different scales** (always normalize!)
- You need **fast real-time predictions**
- Data has **lots of irrelevant features**

---

## Code Examples

### Setup — Install Required Libraries

```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

---

### Example 1: Basic KNN Classification (Iris Dataset)

```python
# ============================================
# KNN Classification — Iris Flower Dataset
# ============================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
iris = load_iris()
X = iris.data      # Features: sepal/petal length & width
y = iris.target    # Labels: 0=setosa, 1=versicolor, 2=virginica

print("Dataset shape:", X.shape)      # (150, 4)
print("Classes:", iris.target_names)  # ['setosa' 'versicolor' 'virginica']

# Step 2: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Scale features (CRITICAL for KNN — distance-based!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Step 4: Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Step 5: Predict
y_pred = knn.predict(X_test)

# Step 6: Evaluate
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # New sample
new_flower_scaled = scaler.transform(new_flower)
prediction = knn.predict(new_flower_scaled)
print(f"\nNew flower prediction: {iris.target_names[prediction[0]]}")
```

**Expected Output:**
```
Dataset shape: (150, 4)
Classes: ['setosa' 'versicolor' 'virginica']

Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support
     setosa       1.00      1.00      1.00        10
 versicolor       1.00      1.00      1.00         9
  virginica       1.00      1.00      1.00        11

New flower prediction: setosa
```

---

### Example 2: Finding the Best K (Elbow Method)

```python
# ============================================
# Finding the Optimal K Value
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Test K from 1 to 30
k_values    = range(1, 31)
train_errors = []
test_errors  = []
cv_scores    = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_errors.append(1 - knn.score(X_train, y_train))
    test_errors.append(1 - knn.score(X_test, y_test))
    cv_scores.append(cross_val_score(knn, X_scaled, y, cv=5).mean())

# Find best K
best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k} with CV Accuracy: {max(cv_scores)*100:.2f}%")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, train_errors, 'b-o', label='Train Error', markersize=4)
plt.plot(k_values, test_errors,  'r-o', label='Test Error',  markersize=4)
plt.axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Train vs Test Error by K')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_values, cv_scores, 'g-o', markersize=4)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('CV Accuracy by K (Elbow Method)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_elbow.png', dpi=150)
plt.show()
```

---

### Example 3: Visualizing KNN Decision Boundary

```python
# ============================================
# KNN Decision Boundary Visualization
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate 2D data
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, random_state=42
)
X = StandardScaler().fit_transform(X)

# Compare K=1, K=5, K=15, K=30
k_values = [1, 5, 15, 30]
colors   = ListedColormap(['#FFAAAA', '#AAAAFF'])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, k in zip(axes, k_values):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    # Mesh grid
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=colors)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors,
               edgecolors='black', s=30)
    ax.set_title(f'K = {k}\nAccuracy: {model.score(X, y)*100:.1f}%')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('KNN Decision Boundaries for Different K Values',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_boundaries.png', dpi=150)
plt.show()
```

---

### Example 4: KNN for Regression

```python
# ============================================
# KNN Regression — Predicting House Prices
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic house price data
np.random.seed(42)
n = 200
house_size = np.random.uniform(500, 3500, n)            # sq ft
house_price = 50000 + 100 * house_size + np.random.normal(0, 20000, n)  # price

X = house_size.reshape(-1, 1)
y = house_price

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_s, y_train)

# Evaluate
y_pred = knn_reg.predict(X_test_s)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.0f}")
print(f"R² Score: {r2:.4f}")

# Predict a new house
new_house = scaler.transform([[2000]])  # 2000 sq ft
predicted_price = knn_reg.predict(new_house)[0]
print(f"\nPredicted price for 2000 sq ft house: ${predicted_price:,.0f}")

# Plot
plt.figure(figsize=(10, 5))
X_range = np.linspace(500, 3500, 300).reshape(-1, 1)
X_range_s = scaler.transform(X_range)
y_range_pred = knn_reg.predict(X_range_s)

plt.scatter(X_train, y_train, alpha=0.4, label='Training data', color='steelblue', s=20)
plt.scatter(X_test,  y_test,  alpha=0.4, label='Test data',     color='orange',    s=20)
plt.plot(X_range, y_range_pred, 'r-', linewidth=2, label='KNN Prediction')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('KNN Regression — House Price Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('knn_regression.png', dpi=150)
plt.show()
```

**Expected Output:**
```
RMSE: $21,543
R² Score: 0.9412

Predicted price for 2000 sq ft house: $249,876
```

---

### Example 5: KNN with Different Distance Metrics

```python
# ============================================
# Comparing Distance Metrics in KNN
# ============================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Wine dataset
wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
y = wine.target

# Test different distance metrics
metrics = {
    'Euclidean':  {'metric': 'euclidean'},
    'Manhattan':  {'metric': 'manhattan'},
    'Minkowski':  {'metric': 'minkowski', 'p': 3},
    'Chebyshev':  {'metric': 'chebyshev'},
}

results = []
for name, params in metrics.items():
    knn = KNeighborsClassifier(n_neighbors=5, **params)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    results.append({
        'Metric':   name,
        'Mean Acc': f"{scores.mean()*100:.2f}%",
        'Std Dev':  f"±{scores.std()*100:.2f}%"
    })

df = pd.DataFrame(results)
print("Distance Metric Comparison (Wine Dataset, K=5):")
print(df.to_string(index=False))
```

**Expected Output:**
```
Distance Metric Comparison (Wine Dataset, K=5):
     Metric Mean Acc Std Dev
  Euclidean   96.07%  ±2.14%
  Manhattan   97.19%  ±1.87%
  Minkowski   95.51%  ±2.43%
  Chebyshev   93.82%  ±3.21%
```

---

### Example 6: Real-World — Disease Prediction

```python
# ============================================
# KNN — Diabetes Disease Prediction
# ============================================

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigree','Age','Outcome']

df = pd.read_csv(url, names=cols)
print("Dataset shape:", df.shape)
print(df.head())

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Replace zero values (invalid) with column median
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    X[col] = X[col].replace(0, X[col].median())

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Find best K using GridSearchCV
param_grid = {'n_neighbors': range(1, 21), 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_*100:.2f}%")

# Final evaluation
best_knn = grid.best_estimator_
y_pred   = best_knn.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'])
plt.title('KNN Confusion Matrix — Diabetes Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('knn_confusion.png', dpi=150)
plt.show()
```

---

### Example 7: Quick Template — Copy & Use

```python
# ============================================
# KNN Quick Start Template
# ============================================

from sklearn.neighbors import KNeighborsClassifier   # or KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# --- YOUR DATA GOES HERE ---
# X = features (2D array)
# y = labels / target values

# 1. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale (ALWAYS do this for KNN!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3. Find best K
k_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    k_scores.append(score)
best_k = np.argmax(k_scores) + 1
print(f"Best K: {best_k}")

# 4. Train with best K
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# 5. Evaluate
print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")
```

---

## Pros & Cons

### ✅ Advantages

| Advantage | Description |
|-----------|-------------|
| **Simple to understand** | No complex math — just distances and voting |
| **No training phase** | Instantly usable after storing data |
| **Non-linear boundaries** | Naturally handles complex decision regions |
| **No assumptions** | Works without assuming data distribution |
| **Versatile** | Works for both classification and regression |

### ❌ Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Slow predictions** | Must compute distance to ALL training points |
| **High memory usage** | Stores all training data |
| **Sensitive to scale** | Always normalize features! |
| **Curse of dimensionality** | Performance drops with too many features |
| **Sensitive to noise** | Outliers directly affect predictions |
| **Choosing K is tricky** | Bad K = bad results |

---

## KNN vs Other Algorithms

| Feature | KNN | SVM | Decision Tree | Logistic Regression |
|---------|-----|-----|---------------|---------------------|
| Training speed | ⚡ Instant | 🐢 Slow | ⚡ Fast | ⚡ Fast |
| Prediction speed | 🐢 Slow | ⚡ Fast | ⚡ Fast | ⚡ Fast |
| Interpretability | ✅ Easy | ❌ Hard | ✅ Easy | ✅ Easy |
| Non-linear data | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| Feature scaling needed | ✅ YES | ✅ YES | ❌ No | ✅ Yes |
| Large datasets | ❌ Poor | ❌ Poor | ✅ OK | ✅ Good |

---

## Summary

```
┌─────────────────────────────────────────────────────┐
│             KNN Quick Reference                     │
├─────────────────┬───────────────────────────────────┤
│ Full Name       │ K-Nearest Neighbors                │
│ Type            │ Supervised Learning (lazy learner) │
│ Tasks           │ Classification, Regression         │
│ Key idea        │ "You are your neighbors"           │
│ Key parameter   │ K (number of neighbors)            │
│ Distance metric │ Euclidean (default), Manhattan...  │
│ Feature scaling │ MANDATORY (StandardScaler)         │
│ Best for        │ Small-medium datasets              │
│ Avoid for       │ Large datasets, high dimensions    │
│ Good starting K │ √N (square root of samples)        │
└─────────────────┴───────────────────────────────────┘
```

### ⚠️ The 3 Golden Rules of KNN

```
Rule 1: ALWAYS scale your features before using KNN
        → Use StandardScaler or MinMaxScaler

Rule 2: ALWAYS tune K using cross-validation
        → Use GridSearchCV or the elbow method

Rule 3: ALWAYS check for and handle missing values
        → KNN breaks with NaN values in data
```

---

> 📚 **Further Reading:**
> - [scikit-learn KNN Docs](https://scikit-learn.org/stable/modules/neighbors.html)
> - [KNN Explained — Towards Data Science](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
>
> 💡 **Pro Tip:** If KNN is too slow for large data, try using a **Ball Tree** or **KD Tree** index:
> ```python
> KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
> ```