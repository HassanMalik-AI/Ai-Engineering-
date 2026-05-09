# 📈 Regression in Supervised Learning

> A complete beginner-friendly guide to understanding, building, and evaluating regression models in machine learning — explained in plain English.

![Topic](https://img.shields.io/badge/Topic-Supervised%20Learning-blue)
![Type](https://img.shields.io/badge/Type-Regression-orange)
![Level](https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green)
![Language](https://img.shields.io/badge/Language-Python-yellow?logo=python)

---

## 📋 Table of Contents

- [What is Supervised Learning?](#-what-is-supervised-learning)
- [What is Regression?](#-what-is-regression)
- [Regression vs Classification](#-regression-vs-classification)
- [Types of Regression Problems](#-types-of-regression-problems)
- [How Regression Works](#-how-regression-works)
- [Regression Algorithms](#-regression-algorithms)
- [Key Concepts](#-key-concepts)
- [The ML Pipeline Step by Step](#-the-ml-pipeline-step-by-step)
- [Evaluation Metrics](#-evaluation-metrics)
- [Handling Common Problems](#-handling-common-problems)
- [Choosing the Right Algorithm](#-choosing-the-right-algorithm)
- [Quick Code Examples](#-quick-code-examples)
- [Real-World Use Cases](#-real-world-use-cases)
- [Glossary](#-glossary)

---

## 🤔 What is Supervised Learning?

**Supervised learning** is a type of machine learning where you train a computer using examples that already have correct answers.

Think of it like a student learning from a textbook with an answer key:
- The student reads a question (input)
- Checks the answer (label)
- Learns the pattern connecting questions to answers
- Later, the student can answer questions they have never seen before

In machine learning:

| Term | What it means | Example |
|------|-------------|---------|
| **Input (Features)** | The information given to the model | House size, location, age |
| **Label (Target)** | The correct answer to predict | House price: $320,000 |
| **Model** | The learned relationship between input and output | `price = f(size, location, age)` |
| **Prediction** | The model's output for new, unseen data | Predicted price: $315,000 |

### Two main tasks in supervised learning

```
Supervised Learning
      │
      ├── Classification → Predicts a CATEGORY
      │     └── "Is this email spam or not spam?"
      │
      └── Regression → Predicts a NUMBER
            └── "What will the house price be?"
```

> This guide focuses entirely on **Regression**.

---

## 🔢 What is Regression?

**Regression** is the task of predicting a **continuous numerical value** — a real number — rather than a category.

Instead of asking "which box does this fall into?", regression asks **"how much?" or "how many?"**.

### Simple real-world analogy

Imagine an experienced estate agent who has sold hundreds of houses:
- They remember that a 3-bedroom house in a good school district sold for £280,000
- A similar 4-bedroom house in the same area sold for £340,000
- Over thousands of deals, they build an intuition for pricing

A regression model does the same — it learns from historical data and then predicts the price of a new house it has never seen.

### What the model actually learns

The model finds the best **mathematical relationship** (a curve or line) that connects the input features to the output number.

```
Price (£)
  │                                        ●
  │                              ●    ●
  │                    ●    ●
  │           ●   ●
  │      ●
  └─────────────────────────────────────── Size (m²)
         ↑
    Regression line fits through the middle of all the dots
```

Given a new house size, you read off the predicted price from the line.

---

## 🔀 Regression vs Classification

| | Regression | Classification |
|--|------------|---------------|
| **Output** | A number (continuous) | A category (discrete) |
| **Question** | How much? How many? | Which group? |
| **Example output** | £342,500 | "Expensive" / "Affordable" |
| **Example task** | Predict tomorrow's temperature | Predict if it will rain (yes/no) |
| **Error measure** | How far off was the number? | Was the category correct? |
| **Algorithms** | Linear Regression, SVR, XGBoost | Logistic Regression, SVM, Random Forest |

> **Tip:** You can turn a regression into a classification. Predict the price (regression), then apply a threshold: price > £500k → "Luxury", else → "Standard". But the reverse is harder.

---

## 🗂️ Types of Regression Problems

### 1. Simple Linear Regression
One input feature predicts one output value. The relationship is assumed to be a straight line.

```
Output = (weight × input) + bias

Example:
  Salary = (500 × YearsExperience) + 30,000
```

```
Salary
  │                         ●
  │                    ●
  │               ●
  │          ●
  │     ●
  └────────────────────────── Years of Experience
```

**Use when:** One feature drives the outcome, and their relationship looks roughly linear.

---

### 2. Multiple Linear Regression
Multiple input features predict one output. Most real-world problems fall here.

```
House Price = (200 × Size_m²) + (15,000 × Bedrooms) + (50,000 × GardenYN) - (1,000 × AgeYears) + base
```

**Use when:** Several features together explain the outcome.

---

### 3. Polynomial Regression
When the relationship between input and output is a **curve**, not a straight line.

```
Linear fit:     ─────────────────   (misses the curve)
Polynomial fit: ───╮───────╭────    (follows the curve)
```

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# Then apply Linear Regression on X_poly
```

**Use when:** A straight line clearly underfits the data.

---

### 4. Ridge and Lasso Regression (Regularised Linear Regression)
Linear regression with a penalty added to prevent overfitting and handle many correlated features.

```
Ridge (L2): Shrinks all weights toward zero equally
Lasso (L1): Shrinks some weights all the way to zero (automatic feature selection)
```

**Use when:** You have many features, some of which may be irrelevant or correlated.

---

### 5. Time Series Regression
Predicting a future value based on past values over time.

```
Past                  Now      Future
  │ ╲  ╱╲  ╱╲  ╱╲  ╱ │  ╌╌╌╌╌╌→
  │  ╲╱  ╲╱  ╲╱  ╲╱   │   (predict)
  └────────────────────────────────── Time
```

**Examples:** Stock prices, electricity demand, website traffic, temperature forecasting.

---

### 6. Quantile Regression
Instead of predicting the average outcome, predicts a specific quantile (e.g. the 90th percentile).

**Use when:** You want to know the worst-case or best-case scenario, not just the average. For example, predicting delivery time at the 95th percentile for logistics planning.

---

## ⚙️ How Regression Works

### The core idea: minimise the error

The model tries different lines/curves, measures how wrong each one is, and keeps adjusting until it finds the best fit.

```
Step 1: Start with a random line
        Price = 100 × Size + 50,000   ← random starting guess

Step 2: Make predictions for all training samples
        Actual price:    £300,000
        Predicted price: £260,000
        Error:           -£40,000

Step 3: Calculate total error across all samples (Loss Function)
        MSE = average of (all errors²)

Step 4: Adjust the weights to reduce error (Gradient Descent)
        New weight = old weight - (learning_rate × gradient)

Step 5: Repeat steps 2–4 until error stops decreasing
```

### What is gradient descent?

Imagine you are blindfolded on a hilly landscape and want to reach the lowest valley. You feel the slope under your feet and take a small step downhill. You repeat this until you cannot go any lower. That lowest point is where the model's error is at its minimum.

```
Error
  │ ╲
  │  ╲
  │   ╲___
  │       ╲___
  │           ╲____
  │                ╲____★   ← Minimum error (best weights)
  └────────────────────────── Model weights
```

---

## 🧮 Regression Algorithms

### 1. Linear Regression

**What it is:** Fits a straight line through the data by finding the best weights for each feature.

**How it thinks:** "What multipliers, when applied to each feature, give me the most accurate predictions?"

**The equation:**

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

where:
  ŷ  = predicted value
  w  = learned weights (coefficients)
  x  = input features
  b  = bias (intercept)
```

**Best for:**
- When the relationship between features and target is linear
- When you need to explain which features matter most
- As a baseline before trying complex models

**Pros:** Fast, interpretable, no hyperparameters to tune
**Cons:** Cannot capture non-linear patterns, sensitive to outliers

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# See which features matter most
for feature, weight in zip(feature_names, model.coef_):
    print(f"{feature}: {weight:.2f}")
```

---

### 2. Ridge Regression (L2 Regularisation)

**What it is:** Linear regression with a penalty that shrinks all weights toward zero. Prevents overfitting.

**How it thinks:** "Find the best line, but don't let any single weight get too large."

**The equation:**

```
Loss = MSE + α × (sum of all weights²)
         ↑               ↑
   standard error    regularisation penalty
```

`α` (alpha) controls how strong the penalty is:
- `α = 0` → same as regular linear regression
- Large `α` → all weights shrink toward zero (underfitting risk)

**Best for:**
- When you have many correlated features (multicollinearity)
- When linear regression is overfitting

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 3. Lasso Regression (L1 Regularisation)

**What it is:** Like Ridge, but the penalty can shrink some weights all the way to exactly zero — effectively removing useless features.

**How it thinks:** "Find the best line and eliminate features that don't help."

**The equation:**

```
Loss = MSE + α × (sum of absolute values of weights)
```

**Key difference from Ridge:**

```
Ridge: weights → very small (but never zero)
Lasso: weights → zero (automatic feature selection)
```

**Best for:**
- When you have many features and suspect only a few are useful
- When you want automatic feature selection

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Features with zero coefficients were automatically removed
important_features = [f for f, w in zip(feature_names, model.coef_) if w != 0]
print("Features used:", important_features)
```

---

### 4. ElasticNet

**What it is:** A combination of Ridge (L2) and Lasso (L1). Gets the benefits of both.

**Best for:** When you have many features and don't know whether Ridge or Lasso will work better.

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
# l1_ratio=0 → pure Ridge, l1_ratio=1 → pure Lasso, 0.5 → balanced mix
model.fit(X_train, y_train)
```

---

### 5. Decision Tree Regressor

**What it is:** Splits data into groups through yes/no questions, then predicts the **average value** of each group.

**How it thinks:**

```
Is house size > 100m²?
    ├── Yes → Is it in London?
    │           ├── Yes → Predict: £520,000 (average of that group)
    │           └── No  → Predict: £280,000
    └── No  → Is it a new build?
                ├── Yes → Predict: £210,000
                └── No  → Predict: £160,000
```

**Best for:**
- When you need an explainable model
- When relationships are non-linear
- Mixed data types (numbers + categories)

**Pros:** Easy to visualise, no feature scaling needed
**Cons:** Prone to overfitting, predictions are step-like (not smooth)

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 6. Random Forest Regressor

**What it is:** Builds hundreds of decision trees on random subsets of the data and averages their predictions.

**How it thinks:** "Ask 200 different trees for their price estimate and average the answers."

```
Input
  ├── Tree 1  → £310,000
  ├── Tree 2  → £325,000
  ├── Tree 3  → £298,000
  ├── Tree 4  → £318,000
  └── Tree 5  → £305,000
              ↓
         Average: £311,200  ← Final prediction
```

**Best for:**
- General-purpose regression
- When you want high accuracy with minimal tuning
- Handling missing data and mixed feature types

**Pros:** Very accurate, resistant to overfitting, shows feature importance
**Cons:** Slower than a single tree, less interpretable

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# See which features matter most
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values(ascending=False).plot(kind='bar')
```

---

### 7. Gradient Boosting Regressor (XGBoost / LightGBM)

**What it is:** Builds trees one by one, where each new tree focuses on fixing the errors made by all previous trees.

**How it thinks:**

```
Round 1: Tree predicts £300,000  → Actual: £350,000 → Error: £50,000
Round 2: New tree predicts the  £50,000 error
Round 3: New tree refines further
...
Final:   Sum of all tree predictions → £348,000
```

**Best for:**
- Tabular/structured data (databases, spreadsheets)
- When maximum accuracy is the goal
- Kaggle-style competitions

**Pros:** State-of-the-art accuracy on tabular data, handles missing values naturally
**Cons:** Many hyperparameters, can overfit, slower to train

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=20,
          verbose=False)
predictions = model.predict(X_test)
```

---

### 8. Support Vector Regressor (SVR)

**What it is:** Tries to fit a line/curve within a tube of acceptable error (epsilon). Points inside the tube do not contribute to the loss.

**How it thinks:**

```
Target value
  │    ═══════════════ ← upper boundary of tube (+ε)
  │    ─────────────── ← prediction line
  │    ═══════════════ ← lower boundary of tube (-ε)
  │
  └──────────────────────── Input
  
Points INSIDE the tube: no penalty
Points OUTSIDE the tube: penalised
```

**Best for:**
- Small to medium datasets with many features
- When you want robustness to outliers

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X_train_s, y_train)
predictions = model.predict(X_test_s)
```

---

### 9. K-Nearest Neighbors Regressor (KNN)

**What it is:** Predicts the value of a new point by averaging the values of its K nearest neighbours in the training data.

**How it thinks:**

```
New house: 120m², 3 bed, Zone 3

Find 5 most similar houses in training data:
  House A: £310,000
  House B: £295,000
  House C: £325,000
  House D: £308,000
  House E: £302,000

Prediction = Average = £308,000
```

**Best for:** Small datasets, non-linear relationships, quick baselines

**Pros:** No training phase, naturally handles non-linearity
**Cons:** Very slow at prediction time for large datasets, needs feature scaling

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 10. Neural Network Regressor

**What it is:** Multiple layers of interconnected nodes that learn complex, non-linear relationships.

**How it thinks:** Each layer learns a progressively more abstract transformation of the input until the final layer outputs a single number.

**Best for:**
- Very large datasets (>100k samples)
- Images, audio, text → regression tasks
- When other algorithms hit a ceiling

**Pros:** Can learn extremely complex patterns
**Cons:** Needs lots of data and compute, acts like a black box, many hyperparameters

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Single output neuron for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_data=(X_val, y_val))
```

> **Note:** The output layer has **no activation function** (or `linear`) — this allows the network to output any real number, not just values between 0 and 1.

---

## 🔑 Key Concepts

### Features, Target, and Samples

```
A single row in your dataset = one sample

     Size_m²  Bedrooms  Age_years  Zone  │  Price (target)
     ───────────────────────────────────────────────────────
     85       3         12         2     │  £285,000
     120      4         5          1     │  £420,000
     60       2         30         3     │  £195,000
     ↑
     Features (inputs the model uses)
```

---

### The Cost Function (Loss Function)

This is the number the model tries to minimise during training. It measures how wrong the predictions are.

**Mean Squared Error (MSE)** — the most common regression loss:

```
MSE = (1/n) × Σ (actual - predicted)²

Example:
  Actual:    £300,000
  Predicted: £280,000
  Error:     -£20,000
  Squared:    400,000,000

  Squaring makes all errors positive and punishes large errors more heavily.
```

---

### Train / Validation / Test Split

Always split your data into three non-overlapping sets:

```
Full Dataset (100%)
    ├── Training Set   (70%) → The model learns from this
    ├── Validation Set (15%) → Tune the model and pick hyperparameters
    └── Test Set       (15%) → Final, honest evaluation (used ONCE at the end)
```

**Why three splits?**

| If you evaluate on... | The problem |
|-----------------------|------------|
| Training data | The model looks great but it just memorised the answers |
| Validation data (repeatedly) | You keep tweaking until you accidentally overfit to validation |
| Test data (at the end, once) | This gives a truly honest measure of real-world performance |

```python
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42
)
```

---

### Overfitting vs Underfitting

**Underfitting** — the model is too simple and misses the pattern.

```
Actual data:   curved relationship  ╰───╮
Model learns:  straight line        ────────
Result: always wrong by a lot
```

**Overfitting** — the model memorised the training data, including its noise.

```
Training RMSE:    £5,000   ← looks great
Validation RMSE:  £45,000  ← terrible
The model wiggled through every training point but fails on new data
```

**Just right** — generalises well.

```
Training RMSE:    £18,000
Validation RMSE:  £21,000  ← slightly higher is normal and acceptable
```

**How to fix underfitting:**
- Use a more complex model
- Add more features
- Reduce regularisation strength

**How to fix overfitting:**
- Get more training data (best solution)
- Simplify the model (reduce depth, fewer layers)
- Add regularisation (Ridge, Lasso, Dropout)
- Use early stopping during training

---

### Feature Scaling

Many regression algorithms are sensitive to the scale of features. Always scale unless you are using a tree-based model.

```
Before scaling:             After scaling (StandardScaler):
  Size:   50 – 300 m²         Size:   -1.5 to +1.5
  Age:    0 – 100 years        Age:    -1.5 to +1.5
  Price:  100k – 1M £          Price:  -1.5 to +1.5

Without scaling, large-valued features dominate the model.
```

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1 (best for most algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on training data
X_test_scaled  = scaler.transform(X_test)        # same transform on test

# MinMaxScaler: scales to [0, 1] range (useful for neural networks)
scaler = MinMaxScaler()
```

> **Critical rule:** Always `fit` the scaler on training data only. If you fit on test data, you leak information from the test set into your model.

---

### Target Scaling

Sometimes the target variable (the value you are predicting) also benefits from scaling, especially for neural networks and algorithms sensitive to large numbers.

```python
from sklearn.preprocessing import StandardScaler

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

# After prediction, reverse the scaling to get real prices back
predictions_scaled = model.predict(X_test)
predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
```

---

### Feature Engineering

Creating new features from existing ones to help the model learn better.

```python
# Example: house price dataset
df['price_per_m2']   = df['price'] / df['size_m2']          # new ratio feature
df['house_age_sq']   = df['age_years'] ** 2                  # quadratic relationship
df['total_rooms']    = df['bedrooms'] + df['bathrooms']      # combine features
df['is_new_build']   = (df['age_years'] < 5).astype(int)     # binary flag
df['zone_size']      = df['zone'] * df['size_m2']            # interaction term
```

Good features can improve model performance more than switching algorithms.

---

### Cross-Validation

Instead of one fixed train/validation split, rotate the validation set across K folds to get a more reliable score.

```
5-Fold Cross-Validation:

Fold 1: [  VAL  ] [train ] [train ] [train ] [train ]
Fold 2: [train ] [  VAL  ] [train ] [train ] [train ]
Fold 3: [train ] [train ] [  VAL  ] [train ] [train ]
Fold 4: [train ] [train ] [train ] [  VAL  ] [train ]
Fold 5: [train ] [train ] [train ] [train ] [  VAL  ]

Final RMSE = mean of 5 fold scores
```

```python
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -scores
print(f"RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
```

---

## 🔄 The ML Pipeline Step by Step

```
Step 1: Define the Problem
        └── What number am I predicting? What is success?

Step 2: Collect & Understand Data
        └── How many rows? Outliers? Missing values? Distribution of target?

Step 3: Exploratory Data Analysis (EDA)
        └── Correlations, scatter plots, histograms, boxplots

Step 4: Preprocess the Data
        ├── Handle missing values
        ├── Encode categorical variables
        ├── Scale features
        └── Remove or cap outliers

Step 5: Feature Engineering
        └── Create new features, select the most useful ones

Step 6: Split Data
        └── Train / Validation / Test

Step 7: Train Multiple Models
        └── Start with Linear Regression as baseline, then try Random Forest, XGBoost

Step 8: Evaluate & Compare Models
        └── RMSE, MAE, R² on validation set

Step 9: Tune the Best Model
        └── Hyperparameter search (Grid Search, Random Search)

Step 10: Final Evaluation on Test Set
         └── Do this ONCE — this is your honest performance estimate

Step 11: Interpret the Model
         └── Feature importance, residual analysis

Step 12: Deploy
         └── Save model, serve predictions via API
```

### Complete pipeline code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- Load data ---
df = pd.read_csv("house_prices.csv")

# --- Basic EDA ---
print(df.describe())
print(df.isnull().sum())

# --- Features & target ---
X = df.drop("price", axis=1)
y = df["price"]

# --- Handle missing values ---
X.fillna(X.median(), inplace=True)

# --- Encode categoricals ---
X = pd.get_dummies(X, drop_first=True)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scale ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- Train & compare models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":  mean_absolute_error(y_test, y_pred),
        "R²":   r2_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T
print(results_df.sort_values("RMSE"))

# --- Save the best model ---
best_model = models["Gradient Boosting"]
joblib.dump(best_model, "best_regressor.pkl")
joblib.dump(scaler, "scaler.pkl")
```

---

## 📏 Evaluation Metrics

### Mean Absolute Error (MAE)

**What it means:** On average, how many units is the prediction off by? Easy to understand because it is in the same units as the target.

```
MAE = (1/n) × Σ |actual - predicted|

Example:
  House 1: Actual £300k, Predicted £285k → Error £15k
  House 2: Actual £450k, Predicted £470k → Error £20k
  House 3: Actual £200k, Predicted £195k → Error £5k

  MAE = (15k + 20k + 5k) / 3 = £13,333
```

**Interpretation:** "On average, the model's predictions are off by £13,333."

**When to use:** When outliers exist and you do not want them to dominate the score. MAE treats all errors equally regardless of size.

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: £{mae:,.0f}")
```

---

### Mean Squared Error (MSE)

**What it means:** The average of squared errors. Because errors are squared, large errors are penalised much more heavily than small ones.

```
MSE = (1/n) × Σ (actual - predicted)²

Same example:
  MSE = (15² + 20² + 5²) / 3 = (225 + 400 + 25) / 3 = 216.67 (in £1000²)
```

**When to use:** When large errors are especially unacceptable and should be heavily penalised.

**Drawback:** Not in the same units as the target (it is squared), so hard to interpret directly. Use RMSE instead.

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
```

---

### Root Mean Squared Error (RMSE)

**What it means:** The square root of MSE — back in the same units as the target. The most commonly reported regression metric.

```
RMSE = √MSE = √216.67 ≈ £14,720 (in our example)
```

**Interpretation:** "A typical prediction error is about £14,720."

**When to use:** Almost always. RMSE is the standard regression metric. Penalises large errors more than MAE.

```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: £{rmse:,.0f}")
```

---

### R² Score (Coefficient of Determination)

**What it means:** How much of the variation in the target does the model explain? Ranges from -∞ to 1.

```
R² = 1 - (Sum of squared errors of model / Sum of squared errors of baseline)

Baseline = always predicting the mean value

R² = 1.0  → Perfect predictions (explains 100% of variation)
R² = 0.8  → Model explains 80% of variation → Good
R² = 0.5  → Model explains 50% → Moderate
R² = 0.0  → Model is no better than predicting the mean every time
R² < 0.0  → Model is WORSE than the mean baseline → Something is wrong
```

**When to use:** When you want a relative, scale-free measure of fit. Easy to explain to non-technical stakeholders: "Our model explains 87% of the variation in house prices."

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}  ({r2*100:.1f}% of variation explained)")
```

---

### Mean Absolute Percentage Error (MAPE)

**What it means:** Average percentage error relative to the actual value. Scale-independent, making it easy to compare across different problems.

```
MAPE = (100/n) × Σ |actual - predicted| / actual

Example:
  Actual £300k, Predicted £285k → Error = 5.0%
  Actual £100k, Predicted £95k  → Error = 5.0%  ← same percentage

MAPE of 8% means: "On average, predictions are 8% off."
```

**When to use:** When you want to express error as a percentage, especially useful for stakeholder communication.

**Limitation:** Breaks when actual values are zero or very small.

```python
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape*100:.2f}%")
```

---

### Residual Analysis

A **residual** is the difference between the actual and predicted value.

```
Residual = Actual - Predicted

Good model residuals look like:
  ●●●●   ●●●                    ← randomly scattered around zero
  ────────────────────── zero line
      ●●    ●●●●●               ← no pattern

Bad model residuals (pattern = the model is missing something):
  ●●●                           ← all errors on one side
  ────────────────────── zero line
               ●●●●●            ← systematic bias
```

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Residuals vs Predicted values
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel("Predicted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Predicted")

# Plot 2: Distribution of residuals
axes[1].hist(residuals, bins=30, edgecolor='black')
axes[1].set_xlabel("Residual value")
axes[1].set_title("Residual distribution (should be bell-shaped)")

plt.tight_layout()
plt.show()
```

---

### Metric Summary

| Metric | Range | Unit | Best when |
|--------|-------|------|-----------|
| **MAE** | 0 → ∞ | Same as target | Outliers present, easy interpretation |
| **MSE** | 0 → ∞ | Target² (squared) | Large errors must be penalised heavily |
| **RMSE** | 0 → ∞ | Same as target | Standard choice, penalises large errors |
| **R²** | -∞ → 1 | Unitless | Comparing models, explaining to stakeholders |
| **MAPE** | 0% → ∞ | Percentage | Comparing across different scales |

> **Rule of thumb:** Always report RMSE + R² together. Add MAE if outliers are a concern. Add MAPE if stakeholders think in percentages.

---

## 🛠️ Handling Common Problems

### Problem 1: Outliers in the Target Variable

**Signs:** A few extreme values are pulling the model's predictions off for everyone else.

```
Without outlier treatment:
  Price distribution: 95% of houses £100k–£600k, 5% > £5M
  → Model spends too much effort predicting luxury houses
  → Predictions for normal houses are worse

Detection:
```

```python
import matplotlib.pyplot as plt

# Visualise the distribution
plt.hist(y, bins=50)
plt.title("Target distribution")
plt.show()

# Find extreme outliers using IQR
Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR
outliers = y[(y < lower) | (y > upper)]
print(f"Outliers found: {len(outliers)} ({len(outliers)/len(y)*100:.1f}%)")

# Option A — Remove outliers
df_clean = df[(df['price'] >= lower) & (df['price'] <= upper)]

# Option B — Cap outliers (Winsorisation)
y_capped = y.clip(lower=lower, upper=upper)

# Option C — Log-transform the target (often works well for prices)
import numpy as np
y_log = np.log1p(y)  # log(1 + y) to handle zeros
# After prediction, reverse: predictions = np.expm1(y_log_pred)
```

---

### Problem 2: Non-Linear Relationships

**Signs:** Your model underfits even on training data. A straight line clearly cannot fit the data.

```python
# Option A — Polynomial features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])
pipe.fit(X_train, y_train)

# Option B — Switch to a non-linear model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()

# Option C — Log-transform skewed features
df['size_log'] = np.log1p(df['size_m2'])
```

---

### Problem 3: Multicollinearity (Correlated Features)

**What it is:** Two or more features are highly correlated with each other. This makes the model unstable — small changes in data lead to large changes in weights.

```python
# Detect correlations
import seaborn as sns
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Solution A — Remove one of the correlated features
X.drop('feature_b', axis=1, inplace=True)

# Solution B — Use Ridge regression (handles multicollinearity well)
from sklearn.linear_model import Ridge
model = Ridge(alpha=10.0)

# Solution C — Use PCA to combine correlated features
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
```

---

### Problem 4: Missing Values

```python
# Check missing values
print(df.isnull().sum().sort_values(ascending=False))

# Strategy A — Fill with median (numerical, robust to outliers)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Strategy B — Fill with mean
imputer = SimpleImputer(strategy='mean')

# Strategy C — KNN imputation (uses similar rows to fill gaps)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Strategy D — Add a flag column to tell the model a value was missing
X['size_was_missing'] = X['size_m2'].isnull().astype(int)
X['size_m2'].fillna(X['size_m2'].median(), inplace=True)
```

---

### Problem 5: Hyperparameter Tuning

Hyperparameters are settings you choose before training. Finding the best combination improves accuracy.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators':    randint(50, 300),
    'max_depth':       [3, 5, 7, 10, None],
    'learning_rate':   uniform(0.01, 0.2),
    'subsample':       uniform(0.6, 0.4),
    'min_samples_split': randint(2, 20)
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Best CV RMSE:", -search.best_score_)
```

---

### Problem 6: Skewed Target Distribution

**Signs:** Your target variable has a long tail — most values are clustered at the low end with a few very large values.

```
Skewed (bad for regression):   Bell-shaped (ideal):
  ████                              ████
  ████                           ████████
  ████████                    ██████████████
  ██████████████     vs.    ████████████████████
  0   100k  1M  10M         0  200k  400k  600k
```

```python
import numpy as np
import matplotlib.pyplot as plt

# Check skewness
print(f"Skewness: {y.skew():.2f}")   # > 1 or < -1 = significantly skewed

# Fix with log transform
y_log = np.log1p(y)
print(f"Skewness after log: {y_log.skew():.2f}")

# Train on log-transformed target
model.fit(X_train, np.log1p(y_train))

# Reverse log transform after prediction
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
```

---

## 🧭 Choosing the Right Algorithm

### Decision guide

```
Start
  │
  ├── Do you need to EXPLAIN the predictions?
  │     ├── Yes → Linear Regression, Ridge, Lasso, Decision Tree
  │     └── No  → Random Forest, XGBoost, Neural Network
  │
  ├── How much data do you have?
  │     ├── Small (<1,000 rows)
  │     │     └── Linear Regression, Ridge, Lasso, SVR, KNN
  │     ├── Medium (1k – 100k rows)
  │     │     └── Random Forest, XGBoost, LightGBM
  │     └── Large (>100k rows)
  │           └── LightGBM, XGBoost, Neural Networks
  │
  ├── Are there many irrelevant features?
  │     ├── Yes → Lasso (auto feature selection), Random Forest
  │     └── No  → Any algorithm
  │
  ├── Is the relationship non-linear?
  │     ├── Yes → Random Forest, XGBoost, Neural Network, Polynomial Regression
  │     └── No  → Linear Regression, Ridge, Lasso
  │
  └── General rule → Start simple, measure, then go more complex
```

### Quick comparison table

| Algorithm | Speed | Accuracy | Interpretable | Handles Non-linearity | Needs Scaling |
|-----------|-------|----------|---------------|----------------------|---------------|
| Linear Regression | ⚡⚡⚡ | ★★★ | ★★★★★ | ★ | Yes |
| Ridge / Lasso | ⚡⚡⚡ | ★★★ | ★★★★ | ★ | Yes |
| Decision Tree | ⚡⚡ | ★★★ | ★★★★★ | ★★★★ | No |
| Random Forest | ⚡⚡ | ★★★★ | ★★ | ★★★★ | No |
| XGBoost | ⚡⚡ | ★★★★★ | ★★ | ★★★★★ | No |
| SVR | ⚡ | ★★★★ | ★ | ★★★★ | Yes |
| KNN | ⚡ | ★★★ | ★★★ | ★★★★ | Yes |
| Neural Network | ⚡ | ★★★★★ | ★ | ★★★★★ | Yes |

---

## 💻 Quick Code Examples

### Minimal working example

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load built-in dataset (California house prices)
data = fetch_california_housing()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"RMSE:  {rmse:.4f}")
print(f"R²:    {r2:.4f}")
```

---

### Predict on new data

```python
import numpy as np
import joblib

# Load saved model and scaler
model  = joblib.load("best_regressor.pkl")
scaler = joblib.load("scaler.pkl")

# New house: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long]
new_house = np.array([[5.0, 15.0, 6.5, 1.1, 800.0, 3.0, 37.5, -122.2]])

# Scale and predict
new_house_scaled = scaler.transform(new_house)
predicted_price  = model.predict(new_house_scaled)
print(f"Predicted price: ${predicted_price[0] * 100_000:,.0f}")
```

---

### Feature importance

```python
import pandas as pd
import matplotlib.pyplot as plt

feature_names = data.feature_names
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh', figsize=(8, 5))
plt.title("Feature importance")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()
```

---

### Save and load model

```python
import joblib

# Save
joblib.dump(model, "regressor.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load
model  = joblib.load("regressor.pkl")
scaler = joblib.load("scaler.pkl")
```

---

## 🌍 Real-World Use Cases

| Domain | Task | Input Features | Output |
|--------|------|---------------|--------|
| Real estate | House price prediction | Size, location, bedrooms, age | Price in £ |
| Finance | Stock price forecasting | Historical prices, volume, indicators | Next day price |
| Healthcare | Hospital stay duration | Age, diagnosis, treatment, vitals | Days in hospital |
| Energy | Electricity demand | Temperature, day of week, season | kWh demand |
| Retail | Sales forecasting | Season, promotions, past sales | Units sold |
| HR | Salary estimation | Experience, education, role, skills | Annual salary |
| Transport | Delivery time | Distance, traffic, weather, package size | Minutes |
| Climate | Temperature forecasting | Pressure, humidity, wind speed, location | °C |
| Agriculture | Crop yield prediction | Rainfall, soil type, temperature, seed | kg per hectare |
| Marketing | Customer lifetime value | Purchase history, demographics, engagement | Revenue in £ |

---

## 📖 Glossary

| Term | Simple Definition |
|------|-----------------|
| **Bias** | The model's consistent tendency to be too high or too low (the intercept) |
| **Coefficient** | The weight the model assigns to a feature — how much it affects the output |
| **Cost Function** | A formula that measures how wrong the model's predictions are |
| **Cross-validation** | Testing a model on multiple train/val splits for a more reliable score |
| **ElasticNet** | A mix of Ridge (L2) and Lasso (L1) regularisation |
| **Feature Engineering** | Creating new features from existing ones to improve model performance |
| **Feature Scaling** | Transforming features to the same scale so large values don't dominate |
| **Gradient Descent** | The algorithm that adjusts model weights step by step to reduce error |
| **Hyperparameter** | A setting chosen before training (e.g. learning rate, tree depth) |
| **Intercept** | The predicted value when all input features are zero (baseline) |
| **L1 Regularisation** | A penalty that can shrink some weights to zero (Lasso) |
| **L2 Regularisation** | A penalty that shrinks all weights toward zero without removing them (Ridge) |
| **Lasso** | Linear regression with L1 regularisation — performs automatic feature selection |
| **Learning Rate** | How large a step gradient descent takes on each update |
| **Loss Function** | Same as cost function — measures prediction error during training |
| **MAE** | Mean Absolute Error — average of absolute prediction errors |
| **MAPE** | Mean Absolute Percentage Error — average percentage error |
| **MSE** | Mean Squared Error — average of squared prediction errors |
| **Multicollinearity** | When two or more input features are highly correlated with each other |
| **Overfitting** | When a model memorises training data and fails to generalise to new data |
| **Polynomial Regression** | Linear regression applied to polynomial features to capture curves |
| **R² Score** | How much of the target's variation is explained by the model (0–1) |
| **Regularisation** | A technique to prevent overfitting by penalising model complexity |
| **Residual** | The difference between the actual and predicted value for one sample |
| **Ridge** | Linear regression with L2 regularisation — handles correlated features |
| **RMSE** | Root Mean Squared Error — square root of MSE, in the same units as target |
| **Supervised Learning** | Learning from labelled examples where correct answers are provided |
| **Target Variable** | The number the model is trying to predict |
| **Training Set** | Data the model learns from |
| **Underfitting** | When a model is too simple and misses patterns in the data |
| **Validation Set** | Data used to tune the model during development |
| **Weight** | The importance assigned to a feature by the model during training |

---

## 📚 Further Reading

- [Scikit-learn Regression Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [StatQuest — Linear Regression (YouTube)](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Google Machine Learning Crash Course — Regression](https://developers.google.com/machine-learning/crash-course/descending-into-ml/linear-regression)

---

<p align="center">
  Written in plain English · No maths degree required · Happy learning 🎓
</p>