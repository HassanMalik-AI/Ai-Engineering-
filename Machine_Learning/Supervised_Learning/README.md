# 🎓 Supervised Learning — Complete Guide

> A thorough, beginner-friendly documentation of Supervised Learning in Machine Learning — covering every concept, algorithm, workflow, and best practice in plain English.

![Topic](https://img.shields.io/badge/Topic-Machine%20Learning-blueviolet)
![Type](https://img.shields.io/badge/Type-Supervised%20Learning-blue)
![Level](https://img.shields.io/badge/Level-Beginner%20to%20Advanced-green)
![Language](https://img.shields.io/badge/Language-Python-yellow?logo=python)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost%20%7C%20TensorFlow-orange)

---

## 📋 Table of Contents

- [What is Machine Learning?](#-what-is-machine-learning)
- [What is Supervised Learning?](#-what-is-supervised-learning)
- [How Supervised Learning Works](#-how-supervised-learning-works)
- [Types of Supervised Learning](#-types-of-supervised-learning)
- [Key Terminology](#-key-terminology)
- [The Data — Foundation of Everything](#-the-data--foundation-of-everything)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Splitting the Data](#-splitting-the-data)
- [Supervised Learning Algorithms](#-supervised-learning-algorithms)
- [Training a Model](#-training-a-model)
- [Evaluating a Model](#-evaluating-a-model)
- [Overfitting and Underfitting](#-overfitting-and-underfitting)
- [Regularisation](#-regularisation)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [The Complete ML Pipeline](#-the-complete-ml-pipeline)
- [Model Interpretability](#-model-interpretability)
- [Saving and Deploying Models](#-saving-and-deploying-models)
- [Common Mistakes to Avoid](#-common-mistakes-to-avoid)
- [Supervised vs Other Learning Types](#-supervised-vs-other-learning-types)
- [Real-World Use Cases](#-real-world-use-cases)
- [Choosing the Right Algorithm](#-choosing-the-right-algorithm)
- [Quick Code Examples](#-quick-code-examples)
- [Glossary](#-glossary)
- [Further Reading](#-further-reading)

---

## 🤖 What is Machine Learning?

**Machine learning** is a way of teaching computers to learn from examples — without manually writing rules for every situation.

### Traditional programming vs machine learning

```
Traditional Programming:
  Rules + Data → Computer → Output

  Example:
    IF email contains "free money" AND "click here" → mark as SPAM
    (You write every rule manually)

Machine Learning:
  Data + Outputs → Computer → Rules (learned automatically)

  Example:
    Show the computer 100,000 emails labelled SPAM or NOT SPAM
    The computer figures out the patterns on its own
```

### Three main types of machine learning

```
Machine Learning
      │
      ├── Supervised Learning   → Learns from labelled examples
      │     └── "Here are 10,000 house prices. Learn to predict new ones."
      │
      ├── Unsupervised Learning → Finds hidden patterns, no labels
      │     └── "Here are 10,000 customers. Group similar ones together."
      │
      └── Reinforcement Learning → Learns by trial and error with rewards
            └── "Play this game. Get points for winning. Figure out the rules."
```

> This guide covers **Supervised Learning** in complete detail.

---

## 🎓 What is Supervised Learning?

**Supervised learning** is the most widely used form of machine learning. You train a model using a dataset where every example already has a correct answer (called a **label**).

The model learns the relationship between the inputs and the correct answers. After training, it can predict answers for new data it has never seen before.

### The classroom analogy

Think of supervised learning like studying for an exam using past papers:

```
Past Papers (Training Data):
  Question: "House is 120m², 3 beds, Zone 2"  →  Answer: "£380,000"
  Question: "House is 60m², 1 bed, Zone 5"   →  Answer: "£165,000"
  Question: "House is 200m², 5 beds, Zone 1" →  Answer: "£950,000"

After studying thousands of past papers, the student (model)
can predict the price of a new house it has never seen before.
```

### What makes it "supervised"?

The word **supervised** comes from the fact that the training process is guided (supervised) by the correct answers. Every training example tells the model:

- Here is the input (features)
- Here is the correct output (label)
- Adjust yourself until you can predict this correctly

```
Input → Model → Prediction
                    ↓
              Compare with
              correct label
                    ↓
              Calculate error
                    ↓
              Adjust model weights
                    ↓
              Repeat thousands of times
```

---

## ⚙️ How Supervised Learning Works

### The complete learning loop

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
│                                                             │
│  Training Data                                              │
│  (with labels)                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐    makes     ┌────────────┐                   │
│  │  Model  │─────────────▶│ Prediction │                   │
│  └─────────┘              └────────────┘                   │
│       ▲                         │                          │
│       │                         ▼                          │
│       │   adjust weights  ┌────────────┐                   │
│       └───────────────────│   Error    │◀── Correct Label  │
│                           └────────────┘                   │
│                                                             │
│  Repeat until error is small enough                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION PHASE                          │
│                                                             │
│  New, unseen data ──▶ Trained Model ──▶ Prediction         │
│  (no label needed)                                          │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-step walkthrough

**Step 1 — Feed data in:** The model receives a set of features (input values) for one training example.

**Step 2 — Make a prediction:** Based on its current internal settings (weights), it produces an output.

**Step 3 — Measure the error:** It compares its prediction to the correct label and calculates how wrong it was. This measurement is called the **loss** or **cost**.

**Step 4 — Adjust the weights:** Using an algorithm called **gradient descent**, the model tweaks its internal weights to reduce the error.

**Step 5 — Repeat:** This cycle repeats across all training samples, many times over (each full pass is called an **epoch**), until the model is accurate enough.

---

## 🗂️ Types of Supervised Learning

Supervised learning has two main task types depending on what kind of output you want to predict.

### 1. Classification — predict a category

The model predicts which **group** or **class** something belongs to.

```
Input → Model → One (or more) categories

Examples:
  Email → Spam or Not Spam
  Image → Cat, Dog, Bird, or Fish
  Patient data → Has disease or Does not have disease
  Transaction → Fraud or Legitimate
```

**Sub-types of classification:**

| Sub-type | Description | Example |
|----------|-------------|---------|
| **Binary** | Exactly 2 possible classes | Spam / Not Spam |
| **Multi-class** | 3 or more classes, predict one | Digit: 0, 1, 2…9 |
| **Multi-label** | Predict multiple classes at once | Movie genres: Action AND Comedy |
| **Imbalanced** | One class has far more samples | Fraud detection (1% fraud, 99% normal) |

---

### 2. Regression — predict a number

The model predicts a **continuous numerical value** — a real number.

```
Input → Model → A number

Examples:
  House features → £342,500
  Weather data   → 24.5°C
  Sales history  → 8,450 units next month
  CV features    → £62,000 salary
```

**Sub-types of regression:**

| Sub-type | Description | Example |
|----------|-------------|---------|
| **Simple linear** | One input feature | Experience → Salary |
| **Multiple linear** | Many input features | Size + Location + Age → Price |
| **Polynomial** | Curved (non-linear) relationship | Speed → Fuel consumption |
| **Time series** | Predict based on past values over time | Yesterday's sales → Today's sales |

---

### Classification vs Regression at a glance

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│                 │   Classification     │     Regression       │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Output type     │ Category / Label     │ Continuous number    │
│ Question        │ Which group?         │ How much / how many? │
│ Output example  │ "Spam"               │ £342,500             │
│ Error measure   │ Was it correct?      │ How far off?         │
│ Metric          │ Accuracy, F1         │ RMSE, MAE, R²        │
│ Algorithm ex.   │ Logistic Regression  │ Linear Regression    │
└─────────────────┴──────────────────────┴──────────────────────┘
```

---

## 📖 Key Terminology

Understanding the vocabulary is half the battle. Here are the core terms you will see everywhere.

### Data terms

| Term | Plain English definition | Example |
|------|------------------------|---------|
| **Dataset** | Your complete collection of examples | 50,000 house sale records |
| **Sample / Instance** | One single example in the dataset | One house's details |
| **Feature / Input** | A piece of information about a sample | Size in m², number of bedrooms |
| **Label / Target** | The correct answer for that sample | Sold price: £285,000 |
| **Dimensionality** | The number of features in your dataset | 15 features = 15 dimensions |

### Model terms

| Term | Plain English definition |
|------|------------------------|
| **Model** | The mathematical function that maps inputs to outputs |
| **Weight / Parameter** | A number inside the model that is adjusted during training |
| **Bias (intercept)** | The baseline prediction when all features are zero |
| **Hypothesis** | The function the model uses to make predictions |
| **Loss / Cost** | A number measuring how wrong the model's predictions are |
| **Gradient Descent** | The algorithm that adjusts weights to reduce loss |
| **Learning Rate** | How big a step the model takes when adjusting weights |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A small subset of training data processed at once |
| **Convergence** | When the loss stops improving — training is done |

### Evaluation terms

| Term | Plain English definition |
|------|------------------------|
| **Training error** | How wrong the model is on training data |
| **Validation error** | How wrong the model is on the held-out validation set |
| **Generalisation** | How well the model performs on new, unseen data |
| **Overfitting** | Model memorised training data, fails on new data |
| **Underfitting** | Model is too simple and misses the patterns |
| **Baseline** | A simple benchmark to beat (e.g. always predict the mean) |

---

## 📦 The Data — Foundation of Everything

In supervised learning, the quality and quantity of your data matters more than which algorithm you pick. A simple model with great data beats a complex model with bad data almost every time.

### What good data looks like

```
Good data characteristics:
  ✓ Enough samples (rule of thumb: at least 10× the number of features)
  ✓ Correct and accurate labels
  ✓ Representative of the real world (no hidden biases)
  ✓ Balanced classes (for classification)
  ✓ Consistent format (no mixed units, no typos)
  ✓ Low percentage of missing values
```

### Data quantity guidelines

```
Task complexity          Minimum samples needed
─────────────────────────────────────────────
Simple linear problem    100 – 1,000
Moderate complexity      1,000 – 10,000
Complex patterns         10,000 – 100,000
Images / Text / Audio    100,000 – millions
```

### Understanding your data (EDA)

Before training any model, explore your data visually and statistically. This is called **Exploratory Data Analysis (EDA)**.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

# Basic statistics
print(df.shape)           # How many rows and columns?
print(df.head())          # First 5 rows
print(df.describe())      # Mean, std, min, max for each column
print(df.dtypes)          # Data type of each column
print(df.isnull().sum())  # Missing values per column

# Distribution of target variable
plt.figure(figsize=(8, 4))
df['target'].hist(bins=50)
plt.title("Target distribution")
plt.xlabel("Target value")
plt.show()

# Correlation between features and target
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature correlation matrix")
plt.show()

# Check class balance (for classification)
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True) * 100)
```

### Data types you will encounter

| Data type | Description | Examples | Notes |
|-----------|-------------|---------|-------|
| **Numerical** | Numbers | Age, income, temperature | Ready to use (after scaling) |
| **Categorical** | Text labels | City, colour, gender | Must encode to numbers |
| **Ordinal** | Ordered categories | Low/Medium/High | Encode preserving order |
| **Binary** | Two values only | Yes/No, True/False | Encode as 0 and 1 |
| **Text** | Free-form sentences | Reviews, emails | Needs NLP preprocessing |
| **Image** | Pixel grids | Photos, scans | Needs CNN or flattening |
| **Time series** | Values over time | Stock prices, sensor data | Order matters |

---

## 🧹 Data Preprocessing

Raw data is almost never ready for a machine learning model. Preprocessing turns messy real-world data into clean, model-ready inputs.

### Step 1 — Handle missing values

Missing values appear as `NaN` (Not a Number) in pandas. You must deal with them before training.

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# Check what is missing
print(df.isnull().sum())
print(f"Missing: {df.isnull().sum().sum()} values ({df.isnull().mean().mean()*100:.1f}% of data)")

# Strategy A — Drop rows with any missing value (only if very few are missing)
df_clean = df.dropna()

# Strategy B — Fill with median (best for numerical — robust to outliers)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Strategy C — Fill with mean (for normally distributed features)
imputer = SimpleImputer(strategy='mean')

# Strategy D — Fill with most common value (for categorical features)
imputer = SimpleImputer(strategy='most_frequent')

# Strategy E — KNN imputation (use similar rows to fill gaps — most accurate)
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Strategy F — Add a flag column to tell the model a value was missing
df['age_missing'] = df['age'].isnull().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)
```

**Which strategy to use:**

| Missing % | Recommended strategy |
|-----------|-------------------|
| < 5% | Drop rows OR fill with median/mean |
| 5% – 30% | Fill with median, mean, or KNN |
| > 30% | Consider dropping the column entirely |

---

### Step 2 — Encode categorical variables

Machine learning models only understand numbers. Text categories must be converted.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# --- One-Hot Encoding (for nominal categories with no order) ---
# Red → [1, 0, 0]    Blue → [0, 1, 0]    Green → [0, 0, 1]
X_encoded = pd.get_dummies(X, columns=['colour', 'city'], drop_first=True)

# Or with sklearn
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_ohe = enc.fit_transform(X[['colour', 'city']])

# --- Label Encoding (for binary categories) ---
# Male → 0    Female → 1
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])

# --- Ordinal Encoding (for ordered categories) ---
# Low → 0    Medium → 1    High → 2
enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
X['education'] = enc.fit_transform(X[['education']])
```

**When to use which:**

| Method | Use when | Warning |
|--------|----------|---------|
| One-Hot | Nominal (no order): colours, cities | Creates many columns for high-cardinality |
| Label Encoding | Binary only, or tree models | Implies false order for multi-class |
| Ordinal Encoding | There is a real order: Low/Med/High | Must define the correct order |

---

### Step 3 — Feature scaling

Most algorithms are sensitive to the scale of your features. Without scaling, a feature with large values (e.g. income: £50,000) will dominate a feature with small values (e.g. age: 35).

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler — mean=0, std=1 (best general-purpose choice)
# Good for: Linear models, SVM, Neural Networks, KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on training data
X_test_scaled  = scaler.transform(X_test)        # apply same transform to test

# MinMaxScaler — scales to [0, 1] range
# Good for: Neural networks, image pixel values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# RobustScaler — uses median and IQR, not affected by outliers
# Good for: Data with many outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Critical rule — never fit on test data:**

```python
# WRONG — leaks test information into the model
scaler.fit(X_test)  # ← NEVER do this

# CORRECT — fit once on training data only
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # same scaler, no re-fitting
```

**Which models need scaling:**

| Needs Scaling | Does NOT Need Scaling |
|---------------|-----------------------|
| Linear / Logistic Regression | Decision Tree |
| SVM / SVR | Random Forest |
| Neural Networks | XGBoost / LightGBM |
| KNN | Gradient Boosting |

---

### Step 4 — Handle outliers

Outliers are extreme values that can distort what the model learns.

```python
import numpy as np

# Detect outliers using IQR (Interquartile Range)
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['income'] < lower) | (df['income'] > upper)]
print(f"Outliers found: {len(outliers)}")

# Option A — Remove outliers
df_clean = df[(df['income'] >= lower) & (df['income'] <= upper)]

# Option B — Cap outliers (Winsorisation)
df['income'] = df['income'].clip(lower=lower, upper=upper)

# Option C — Log transform (compresses large values)
df['income_log'] = np.log1p(df['income'])
```

---

## 🔨 Feature Engineering

**Feature engineering** is the art of creating new, more informative features from your existing data. It is often the single most impactful thing you can do to improve model performance.

### Why it matters

```
Raw features:       birth_date, join_date
Engineered feature: customer_age_years = (join_date - birth_date).days / 365

Raw features:       order_count, complaint_count
Engineered feature: complaint_rate = complaint_count / order_count

Raw features:       purchase_date
Engineered features: day_of_week, is_weekend, month, quarter, days_since_purchase
```

### Common techniques

```python
import pandas as pd
import numpy as np

# --- Ratio features ---
df['price_per_m2']    = df['price'] / df['size_m2']
df['debt_to_income']  = df['total_debt'] / df['annual_income']

# --- Interaction features (combine two features) ---
df['size_x_quality']  = df['size_m2'] * df['quality_score']

# --- Polynomial features (capture curved relationships) ---
df['size_squared']    = df['size_m2'] ** 2
df['age_squared']     = df['age_years'] ** 2

# --- Binning (convert number to category) ---
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                          labels=['youth', 'young_adult', 'adult', 'senior'])

# --- Date/time features ---
df['join_date'] = pd.to_datetime(df['join_date'])
df['join_year']      = df['join_date'].dt.year
df['join_month']     = df['join_date'].dt.month
df['join_day_of_week'] = df['join_date'].dt.dayofweek
df['is_weekend']     = df['join_day_of_week'].isin([5, 6]).astype(int)
df['days_as_customer'] = (pd.Timestamp.today() - df['join_date']).dt.days

# --- Aggregation features (for grouped data) ---
group_stats = df.groupby('city')['price'].agg(['mean', 'std', 'count'])
group_stats.columns = ['city_avg_price', 'city_price_std', 'city_listing_count']
df = df.merge(group_stats, on='city')

# --- Text length (simple text feature) ---
df['review_length'] = df['review_text'].str.len()
df['word_count']    = df['review_text'].str.split().str.len()
```

### Feature selection — removing useless features

Too many irrelevant features add noise and slow training. Remove them.

```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Method A — Select K best features by statistical test
selector = SelectKBest(score_func=f_classif, k=10)  # f_regression for regression
X_selected = selector.fit_transform(X_train, y_train)
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

# Method B — Feature importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importance.nlargest(10).index.tolist()

# Method C — Remove low variance features (nearly constant columns)
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)

# Method D — Remove highly correlated features (keep one of each pair)
corr_matrix = pd.DataFrame(X).corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
X_reduced = pd.DataFrame(X).drop(columns=to_drop)
```

---

## ✂️ Splitting the Data

Before training, divide your data into separate sets so you can evaluate the model honestly.

### The three-way split

```
Full Dataset (100%)
      │
      ├── Training Set   (70%) ── The model LEARNS from this
      │                          (fits weights and parameters)
      │
      ├── Validation Set (15%) ── Used to TUNE the model
      │                          (pick hyperparameters, compare algorithms)
      │
      └── Test Set       (15%) ── Final HONEST evaluation
                                  (used ONCE at the very end — never during development)
```

**Why three sets and not two?**

```
If you tune on validation data repeatedly → you overfit to validation data
If you then use test data → you get a biased, over-optimistic score
The test set MUST be completely invisible during all development
```

```python
from sklearn.model_selection import train_test_split

# Step 1 — separate out the test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y           # keeps class proportions equal in every split
)

# Step 2 — split the rest into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.176,     # 0.176 × 85% ≈ 15% of total dataset
    random_state=42,
    stratify=y_trainval
)

print(f"Training:   {len(X_train):,} samples")
print(f"Validation: {len(X_val):,} samples")
print(f"Test:       {len(X_test):,} samples")
```

### K-Fold Cross-Validation

Instead of one fixed split, rotate the validation window across K portions of the data. This gives a much more reliable estimate of real performance.

```
5-Fold Cross-Validation — data split into 5 equal parts:

Fold 1: [ VAL ] [train] [train] [train] [train]  → Score 1
Fold 2: [train] [ VAL ] [train] [train] [train]  → Score 2
Fold 3: [train] [train] [ VAL ] [train] [train]  → Score 3
Fold 4: [train] [train] [train] [ VAL ] [train]  → Score 4
Fold 5: [train] [train] [train] [train] [ VAL ]  → Score 5

Final Score = Mean of all 5 scores  ±  Standard Deviation
```

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Classification
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
print(f"F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Regression
scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -scores
print(f"RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
```

---

## 🧮 Supervised Learning Algorithms

### Classification algorithms

#### Logistic Regression

Despite its name, this is a **classification** algorithm. It estimates the probability that an input belongs to a class.

```
Output = sigmoid(w₁x₁ + w₂x₂ + ... + b)
sigmoid(z) = 1 / (1 + e⁻ᶻ)  →  always outputs a value between 0 and 1

If output > 0.5 → Class 1
If output ≤ 0.5 → Class 0
```

**Best for:** Binary classification, when you need probability scores, linearly separable data
**Pros:** Fast, highly interpretable, probability outputs
**Cons:** Cannot learn non-linear patterns without feature engineering

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
predictions  = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # confidence scores

# Which features drive predictions?
for feature, coef in zip(feature_names, model.coef_[0]):
    print(f"{feature:20s}: {coef:+.4f}")
```

---

#### Decision Tree Classifier

Builds a tree of yes/no questions to divide data into classes. Very easy to visualise and explain.

```
Is annual income > £40,000?
      ├── YES → Is credit score > 700?
      │           ├── YES → APPROVED ✓
      │           └── NO  → REJECTED ✗
      └── NO  → Is existing debt < £5,000?
                  ├── YES → APPROVED ✓
                  └── NO  → REJECTED ✗
```

**Best for:** When decisions need to be explained, mixed data types
**Pros:** No scaling needed, easy to visualise, handles non-linearity
**Cons:** Prone to overfitting if tree is too deep

```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

# Print the decision rules as text
print(export_text(model, feature_names=feature_names))
```

---

#### Random Forest Classifier

Builds hundreds of decision trees on random subsets of the data and takes a majority vote.

```
Input
  ├── Tree 1   → Class A
  ├── Tree 2   → Class A
  ├── Tree 3   → Class B
  ├── Tree 4   → Class A
  └── Tree 5   → Class A
              ↓
       Majority vote: Class A (4 out of 5)
```

**Best for:** General-purpose classification, feature importance, handling missing data
**Pros:** Very accurate, robust to overfitting, shows feature importance
**Cons:** Less interpretable than a single tree, slower

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    class_weight='balanced',    # handles class imbalance
    random_state=42,
    n_jobs=-1                   # use all CPU cores
)
model.fit(X_train, y_train)
```

---

#### Gradient Boosting (XGBoost / LightGBM)

Builds trees one by one, where each new tree fixes the errors of the previous ones.

```
Round 1: Predict  →  Error: 45%
Round 2: New tree focuses on the 45% errors  →  Remaining error: 28%
Round 3: New tree focuses on the 28% errors  →  Remaining error: 18%
...
Round N: Combined prediction of all trees    →  Final error: 5%
```

**Best for:** Tabular data, maximum accuracy, Kaggle-style competitions
**Pros:** State-of-the-art on tabular data, handles missing values natively
**Cons:** Many hyperparameters, can overfit, slower to interpret

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)
```

---

#### Support Vector Machine (SVM)

Finds the widest possible "street" (margin) between classes in high-dimensional space.

```
Class A points:   ● ● ●
                ════════════  ← maximum margin boundary
Class B points:     ○ ○ ○

New point above the line → Class A
New point below the line → Class B
```

**Best for:** High-dimensional data, text classification, small-to-medium datasets
**Pros:** Works well in many dimensions, robust to some outliers
**Cons:** Slow for large datasets, requires scaling, hard to interpret

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)
```

---

#### K-Nearest Neighbors (KNN) Classifier

Classifies a new point by looking at its K nearest neighbours and taking a majority vote.

```
New point: ★   (K = 5 nearest neighbours)
Neighbours found: ● ● ● ○ ●   (3 Class A, 2 Class B)
Prediction: Class A (majority)
```

**Best for:** Small datasets, when decision boundaries are irregular
**Pros:** No training phase, simple and intuitive
**Cons:** Slow prediction on large data, sensitive to irrelevant features

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
model.fit(X_train_scaled, y_train)
```

---

#### Naive Bayes Classifier

Uses Bayes' theorem (probability theory) to classify data. Assumes features are independent.

```
P(Spam | words) ∝ P(words | Spam) × P(Spam)

"Given that this email contains 'free', 'money', 'click here',
 what is the probability it is spam?"
```

**Best for:** Text classification, when data is limited, real-time prediction
**Pros:** Extremely fast, works with little data, great for NLP
**Cons:** Independence assumption is rarely true in practice

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
model = GaussianNB()

# For word counts / text features
model = MultinomialNB(alpha=1.0)

model.fit(X_train, y_train)
```

---

### Regression algorithms

#### Linear Regression

Fits the best straight line through the data by learning one weight per feature.

```
Predicted value = w₁×feature₁ + w₂×feature₂ + ... + wₙ×featureₙ + bias

House price = (200 × size_m²) + (15000 × bedrooms) + (50000 × has_garden) + 80000
                  ↑                    ↑                      ↑
              learned weight      learned weight          learned weight
```

**Best for:** Linear relationships, baseline model, when interpretability is critical
**Pros:** Fastest training, fully interpretable, probability theory foundation
**Cons:** Cannot capture non-linear patterns

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Interpret the model
for name, weight in zip(feature_names, model.coef_):
    print(f"{name:20s}: {weight:+.2f}")
print(f"Base prediction (intercept): {model.intercept_:.2f}")
```

---

#### Ridge Regression (L2 Regularisation)

Linear regression with a penalty that prevents any single weight from getting too large.

```
Total loss = prediction error + α × (sum of all weights²)
                                ↑
                   Controls penalty strength (hyperparameter)
```

**Best for:** Multicollinearity (correlated features), preventing overfitting of linear models
**Pros:** Stable when features are correlated, simple to tune
**Cons:** Does not remove useless features

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train)
```

---

#### Lasso Regression (L1 Regularisation)

Like Ridge, but can shrink some weights all the way to zero — automatically removing useless features.

```
Total loss = prediction error + α × (sum of absolute weights)

Ridge:   weights approach 0 but never reach it  → keeps all features
Lasso:   weights can equal 0 exactly            → automatic feature removal
```

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train_scaled, y_train)

# See which features were kept vs removed
kept = [f for f, w in zip(feature_names, model.coef_) if w != 0]
removed = [f for f, w in zip(feature_names, model.coef_) if w == 0]
print(f"Kept {len(kept)} features, removed {len(removed)}")
```

---

#### Random Forest Regressor

Hundreds of decision trees each predict a number; their average is the final prediction.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

#### Gradient Boosting Regressor (XGBoost)

Sequential trees, each correcting the residual errors of all previous trees.

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=30,
          verbose=False)
```

---

### Algorithm comparison at a glance

| Algorithm | Task | Speed | Accuracy | Interpretable | Needs Scaling |
|-----------|------|-------|----------|---------------|---------------|
| Logistic Regression | C | ⚡⚡⚡ | ★★★ | ★★★★★ | Yes |
| Decision Tree | C / R | ⚡⚡ | ★★★ | ★★★★★ | No |
| Random Forest | C / R | ⚡⚡ | ★★★★ | ★★ | No |
| XGBoost / LightGBM | C / R | ⚡⚡ | ★★★★★ | ★★ | No |
| SVM / SVR | C / R | ⚡ | ★★★★ | ★ | Yes |
| KNN | C / R | ⚡ | ★★★ | ★★★ | Yes |
| Naive Bayes | C | ⚡⚡⚡ | ★★★ | ★★★ | No |
| Linear Regression | R | ⚡⚡⚡ | ★★★ | ★★★★★ | Yes |
| Ridge / Lasso | R | ⚡⚡⚡ | ★★★ | ★★★★ | Yes |
| Neural Network | C / R | ⚡ | ★★★★★ | ★ | Yes |

*C = Classification, R = Regression*

---

## 🏋️ Training a Model

### What happens inside training

During training the model repeatedly:

1. **Forward pass** — computes a prediction using current weights
2. **Loss calculation** — measures how wrong the prediction is
3. **Backward pass** — calculates how much each weight contributed to the error
4. **Weight update** — adjusts each weight slightly to reduce the error

### The loss function

The loss function is the measuring stick for how wrong the model is.

| Task | Loss Function | Formula |
|------|-------------|---------|
| Binary classification | Binary Cross-Entropy | `-[y·log(p) + (1-y)·log(1-p)]` |
| Multi-class classification | Categorical Cross-Entropy | `-Σ yᵢ·log(pᵢ)` |
| Regression | Mean Squared Error (MSE) | `(1/n) Σ (actual - predicted)²` |
| Regression (robust) | Mean Absolute Error (MAE) | `(1/n) Σ |actual - predicted|` |

### Gradient descent — explained simply

Imagine you are standing on a foggy hillside and want to reach the lowest valley (minimum loss), but you cannot see far ahead. You feel which direction slopes downward beneath your feet and take one small step in that direction. Repeat until you cannot go any lower.

```
Loss
  │╲
  │ ╲
  │  ╲____
  │       ╲___
  │           ╲____★  ← Minimum loss (best weights)
  └────────────────── Weights

Each arrow → is one gradient descent step (size = learning rate)
```

### Learning rate — a critical hyperparameter

```
Too high learning rate:
  Steps are too big → overshoots the minimum → loss bounces around
  │     ★           │    ★                   │         ★
  │  ↗  ↘  ↗  ↘   →   oscillates forever   → model never converges
  └─────────────

Too low learning rate:
  Steps are tiny → training takes forever
  │╲............... very slow progress

Just right:
  │╲
  │ ╲___★  converges smoothly and quickly
  └────────
```

```python
# Common starting values for learning rate:
# Gradient Boosting:  0.01 – 0.1
# Neural Networks:    0.0001 – 0.01
# Logistic Regression: determined by solver (no manual setting needed)
```

---

## 📊 Evaluating a Model

### Classification metrics

#### Confusion Matrix

The foundation of all classification metrics. Shows where predictions are correct and where they are wrong.

```
                     Predicted
                  Positive   Negative
Actual  Positive [ TP = 85 | FN = 15 ]
        Negative [ FP = 10 | TN = 90 ]

TP = True Positive  → Said Yes, Was Yes   ✓
TN = True Negative  → Said No,  Was No    ✓
FP = False Positive → Said Yes, Was No    ✗ (false alarm)
FN = False Negative → Said No,  Was Yes   ✗ (missed it)
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
```

#### Accuracy

```
Accuracy = (TP + TN) / Total predictions

When to use:  Balanced classes
When to avoid: Imbalanced classes (99% "Not Fraud" → 99% accuracy by always saying Not Fraud)
```

#### Precision and Recall

```
Precision = TP / (TP + FP)   "Of all my Positive predictions, how many were right?"
Recall    = TP / (TP + FN)   "Of all actual Positives, how many did I find?"

High Precision → Few false alarms (good for: spam filter, legal decisions)
High Recall    → Few missed cases (good for: cancer screening, fraud detection)
```

#### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Balances precision and recall into one number.
Range: 0 (worst) to 1 (best)
Use when: Classes are imbalanced OR both precision and recall matter
```

#### ROC-AUC Score

```
AUC = 0.5  → No better than random guessing
AUC = 0.7  → Acceptable
AUC = 0.8  → Good
AUC = 0.9  → Excellent
AUC = 1.0  → Perfect (check for data leakage)
```

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

print(classification_report(y_test, y_pred, target_names=class_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

---

### Regression metrics

#### MAE — Mean Absolute Error

```
MAE = (1/n) × Σ |actual - predicted|

Average of absolute errors — in the same units as the target.
"On average, predictions are off by £12,400"
Use when: Outliers are present, easy interpretation needed
```

#### RMSE — Root Mean Squared Error

```
RMSE = √[ (1/n) × Σ (actual - predicted)² ]

Squares errors before averaging → large errors penalised more.
"A typical prediction error is £18,700"
Use when: Large errors are especially unacceptable
```

#### R² Score

```
R² = 1 − (model error / baseline error)

R² = 1.0  → Perfect (explains 100% of variation)
R² = 0.85 → Explains 85% of variation (good)
R² = 0.0  → No better than always predicting the mean
R² < 0.0  → Worse than predicting the mean (something is wrong)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}  ({r2*100:.1f}% of variation explained)")
```

---

### Metric selection guide

| Situation | Recommended metrics |
|-----------|-------------------|
| Balanced classification | Accuracy, F1 |
| Imbalanced classification | F1, ROC-AUC, Precision-Recall curve |
| False positives are costly | Precision |
| False negatives are costly | Recall |
| Regression, general | RMSE + R² |
| Regression with outliers | MAE + R² |
| Regression for stakeholders | MAPE (percentage error) |

---

## 📉 Overfitting and Underfitting

This is the central challenge of machine learning — finding the right model complexity.

### The bias-variance trade-off

```
                Complexity of model
Simple ─────────────────────────────────── Complex

Underfitting              ★ Just right              Overfitting
(High bias)                                          (High variance)
Model too simple          Good generalisation        Model too complex
Misses patterns           Low train + val error      Memorises training data
                                                     Fails on new data
```

### How to detect each

```
Underfitting:
  Training accuracy:    70%
  Validation accuracy:  69%
  → Both are low — model needs to be more complex

Overfitting:
  Training accuracy:    99%
  Validation accuracy:  72%
  → Large gap — model memorised training data

Just right:
  Training accuracy:    91%
  Validation accuracy:  88%
  → Small gap, both high — good generalisation
```

### How to fix underfitting

- Use a more complex algorithm (e.g. switch from linear to tree-based)
- Add more features or engineer new ones
- Reduce regularisation strength (smaller alpha/C)
- Train for more epochs
- Remove noise from labels

### How to fix overfitting

```python
# 1. Get more training data (best solution)
# 2. Reduce model complexity
model = DecisionTreeClassifier(max_depth=4)   # limit tree depth
model = RandomForestClassifier(max_depth=8)   # limit forest depth

# 3. Add regularisation (see Regularisation section)
model = LogisticRegression(C=0.1)             # smaller C = more regularisation
model = Ridge(alpha=10.0)                     # larger alpha = more regularisation

# 4. Early stopping (stop training when validation loss stops improving)
model = XGBClassifier(early_stopping_rounds=20)

# 5. Dropout (neural networks)
keras.layers.Dropout(0.5)

# 6. Cross-validation to detect overfitting early
scores = cross_val_score(model, X_train, y_train, cv=5)
```

---

## 🛡️ Regularisation

Regularisation adds a penalty to the model for being too complex. It forces the model to learn simpler patterns that generalise better.

### L1 Regularisation (Lasso)

```
Total loss = prediction error + α × Σ|wᵢ|

Effect: Some weights become exactly zero → automatic feature removal
When α = 0: no regularisation (pure linear regression)
When α is large: many features removed (heavy regularisation)
```

### L2 Regularisation (Ridge)

```
Total loss = prediction error + α × Σwᵢ²

Effect: All weights shrink toward zero but never reach it
When α = 0: no regularisation
When α is large: all weights are very small
```

### ElasticNet — best of both

```
Total loss = prediction error + α × [ρ × Σ|wᵢ| + (1-ρ) × Σwᵢ²]
                                       ↑ L1 part       ↑ L2 part
ρ (l1_ratio) controls the mix
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Test different regularisation strengths
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Alpha={alpha:6.3f}  →  R²: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Dropout (Neural Networks)

During each training step, randomly switch off a fraction of neurons. Forces the network to not rely on any single neuron.

```
Normal layer:  [●] [●] [●] [●] [●] [●]
After dropout: [●] [○] [●] [●] [○] [●]   ← 2 neurons switched off at random

Next step:     [●] [●] [○] [●] [●] [○]   ← different neurons switched off

Effect: Network learns redundant representations → less overfitting
```

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),           # 30% of neurons off during training
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])
```

---

## 🎛️ Hyperparameter Tuning

**Parameters** are learned by the model during training (weights, biases).
**Hyperparameters** are settings YOU choose before training begins.

Finding the best hyperparameters is called **hyperparameter tuning** or **model selection**.

### Common hyperparameters per algorithm

| Algorithm | Key hyperparameters |
|-----------|-------------------|
| Logistic Regression | `C` (regularisation strength), `solver` |
| Decision Tree | `max_depth`, `min_samples_leaf`, `min_samples_split` |
| Random Forest | `n_estimators`, `max_depth`, `max_features` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| SVM | `C`, `kernel`, `gamma` |
| KNN | `n_neighbors`, `weights`, `metric` |
| Neural Network | `learning_rate`, `batch_size`, `n_layers`, `dropout_rate` |

### Method 1 — Grid Search (exhaustive)

Tries every possible combination. Thorough but slow for many hyperparameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [3, 5, 10, None],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:  ", grid_search.best_score_)
best_model = grid_search.best_estimator_
```

### Method 2 — Random Search (efficient)

Randomly samples combinations from a defined search space. Often finds good results faster.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators':     randint(50, 500),
    'max_depth':        [3, 5, 7, 10, None],
    'max_features':     ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_leaf': randint(1, 20),
    'learning_rate':    uniform(0.01, 0.2)   # for gradient boosting
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,          # try 50 random combinations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
print("Best params:", random_search.best_params_)
```

### Method 3 — Bayesian Optimisation (smartest)

Uses results of previous trials to intelligently choose the next set of hyperparameters to try.

```python
# pip install optuna
import optuna

def objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
        'max_depth':     trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':     trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = XGBClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print("Best params:", study.best_params)
```

---

## 🔄 The Complete ML Pipeline

Putting it all together — from raw data to deployed model.

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Define the Problem                                         │
│  What am I predicting? Classification or Regression?               │
│  What is the success metric? What does "good enough" mean?          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Collect Data                                               │
│  Gather labelled examples. More quality data = better model.        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Explore the Data (EDA)                                     │
│  Distributions, missing values, outliers, class balance,            │
│  correlations between features and target.                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Preprocess the Data                                        │
│  Handle missing values, encode categoricals, scale features,        │
│  remove or cap outliers.                                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: Feature Engineering                                        │
│  Create new features, combine existing ones, select the best.       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: Split the Data                                             │
│  Train (70%) / Validation (15%) / Test (15%)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: Train a Baseline Model                                     │
│  Start simple (Logistic/Linear Regression). Measure performance.    │
│  This is your benchmark to beat.                                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 8: Train Multiple Models                                      │
│  Try Random Forest, XGBoost, SVM. Compare on validation set.       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 9: Tune the Best Model                                        │
│  Hyperparameter search. Cross-validation.                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 10: Final Evaluation on Test Set                              │
│  Do this ONCE. This is the honest measure of real-world performance.│
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 11: Interpret the Model                                       │
│  Feature importance. Error analysis. Explain predictions.           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 12: Deploy and Monitor                                        │
│  Save model. Serve predictions. Monitor for data drift.             │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete runnable pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ── 1. Load data ──────────────────────────────────────────────────
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ── 2. Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Build pipelines (preprocessing + model in one object) ───────
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

models = {
    "Logistic Regression": Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ('pre', preprocessor),
        ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
}

# ── 4. Train and compare with cross-validation ─────────────────────
results = {}
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    results[name] = {'AUC Mean': scores.mean(), 'AUC Std': scores.std()}
    print(f"{name:25s}  AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 5. Train best model on full training set ───────────────────────
best_pipeline = models["Gradient Boosting"]
best_pipeline.fit(X_train, y_train)

# ── 6. Final evaluation on test set (done ONCE) ────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n── Final Test Set Results ──")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── 7. Save ────────────────────────────────────────────────────────
joblib.dump(best_pipeline, "model_pipeline.pkl")
print("Model saved to model_pipeline.pkl")
```

---

## 🔍 Model Interpretability

Knowing that your model works is not enough. Understanding **why** it makes certain predictions builds trust, helps debugging, and is required in regulated industries.

### Feature importance (tree models)

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

importances.head(15).plot(kind='barh', figsize=(8, 6))
plt.title("Feature importance — top 15")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()
```

### SHAP values — explain individual predictions

SHAP (SHapley Additive exPlanations) shows how much each feature contributed to one specific prediction.

```python
# pip install shap
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot — global feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Waterfall plot — explain one prediction
shap.plots.waterfall(shap_values[0])
```

### Partial Dependence Plots (PDP)

Shows how changing one feature affects the model's predictions, holding everything else constant.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1, 2],
    feature_names=feature_names, ax=ax
)
plt.tight_layout()
plt.show()
```

---

## 💾 Saving and Deploying Models

### Save and load with joblib

```python
import joblib

# Save the model (and scaler if separate)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load later
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Predict on new data
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
```

### Save as a sklearn Pipeline (recommended)

When preprocessing is inside the Pipeline, saving and loading is one step.

```python
import joblib

# Save entire pipeline (preprocessing + model)
joblib.dump(pipeline, "full_pipeline.pkl")

# Load and use — no need to scale separately
pipeline = joblib.load("full_pipeline.pkl")
prediction = pipeline.predict(raw_new_data)  # pipeline handles scaling internally
```

### Serve predictions via a simple API

```python
# pip install flask
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
pipeline = joblib.load("full_pipeline.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']           # list of feature values
    features = np.array(data).reshape(1, -1)  # reshape to 2D array
    prediction  = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0].max()
    return jsonify({
        'prediction':  int(prediction),
        'confidence':  round(float(probability), 4)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Test: curl -X POST http://localhost:5000/predict \
#       -H "Content-Type: application/json" \
#       -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## ⚠️ Common Mistakes to Avoid

### Mistake 1 — Data leakage

Using information from the future or from the test set during training. This causes falsely optimistic results that collapse in production.

```python
# WRONG — fitting scaler on all data before splitting
scaler.fit(X)             # ← leaks test statistics into training
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# CORRECT — fit only on training data, transform test separately
X_train, X_test = train_test_split(X)
scaler.fit(X_train)               # ← only training data
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)  # ← same scaler applied
```

### Mistake 2 — Evaluating on training data

```python
# WRONG — this just measures memorisation, not learning
score = model.score(X_train, y_train)   # always high — meaningless

# CORRECT — always evaluate on held-out data
score = model.score(X_test, y_test)
```

### Mistake 3 — Wrong metric for imbalanced data

```python
# WRONG — 99% accuracy on 99%/1% imbalanced data is meaningless
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# CORRECT — use F1, ROC-AUC, or Precision-Recall for imbalanced data
print(f"F1:      {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Mistake 4 — Not using cross-validation

```python
# WRONG — one random split gives unreliable results
X_train, X_val = train_test_split(X, test_size=0.2)
score = model.score(X_val, y_val)  # could be lucky or unlucky split

# CORRECT — average over 5 different splits
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Mistake 5 — Touching the test set too early

```
The test set is sacred. It should be opened ONCE — at the very end.
If you look at it and adjust your model, you are overfitting to it.
Once used, it is no longer an honest measure of generalisation.
```

### Mistake 6 — Skipping EDA

Training a model on data you do not understand leads to:
- Invisible data quality issues
- Wrong choice of algorithm
- Missing obvious feature engineering opportunities
- Surprises in production

**Always explore your data first.**

---

## 🔀 Supervised vs Other Learning Types

| | Supervised | Unsupervised | Reinforcement |
|--|------------|-------------|---------------|
| **Labels needed?** | Yes — every example | No labels at all | Reward signal only |
| **Goal** | Learn input → output mapping | Find hidden structure | Learn optimal actions |
| **Feedback** | Correct answer for each sample | No external feedback | Reward/penalty per action |
| **Examples** | Classification, Regression | Clustering, PCA, Autoencoders | Game playing, robotics |
| **Algorithms** | SVM, Random Forest, XGBoost | K-Means, DBSCAN, PCA | Q-Learning, PPO, A3C |
| **Difficulty of labels** | Expensive to collect | Not needed | Complex to define |
| **Typical use** | Prediction from historical data | Exploration, compression | Sequential decision making |

---

## 🌍 Real-World Use Cases

| Industry | Task | Type | Input features | Output |
|----------|------|------|---------------|--------|
| Email | Spam detection | Classification | Word frequencies, sender info | Spam / Not Spam |
| Healthcare | Disease diagnosis | Classification | Test results, symptoms, age | Disease category |
| Finance | Credit scoring | Classification | Income, debt, history | Approved / Rejected |
| Finance | Fraud detection | Classification | Transaction patterns | Fraud / Legitimate |
| Real estate | Price prediction | Regression | Size, location, age, rooms | Price in £ |
| HR | Salary estimation | Regression | Experience, education, role | Annual salary |
| Retail | Churn prediction | Classification | Purchase history, activity | Will churn / Won't |
| Retail | Demand forecasting | Regression | Season, promotions, history | Units sold |
| Transport | Delivery time | Regression | Distance, traffic, weight | Minutes |
| NLP | Sentiment analysis | Classification | Review text | Positive / Negative |
| CV / Vision | Object detection | Classification | Pixel values | Object category |
| Agriculture | Crop yield | Regression | Rainfall, soil, temperature | kg per hectare |

---

## 🧭 Choosing the Right Algorithm

```
What is my output?
      │
      ├── A category → Classification
      │       │
      │       ├── Do I need probability scores?
      │       │     └── Yes → Logistic Regression, Random Forest, XGBoost
      │       │
      │       ├── Do I need to explain the decision?
      │       │     └── Yes → Logistic Regression, Decision Tree
      │       │
      │       ├── Is it text data?
      │       │     └── Yes → Naive Bayes, Logistic Regression, BERT
      │       │
      │       ├── Small dataset (<1,000 rows)?
      │       │     └── Yes → SVM, Logistic Regression, KNN
      │       │
      │       └── General best first attempt → Random Forest, then XGBoost
      │
      └── A number → Regression
              │
              ├── Need to explain predictions?
              │     └── Yes → Linear Regression, Ridge, Lasso
              │
              ├── Many irrelevant features?
              │     └── Yes → Lasso (auto feature removal)
              │
              ├── Non-linear relationship?
              │     └── Yes → Random Forest, XGBoost, Neural Network
              │
              ├── Small dataset (<1,000 rows)?
              │     └── Yes → Ridge, Lasso, SVR
              │
              └── General best first attempt → Random Forest, then XGBoost
```

**Golden rule:** Always start with the simplest model that could work. Measure. Then step up in complexity only if needed.

---

## 💻 Quick Code Examples

### End-to-end classification — 20 lines

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### End-to-end regression — 20 lines

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

### Handle imbalanced classes

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
```

### Build a reusable sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numerical_features = ['age', 'income', 'loan_amount']
categorical_features = ['employment_type', 'city']

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=200, random_state=42))
])

full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

---

## 📖 Glossary

| Term | Plain English definition |
|------|------------------------|
| **Accuracy** | Percentage of correct predictions out of all predictions |
| **Algorithm** | A set of mathematical rules the model uses to learn |
| **Batch** | A small group of training samples processed together before updating weights |
| **Bias (model)** | The baseline output when all features are zero (intercept) |
| **Bias-variance trade-off** | The balance between underfitting (high bias) and overfitting (high variance) |
| **Classification** | Predicting which category an input belongs to |
| **Confusion matrix** | A table showing correct vs incorrect predictions broken down by class |
| **Convergence** | When the model's loss stops improving significantly — training complete |
| **Cross-validation** | Evaluating a model by rotating the validation window across multiple data splits |
| **Data leakage** | Accidentally using test or future information during training |
| **Decision boundary** | The line or surface that separates different classes |
| **Dropout** | Randomly switching off neurons during training to prevent overfitting |
| **Early stopping** | Stopping training when validation performance stops improving |
| **Epoch** | One full pass through the entire training dataset |
| **Feature** | A single input variable used by the model to make predictions |
| **Feature engineering** | Creating new input features from existing data to improve the model |
| **Feature importance** | A score showing how much each feature contributed to the model's predictions |
| **Feature scaling** | Transforming features to a common scale so large values don't dominate |
| **F1 score** | The harmonic mean of precision and recall — balances both |
| **Gradient descent** | The algorithm that adjusts model weights step by step to minimise loss |
| **Hyperparameter** | A setting chosen before training (e.g. number of trees, learning rate) |
| **Imbalanced dataset** | When one class has far more training examples than others |
| **Label** | The correct answer for a training example |
| **Lasso** | Linear regression with L1 regularisation — removes useless features |
| **Learning rate** | How big a step gradient descent takes when adjusting weights |
| **Loss function** | A formula measuring how wrong the model's predictions are |
| **MAE** | Mean Absolute Error — average absolute difference between predictions and actuals |
| **Model** | The mathematical function that maps inputs to predicted outputs |
| **Multicollinearity** | When two or more input features are highly correlated |
| **Overfitting** | Model memorises training data but fails to generalise to new data |
| **Parameter** | A value inside the model that is learned during training (weight, bias) |
| **Pipeline** | A chain of preprocessing and modelling steps packaged as one object |
| **Precision** | Of all Positive predictions, what fraction were actually Positive |
| **R² score** | Fraction of target variation explained by the model (0 to 1) |
| **Recall** | Of all actual Positives, what fraction did the model correctly find |
| **Regression** | Predicting a continuous numerical value |
| **Regularisation** | Adding a penalty to prevent overfitting by discouraging complexity |
| **Ridge** | Linear regression with L2 regularisation — shrinks all weights |
| **RMSE** | Root Mean Squared Error — typical prediction error in the same units as the target |
| **ROC-AUC** | A threshold-independent measure of how well a classifier separates classes |
| **Sample** | One individual example in the dataset (one row) |
| **SHAP** | A method to explain individual predictions by showing each feature's contribution |
| **Stratified split** | A split that maintains the same class proportions in every subset |
| **Supervised learning** | Machine learning from labelled examples with known correct answers |
| **Test set** | Data held back for the final, honest evaluation — used once at the very end |
| **Training set** | Data the model learns from during the training phase |
| **Underfitting** | Model is too simple and misses important patterns in the data |
| **Validation set** | Data used to tune the model and compare algorithms during development |
| **Weight** | A number inside the model that is multiplied with a feature during prediction |

---

## 📚 Further Reading

### Books

- **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** — Aurélien Géron *(best overall book)*
- **Pattern Recognition and Machine Learning** — Christopher Bishop *(deeper theory)*
- **The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman *(free PDF online)*

### Online courses

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) — free, practical
- [fast.ai Practical Deep Learning](https://course.fast.ai) — free, top-down approach
- [Andrew Ng — Machine Learning Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction)

### Documentation

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Hyperparameter Optimisation](https://optuna.readthedocs.io/)

---

<p align="center">
  Written in plain English · No maths degree required · Start simple, measure, then improve 🚀
</p>