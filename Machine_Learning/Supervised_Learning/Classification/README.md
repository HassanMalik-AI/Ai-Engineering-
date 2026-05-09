# 📚 Classification in Supervised Learning

> A complete beginner-friendly guide to understanding, building, and evaluating classification models in machine learning.

![Topic](https://img.shields.io/badge/Topic-Supervised%20Learning-blue)
![Type](https://img.shields.io/badge/Type-Classification-orange)
![Level](https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green)
![Language](https://img.shields.io/badge/Language-Python-yellow?logo=python)

---

## 📋 Table of Contents

- [What is Supervised Learning?](#-what-is-supervised-learning)
- [What is Classification?](#-what-is-classification)
- [Types of Classification](#-types-of-classification)
- [How Classification Works](#-how-classification-works)
- [Classification Algorithms](#-classification-algorithms)
- [Key Concepts](#-key-concepts)
- [The ML Pipeline](#-the-ml-pipeline-step-by-step)
- [Evaluation Metrics](#-evaluation-metrics)
- [Handling Common Problems](#-handling-common-problems)
- [Choosing the Right Algorithm](#-choosing-the-right-algorithm)
- [Quick Code Examples](#-quick-code-examples)
- [Real-World Use Cases](#-real-world-use-cases)
- [Glossary](#-glossary)

---

## 🤔 What is Supervised Learning?

**Supervised learning** is a type of machine learning where you teach a computer using examples that already have correct answers.

Think of it like teaching a child using flashcards:
- You show a picture of a cat and say "this is a cat"
- You show a picture of a dog and say "this is a dog"
- After seeing enough examples, the child can identify cats and dogs on their own

In supervised learning:
- The **input** is the data (e.g. an image, an email, a patient's test results)
- The **label** is the correct answer (e.g. "cat", "spam", "has diabetes")
- The **model** learns the relationship between inputs and labels
- After training, it can predict labels for new, unseen inputs

### Two main tasks in supervised learning

| Task | Question it answers | Example |
|------|-------------------|---------|
| **Classification** | Which category does this belong to? | Is this email spam or not spam? |
| **Regression** | What number/value is this? | What will the house price be? |

> This guide focuses entirely on **Classification**.

---

## 🏷️ What is Classification?

**Classification** is the task of predicting which **category** or **class** a piece of data belongs to.

The model learns from labelled training data and then assigns a class label to new data it has never seen before.

### Simple real-world analogy

Imagine a doctor looking at an X-ray:
- They have seen thousands of X-rays before (training data)
- Each X-ray was labelled "healthy" or "has fracture" (labels)
- Now they can look at a new X-ray and predict which category it falls into

A classification model does exactly the same thing — but with numbers and mathematics.

### What the model actually learns

The model learns a **decision boundary** — an imaginary line (or curve) that separates different classes in the data.

```
        ● ● ●          ← Class A (e.g. spam emails)
    ────────────────    ← Decision boundary
        ○ ○ ○          ← Class B (e.g. normal emails)
```

Anything above the line → predicted as Class A
Anything below the line → predicted as Class B

---

## 🗂️ Types of Classification

### 1. Binary Classification
Only **two possible classes** — yes or no, true or false, 0 or 1.

```
Input → Model → One of two outcomes
                ├── Class 0 (e.g. "Not Spam")
                └── Class 1 (e.g. "Spam")
```

**Examples:**
- Email: Spam or Not Spam
- Medical: Disease Present or Absent
- Credit: Loan Approved or Rejected
- Sentiment: Positive or Negative review

---

### 2. Multi-Class Classification
**More than two classes**, but each input belongs to exactly **one** class.

```
Input → Model → One of many outcomes
                ├── Class A (e.g. "Cat")
                ├── Class B (e.g. "Dog")
                ├── Class C (e.g. "Bird")
                └── Class D (e.g. "Fish")
```

**Examples:**
- Handwritten digit recognition: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9
- Weather prediction: Sunny, Rainy, Cloudy, Snowy
- News article topic: Sports, Politics, Technology, Entertainment

---

### 3. Multi-Label Classification
Each input can belong to **more than one class** at the same time.

```
Input → Model → Multiple labels simultaneously
                ├── Label A ✓ (e.g. "Action")
                ├── Label B ✓ (e.g. "Comedy")
                ├── Label C ✗ (e.g. "Horror")
                └── Label D ✓ (e.g. "Sci-Fi")
```

**Examples:**
- A movie can be both "Action" AND "Comedy"
- A photo can contain "Dog", "Beach", AND "Sunset"
- A research paper can belong to "NLP" AND "Computer Vision"

---

### 4. Imbalanced Classification
A special case where one class has **far more examples** than others.

```
Training data:
  Class A (Fraud):     ████ 500 samples      (1%)
  Class B (Not Fraud): ████████████ 49,500   (99%)
```

This is tricky because the model tends to always predict the majority class and still get high accuracy. Special techniques are needed (covered in [Handling Common Problems](#-handling-common-problems)).

---

## ⚙️ How Classification Works

### Step-by-step process

```
Raw Data
    ↓
Preprocessing (clean, encode, scale)
    ↓
Feature Extraction (select useful inputs)
    ↓
Model Training (learn from labelled examples)
    ↓
Model Evaluation (check accuracy on unseen data)
    ↓
Prediction (classify new inputs)
```

### What happens during training

1. The model receives an input (e.g. an email's words)
2. It makes a prediction (e.g. "70% chance of spam")
3. It compares its prediction to the real label (e.g. "this was actually spam")
4. It calculates the **error** (how wrong it was)
5. It adjusts its internal parameters to reduce the error
6. This repeats thousands of times until the model gets good

This process of adjusting based on errors is called **learning**.

---

## 🧮 Classification Algorithms

### 1. Logistic Regression

**What it is:** Despite the name, this is a classification algorithm. It calculates the probability of an input belonging to a class.

**How it thinks:** "Given these features, what is the probability this belongs to Class 1?"

**Output:** A probability between 0 and 1. If > 0.5 → Class 1, else → Class 0.

**Best for:**
- Binary classification
- When you need probability scores
- When data is roughly linearly separable

**Pros:** Fast, easy to interpret, works well on small datasets
**Cons:** Can't capture complex non-linear patterns

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

---

### 2. K-Nearest Neighbors (KNN)

**What it is:** Classifies a new point by looking at its K closest neighbours in the training data and taking a majority vote.

**How it thinks:** "What class do most of my nearest neighbours belong to? I'll be that class too."

```
New point: ★

Neighbours (K=3):  ● ● ○
                   2 Class A, 1 Class B
                   → Predict Class A
```

**Best for:**
- Small to medium datasets
- When decision boundaries are irregular

**Pros:** Simple, no training phase, naturally handles multi-class
**Cons:** Slow for large datasets, sensitive to irrelevant features

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 3. Decision Tree

**What it is:** A tree-like structure that asks a series of yes/no questions to arrive at a prediction.

**How it thinks:** Like a game of 20 questions — splitting data step by step.

```
Is Age > 30?
    ├── Yes → Is Income > 50k?
    │           ├── Yes → APPROVED ✓
    │           └── No  → REJECTED ✗
    └── No  → Is Credit Score > 700?
                ├── Yes → APPROVED ✓
                └── No  → REJECTED ✗
```

**Best for:**
- When you need to explain decisions to non-technical stakeholders
- Mixed data types (numbers + categories)

**Pros:** Very easy to understand and visualise, no scaling needed
**Cons:** Tends to overfit (memorise training data)

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 4. Random Forest

**What it is:** A collection (ensemble) of many decision trees. Each tree votes, and the majority wins.

**How it thinks:** "Let's ask 100 different experts and go with the majority answer."

```
Input
  ├── Tree 1 → Class A
  ├── Tree 2 → Class A
  ├── Tree 3 → Class B
  ├── Tree 4 → Class A
  └── Tree 5 → Class A
              ↓
         Majority Vote
              ↓
         Class A (4/5 votes)
```

**Best for:**
- General-purpose classification
- When you want high accuracy with less tuning

**Pros:** Very accurate, resistant to overfitting, handles missing data well
**Cons:** Harder to interpret, slower than a single decision tree

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 5. Support Vector Machine (SVM)

**What it is:** Finds the best possible line (or hyperplane) to separate two classes, maximising the gap between them.

**How it thinks:** "Draw the widest possible street between the two groups of points."

```
● ●  ●                 Class A
   ════════════         ← Maximum margin boundary
       ○  ○  ○         Class B
```

**Best for:**
- High-dimensional data (many features)
- Image and text classification
- When classes are clearly separable

**Pros:** Works well in high dimensions, robust to outliers
**Cons:** Slow on large datasets, needs careful tuning

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 6. Naive Bayes

**What it is:** Uses probability theory (Bayes' Theorem) to classify data. It assumes all features are independent of each other.

**How it thinks:** "Given these words in an email, what is the probability it is spam?"

**Best for:**
- Text classification (spam detection, sentiment analysis)
- When you have very little training data
- Real-time prediction (it's very fast)

**Pros:** Extremely fast, works well with small data, great for text
**Cons:** The "naive" independence assumption is often wrong in practice

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 7. Gradient Boosting (XGBoost / LightGBM)

**What it is:** Builds trees one at a time, where each new tree corrects the mistakes of the previous ones.

**How it thinks:** "The last model got these examples wrong — let me focus on fixing those."

**Best for:**
- Tabular/structured data (spreadsheets, databases)
- Kaggle competitions and production systems
- When maximum accuracy is the goal

**Pros:** State-of-the-art accuracy on tabular data, handles missing values
**Cons:** Many hyperparameters to tune, can overfit if not careful

```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 8. Neural Networks

**What it is:** Layers of interconnected nodes inspired by the human brain. Each layer learns more abstract features.

**How it thinks:** Layer 1 detects edges → Layer 2 detects shapes → Layer 3 detects objects.

**Best for:**
- Images, audio, and text (with specialised architectures)
- Very large datasets
- When other algorithms fall short

**Pros:** Can learn extremely complex patterns, state-of-the-art on many tasks
**Cons:** Needs lots of data and compute, acts like a "black box"

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

---

## 🔑 Key Concepts

### Features vs. Labels

| Term | What it means | Example |
|------|-------------|---------|
| **Feature** | An input variable used to make predictions | Age, Income, Credit Score |
| **Label** | The correct answer / output | "Approved" or "Rejected" |
| **Sample** | One row of data (one example) | A single customer's record |
| **Dataset** | All samples together | 10,000 customer records |

---

### Training Set, Validation Set, Test Set

It is essential to split your data into three parts:

```
Full Dataset (100%)
    ├── Training Set (70%) → Model learns from this
    ├── Validation Set (15%) → Tune the model, pick hyperparameters
    └── Test Set (15%) → Final, honest evaluation (never seen during training)
```

**Why three splits?**
- If you evaluate on the same data you trained on, the model looks better than it really is
- Validation set helps you improve the model without "cheating" with test data
- Test set gives you a final, unbiased measure of real-world performance

```python
from sklearn.model_selection import train_test_split

# First split off the test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Then split training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42
    # 0.176 of 85% ≈ 15% of total
)
```

---

### Overfitting vs. Underfitting

**Underfitting** — the model is too simple. It hasn't learned enough from the data.

```
Training accuracy:   60%   ← bad
Validation accuracy: 59%   ← similar but still bad
→ Model needs to be more complex
```

**Overfitting** — the model has memorised the training data but fails on new data.

```
Training accuracy:   99%   ← looks great
Validation accuracy: 65%   ← much worse
→ Model is memorising, not generalising
```

**Just right** — the model generalises well.

```
Training accuracy:   92%
Validation accuracy: 89%   ← close, slightly lower is normal
→ Good balance
```

**How to fix overfitting:**
- Add more training data
- Simplify the model (reduce depth, fewer features)
- Use regularisation (L1, L2)
- Use dropout (for neural networks)
- Use cross-validation

---

### Cross-Validation

Instead of a single train/validation split, use **K-Fold Cross-Validation** to get a more reliable estimate.

```
Data split into 5 folds:

Fold 1: [TEST] [train] [train] [train] [train]
Fold 2: [train] [TEST] [train] [train] [train]
Fold 3: [train] [train] [TEST] [train] [train]
Fold 4: [train] [train] [train] [TEST] [train]
Fold 5: [train] [train] [train] [train] [TEST]

Final score = average of 5 results
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

---

### Feature Scaling

Many algorithms are sensitive to the scale of your features. Always scale your data before training (except for tree-based models).

```
Before scaling:        After scaling:
Age:   [18, 65, 42]   Age:   [0.0, 1.0, 0.51]
Income:[20k, 150k, 60k] Income:[0.0, 1.0, 0.31]
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on training data
X_test_scaled  = scaler.transform(X_test)        # apply same transform to test
```

> **Important:** Never fit the scaler on the test set. This would leak information from the test data into your model.

---

## 🔄 The ML Pipeline Step by Step

```
Step 1: Define the Problem
        └── What classes do I want to predict?

Step 2: Collect & Understand Data
        └── How many samples? Any class imbalance? Missing values?

Step 3: Preprocess the Data
        ├── Handle missing values
        ├── Encode categorical variables
        └── Scale numerical features

Step 4: Split the Data
        └── Train / Validation / Test sets

Step 5: Choose & Train a Model
        └── Start simple (Logistic Regression), then try more complex

Step 6: Evaluate the Model
        └── Accuracy, Precision, Recall, F1, ROC-AUC

Step 7: Improve the Model
        ├── Tune hyperparameters
        ├── Try different algorithms
        └── Add/remove features (feature engineering)

Step 8: Test on Test Set
        └── Final evaluation — do this ONCE at the very end

Step 9: Deploy
        └── Save model, serve predictions
```

### Full pipeline code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- Step 1: Load data ---
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# --- Step 2: Encode labels ---
le = LabelEncoder()
y = le.fit_transform(y)

# --- Step 3: Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 4: Scale ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Step 5: Train ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Step 7: Save ---
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

---

## 📏 Evaluation Metrics

### Confusion Matrix

The foundation of all classification metrics. Shows exactly what the model predicted vs. what was actually true.

```
                  Predicted
                  Positive  Negative
Actual  Positive [  TP   |   FN  ]
        Negative [  FP   |   TN  ]

TP = True Positive  → Predicted Positive, Actually Positive ✓
TN = True Negative  → Predicted Negative, Actually Negative ✓
FP = False Positive → Predicted Positive, Actually Negative ✗ (False Alarm)
FN = False Negative → Predicted Negative, Actually Positive ✗ (Missed it)
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot()
plt.show()
```

---

### Accuracy

**What it means:** Out of all predictions, how many were correct?

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Example:** 90 correct out of 100 → Accuracy = 90%

**When to use:** When classes are balanced (roughly equal number of samples per class)

**When NOT to use:** Imbalanced datasets. A model that always says "Not Fraud" gets 99% accuracy on a dataset where only 1% of transactions are fraud — but it's completely useless.

---

### Precision

**What it means:** Of all the times the model said "Positive", how often was it actually right?

```
Precision = TP / (TP + FP)
```

**Real-world meaning:** If the model flags 100 emails as spam, precision tells you how many of those 100 were actually spam.

**High precision is important when:** False alarms are costly (e.g. blocking a legitimate email, flagging an innocent person).

---

### Recall (Sensitivity)

**What it means:** Of all actual Positives, how many did the model find?

```
Recall = TP / (TP + FN)
```

**Real-world meaning:** Out of all actual cancer cases, how many did the model detect?

**High recall is important when:** Missing a positive is dangerous (e.g. missing a cancer diagnosis, missing a fraud transaction).

---

### F1 Score

**What it means:** The balanced average of Precision and Recall. Useful when you care about both.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Range:** 0 (worst) to 1 (best)

**When to use:** Imbalanced datasets, or when you want a single number that balances precision and recall.

---

### Precision vs. Recall Trade-off

You usually cannot maximise both at the same time. Changing the decision threshold shifts the balance.

```
High Threshold (e.g. 0.9 to call it Spam):
  → High Precision (only very sure spam is flagged)
  → Low Recall (misses lots of spam)

Low Threshold (e.g. 0.3 to call it Spam):
  → Low Precision (lots of legitimate emails flagged)
  → High Recall (catches almost all spam)
```

---

### ROC-AUC Score

**What it means:** Measures how well the model separates classes across all possible thresholds.

**Range:** 0.5 (random guessing) to 1.0 (perfect)

```
AUC = 0.5  → No better than random chance
AUC = 0.7  → Acceptable
AUC = 0.8  → Good
AUC = 0.9  → Excellent
AUC = 1.0  → Perfect (likely overfitting)
```

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

### Metric Summary — When to use which

| Metric | Use when |
|--------|---------|
| **Accuracy** | Classes are balanced |
| **Precision** | False positives are costly (e.g. spam filter, legal flagging) |
| **Recall** | False negatives are costly (e.g. disease screening, fraud detection) |
| **F1 Score** | You want a balance of precision and recall |
| **ROC-AUC** | You want a threshold-independent measure of separability |

---

## 🛠️ Handling Common Problems

### Problem 1: Imbalanced Classes

**Signs:** One class has far more examples than others.

**Solutions:**

```python
# Option A — Oversample the minority class (SMOTE)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Option B — Undersample the majority class
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Option C — Class weights (tell the model to care more about minority class)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```

---

### Problem 2: Overfitting

**Signs:** Training accuracy >> Validation accuracy

**Solutions:**

```python
# Reduce model complexity
model = DecisionTreeClassifier(max_depth=4)  # limit tree depth

# Add regularisation (Logistic Regression)
model = LogisticRegression(C=0.1)  # smaller C = stronger regularisation

# Use dropout (Neural Networks)
keras.layers.Dropout(0.5)

# Collect more training data (best solution)
```

---

### Problem 3: Missing Values

```python
# Check for missing values
print(df.isnull().sum())

# Strategy A — Fill with mean (numerical)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Strategy B — Fill with most frequent value (categorical)
imputer = SimpleImputer(strategy='most_frequent')

# Strategy C — Drop rows with missing values
df.dropna(inplace=True)
```

---

### Problem 4: Categorical Features

Models only understand numbers. Convert text categories to numbers first.

```python
# Option A — One-Hot Encoding (for nominal categories: colour, city)
# Red → [1, 0, 0], Blue → [0, 1, 0], Green → [0, 0, 1]
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
X_encoded = enc.fit_transform(X[['colour']])

# Option B — Label Encoding (for ordinal categories: Low < Medium < High)
# Low → 0, Medium → 1, High → 2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['size'] = le.fit_transform(X['size'])

# Option C — Pandas get_dummies (quick one-hot encoding)
X = pd.get_dummies(X, columns=['colour', 'city'])
```

---

### Problem 5: Hyperparameter Tuning

Hyperparameters are settings you choose before training. Finding the best ones is called **hyperparameter tuning**.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)
```

---

## 🧭 Choosing the Right Algorithm

Use this flowchart to guide your algorithm selection:

```
Start
  │
  ├── How much data do you have?
  │     │
  │     ├── Small (<1,000 samples)
  │     │     └── Try: Logistic Regression, SVM, Naive Bayes, KNN
  │     │
  │     ├── Medium (1k – 100k samples)
  │     │     └── Try: Random Forest, XGBoost, SVM
  │     │
  │     └── Large (>100k samples)
  │           └── Try: XGBoost, LightGBM, Neural Networks
  │
  ├── Do you need to explain the predictions?
  │     ├── Yes → Logistic Regression, Decision Tree
  │     └── No  → Random Forest, XGBoost, Neural Networks
  │
  ├── What type of data is it?
  │     ├── Text → Naive Bayes, Logistic Regression, BERT
  │     ├── Images → Convolutional Neural Networks (CNN)
  │     ├── Tabular → XGBoost, Random Forest, Logistic Regression
  │     └── Time Series → RNN, LSTM, Gradient Boosting
  │
  └── General advice: Start simple, measure, then go complex
```

### Quick comparison table

| Algorithm | Speed | Accuracy | Interpretability | Handles Non-linearity |
|-----------|-------|----------|-----------------|----------------------|
| Logistic Regression | ⚡⚡⚡ | ★★★ | ★★★★★ | ★★ |
| KNN | ⚡ | ★★★ | ★★★★ | ★★★★ |
| Decision Tree | ⚡⚡ | ★★★ | ★★★★★ | ★★★ |
| Random Forest | ⚡⚡ | ★★★★ | ★★ | ★★★★ |
| SVM | ⚡ | ★★★★ | ★★ | ★★★★ |
| Naive Bayes | ⚡⚡⚡ | ★★★ | ★★★ | ★★ |
| XGBoost | ⚡⚡ | ★★★★★ | ★★ | ★★★★★ |
| Neural Network | ⚡ | ★★★★★ | ★ | ★★★★★ |

---

## 💻 Quick Code Examples

### Minimal working example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load a built-in dataset (Iris flowers — 3 classes)
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred,
      target_names=['Setosa', 'Versicolor', 'Virginica']))
```

**Expected output:**

```
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00        10
  Versicolor       1.00      1.00      1.00         9
   Virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
```

---

### Save and load a trained model

```python
import joblib

# Save model
joblib.dump(model, 'classifier.pkl')

# Load model later
loaded_model = joblib.load('classifier.pkl')
new_predictions = loaded_model.predict(X_new)
```

---

### Predict on new data

```python
import numpy as np

# One new sample (same feature order as training data)
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict class
predicted_class = model.predict(new_sample)
print("Predicted class:", predicted_class[0])

# Predict probabilities for each class
probabilities = model.predict_proba(new_sample)
print("Probabilities:", probabilities)
# Output: [[0.97, 0.02, 0.01]]  → 97% confident it is Class 0
```

---

## 🌍 Real-World Use Cases

| Domain | Task | Input Features | Classes |
|--------|------|---------------|---------|
| Email | Spam detection | Word frequencies, sender info | Spam / Not Spam |
| Healthcare | Disease diagnosis | Test results, symptoms, age | Disease A / B / Healthy |
| Finance | Credit scoring | Income, debt, history | Approve / Reject |
| E-commerce | Churn prediction | Purchase history, activity | Will Churn / Won't Churn |
| Computer Vision | Image recognition | Pixel values | Cat / Dog / Bird... |
| NLP | Sentiment analysis | Review text | Positive / Negative / Neutral |
| Cybersecurity | Intrusion detection | Network traffic patterns | Attack / Normal |
| Agriculture | Crop disease | Leaf image pixels | Healthy / Rust / Blight |

---

## 📖 Glossary

| Term | Simple Definition |
|------|-----------------|
| **Algorithm** | A set of rules the computer follows to learn from data |
| **Class** | A category the model predicts (e.g. "Spam", "Cat", "Fraud") |
| **Classification** | Predicting which category a data point belongs to |
| **Confusion Matrix** | A table showing correct vs. incorrect predictions by class |
| **Cross-validation** | Evaluating a model by training and testing on multiple data splits |
| **Decision Boundary** | The line/surface that separates different classes |
| **Epoch** | One full pass through the entire training dataset |
| **Feature** | An input variable used to make predictions (e.g. Age, Income) |
| **F1 Score** | The harmonic mean of Precision and Recall |
| **Hyperparameter** | A setting chosen before training (e.g. number of trees) |
| **Label** | The correct answer / output class for a training sample |
| **Overfitting** | When a model memorises training data but fails on new data |
| **Precision** | Of all Positive predictions, how many were actually Positive |
| **Recall** | Of all actual Positives, how many did the model find |
| **Regularisation** | A technique to prevent overfitting by penalising complexity |
| **ROC-AUC** | A score measuring class separability across all thresholds |
| **Supervised Learning** | Learning from labelled examples with correct answers |
| **Training Set** | Data used to teach the model |
| **Test Set** | Data used for the final, honest evaluation |
| **Underfitting** | When a model is too simple and misses patterns in the data |
| **Validation Set** | Data used to tune the model during development |

---

## 📚 Further Reading

- [Scikit-learn User Guide — Classification](https://scikit-learn.org/stable/supervised_learning.html)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [fast.ai Practical Deep Learning](https://course.fast.ai/)

---

<p align="center">
  Written in plain English · No maths degree required · Happy learning 🎓
</p>