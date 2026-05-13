# 🧠 Naive Bayes — Complete Guide in Easy Words

## What Is Naive Bayes?

Naive Bayes is a **supervised machine learning algorithm** used for **classification tasks**. It is based on **Bayes' Theorem** from probability theory.

> **Simple analogy:** Imagine you're a doctor. A patient comes in with a fever and cough. Based on past experience (data), you calculate the probability that they have flu vs. cold vs. COVID. You pick the disease with the **highest probability**. That's exactly what Naive Bayes does!

---

## Why "Naive"?

It's called **"Naive"** because it makes a big assumption:

> 🔑 **All features are independent of each other** — knowing one feature tells you nothing about another.

In reality, this is rarely true (e.g., fever and cough are related in flu). But despite this "naive" assumption, the algorithm works **surprisingly well** in practice!

---

## Bayes' Theorem (The Core Formula)

$$P(Class \mid Features) = \frac{P(Features \mid Class) \times P(Class)}{P(Features)}$$

In plain English:

```
Probability of Class given what we see = 
    (How likely we'd see these features in that class) 
  × (How common that class is)
  ÷ (How common these features are overall)
```

### Breaking It Down

| Term | Name | Meaning |
|------|------|---------|
| `P(Class \| Features)` | **Posterior** | What we want — probability of class given data |
| `P(Features \| Class)` | **Likelihood** | How likely are these features in this class? |
| `P(Class)` | **Prior** | How common is this class in training data? |
| `P(Features)` | **Evidence** | Overall probability of seeing these features |

> Since `P(Features)` is the same for all classes, we just compare:
> `P(Class | Features) ∝ P(Features | Class) × P(Class)`

---

## Real-World Example (Step by Step)

### Problem: Will it rain today? ☁️

**Training Data:**

| Outlook | Humidity | Wind  | Rain? |
|---------|----------|-------|-------|
| Sunny   | High     | Weak  | No    |
| Sunny   | High     | Strong| No    |
| Cloudy  | High     | Weak  | Yes   |
| Rainy   | High     | Weak  | Yes   |
| Rainy   | Normal   | Weak  | Yes   |
| Rainy   | Normal   | Strong| No    |
| Cloudy  | Normal   | Strong| Yes   |
| Sunny   | High     | Weak  | No    |
| Sunny   | Normal   | Weak  | Yes   |
| Rainy   | Normal   | Weak  | Yes   |
| Sunny   | Normal   | Strong| Yes   |
| Cloudy  | High     | Strong| Yes   |
| Cloudy  | Normal   | Weak  | Yes   |
| Rainy   | High     | Strong| No    |

**New Day:** Outlook=Sunny, Humidity=High, Wind=Weak → Will it Rain?

**Step 1: Calculate Priors**
```
P(Rain=Yes) = 9/14 = 0.64
P(Rain=No)  = 5/14 = 0.36
```

**Step 2: Calculate Likelihoods**
```
P(Sunny | Yes) = 2/9 = 0.22    P(Sunny | No)  = 3/5 = 0.60
P(High  | Yes) = 3/9 = 0.33    P(High  | No)  = 4/5 = 0.80
P(Weak  | Yes) = 6/9 = 0.67    P(Weak  | No)  = 2/5 = 0.40
```

**Step 3: Multiply**
```
Score(Yes) = 0.64 × 0.22 × 0.33 × 0.67 = 0.031
Score(No)  = 0.36 × 0.60 × 0.80 × 0.40 = 0.069
```

**Decision:** `No > Yes` → **Prediction: No Rain ☀️**

---

## Types of Naive Bayes

| Type | Best For | Feature Distribution |
|------|----------|----------------------|
| **Gaussian NB** | Continuous data | Normal/bell-curve distribution |
| **Multinomial NB** | Text classification, word counts | Discrete counts |
| **Bernoulli NB** | Binary features (yes/no) | Binary (0 or 1) |
| **Complement NB** | Imbalanced text datasets | Complement of class stats |

---

## Code Example

### Example 1: Gaussian Naive Bayes (Continuous Data)

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
# Features: sepal length, sepal width, petal length, petal width
# Classes:  0=Setosa, 1=Versicolor, 2=Virginica

# ── 2. Split Data ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Train Model ───────────────────────────────────────────────────────────
model = GaussianNB()
model.fit(X_train, y_train)

# ── 4. Predict ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── 6. Predict a new flower ──────────────────────────────────────────────────
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # likely Setosa
prediction = model.predict(new_flower)
proba = model.predict_proba(new_flower)

print(f"\nNew flower prediction : {iris.target_names[prediction[0]]}")
print(f"Class probabilities   : {dict(zip(iris.target_names, proba[0].round(4)))}")
```

**Output:**
```
Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.89      0.94         9
   virginica       0.92      1.00      0.96        11

New flower prediction : setosa
Class probabilities   : {'setosa': 0.9998, 'versicolor': 0.0002, 'virginica': 0.0}
```

---

### Example 2: Multinomial Naive Bayes (Text Classification / Spam Detection)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# ── 1. Sample Email Data ─────────────────────────────────────────────────────
emails = [
    "Win free money now click here",
    "Congratulations you won a prize",
    "Buy cheap pills online discount",
    "Free offer claim your reward today",
    "Limited time deal win cash now",
    "Meeting tomorrow at 10am in office",
    "Please review the attached report",
    "Can we reschedule the project call",
    "Your invoice is ready for download",
    "Team lunch is planned for Friday",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = spam, 0 = not spam

# ── 2. Build Pipeline (Vectorizer + Naive Bayes) ─────────────────────────────
pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),   # text → word count matrix
    ("classifier", MultinomialNB()),     # train Naive Bayes
])

pipeline.fit(emails, labels)

# ── 3. Predict New Emails ────────────────────────────────────────────────────
new_emails = [
    "Free money win prize now",        # sounds spammy
    "Can you join the standup call",   # sounds normal
]

predictions = pipeline.predict(new_emails)
probabilities = pipeline.predict_proba(new_emails)

for email, pred, prob in zip(new_emails, predictions, probabilities):
    label = "🚨 SPAM" if pred == 1 else "✅ NOT SPAM"
    print(f"\nEmail    : '{email}'")
    print(f"Result   : {label}")
    print(f"Confidence → Not Spam: {prob[0]:.2%} | Spam: {prob[1]:.2%}")
```

**Output:**
```
Email    : 'Free money win prize now'
Result   : 🚨 SPAM
Confidence → Not Spam: 2.31% | Spam: 97.69%

Email    : 'Can you join the standup call'
Result   : ✅ NOT SPAM
Confidence → Not Spam: 95.14% | Spam: 4.86%
```

---

### Example 3: Naive Bayes from Scratch (Pure Python)

```python
import numpy as np

class NaiveBayesFromScratch:
    """Gaussian Naive Bayes implemented from scratch."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.stds = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)   # P(Class)
            self.means[c]  = X_c.mean(axis=0)     # mean per feature
            self.stds[c]   = X_c.std(axis=0)      # std per feature

    def _gaussian_likelihood(self, x, mean, std):
        """P(feature | class) using Gaussian distribution."""
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2 + 1e-9))
        return exponent / (np.sqrt(2 * np.pi) * std + 1e-9)

    def _predict_one(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])                              # log P(Class)
            likelihood = np.sum(
                np.log(self._gaussian_likelihood(x, self.means[c], self.stds[c]))
            )                                                           # log P(X|Class)
            posteriors[c] = prior + likelihood
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


# ── Test it ───────────────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = NaiveBayesFromScratch()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print(f"Accuracy (from scratch): {accuracy_score(y_test, y_pred):.4f}")
# Output: Accuracy (from scratch): 0.9667
```

---

## Evaluation Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | Correct / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are real? |
| **Recall** | TP / (TP + FN) | Of actual positives, how many did we catch? |
| **F1 Score** | 2 × P×R / (P+R) | Balance of Precision and Recall |

```
Confusion Matrix:
                 Predicted Positive   Predicted Negative
Actual Positive       TP                   FN
Actual Negative       FP                   TN
```

---

## Pros and Cons

### ✅ Advantages

- **Fast** — Trains and predicts extremely quickly
- **Works with small data** — Doesn't need huge datasets
- **Multi-class** — Naturally handles multiple classes
- **Text friendly** — Excellent for spam/sentiment/NLP tasks
- **Interpretable** — Easy to understand probabilities

### ❌ Disadvantages

- **Independence assumption** — Features are rarely truly independent
- **Zero probability problem** — If a feature never appears in training, probability = 0 (fixed with Laplace smoothing)
- **Continuous features** — Assumes Gaussian distribution which may not hold
- **Not great for complex patterns** — Simpler than neural networks or tree models

---

## Laplace Smoothing (Fixing Zero Probability)

If a word never appeared in training spam emails, `P(word | spam) = 0`, which kills the entire product.

**Fix:** Add a small count (usually 1) to every feature:

$$P(w_i \mid Class) = \frac{\text{count}(w_i, Class) + 1}{\text{count}(Class) + |Vocabulary|}$$

```python
# In sklearn, set var_smoothing for Gaussian NB
model = GaussianNB(var_smoothing=1e-9)

# For MultinomialNB, use alpha (Laplace smoothing)
model = MultinomialNB(alpha=1.0)   # alpha=1 is standard Laplace smoothing
```

---

## When to Use Naive Bayes

✅ **Perfect for:**
- Email spam detection
- Sentiment analysis (positive / negative review)
- Document / news classification
- Medical diagnosis (disease prediction)
- Real-time predictions (very fast)
- Small training datasets

❌ **Avoid when:**
- Features are highly correlated (e.g., pixels in images)
- You need very high accuracy on complex tasks
- You have rich structured/tabular data (try XGBoost instead)

---

## Quick Comparison: Which Naive Bayes to Use?

```
Your Data Type?
      │
      ├── Continuous numbers (height, weight, temperature)
      │         └──→ GaussianNB
      │
      ├── Word counts / frequency (emails, documents)
      │         └──→ MultinomialNB
      │
      └── Binary features (word present/absent, yes/no)
                └──→ BernoulliNB
```

---

## Summary

```
Training:
  For each class → calculate P(Class) and P(each feature | Class)

Prediction:
  For new input → multiply all probabilities together → pick highest class

Key Formula:
  P(Class | Features) ∝ P(Class) × P(f1|Class) × P(f2|Class) × ... × P(fn|Class)
```

> Naive Bayes is **simple, fast, and effective** — especially for text tasks.
> It's the algorithm that powered the first generation of spam filters and still holds up today!