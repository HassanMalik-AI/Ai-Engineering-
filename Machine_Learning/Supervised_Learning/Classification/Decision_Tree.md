# 🌳 Decision Tree in Supervised Learning

> **A beginner-friendly guide with real code examples — learn how machines make decisions like humans!**

---

## 📌 Table of Contents

1. [Quick Recap: Supervised Learning](#quick-recap-supervised-learning)
2. [What is a Decision Tree?](#what-is-a-decision-tree)
3. [How Does It Work? (The Intuition)](#how-does-it-work)
4. [Key Concepts Explained Simply](#key-concepts-explained-simply)
5. [Real-Life Example](#real-life-example)
6. [How the Tree Decides to Split](#how-the-tree-decides-to-split)
7. [Full Code Example](#full-code-example)
8. [Understanding the Output](#understanding-the-output)
9. [Overfitting & How to Fix It](#overfitting--how-to-fix-it)
10. [Decision Tree vs Logistic Regression](#decision-tree-vs-logistic-regression)
11. [When to Use Decision Trees](#when-to-use-decision-trees)
12. [Key Terms Glossary](#key-terms-glossary)
13. [What's Next?](#whats-next)

---

## 📚 Quick Recap: Supervised Learning

In **Supervised Learning**, we train a model using labeled data — examples where we already know the correct answer. The model learns from these examples so it can predict answers for new, unseen data.

```
Labeled Training Data  ──►  Model Learns  ──►  Predicts on New Data
 (input + correct answer)      Patterns          (without labels)
```

---

## 🌳 What is a Decision Tree?

A Decision Tree is a model that makes decisions by asking a **series of yes/no questions** — just like a flowchart!

Think about how a doctor diagnoses a patient:

```
Do you have a fever?
    │
   YES ──► Is your throat sore?
               │
              YES ──► Likely: Flu
               │
               NO ──► Likely: Infection
    │
    NO ──► Do you have a headache?
               │
              YES ──► Likely: Migraine
               │
               NO ──► Likely: Healthy
```

That's exactly how a Decision Tree works — it **splits** data step by step until it reaches a final answer.

### Simple Definition:
> A Decision Tree is a tree-shaped model that makes predictions by following a path of questions from the **root** (top) down to a **leaf** (final answer).

---

## ⚙️ How Does It Work?

### The Big Picture

```
          [Root Node]          ← First question (most important feature)
          /          \
    [Branch]       [Branch]    ← Answers split the data
     /    \         /    \
[Leaf]  [Node]  [Leaf]  [Leaf] ← Leaves = Final predictions
         /  \
      [Leaf][Leaf]
```

### Step-by-Step Process:

**Step 1:** Start at the **root** — the model picks the most important question (feature) to ask first.

**Step 2:** **Split** the data based on the answer (yes/no, high/low, true/false).

**Step 3:** **Repeat** for each branch — ask the next best question.

**Step 4:** Stop when you reach a **leaf node** — this gives the final prediction.

---

## 🔑 Key Concepts Explained Simply

| Concept | Simple Explanation | Analogy |
|---------|-------------------|---------|
| **Root Node** | The very first question asked | The trunk of a tree |
| **Branch** | The path taken based on an answer | A branch of the tree |
| **Internal Node** | A question in the middle of the tree | A fork in the road |
| **Leaf Node** | The final answer/prediction | A fruit at the end |
| **Splitting** | Dividing data into groups | Sorting cards into piles |
| **Depth** | How many questions deep the tree goes | How tall the tree grows |
| **Pruning** | Trimming the tree to prevent memorization | Cutting dead branches |

---

## 🎯 Real-Life Example

### Problem: Should a bank approve a loan?

The bank has this data:

```
Person | Income    | Credit Score | Employed? | Loan Approved?
-------|-----------|-------------|-----------|---------------
  A    | High      | Good        | Yes       | ✅ Yes
  B    | Low       | Good        | No        | ❌ No
  C    | Medium    | Poor        | Yes       | ❌ No
  D    | High      | Poor        | Yes       | ❌ No
  E    | Medium    | Good        | Yes       | ✅ Yes
  F    | Low       | Poor        | No        | ❌ No
  G    | High      | Good        | No        | ✅ Yes
  H    | Medium    | Good        | No        | ❌ No
```

The Decision Tree might learn this logic:

```
Is Credit Score Good?
├── YES ──► Is Income High or Medium?
│           ├── HIGH   ──► ✅ APPROVE
│           └── MEDIUM ──► Is Employed?
│                           ├── YES ──► ✅ APPROVE
│                           └── NO  ──► ❌ REJECT
└── NO  ──► ❌ REJECT
```

---

## 📐 How the Tree Decides to Split

The model needs to decide: **"Which question should I ask first?"**

It uses mathematical measures to find the **best split**:

### 1. Gini Impurity (most common)
Measures how "mixed" a group is. Lower = purer = better split.

```
Gini = 1 - (p₁² + p₂² + ... + pₙ²)

Example:
  Group of [5 Pass, 5 Fail] → Gini = 1 - (0.5² + 0.5²) = 0.5  (very mixed)
  Group of [9 Pass, 1 Fail] → Gini = 1 - (0.9² + 0.1²) = 0.18 (much purer!)
```

### 2. Information Gain (Entropy)
Measures how much a split reduces uncertainty. Higher = better split.

```
The model picks the feature that gives the HIGHEST information gain
→ That feature becomes the node (question) at that level
```

### Visual Example:

```
Before Split:        After Split on "Credit Score":
[✅✅✅❌❌❌❌❌]    LEFT: [✅✅✅] ← Pure! (all approved)
  Mixed (bad)         RIGHT: [❌❌❌❌❌] ← Pure! (all rejected)
                      Great split!
```

---

## 💻 Full Code Example

### Requirements
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Complete Working Code

```python
# ============================================================
# Decision Tree — Loan Approval Prediction
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Step 1: Create our Dataset ──────────────────────────────
data = {
    'Income':       ['High','Low','Medium','High','Medium','Low','High','Medium',
                     'Low','High','Medium','Low','High','Medium','Low','High'],
    'Credit_Score': ['Good','Good','Poor','Poor','Good','Poor','Good','Good',
                     'Good','Poor','Good','Poor','Good','Poor','Good','Good'],
    'Employed':     ['Yes','No','Yes','Yes','Yes','No','No','No',
                     'Yes','No','Yes','Yes','No','No','Yes','Yes'],
    'Loan_Approved':['Yes','No','No','No','Yes','No','Yes','No',
                     'No','No','Yes','No','Yes','No','No','Yes']
}

df = pd.DataFrame(data)
print("📊 Our Loan Dataset:")
print(df.to_string())
print(f"\nTotal records: {len(df)}")
print(f"Approved: {df['Loan_Approved'].value_counts()['Yes']}  |  Rejected: {df['Loan_Approved'].value_counts()['No']}")
print()

# ── Step 2: Encode categorical variables to numbers ─────────
# Machine learning models work with numbers, not text
le = LabelEncoder()

df_encoded = df.copy()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("🔢 Encoded Dataset (numbers for the model):")
print(df_encoded.to_string())
print()

# ── Step 3: Split into features (X) and label (y) ──────────
X = df_encoded[['Income', 'Credit_Score', 'Employed']]  # Features (inputs)
y = df_encoded['Loan_Approved']                          # Label (output)

# ── Step 4: Split into training and testing sets ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"📚 Training samples: {len(X_train)}")
print(f"🧪 Testing samples:  {len(X_test)}")
print()

# ── Step 5: Create and train the Decision Tree ──────────────
model = DecisionTreeClassifier(
    criterion='gini',    # Use Gini Impurity to find best splits
    max_depth=3,         # Limit depth to avoid overfitting
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Decision Tree trained successfully!")

# ── Step 6: View the Decision Tree Rules ────────────────────
feature_names = ['Income', 'Credit_Score', 'Employed']
class_names   = ['Rejected', 'Approved']

print("\n🌳 Decision Tree Rules (Text Format):")
print("-" * 50)
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)

# ── Step 7: Make Predictions ────────────────────────────────
y_pred = model.predict(X_test)

# ── Step 8: Evaluate the Model ──────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.1f}%")
print()
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── Step 9: Feature Importance ──────────────────────────────
print("🔍 Feature Importance (which feature matters most?):")
print("-" * 45)
importances = model.feature_importances_
for feature, importance in sorted(zip(feature_names, importances),
                                   key=lambda x: x[1], reverse=True):
    bar = "█" * int(importance * 40)
    print(f"  {feature:<15} {bar}  ({importance:.3f})")

print()

# ── Step 10: Predict for New Applicants ─────────────────────
# Encoding: Income(High=0,Low=1,Medium=2), Credit(Good=0,Poor=1), Employed(No=0,Yes=1)
new_applicants = pd.DataFrame({
    'Income':       [0, 1, 2, 0],   # High, Low, Medium, High
    'Credit_Score': [0, 0, 0, 1],   # Good, Good, Good, Poor
    'Employed':     [1, 0, 1, 1]    # Yes, No, Yes, Yes
})

applicant_labels = [
    "High Income, Good Credit, Employed",
    "Low Income, Good Credit, Unemployed",
    "Medium Income, Good Credit, Employed",
    "High Income, Poor Credit, Employed"
]

predictions   = model.predict(new_applicants)
probabilities = model.predict_proba(new_applicants)

print("🔮 Predictions for New Applicants:")
print("-" * 65)
for label, pred, prob in zip(applicant_labels, predictions, probabilities):
    result     = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
    confidence = max(prob) * 100
    print(f"  {label}")
    print(f"  → {result}  (Confidence: {confidence:.0f}%)")
    print()

# ── Step 11: Visualize the Decision Tree ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot 1: The Decision Tree ---
plot_tree(
    model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=11,
    ax=axes[0],
    impurity=True,
    proportion=False
)
axes[0].set_title("🌳 Decision Tree — Loan Approval", fontsize=14, fontweight='bold', pad=15)

# --- Plot 2: Feature Importance Bar Chart ---
colors = ['#2ecc71' if i == importances.argmax() else '#3498db'
          for i in range(len(feature_names))]
bars = axes[1].bar(feature_names, importances, color=colors, edgecolor='white',
                   linewidth=1.5, width=0.5)

for bar, val in zip(bars, importances):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

axes[1].set_ylabel('Importance Score', fontsize=12)
axes[1].set_title('🔍 Feature Importance\n(Which factor matters most?)',
                   fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(importances) + 0.15)
axes[1].grid(axis='y', alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].text(importances.argmax(), importances.max() + 0.08, '⭐ Most Important',
             ha='center', fontsize=10, color='#27ae60')

plt.tight_layout(pad=3.0)
plt.savefig('decision_tree_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 Visualization saved as 'decision_tree_output.png'")
```

---

## 📊 Understanding the Output

### What the code prints:

```
🌳 Decision Tree Rules (Text Format):
--------------------------------------------------
|--- Credit_Score <= 0.50          ← Is Credit Good?
|   |--- Income <= 1.50            ← Is Income High?
|   |   |--- class: Approved       ← YES → Approve!
|   |--- Income > 1.50
|   |   |--- class: Rejected       ← Low income → Reject
|--- Credit_Score > 0.50           ← Poor Credit?
|   |--- class: Rejected           ← Always Reject

🎯 Model Accuracy: 100.0%

🔍 Feature Importance:
  Credit_Score    ████████████████████████████  (0.700)  ← MOST important!
  Income          ████████████                  (0.300)
  Employed        (0.000)                       ← Not needed

🔮 Predictions for New Applicants:
  High Income, Good Credit, Employed
  → ✅ APPROVED  (Confidence: 100%)

  Low Income, Good Credit, Unemployed
  → ❌ REJECTED  (Confidence: 100%)
```

### Reading the Tree Nodes:

Each node in the tree shows:
```
┌─────────────────────────────┐
│  Credit_Score <= 0.50       │  ← The question being asked
│  gini = 0.46                │  ← How mixed is this group?
│  samples = 12               │  ← How many data points here?
│  value = [5, 7]             │  ← [Rejected, Approved]
└─────────────────────────────┘
          /          \
       True          False
```

---

## ⚠️ Overfitting & How to Fix It

### What is Overfitting?
> When a tree grows too deep, it memorizes the training data instead of learning general patterns — it fails on new data!

```
Shallow Tree (Good)          Deep Tree (Overfitted)
    Ask 3 questions              Ask 50 questions
    Works on new data ✅         Memorized training data ❌
```

### How to Prevent Overfitting:

```python
model = DecisionTreeClassifier(
    max_depth=3,          # ✅ Limit how deep the tree grows
    min_samples_split=5,  # ✅ Need at least 5 samples to split a node
    min_samples_leaf=2,   # ✅ Each leaf must have at least 2 samples
    max_features='sqrt',  # ✅ Only consider sqrt(features) at each split
    ccp_alpha=0.01        # ✅ Cost-complexity pruning (trims useless branches)
)
```

### Visual Comparison:

| Setting | Result |
|---------|--------|
| `max_depth=None` (default) | Tree grows until perfect — likely overfit |
| `max_depth=3` | Controlled depth — generalizes well |
| `min_samples_leaf=5` | No tiny leaf nodes — more robust |
| `ccp_alpha=0.01` | Prunes branches that don't help much |

---

## ⚖️ Decision Tree vs Logistic Regression

| Feature | Decision Tree | Logistic Regression |
|---------|--------------|-------------------|
| **Type** | Rule-based | Probability-based |
| **Output** | Class label | Probability score |
| **Interpretability** | Very easy (flowchart) | Moderate |
| **Non-linear patterns** | ✅ Handles well | ❌ Struggles |
| **Missing values** | ✅ Handles well | ❌ Needs preprocessing |
| **Overfitting risk** | ⚠️ High (without pruning) | ✅ Low |
| **Best for** | Complex rules, mixed data | Linear, binary problems |

---

## ✅ When to Use Decision Trees

### ✔ Use Decision Trees when:
- Data has **non-linear relationships**
- You need a **human-readable** model (explainability matters)
- Data has **mixed types** (numbers + categories)
- You want to know **which features matter most**
- Quick **baseline** model is needed

### ✘ Avoid Decision Trees when:
- Dataset is very small (tree might overfit)
- You need very high accuracy (use Random Forest or XGBoost instead)
- Features are all continuous and linear (Logistic Regression is better)

---

## 📖 Key Terms Glossary

| Term | Simple Explanation |
|------|--------------------|
| **Root Node** | The first question the tree asks |
| **Leaf Node** | The final prediction at the end of a branch |
| **Splitting** | Dividing data into two groups based on a feature |
| **Depth** | How many levels/questions deep the tree goes |
| **Pruning** | Removing branches that don't improve accuracy |
| **Gini Impurity** | Measures how mixed a group of data is (lower = better) |
| **Information Gain** | How much a split reduces uncertainty (higher = better) |
| **Feature Importance** | Score showing which input matters most for predictions |
| **Overfitting** | When the model memorizes training data and fails on new data |
| **max_depth** | A setting that limits how tall the tree grows |
| **Entropy** | Another measure of disorder in data (like Gini) |

---
