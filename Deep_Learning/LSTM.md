# 🧠 LSTM (Long Short-Term Memory) — Complete Guide

> A beginner-friendly, in-depth guide to understanding and implementing LSTMs in Deep Learning — with code examples covering every major concept.

---

## 📚 Table of Contents

1. [What is an LSTM?](#1-what-is-an-lstm)
2. [Why Not Just Use a Regular RNN?](#2-why-not-just-use-a-regular-rnn)
3. [The LSTM Architecture — Explained Simply](#3-the-lstm-architecture--explained-simply)
4. [The Four Gates of LSTM](#4-the-four-gates-of-lstm)
5. [LSTM vs GRU vs RNN — Comparison](#5-lstm-vs-gru-vs-rnn--comparison)
6. [Environment Setup](#6-environment-setup)
7. [Code Example 1 — Basic LSTM in PyTorch](#7-code-example-1--basic-lstm-in-pytorch)
8. [Code Example 2 — Basic LSTM in TensorFlow/Keras](#8-code-example-2--basic-lstm-in-tensorflowkeras)
9. [Code Example 3 — Text Prediction with LSTM](#9-code-example-3--text-prediction-with-lstm)
10. [Code Example 4 — Time Series Forecasting](#10-code-example-4--time-series-forecasting)
11. [Code Example 5 — Sentiment Analysis](#11-code-example-5--sentiment-analysis)
12. [Code Example 6 — Stacked (Deep) LSTM](#12-code-example-6--stacked-deep-lstm)
13. [Code Example 7 — Bidirectional LSTM](#13-code-example-7--bidirectional-lstm)
14. [Hyperparameter Tuning Tips](#14-hyperparameter-tuning-tips)
15. [Common Mistakes and How to Avoid Them](#15-common-mistakes-and-how-to-avoid-them)
16. [When to Use LSTM](#16-when-to-use-lstm)
17. [Summary Cheat Sheet](#17-summary-cheat-sheet)

---

## 1. What is an LSTM?

**LSTM** stands for **Long Short-Term Memory**. It is a special type of **Recurrent Neural Network (RNN)** designed to **remember information over long sequences of data**.

Think of it like your brain reading a book:
- You don't forget the beginning of a chapter by the time you reach the end.
- You remember important details and ignore irrelevant ones.
- You connect earlier events with later ones.

That's exactly what LSTM does — it **selectively remembers and forgets** information as it processes sequences.

**Where is LSTM used?**

| Domain | Use Case |
|--------|----------|
| NLP | Text generation, machine translation, sentiment analysis |
| Finance | Stock price prediction, fraud detection |
| Healthcare | ECG signal analysis, disease prediction |
| Speech | Speech recognition, audio generation |
| IoT | Sensor data prediction, anomaly detection |

---

## 2. Why Not Just Use a Regular RNN?

A regular RNN passes information from one step to the next through a **hidden state**. But it suffers from a major problem called the **Vanishing Gradient Problem**.

### What is the Vanishing Gradient Problem?

When training a neural network, we use **backpropagation** to update weights. In a long sequence, gradients (error signals) are multiplied through many steps — and they become **extremely tiny** (vanish) before reaching the early steps.

```
Step 1 --> Step 2 --> Step 3 --> ... --> Step 100
  ^           ^           ^                 ^
gradient   0.001       0.0001         output (loss computed here)
```

By the time the gradient travels back to Step 1, it's practically **zero** — meaning the network can't learn long-range dependencies.

**LSTM solves this** by using a special "conveyor belt" called the **Cell State** that carries information across many steps with minimal change.

```
Regular RNN:   h1 --> h2 --> h3 --> ... --> hn   (gradient vanishes)

LSTM:          C1 --> C2 --> C3 --> ... --> Cn   (cell state, gradient flows easily)
               h1 --> h2 --> h3 --> ... --> hn   (hidden state)
```

---

## 3. The LSTM Architecture — Explained Simply

An LSTM cell has **two highways** of information:

1. **Cell State (Ct)** — Long-term memory (like a notebook). Information is written to it and erased from it selectively.
2. **Hidden State (ht)** — Short-term memory / working memory (what the LSTM "outputs" at each step).

```
         +--------------------------------------------------+
         |                   LSTM Cell                      |
         |                                                  |
  Ct-1   | -->[Forget Gate]-->[Input Gate]-->[Output Gate]-->  Ct
         |         |               |               |         |
  ht-1   | ---------+-------------+---------------+--------->  ht
         |                                                  |
  xt     | ------------------------------------------------>|
         +--------------------------------------------------+
```

At every time step, the LSTM takes in:
- **xt** — Current input (e.g., a word, a number)
- **ht-1** — Previous hidden state
- **Ct-1** — Previous cell state

And it produces:
- **ht** — New hidden state (output)
- **Ct** — Updated cell state

---

## 4. The Four Gates of LSTM

LSTM uses **gates** — think of them as **filters** that decide what information to keep or throw away. Each gate uses a **sigmoid function** (outputs 0 to 1):
- `0` = block everything (forget)
- `1` = let everything through (remember)

### Gate 1: Forget Gate

**"What should we forget from the cell state?"**

```
ft = sigmoid(Wf . [ht-1, xt] + bf)
```

- Looks at previous hidden state `ht-1` and current input `xt`
- Outputs a number between 0 and 1 for each value in the cell state
- `0` = completely forget, `1` = completely remember

**Example:** Reading the word "he" in a sentence. The LSTM forgets the gender of any previous subject it was tracking.

---

### Gate 2: Input Gate

**"What new information should we write into the cell state?"**

This gate has two parts:

```
it  = sigmoid(Wi . [ht-1, xt] + bi)    <-- How much to update?
Ct~ = tanh(Wc . [ht-1, xt] + bc)      <-- What candidate values to add?
```

- `it` decides which values to update (sigmoid: 0 to 1)
- `Ct~` creates new candidate values (tanh: -1 to 1)

**Example:** Reading "she" — the LSTM adds female gender information to the cell state.

---

### Gate 3: Cell State Update

**"Actually update the cell state"**

```
Ct = ft * Ct-1  +  it * Ct~
        ^               ^
   (forget old)    (add new info)
```

This is the conveyor belt! Old information is selectively erased, new information is selectively added.

---

### Gate 4: Output Gate

**"What should we output as the hidden state?"**

```
ot = sigmoid(Wo . [ht-1, xt] + bo)
ht = ot * tanh(Ct)
```

- `ot` decides which parts of the cell state to output
- `tanh(Ct)` squishes cell state values between -1 and 1
- Multiply them to get the final hidden state output

---

## 5. LSTM vs GRU vs RNN — Comparison

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Memory Type | Hidden state only | Cell state + Hidden state | Hidden state only |
| Gates | None | 4 (Forget, Input, Cell update, Output) | 2 (Reset, Update) |
| Handles Long Sequences | Poor | Excellent | Good |
| Training Speed | Fast | Slower | Faster than LSTM |
| Parameters | Fewest | Most | Medium |
| Vanishing Gradient | Suffers | Handles well | Handles well |
| Best For | Very short sequences | Long sequences, complex patterns | Long sequences, faster training |

**Rule of thumb:**
- Use **GRU** when you need speed and your sequences aren't extremely long.
- Use **LSTM** when you need maximum accuracy on complex long-range dependencies.

---

## 6. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv lstm_env
source lstm_env/bin/activate      # Linux/Mac
lstm_env\Scripts\activate          # Windows

# Install dependencies
pip install torch torchvision       # PyTorch
pip install tensorflow              # TensorFlow/Keras
pip install numpy pandas matplotlib scikit-learn
pip install nltk                    # For NLP examples
```

---

## 7. Code Example 1 — Basic LSTM in PyTorch

This is the simplest possible LSTM — understand the input/output shapes first.

```python
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# LSTM Input Shape: (seq_len, batch_size, input_size)
# ─────────────────────────────────────────────

input_size  = 10   # Number of features in each time step
hidden_size = 20   # Number of neurons in LSTM hidden layer
num_layers  = 1    # Number of stacked LSTM layers
seq_len     = 5    # Length of the sequence
batch_size  = 3    # Number of sequences processed at once

# Define LSTM
lstm = nn.LSTM(input_size=input_size,
               hidden_size=hidden_size,
               num_layers=num_layers,
               batch_first=False)  # If True: (batch, seq, features)

# Create a random input sequence
x = torch.randn(seq_len, batch_size, input_size)

# Initial hidden and cell states (zeros by default)
h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

# Forward pass
output, (hn, cn) = lstm(x, (h0, c0))

print(f"Input shape:       {x.shape}")         # (5, 3, 10)
print(f"Output shape:      {output.shape}")     # (5, 3, 20) — all time steps
print(f"Hidden state (hn): {hn.shape}")         # (1, 3, 20) — last time step
print(f"Cell state  (cn):  {cn.shape}")         # (1, 3, 20) — last time step
```

**Output:**
```
Input shape:       torch.Size([5, 3, 10])
Output shape:      torch.Size([5, 3, 20])
Hidden state (hn): torch.Size([1, 3, 20])
Cell state  (cn):  torch.Size([1, 3, 20])
```

---

## 8. Code Example 2 — Basic LSTM in TensorFlow/Keras

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ─────────────────────────────────────────────
# Keras LSTM Input Shape: (batch_size, timesteps, features)
# ─────────────────────────────────────────────

timesteps  = 10   # Sequence length
n_features = 5    # Number of features per time step
n_samples  = 100  # Number of training samples

# Generate dummy data
X = np.random.randn(n_samples, timesteps, n_features)
y = np.random.randn(n_samples, 1)

# Build a simple LSTM model
model = Sequential([
    LSTM(units=50,                    # Number of LSTM neurons
         activation='tanh',           # Default activation
         input_shape=(timesteps, n_features),
         return_sequences=False),      # Only return last output
    Dense(1)                          # Output layer
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
model.fit(X, y, epochs=5, batch_size=16, verbose=1)

# Predict
X_test = np.random.randn(5, timesteps, n_features)
predictions = model.predict(X_test)
print("Predictions:", predictions.flatten())
```

---

## 9. Code Example 3 — Text Prediction with LSTM

Build a character-level LSTM that learns to predict the next character in a sequence.

```python
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────
# Step 1: Prepare the data
# ─────────────────────────────────────────────
text = "hello world! deep learning with lstm is amazing. " * 50

# Create vocabulary (unique characters)
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# Convert text to indices
data = [char2idx[c] for c in text]

# ─────────────────────────────────────────────
# Step 2: Create sequences
# ─────────────────────────────────────────────
seq_len    = 20    # Input sequence length
batch_size = 32

def create_batches(data, seq_len, batch_size):
    """Create input-target pairs."""
    sequences, targets = [], []
    for i in range(0, len(data) - seq_len - 1, 1):
        seq    = data[i: i + seq_len]
        target = data[i + seq_len]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

X, y = create_batches(data, seq_len, batch_size)
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# ─────────────────────────────────────────────
# Step 3: Define the Model
# ─────────────────────────────────────────────
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm      = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded      = self.embedding(x)          # (batch, seq, embed)
        output, _     = self.lstm(embedded)        # (batch, seq, hidden)
        out           = self.fc(output[:, -1, :])  # Only last time step
        return out

model = CharLSTM(vocab_size=vocab_size, embed_size=32, hidden_size=128, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────
# Step 4: Train
# ─────────────────────────────────────────────
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss   = criterion(output, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.4f}")

# ─────────────────────────────────────────────
# Step 5: Generate Text
# ─────────────────────────────────────────────
def generate_text(model, start_text, length=50):
    model.eval()
    chars_out = list(start_text)
    input_seq = [char2idx[c] for c in start_text]

    with torch.no_grad():
        for _ in range(length):
            x         = torch.tensor([input_seq[-seq_len:]], dtype=torch.long)
            output    = model(x)
            predicted = torch.argmax(output, dim=1).item()
            chars_out.append(idx2char[predicted])
            input_seq.append(predicted)

    return ''.join(chars_out)

print("\nGenerated text:")
print(generate_text(model, "hello", length=100))
```

---

## 10. Code Example 4 — Time Series Forecasting

Predict future values of a sine wave using LSTM.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Step 1: Generate Sine Wave Data
# ─────────────────────────────────────────────
t    = np.linspace(0, 100, 1000)
data = np.sin(t) + 0.1 * np.random.randn(len(t))  # Add noise

# Normalize data to [0, 1]
data_min, data_max = data.min(), data.max()
data_norm = (data - data_min) / (data_max - data_min)

# ─────────────────────────────────────────────
# Step 2: Create Sliding Window Sequences
# ─────────────────────────────────────────────
def sliding_window(data, window=30, horizon=1):
    """
    window  : how many past steps to look at
    horizon : how many steps ahead to predict
    """
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i: i + window])
        y.append(data[i + window: i + window + horizon])
    return np.array(X), np.array(y)

window_size = 30
X, y = sliding_window(data_norm, window=window_size, horizon=1)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape(-1, window_size, 1)
y = y.reshape(-1, 1)

# Train/test split (80/20)
split   = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

# ─────────────────────────────────────────────
# Step 3: Define the LSTM Model
# ─────────────────────────────────────────────
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq, hidden)
        out    = self.fc(out[:, -1, :])  # Last time step
        return out

model     = TimeSeriesLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────
# Step 4: Train
# ─────────────────────────────────────────────
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(50):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred       = model(batch_x)
        loss       = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test)
            val_loss = criterion(val_pred, y_test)
        print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

# ─────────────────────────────────────────────
# Step 5: Plot Results
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    actuals     = y_test.numpy()

# Denormalize
predictions = predictions * (data_max - data_min) + data_min
actuals     = actuals     * (data_max - data_min) + data_min

plt.figure(figsize=(12, 4))
plt.plot(actuals[:200],     label='Actual',    color='blue')
plt.plot(predictions[:200], label='Predicted', color='red', linestyle='--')
plt.title('LSTM Time Series Forecasting')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.savefig('time_series_forecast.png', dpi=150)
plt.show()
```

---

## 11. Code Example 5 — Sentiment Analysis

Classify movie reviews as Positive or Negative using LSTM.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# ─────────────────────────────────────────────
# Step 1: Sample Data
# ─────────────────────────────────────────────
reviews = [
    ("this movie was absolutely fantastic and wonderful", 1),
    ("i loved every single minute of this film", 1),
    ("the acting was superb and the story was great", 1),
    ("amazing plot with brilliant performances throughout", 1),
    ("one of the best movies i have ever seen", 1),
    ("terrible film waste of time completely boring", 0),
    ("the worst movie i have ever seen in my life", 0),
    ("awful acting bad story total disappointment", 0),
    ("boring and dull with no interesting characters", 0),
    ("horrible experience from start to finish", 0),
]

# ─────────────────────────────────────────────
# Step 2: Tokenize and Build Vocabulary
# ─────────────────────────────────────────────
def tokenize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

all_words = [word for text, _ in reviews for word in tokenize(text)]
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in Counter(all_words).most_common():
    if count >= 1:
        vocab[word] = len(vocab)

def encode(text, vocab, max_len=20):
    tokens = tokenize(text)
    ids    = [vocab.get(t, 1) for t in tokens]   # 1 = UNK
    # Pad or truncate to max_len
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# ─────────────────────────────────────────────
# Step 3: Dataset and DataLoader
# ─────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, reviews, vocab, max_len=20):
        self.data = [(encode(text, vocab, max_len), label) for text, label in reviews]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float)

dataset    = SentimentDataset(reviews, vocab)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ─────────────────────────────────────────────
# Step 4: LSTM Classifier
# ─────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, output_dim)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))
        _, (hidden, _)  = self.lstm(embedded)    # Use final hidden state
        out             = self.fc(self.dropout(hidden[-1]))
        return self.sigmoid(out).squeeze(1)

model = SentimentLSTM(vocab_size=len(vocab), embed_dim=32,
                      hidden_dim=64, output_dim=1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────
# Step 5: Train
# ─────────────────────────────────────────────
for epoch in range(30):
    model.train()
    epoch_loss, correct = 0, 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss        = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct    += ((predictions > 0.5).float() == batch_y).sum().item()

    if (epoch + 1) % 10 == 0:
        accuracy = correct / len(dataset) * 100
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.1f}%")

# ─────────────────────────────────────────────
# Step 6: Predict
# ─────────────────────────────────────────────
def predict_sentiment(text, model, vocab):
    model.eval()
    encoded = encode(text, vocab)
    tensor  = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "Positive" if prob > 0.5 else "Negative"
    return label, prob

test_reviews = [
    "this was a wonderful and entertaining experience",
    "horrible film completely boring and awful",
]

for review in test_reviews:
    label, prob = predict_sentiment(review, model, vocab)
    print(f"Review: '{review}'")
    print(f"  Sentiment: {label} (confidence: {prob:.3f})\n")
```

---

## 12. Code Example 6 — Stacked (Deep) LSTM

Stacking multiple LSTM layers lets the model learn **hierarchical patterns**.

```python
import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    """
    Multiple LSTM layers stacked on top of each other.
    Layer 1 learns low-level patterns.
    Layer 2 learns patterns of patterns.
    Layer N learns high-level abstract patterns.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(StackedLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,     # Stacks multiple LSTM layers
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0  # Dropout between layers
        )
        self.bn = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)       # hn: (num_layers, batch, hidden)
        last_hidden  = hn[-1]             # Take output from last layer
        last_hidden  = self.bn(last_hidden)
        return self.fc(last_hidden)

# Compare 1-layer vs 3-layer LSTM
for n_layers in [1, 2, 3]:
    model        = StackedLSTM(input_size=10, hidden_size=64,
                               num_layers=n_layers, output_size=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Layers: {n_layers} | Parameters: {total_params:,}")

# Test forward pass
model = StackedLSTM(input_size=10, hidden_size=64, num_layers=3, output_size=2)
x     = torch.randn(8, 20, 10)   # (batch=8, seq=20, features=10)
out   = model(x)
print(f"\nInput:  {x.shape}")
print(f"Output: {out.shape}")    # (8, 2)
```

---

## 13. Code Example 7 — Bidirectional LSTM

A **Bidirectional LSTM** processes the sequence in **both directions** — forward and backward. This gives it context from both the past and the future at every time step.

```
Forward  LSTM: -->  x1 --> x2 --> x3 --> x4 --> x5
Backward LSTM: <--  x1 <-- x2 <-- x3 <-- x4 <-- x5
                                                    |
                                Concatenate both outputs
```

**Best for:** Tasks where full context matters (e.g., Named Entity Recognition, classification).  
**Not for:** Real-time or streaming tasks where future data isn't available.

```python
import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True    # Key parameter!
        )
        # Output is hidden_size * 2 because we concat forward + backward
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, (hn, _)    = self.lstm(x)
        # hn shape: (num_layers * 2, batch, hidden_size)
        # Concatenate the final forward and backward hidden states
        forward_hidden  = hn[-2]   # Last forward layer
        backward_hidden = hn[-1]   # Last backward layer
        combined        = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.fc(combined)

# Regular vs Bidirectional comparison
regular = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
bidir   = nn.LSTM(input_size=10, hidden_size=32, batch_first=True, bidirectional=True)

x = torch.randn(4, 15, 10)  # (batch, seq, features)

reg_out,   _ = regular(x)
bidir_out, _ = bidir(x)

print(f"Regular LSTM output:       {reg_out.shape}")    # (4, 15, 32)
print(f"Bidirectional LSTM output: {bidir_out.shape}")  # (4, 15, 64)  <- double hidden

# Full model example
model = BidirectionalLSTM(input_size=10, hidden_size=32, num_layers=2, output_size=3)
out   = model(x)
print(f"Classification output:     {out.shape}")        # (4, 3)
```

---

## 14. Hyperparameter Tuning Tips

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| `hidden_size` | 32 – 512 | Start with 64 or 128; larger = more capacity but slower |
| `num_layers` | 1 – 4 | 1-2 for most tasks; 3-4 for very complex patterns |
| `seq_len` | 10 – 500 | Depends on your data; longer = more context |
| `batch_size` | 16 – 128 | Smaller = noisier updates; larger = smoother |
| `learning_rate` | 1e-4 – 1e-2 | Start with `0.001`; use LR scheduler |
| `dropout` | 0.1 – 0.5 | Add between layers to prevent overfitting |
| `embedding_dim` | 32 – 300 | For NLP; 300 for large vocab (or use pretrained) |

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Inside training loop, after validation:
val_loss = compute_validation_loss(model)
scheduler.step(val_loss)  # Reduce LR if val_loss doesn't improve
```

### Gradient Clipping — Always Use This!

```python
# Prevents exploding gradients — very common with RNNs
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 15. Common Mistakes and How to Avoid Them

### Mistake 1: Wrong Input Shape

```python
# WRONG: Forgot to add the features dimension
x = torch.randn(batch_size, seq_len)        # Bad

# CORRECT: (batch, seq, features)
x = torch.randn(batch_size, seq_len, 1)    # Good
```

### Mistake 2: Not Detaching Hidden State

```python
# When reusing hidden states across batches (language modeling)
# WRONG: Gradients flow back through entire history
h, c = lstm(x, (h, c))

# CORRECT: Detach to stop gradient flow
h = h.detach()
c = c.detach()
```

### Mistake 3: Forgetting to Zero Gradients

```python
# WRONG: Gradients accumulate across batches
loss.backward()
optimizer.step()

# CORRECT
optimizer.zero_grad()    # Always before backward()
loss.backward()
optimizer.step()
```

### Mistake 4: Not Switching Train/Eval Mode

```python
# Training (enables Dropout, BatchNorm updates)
model.train()

# Evaluation (disables Dropout, freezes BatchNorm)
model.eval()
with torch.no_grad():    # Also disable gradient computation
    predictions = model(X_test)
```

### Mistake 5: Not Normalizing Input Data

```python
# Raw data often has large scale differences — always normalize
from sklearn.preprocessing import StandardScaler

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)    # Use fit parameters from train only!
```

---

## 16. When to Use LSTM

### Use LSTM when:
- Your data is **sequential** (time series, text, audio, video)
- You need to capture **long-range dependencies** (e.g., subject-verb agreement)
- Temporal order **matters** (shuffling samples would break the task)
- You're doing: language modeling, speech recognition, music generation, ECG analysis

### Don't Use LSTM when:
- Your data has **no temporal structure** — use MLP or CNN instead
- You need **very fast inference** — use GRU or Transformers
- Your sequences are **very long** (1000+ steps) — use Transformer/Attention models
- You have **large compute and lots of data** — Transformers usually outperform LSTMs

### Consider Alternatives:
- **Transformer** — Better for very long sequences, parallelizable
- **GRU** — Simpler than LSTM, often comparable accuracy
- **TCN (Temporal CNN)** — Fast, parallelizable, great for time series
- **1D CNN** — Good for short local patterns in sequences

---

## 17. Summary Cheat Sheet

```
+------------------------------------------------------------------------+
|                          LSTM CHEAT SHEET                              |
+------------------------------------------------------------------------+
|  GATES                                                                 |
|  Forget Gate  : ft  = sigmoid(Wf . [ht-1, xt] + bf)   What to forget  |
|  Input Gate   : it  = sigmoid(Wi . [ht-1, xt] + bi)   What to update  |
|  Candidate    : Ct~ = tanh(Wc . [ht-1, xt] + bc)      New info        |
|  Cell Update  : Ct  = ft*Ct-1 + it*Ct~                Update memory   |
|  Output Gate  : ot  = sigmoid(Wo . [ht-1, xt] + bo)   What to output  |
|  Hidden State : ht  = ot * tanh(Ct)                   Final output    |
+------------------------------------------------------------------------+
|  INPUT SHAPES                                                          |
|  PyTorch (batch_first=True)  : (batch, seq_len, features)             |
|  PyTorch (batch_first=False) : (seq_len, batch, features)             |
|  Keras                       : (batch, timesteps, features)           |
+------------------------------------------------------------------------+
|  VARIANTS                                                              |
|  Stacked LSTM     : num_layers > 1                                     |
|  Bidirectional    : bidirectional=True -> hidden_size * 2             |
|  Regularization   : dropout=0.2 + clip_grad_norm_(max_norm=1.0)       |
+------------------------------------------------------------------------+
|  BEST PRACTICES                                                        |
|  Normalize inputs              Clip gradients (max_norm=1.0)          |
|  Use dropout                   Detach hidden states in TBPTT          |
|  Start small (1 layer)         Switch model.train() / .eval()         |
+------------------------------------------------------------------------+
```

---

## Project Structure

```
lstm-deep-learning/
├── README.md                        <- You are here
├── requirements.txt
├── 01_basic_lstm_pytorch.py
├── 02_basic_lstm_keras.py
├── 03_text_prediction.py
├── 04_time_series_forecasting.py
├── 05_sentiment_analysis.py
├── 06_stacked_lstm.py
├── 07_bidirectional_lstm.py
└── plots/
    └── time_series_forecast.png
```

---

## Further Reading

- [Original LSTM Paper — Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Understanding LSTM Networks — Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)

---
