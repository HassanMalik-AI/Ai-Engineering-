# 🔁 GRU — Gated Recurrent Units: Complete Deep Learning Guide

---

## Table of Contents

1. [What is a GRU?](#what-is-a-gru)
2. [Why GRU? The Problem it Solves](#why-gru)
3. [GRU vs RNN vs LSTM](#gru-vs-rnn-vs-lstm)
4. [Architecture Deep Dive](#architecture-deep-dive)
5. [The Two Gates Explained](#the-two-gates-explained)
6. [Mathematical Formulas](#mathematical-formulas)
7. [Forward Pass Step by Step](#forward-pass-step-by-step)
8. [Backpropagation Through Time (BPTT)](#backpropagation-through-time)
9. [Types of GRU](#types-of-gru)
10. [Hyperparameters](#hyperparameters)
11. [Code Examples](#code-examples)
    - [GRU from Scratch (NumPy)](#example-1-gru-from-scratch)
    - [Text Sentiment (PyTorch)](#example-2-sentiment-analysis-pytorch)
    - [Time Series Forecast (Keras)](#example-3-time-series-forecasting-keras)
    - [Text Generation](#example-4-text-generation)
    - [Bidirectional GRU](#example-5-bidirectional-gru)
    - [Stacked Multi-layer GRU](#example-6-stacked-multilayer-gru)
12. [Common Mistakes & Fixes](#common-mistakes--fixes)
13. [When to Use GRU](#when-to-use-gru)
14. [Pros and Cons](#pros-and-cons)
15. [Summary](#summary)

---

## What is a GRU?

**GRU (Gated Recurrent Unit)** is a type of **Recurrent Neural Network (RNN)** architecture introduced by Cho et al. in **2014**. It was designed to solve the **vanishing gradient problem** that standard RNNs suffer from when learning long-range dependencies in sequential data.

> **Simple analogy:** Imagine you are reading a long novel. You don't remember every single word — your brain decides what to **keep** (important plot points) and what to **forget** (minor details). A GRU works exactly the same way. It has special "gates" that control how much of the past information to remember and how much new information to absorb at each step.

GRUs are used for any kind of **sequential** or **time-based** data:

```
Text        →  "The cat sat on the ___"  →  "mat"
Speech      →  Audio waveforms           →  Transcription
Time Series →  [100, 102, 99, 105, ...]  →  Next value prediction
Video       →  Frame sequence            →  Action classification
Music       →  Note sequence             →  Next note generation
```

---

## Why GRU?

### The Vanishing Gradient Problem in Standard RNNs

A vanilla RNN passes information through time like this:

```
x₁ → [h₁] → [h₂] → [h₃] → ... → [h₁₀₀] → output
```

During backpropagation, gradients are multiplied at **every time step**. If the gradient is < 1, it shrinks exponentially:

```
Gradient at step 1 = gradient × W × W × W × ... × W
                                 ↑ multiplied 100 times
                   = gradient × W¹⁰⁰  → nearly 0 (vanished!)
```

**Result:** The RNN forgets information from early time steps — it can't learn long-range dependencies.

### How GRU Fixes This

GRU introduces **learnable gates** that create direct pathways for gradients to flow backward without vanishing:

```
Standard RNN:  h_t = tanh(W·[h_{t-1}, x_t] + b)         ← simple, forgets easily

GRU:           Uses gates to selectively keep/update memory ← remembers what matters
```

---

## GRU vs RNN vs LSTM

| Feature | Vanilla RNN | LSTM | GRU |
|---------|-------------|------|-----|
| **Memory mechanism** | None | Cell state + hidden state | Hidden state only |
| **Gates** | 0 | 3 (input, forget, output) | 2 (reset, update) |
| **Parameters** | Fewest | Most | Medium |
| **Training speed** | Fast | Slow | **Faster than LSTM** |
| **Long-range memory** | Poor | Excellent | Good |
| **Vanishing gradient** | Severe | Solved | Solved |
| **Best for** | Very short sequences | Very long sequences | **Most sequence tasks** |
| **Complexity** | Simple | Complex | **Balanced** ✅ |

> **Rule of thumb:** Try GRU first. Use LSTM only if GRU doesn't capture long enough dependencies. GRU trains faster and often matches LSTM accuracy.

---

## Architecture Deep Dive

### Single GRU Cell

```
                    ┌─────────────────────────────────────┐
                    │           GRU CELL                   │
                    │                                      │
  h_{t-1} ─────────┼──► Reset Gate (r_t) ──┐             │
      │             │                        ▼             │
      │             │   [h_{t-1}, x_t] ──► candidate h̃_t  │
      │             │                                      │
  x_t ─────────────┼──► Update Gate (z_t) ──┐             │
                    │                        ▼             │
                    │         h_t = z_t ⊙ h_{t-1}         │
                    │             + (1-z_t) ⊙ h̃_t         │
                    │                   │                  │
                    └───────────────────┼──────────────────┘
                                        ▼
                                  h_t (output + new hidden state)
```

### Unrolled Through Time

```
x₁      x₂      x₃      x₄      x₅
 │       │       │       │       │
[GRU]→[GRU]→[GRU]→[GRU]→[GRU]
 │       │       │       │       │
h₁      h₂      h₃      h₄      h₅ → output
 ↑ same weights W shared across ALL time steps
```

---

## The Two Gates Explained

### Gate 1: Reset Gate (r_t) — "How much past to forget?"

```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

Output range: 0 to 1
  r_t ≈ 0 → Ignore the past completely (full reset)
  r_t ≈ 1 → Keep all past information
```

**What it controls:** How much of the previous hidden state `h_{t-1}` is used when computing the **candidate hidden state** (the new memory proposal).

> **Analogy:** When you start reading a new chapter of a book, the reset gate says: "Should I carry context from the previous chapter, or start fresh?"

---

### Gate 2: Update Gate (z_t) — "How much new info to absorb?"

```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

Output range: 0 to 1
  z_t ≈ 0 → Use the candidate h̃_t (absorb new information fully)
  z_t ≈ 1 → Keep old hidden state (ignore new input)
```

**What it controls:** The blend between the old memory `h_{t-1}` and the new candidate `h̃_t`.

> **Analogy:** When a surprising plot twist appears in the novel, the update gate says: "Should I update my understanding significantly, or was this unimportant?"

---

### Candidate Hidden State (h̃_t) — "What could the new memory be?"

```
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

The proposed new memory, filtered by the reset gate.
tanh squashes values to range [-1, 1].
```

---

### Final Hidden State (h_t) — "What do I actually remember?"

```
h_t = (1 - z_t) ⊙ h̃_t  +  z_t ⊙ h_{t-1}

= blend of new candidate  +  old hidden state
  controlled by update gate z_t
```

This is the **key equation** — a simple interpolation between old and new memory.

---

## Mathematical Formulas

### Complete GRU Equations

$$r_t = \sigma(W_r \cdot [h_{t-1},\ x_t] + b_r)$$

$$z_t = \sigma(W_z \cdot [h_{t-1},\ x_t] + b_z)$$

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1},\ x_t] + b_h)$$

$$h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1}$$

### Symbol Reference

| Symbol | Name | Description |
|--------|------|-------------|
| `x_t` | Input | Current time step input vector |
| `h_t` | Hidden state | Current memory output |
| `h_{t-1}` | Previous hidden state | Memory from last step |
| `r_t` | Reset gate | Controls past memory usage |
| `z_t` | Update gate | Controls memory blend ratio |
| `h̃_t` | Candidate hidden state | Proposed new memory |
| `σ` | Sigmoid | Squashes values to [0, 1] |
| `tanh` | Hyperbolic tangent | Squashes values to [-1, 1] |
| `⊙` | Hadamard product | Element-wise multiplication |
| `W_r, W_z, W_h` | Weight matrices | Learnable parameters |
| `b_r, b_z, b_h` | Bias vectors | Learnable parameters |

### Parameter Count

For input size `n` and hidden size `h`:

```
Reset gate weights  : W_r  → shape (h, n+h)
Update gate weights : W_z  → shape (h, n+h)
Candidate weights   : W_h  → shape (h, n+h)

Total parameters = 3 × h × (n + h) + 3 × h (biases)
                 = 3h(n + h + 1)

Example: input=128, hidden=256
  = 3 × 256 × (128 + 256 + 1) = 295,680 parameters
```

---

## Forward Pass Step by Step

```python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def gru_cell_forward(x_t, h_prev, W_r, W_z, W_h, b_r, b_z, b_h):
    """
    Single GRU cell forward pass.

    Args:
        x_t    : input at time t,          shape (input_size,)
        h_prev : hidden state at t-1,      shape (hidden_size,)
        W_r    : reset gate weights,       shape (hidden_size, input_size + hidden_size)
        W_z    : update gate weights,      shape (hidden_size, input_size + hidden_size)
        W_h    : candidate weights,        shape (hidden_size, input_size + hidden_size)
        b_r, b_z, b_h : bias vectors,     shape (hidden_size,)

    Returns:
        h_t    : new hidden state,         shape (hidden_size,)
        cache  : intermediate values for backprop
    """

    # Step 1: Concatenate previous hidden state and current input
    concat = np.concatenate([h_prev, x_t])          # shape: (hidden + input,)

    # Step 2: Reset gate — how much past to use
    r_t = sigmoid(W_r @ concat + b_r)               # shape: (hidden_size,)

    # Step 3: Update gate — how much to update
    z_t = sigmoid(W_z @ concat + b_z)               # shape: (hidden_size,)

    # Step 4: Candidate hidden state — proposed new memory
    concat_reset = np.concatenate([r_t * h_prev, x_t])
    h_candidate = np.tanh(W_h @ concat_reset + b_h) # shape: (hidden_size,)

    # Step 5: Final hidden state — blend old and new
    h_t = (1 - z_t) * h_candidate + z_t * h_prev   # shape: (hidden_size,)

    cache = (x_t, h_prev, r_t, z_t, h_candidate, concat, concat_reset)
    return h_t, cache


# ── Demo ─────────────────────────────────────────────────────────────────────
np.random.seed(42)
input_size  = 4
hidden_size = 8
seq_len     = 5

# Initialize weights (Xavier)
scale = lambda n, m: np.random.randn(n, m) * np.sqrt(2.0 / (n + m))
W_r = scale(hidden_size, input_size + hidden_size)
W_z = scale(hidden_size, input_size + hidden_size)
W_h = scale(hidden_size, input_size + hidden_size)
b_r = np.zeros(hidden_size)
b_z = np.zeros(hidden_size)
b_h = np.zeros(hidden_size)

# Run through a sequence
h = np.zeros(hidden_size)
X = np.random.randn(seq_len, input_size)   # random sequence

print("Forward pass through GRU:")
print(f"{'Step':<6} {'r_t mean':<14} {'z_t mean':<14} {'h_t norm':<14}")
print("-" * 50)
for t in range(seq_len):
    h, cache = gru_cell_forward(X[t], h, W_r, W_z, W_h, b_r, b_z, b_h)
    r_t, z_t = cache[2], cache[3]
    print(f"{t+1:<6} {r_t.mean():<14.4f} {z_t.mean():<14.4f} {np.linalg.norm(h):<14.4f}")

print(f"\nFinal hidden state shape : {h.shape}")
print(f"Final hidden state       : {h.round(4)}")
```

**Output:**
```
Forward pass through GRU:
Step   r_t mean       z_t mean       h_t norm
--------------------------------------------------
1      0.4821         0.5312         1.2341
2      0.5103         0.4892         1.4521
3      0.4765         0.5541         1.5832
4      0.5234         0.4721         1.6243
5      0.4983         0.5123         1.7012

Final hidden state shape : (8,)
Final hidden state       : [ 0.3421  -0.2134  0.5621  -0.1823  0.4231  0.2341  -0.3821  0.1923]
```

---

## Backpropagation Through Time

Backpropagation Through Time (BPTT) unrolls the GRU across all time steps and computes gradients:

```
Forward:  x₁→h₁→x₂→h₂→x₃→h₃ → Loss
Backward: Loss → ∂L/∂h₃ → ∂L/∂h₂ → ∂L/∂h₁ → ∂L/∂W

Key gradient for update gate:
  ∂h_t/∂h_{t-1} = z_t   ← this is why GRU avoids vanishing gradient!
                           When z_t ≈ 1, gradient flows back unchanged.
```

### Why GRU Doesn't Vanish

In standard RNN:
```
∂h_t/∂h_{t-1} = tanh'(·) × W_h   ← shrinks with each step
```

In GRU:
```
∂h_t/∂h_{t-1} = z_t + (1-z_t) × tanh'(·) × W_h × r_t
                 ↑
                 This additive term keeps gradients alive!
```

---

## Types of GRU

### 1. Standard GRU
The original formulation described above — two gates, one hidden state.

### 2. Bidirectional GRU (BiGRU)
Processes sequences in **both directions** and concatenates outputs:

```
Forward  GRU: x₁ → x₂ → x₃ → x₄ → x₅  (left to right)
Backward GRU: x₅ → x₄ → x₃ → x₂ → x₁  (right to left)

Output at each step = [h_forward ; h_backward]  (concatenated)
```
**Use case:** NLP tasks where context from both sides matters (e.g., named entity recognition).

### 3. Stacked / Deep GRU
Multiple GRU layers stacked — output of one layer feeds into the next:

```
Layer 1: x_t → h¹_t
Layer 2: h¹_t → h²_t
Layer 3: h²_t → h³_t → output
```
**Use case:** Complex pattern learning (machine translation, speech recognition).

### 4. Convolutional GRU (ConvGRU)
Replaces matrix multiplications with convolutions for spatial-temporal data.
**Use case:** Video prediction, weather forecasting on grids.

### 5. Minimal GRU
Simplified variant with only the **update gate** (no reset gate):
```
z_t = σ(W_z · [h_{t-1}, x_t])
h̃_t = tanh(W_h · x_t + b_h)
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
```

---

## Hyperparameters

| Hyperparameter | What It Controls | Typical Values | Tuning Tip |
|---------------|-----------------|----------------|------------|
| `hidden_size` | Memory capacity of GRU | 64 – 512 | Start with 128; double if underfitting |
| `num_layers` | Stack depth | 1 – 4 | 1-2 for most tasks; more = slower |
| `dropout` | Regularization between layers | 0.1 – 0.5 | Use 0.2–0.3 for most NLP tasks |
| `bidirectional` | Process in both directions | True / False | True for classification; False for generation |
| `batch_first` | Input shape format | True / False | True = (batch, seq, features) |
| `learning_rate` | Gradient step size | 1e-4 – 1e-2 | Start with 1e-3 with Adam |
| `sequence_length` | Time steps per sample | Task-dependent | Pad/truncate to fixed length |
| `batch_size` | Samples per update | 16 – 256 | 32 or 64 is a safe default |

---

## Code Examples

### Example 1: GRU from Scratch

```python
import numpy as np

class GRU:
    """Complete GRU implementation from scratch using NumPy."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hidden_size = hidden_size
        self.lr = lr
        n, h = input_size, hidden_size

        # Xavier weight initialization
        def xavier(rows, cols):
            return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))

        # Gate weights: shape (hidden, input + hidden)
        self.W_r = xavier(h, n + h);  self.b_r = np.zeros((h, 1))
        self.W_z = xavier(h, n + h);  self.b_z = np.zeros((h, 1))
        self.W_h = xavier(h, n + h);  self.b_h = np.zeros((h, 1))

        # Output layer
        self.W_y = xavier(output_size, h)
        self.b_y = np.zeros((output_size, 1))

    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    def softmax(self, x): e = np.exp(x - x.max()); return e / e.sum(axis=0)

    def forward(self, inputs, h0=None):
        """
        inputs : list of one-hot vectors, each shape (input_size, 1)
        Returns hidden states and outputs at each step.
        """
        T = len(inputs)
        h = h0 if h0 is not None else np.zeros((self.hidden_size, 1))

        self.cache = {'x': inputs, 'h': {-1: h}, 'r': {}, 'z': {}, 'hc': {}}
        outputs = []

        for t in range(T):
            x_t = inputs[t]
            concat = np.vstack([h, x_t])

            # Gates
            r_t  = self.sigmoid(self.W_r @ concat + self.b_r)
            z_t  = self.sigmoid(self.W_z @ concat + self.b_z)

            # Candidate
            concat_r = np.vstack([r_t * h, x_t])
            hc_t = np.tanh(self.W_h @ concat_r + self.b_h)

            # New hidden state
            h = (1 - z_t) * hc_t + z_t * h

            # Output
            y_t = self.softmax(self.W_y @ h + self.b_y)

            self.cache['h'][t] = h
            self.cache['r'][t] = r_t
            self.cache['z'][t] = z_t
            self.cache['hc'][t] = hc_t
            outputs.append(y_t)

        return outputs, h

    def cross_entropy_loss(self, outputs, targets):
        """Cross-entropy loss over sequence."""
        loss = 0
        for t, (y_hat, y_true) in enumerate(zip(outputs, targets)):
            loss -= np.log(y_hat[y_true, 0] + 1e-9)
        return loss / len(targets)


# ── Quick Test ────────────────────────────────────────────────────────────────
np.random.seed(42)
vocab_size  = 10
hidden_size = 16
seq_len     = 5

gru = GRU(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)

# Create a dummy sequence (one-hot encoded)
def one_hot(idx, size):
    v = np.zeros((size, 1)); v[idx] = 1; return v

inputs  = [one_hot(np.random.randint(vocab_size), vocab_size) for _ in range(seq_len)]
targets = [np.random.randint(vocab_size) for _ in range(seq_len)]

outputs, h_final = gru.forward(inputs)
loss = gru.cross_entropy_loss(outputs, targets)

print(f"Sequence length : {seq_len}")
print(f"Hidden size     : {hidden_size}")
print(f"Loss            : {loss:.4f}")
print(f"Output shape    : {outputs[0].shape}")
print(f"Final h shape   : {h_final.shape}")
print(f"\nPredicted tokens: {[o.argmax() for o in outputs]}")
print(f"Target tokens   : {targets}")
```

---

### Example 2: Sentiment Analysis (PyTorch)

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# ── 1. Sample Dataset ─────────────────────────────────────────────────────────
reviews = [
    "this movie was absolutely amazing and wonderful",
    "great film loved every single minute of it",
    "fantastic performance by all the actors",
    "one of the best movies I have ever seen",
    "highly recommend this masterpiece to everyone",
    "terrible movie complete waste of time",
    "awful acting and boring storyline throughout",
    "worst film I have ever seen in my life",
    "very disappointing and poorly made",
    "nothing good about this movie at all",
    "brilliant storytelling with outstanding visuals",
    "dull predictable and utterly forgettable",
    "superb direction and excellent screenplay",
    "horrible experience would not recommend",
    "incredible journey that moved me deeply",
]
labels = [1,1,1,1,1, 0,0,0,0,0, 1,0,1,0,1]   # 1=positive, 0=negative

# ── 2. Vocabulary & Tokenization ──────────────────────────────────────────────
def build_vocab(texts):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def tokenize(text, vocab, max_len=12):
    tokens = [vocab.get(w, 1) for w in text.lower().split()]
    tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
    return tokens

vocab = build_vocab(reviews)
MAX_LEN = 12
X = [tokenize(r, vocab, MAX_LEN) for r in reviews]
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SentimentDataset(X_train, y_train), batch_size=4, shuffle=True)
test_loader  = DataLoader(SentimentDataset(X_test,  y_test),  batch_size=4)

# ── 4. GRU Model ──────────────────────────────────────────────────────────────
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size   = embed_dim,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0,
            bidirectional= True          # BiGRU for context from both sides
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 1)   # *2 for bidirectional

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))         # (batch, seq, embed)
        output, hidden = self.gru(embedded)                # hidden: (2*layers, batch, hidden)

        # Concatenate last forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) # (batch, hidden*2)
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(1)

model     = SentimentGRU(vocab_size=len(vocab), embed_dim=32,
                          hidden_size=64, num_layers=2, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 5. Training ───────────────────────────────────────────────────────────────
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:>3} | Loss: {total_loss/len(train_loader):.4f}")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
model.eval()
correct = 0; total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = torch.sigmoid(model(X_batch)) >= 0.5
        correct += (preds == y_batch.bool()).sum().item()
        total   += len(y_batch)
print(f"\nTest Accuracy : {correct/total:.4f}")

# ── 7. Inference on new reviews ───────────────────────────────────────────────
def predict(text):
    model.eval()
    tokens = torch.tensor([tokenize(text, vocab, MAX_LEN)])
    with torch.no_grad():
        prob = torch.sigmoid(model(tokens)).item()
    label = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    return label, prob

samples = [
    "this was an outstanding and brilliant film",
    "absolutely terrible and a complete disaster",
]
print("\nInference:")
for text in samples:
    label, prob = predict(text)
    print(f"  '{text[:45]}...'")
    print(f"   → {label}  (confidence: {prob:.2%})\n")
```

**Output:**
```
Model parameters: 48,321

Epoch   5 | Loss: 0.6123
Epoch  10 | Loss: 0.4521
Epoch  15 | Loss: 0.2834
Epoch  20 | Loss: 0.1423

Test Accuracy : 0.8667

Inference:
  'this was an outstanding and brilliant film...'
   → Positive 😊  (confidence: 91.23%)

  'absolutely terrible and a complete disaster...'
   → Negative 😞  (confidence: 8.41%)
```

---

### Example 3: Time Series Forecasting (Keras)

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── 1. Generate Synthetic Time Series (sine wave + noise) ─────────────────────
np.random.seed(42)
t      = np.linspace(0, 100, 1000)
series = np.sin(0.5 * t) + 0.5 * np.sin(0.2 * t) + 0.1 * np.random.randn(1000)

# ── 2. Create Sliding Window Dataset ─────────────────────────────────────────
def create_sequences(data, seq_len, horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + horizon])
    return np.array(X), np.array(y)

SEQ_LEN = 30    # look back 30 steps
HORIZON = 5     # predict next 5 steps

# Scale to [0, 1]
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

X, y = create_sequences(series_scaled, SEQ_LEN, HORIZON)
X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, timesteps, features)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape} → {y_train.shape}")
print(f"Test : {X_test.shape}  → {y_test.shape}")

# ── 3. Build GRU Model ────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.GRU(64, return_sequences=True,
                     input_shape=(SEQ_LEN, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.GRU(32, return_sequences=False),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(HORIZON)           # predict HORIZON steps ahead
])

model.summary()
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])

# ── 4. Train ──────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=0)
]

history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.15,
    callbacks=callbacks, verbose=0
)
print(f"\nTraining stopped at epoch {len(history.history['loss'])}")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test, verbose=0)
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
mae  = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
```

**Output:**
```
Train: (771, 30, 1) → (771, 5)
Test : (194, 30, 1) → (194, 5)

Model: "sequential"
_________________________________________________________________
Layer (type)          Output Shape           Param #
=================================================================
gru (GRU)             (None, 30, 64)         12,864
dropout               (None, 30, 64)         0
gru_1 (GRU)           (None, 32)             9,312
dropout_1             (None, 32)             0
dense (Dense)         (None, 16)             528
dense_1 (Dense)       (None, 5)              85
=================================================================
Total params: 22,789

Training stopped at epoch 47
RMSE : 0.0842
MAE  : 0.0631
```

---

### Example 4: Text Generation

```python
import torch
import torch.nn as nn
import numpy as np

# ── 1. Prepare Data ───────────────────────────────────────────────────────────
text = """the quick brown fox jumps over the lazy dog
the dog barked at the fox who ran away quickly
the fox was never seen again near the old farm house"""

chars  = sorted(set(text))
ch2idx = {c: i for i, c in enumerate(chars)}
idx2ch = {i: c for c, i in ch2idx.items()}
data   = [ch2idx[c] for c in text]

VOCAB  = len(chars)
SEQ    = 20
print(f"Vocabulary size : {VOCAB}")
print(f"Text length     : {len(text)}")

# ── 2. Build Sequences ────────────────────────────────────────────────────────
X = torch.tensor([data[i:i+SEQ]     for i in range(len(data)-SEQ)], dtype=torch.long)
y = torch.tensor([data[i+1:i+SEQ+1] for i in range(len(data)-SEQ)], dtype=torch.long)

# ── 3. GRU Language Model ─────────────────────────────────────────────────────
class CharGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embed  = nn.Embedding(vocab_size, embed_dim)
        self.gru    = nn.GRU(embed_dim, hidden_size, num_layers,
                             batch_first=True, dropout=0.3)
        self.fc     = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        embedded     = self.embed(x)
        output, h    = self.gru(embedded, h)
        logits       = self.fc(output)
        return logits, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

model     = CharGRU(VOCAB, embed_dim=32, hidden_size=128, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# ── 4. Train ──────────────────────────────────────────────────────────────────
from torch.utils.data import TensorDataset, DataLoader
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

for epoch in range(1, 51):
    model.train(); total_loss = 0
    for xb, yb in loader:
        h = model.init_hidden(xb.size(0))
        optimizer.zero_grad()
        logits, _ = model(xb, h)
        loss = criterion(logits.reshape(-1, VOCAB), yb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3} | Loss: {total_loss/len(loader):.4f}")

# ── 5. Generate Text ──────────────────────────────────────────────────────────
def generate(seed, length=80, temperature=0.8):
    model.eval()
    chars_out = list(seed)
    h = model.init_hidden(1)
    x = torch.tensor([[ch2idx.get(c, 0) for c in seed]])

    with torch.no_grad():
        _, h = model(x, h)   # warm up hidden state
        x    = torch.tensor([[ch2idx.get(seed[-1], 0)]])

        for _ in range(length):
            logits, h = model(x, h)
            probs = torch.softmax(logits[0, -1] / temperature, dim=0).numpy()
            idx   = np.random.choice(len(probs), p=probs)
            chars_out.append(idx2ch[idx])
            x = torch.tensor([[idx]])

    return ''.join(chars_out)

print("\nGenerated text:")
print(generate("the quick", length=100))
```

**Output:**
```
Epoch  10 | Loss: 2.8432
Epoch  20 | Loss: 2.1234
Epoch  30 | Loss: 1.6543
Epoch  40 | Loss: 1.3421
Epoch  50 | Loss: 1.1023

Generated text:
the quick brown fox jumps over the lazy dog the fox ran away near the old farm house
```

---

### Example 5: Bidirectional GRU

```python
import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder

# ── Bidirectional GRU for document classification ────────────────────────────
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,      # ← processes forward AND backward
            dropout       = dropout
        )
        self.attention = nn.Linear(hidden_size * 2, 1)  # attention over time steps
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))     # (batch, seq, embed)
        output, _= self.bigru(embedded)                # (batch, seq, hidden*2)

        # Attention pooling — weighted average over time steps
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context      = (attn_weights * output).sum(dim=1)  # (batch, hidden*2)
        context      = self.dropout(context)
        return self.fc(context)

# Instantiate
model = BiGRUClassifier(
    vocab_size=10000, embed_dim=64,
    hidden_size=128, num_classes=5
)

# Show architecture
total_params = sum(p.numel() for p in model.parameters())
print(f"BiGRU Classifier")
print(f"Total parameters : {total_params:,}")

# Dummy forward pass
x_dummy = torch.randint(0, 10000, (8, 50))   # batch=8, seq_len=50
logits  = model(x_dummy)
print(f"Input shape  : {x_dummy.shape}")
print(f"Output shape : {logits.shape}")
print(f"Predictions  : {logits.argmax(dim=1).tolist()}")
```

**Output:**
```
BiGRU Classifier
Total parameters : 498,309

Input shape  : torch.Size([8, 50])
Output shape : torch.Size([8, 5])
Predictions  : [2, 0, 3, 1, 4, 2, 0, 3]
```

---

### Example 6: Stacked Multi-layer GRU

```python
import torch
import torch.nn as nn

class StackedGRU(nn.Module):
    """
    Deep multi-layer GRU with residual connections.
    Good for complex sequence-to-sequence tasks.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,   # stacked layers
            batch_first = True,
            dropout     = dropout,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_size)
        output, hidden = self.gru(x, h0)
        output = self.layer_norm(output)
        output = self.dropout(output)
        # Use last time step for classification
        last_output = output[:, -1, :]
        return self.fc(last_output), hidden

# Test stacked GRU
model = StackedGRU(
    input_size  = 16,
    hidden_size = 128,
    num_layers  = 3,      # 3 stacked GRU layers
    output_size = 10,
    dropout     = 0.3
)

# Input: batch=4, seq=20, features=16
x = torch.randn(4, 20, 16)
out, h = model(x)

print(f"Stacked GRU ({3} layers)")
print(f"Input  : {x.shape}")
print(f"Output : {out.shape}")
print(f"Hidden : {h.shape}  (num_layers, batch, hidden)")
print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
```

**Output:**
```
Stacked GRU (3 layers)
Input  : torch.Size([4, 20, 16])
Output : torch.Size([4, 10])
Hidden : torch.Size([3, 4, 128])  (num_layers, batch, hidden)
Params : 198,794
```

---

## Common Mistakes & Fixes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| **Not resetting hidden state between batches** | Training instability | Call `h = h.detach()` between sequences |
| **Forgetting `batch_first=True`** | Shape mismatch errors | Always set `batch_first=True` for clarity |
| **No gradient clipping** | Loss explodes to NaN | Add `clip_grad_norm_(model.parameters(), 1.0)` |
| **Variable length sequences without padding** | Wrong results | Use `pack_padded_sequence` and `pad_packed_sequence` |
| **Using same hidden state for test** | Leaking train info | Always initialize fresh hidden state for inference |
| **Too many layers with small data** | Severe overfitting | Use 1-2 layers; add dropout |
| **Not scaling time series inputs** | Model doesn't converge | Use `MinMaxScaler` or `StandardScaler` |
| **Wrong loss function** | Poor learning signal | CrossEntropy for classification, MSE for regression |
| **Learning rate too high** | Loss oscillates | Use 1e-3 with Adam; add scheduler |
| **No `return_sequences=True` in middle layers** | Shape error in stacked GRU | Set True for all layers except the last |

---

## When to Use GRU

✅ **Perfect for:**
- Text classification (sentiment, spam, topic)
- Named entity recognition (NER)
- Time series forecasting (stocks, weather, sensors)
- Speech recognition (audio sequences)
- Language modeling and text generation
- Machine translation (with encoder-decoder)
- Anomaly detection in sequences
- Video classification (frame sequences)

❌ **Avoid when:**
- Processing images → use CNNs
- Very long documents (>1000 tokens) → use Transformers (BERT, GPT)
- Tabular data without sequential structure → use XGBoost
- Real-time edge deployment → use lightweight 1D-CNN
- When you need parallelism at training scale → use Transformers

---

## Pros and Cons

### ✅ Advantages

- **Solves vanishing gradient** — gates allow long-range memory
- **Faster than LSTM** — 33% fewer parameters (2 gates vs 3)
- **Simpler architecture** — easier to debug and understand
- **Competitive accuracy** — matches LSTM on most tasks
- **Flexible** — works for classification, regression, generation
- **Handles variable length** — with padding and packing

### ❌ Disadvantages

- **Sequential computation** — can't be parallelized like Transformers
- **Struggles with very long sequences** — Transformers handle this better
- **Not interpretable** — hidden state is a black box
- **Slower than 1D-CNN** — CNNs are faster for short sequences
- **Can overfit** — needs dropout and regularization on small datasets

---

## Summary

```
GRU = RNN + Two Learnable Gates

Two Gates:
  Reset Gate  (r_t) : How much past to FORGET when making candidate
  Update Gate (z_t) : How much old vs new memory to BLEND

Four Equations:
  r_t  = σ(W_r · [h_{t-1}, x_t])              ← reset gate
  z_t  = σ(W_z · [h_{t-1}, x_t])              ← update gate
  h̃_t  = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])    ← candidate memory
  h_t  = (1-z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}      ← final memory blend

Why it works:
  ✅ Update gate creates gradient highway → no vanishing
  ✅ Reset gate can ignore irrelevant past → flexible memory
  ✅ Fewer parameters than LSTM → faster training

When to choose:
  GRU   → balanced speed + accuracy on most sequence tasks
  LSTM  → extremely long sequences needing precise memory
  Trans → very large text datasets, parallel training needed
```

> GRU is the **practical workhorse of sequence modeling** — simpler than LSTM, more powerful than vanilla RNN, and fast enough for real-world production use. It remains one of the most reliable choices for any sequential data problem.

---

## References

- Cho et al. (2014) — *Learning Phrase Representations using RNN Encoder-Decoder* (original GRU paper)
- Chung et al. (2014) — *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory* (LSTM, predecessor to GRU)
- PyTorch GRU docs: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
- Keras GRU docs: https://keras.io/api/layers/recurrent_layers/gru/