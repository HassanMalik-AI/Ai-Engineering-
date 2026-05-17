# 🧠 Recurrent Neural Networks (RNN) — Complete Guide

> A beginner-friendly, in-depth guide to Recurrent Neural Networks with code examples, diagrams, and real-world applications.

---

## 📚 Table of Contents

- [What is an RNN?](#what-is-an-rnn)
- [Why Not a Regular Neural Network?](#why-not-a-regular-neural-network)
- [How RNN Works — Step by Step](#how-rnn-works--step-by-step)
- [The Hidden State — RNN's Memory](#the-hidden-state--rnns-memory)
- [Math Behind RNN](#math-behind-rnn)
- [Types of RNN Architectures](#types-of-rnn-architectures)
- [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
- [LSTM — Long Short-Term Memory](#lstm--long-short-term-memory)
- [GRU — Gated Recurrent Unit](#gru--gated-recurrent-unit)
- [LSTM vs GRU vs RNN — Comparison](#lstm-vs-gru-vs-rnn--comparison)
- [Code Examples](#code-examples)
  - [Simple RNN from Scratch (NumPy)](#1-simple-rnn-from-scratch-numpy)
  - [RNN with PyTorch](#2-rnn-with-pytorch)
  - [LSTM for Text Generation](#3-lstm-for-text-generation)
  - [GRU for Sentiment Analysis](#4-gru-for-sentiment-analysis)
  - [Bidirectional RNN](#5-bidirectional-rnn)
  - [Stacked / Deep RNN](#6-stacked--deep-rnn)
  - [Sequence-to-Sequence (Encoder-Decoder)](#7-sequence-to-sequence-encoder-decoder)
- [Training Tips & Best Practices](#training-tips--best-practices)
- [Common Mistakes](#common-mistakes)
- [Real-World Applications](#real-world-applications)
- [When NOT to Use RNN](#when-not-to-use-rnn)
- [Resources](#resources)

---

## What is an RNN?

Imagine you're reading a sentence word by word. When you read the word **"bank"**, you don't understand it in isolation — you use the previous words to figure out if it means a **river bank** or a **financial bank**.

That's exactly what an **RNN (Recurrent Neural Network)** does. It's a type of neural network designed to handle **sequential data** — data where order and context matter.

**Examples of sequential data:**
- Text (sentences, paragraphs)
- Time series (stock prices, weather)
- Audio (speech, music)
- Video (frames in order)
- DNA sequences

> 💡 **Key Idea:** A regular neural network processes each input independently. An RNN remembers previous inputs using a **hidden state** — its memory.

---

## Why Not a Regular Neural Network?

A standard (feedforward) neural network:
- Takes a **fixed-size input** → produces a **fixed-size output**
- Has **no memory** of previous inputs
- Each input is processed **independently**

**Problem:** Suppose you want to predict the next word in:
> "The cat sat on the ___"

A regular network doesn't know about "cat" or "sat" when predicting the blank. An RNN does — it carries context forward through time.

```
Regular NN:   [Input] → [Hidden Layer] → [Output]   (no memory)

RNN:          [Input₁] → [Hidden₁] → [Output₁]
                              ↓
              [Input₂] → [Hidden₂] → [Output₂]
                              ↓
              [Input₃] → [Hidden₃] → [Output₃]
```

---

## How RNN Works — Step by Step

Think of RNN as a loop that processes one element of a sequence at a time.

```
Step 1:   Word "I"     → RNN → Hidden State h₁
Step 2:   Word "love"  → RNN (+ h₁) → Hidden State h₂
Step 3:   Word "cats"  → RNN (+ h₂) → Hidden State h₃ → Output
```

At each step:
1. The RNN receives the **current input** (e.g., a word)
2. It combines it with the **previous hidden state** (memory)
3. It produces a **new hidden state** and optionally an **output**

The same RNN "cell" (with the same weights) is **reused at every step** — this is called **weight sharing**.

```
┌─────────────────────────────────────┐
│                                     │
│   h(t-1) ──┐                        │
│             ▼                        │
│   x(t) ──► [RNN Cell] ──► h(t) ──► output(t)
│                              │       │
│                              └───────┘ (fed back next step)
│                                     │
└─────────────────────────────────────┘
```

---

## The Hidden State — RNN's Memory

The **hidden state** `h(t)` is the core of an RNN. It acts as the network's memory.

- At time step `t`, the hidden state `h(t)` depends on:
  - The current input `x(t)`
  - The previous hidden state `h(t-1)`

This is why RNNs are called **recurrent** — the output loops back as input.

```python
# Conceptually, what happens at each time step:
h_t = tanh(W_hh * h_prev + W_xh * x_t + bias)
```

Where:
- `W_hh` = weight matrix for hidden-to-hidden connection
- `W_xh` = weight matrix for input-to-hidden connection
- `tanh` = activation function (squishes values between -1 and 1)

---

## Math Behind RNN

### Forward Pass

At each time step `t`:

```
h(t) = tanh(W_hh · h(t-1) + W_xh · x(t) + b_h)
y(t) = W_hy · h(t) + b_y
```

Where:
- `x(t)` → input at time t
- `h(t)` → hidden state at time t
- `y(t)` → output at time t
- `W_hh`, `W_xh`, `W_hy` → learnable weight matrices
- `b_h`, `b_y` → bias vectors
- `tanh` → activation (keeps values in range [-1, 1])

### Loss Function

For sequence tasks, we sum the loss across all time steps:

```
L = Σ L(t)   for t = 1 to T
```

### Backpropagation Through Time (BPTT)

RNNs are trained using **BPTT** — a variant of backpropagation that unfolds the network through time and computes gradients at each step.

---

## Types of RNN Architectures

RNNs are incredibly flexible. Here are the main architectures:

```
1. ONE-TO-ONE          2. ONE-TO-MANY         3. MANY-TO-ONE
   (Standard NN)          (Image Captioning)     (Sentiment Analysis)

   [x] → [y]             [x] → [y1][y2][y3]    [x1][x2][x3] → [y]


4. MANY-TO-MANY (same)   5. MANY-TO-MANY (diff)
   (Video frame labels)     (Machine Translation)

   [x1][x2][x3]            [x1][x2] → [y1][y2][y3]
   [y1][y2][y3]             (Encoder → Decoder)
```

| Architecture | Description | Example Use Case |
|---|---|---|
| One-to-One | Fixed input → Fixed output | Basic classification |
| One-to-Many | Single input → Sequence output | Image captioning |
| Many-to-One | Sequence input → Single output | Sentiment analysis |
| Many-to-Many (synced) | Sequence → Sequence (same length) | Video labeling |
| Many-to-Many (async) | Sequence → Sequence (diff length) | Machine translation |

---

## The Vanishing Gradient Problem

This is the **biggest weakness** of vanilla RNNs.

### What happens?

During BPTT, gradients are multiplied repeatedly as we go backward through time steps. If these gradients are small (< 1), they **shrink exponentially** — becoming so tiny the network stops learning from distant past.

```
Gradient at step 100 = gradient_current × W^100

If W = 0.9:  0.9^100 ≈ 0.000027  (nearly zero! 😱)
If W = 1.1:  1.1^100 ≈ 13780     (explodes! 💥)
```

### What does this mean in practice?

```
Sentence: "The cats that lived in the barn ... [50 words later] ... were happy."

RNN struggles to connect "cats" (plural) with "were" (not "was")
because the gradient from 50 steps back has vanished.
```

### Solutions:
- **LSTM** — Solves vanishing gradient with gates
- **GRU** — Simpler gating mechanism
- **Gradient clipping** — Prevents exploding gradients
- **Truncated BPTT** — Only backprop through last N steps

---

## LSTM — Long Short-Term Memory

LSTM was invented in 1997 by Hochreiter & Schmidhuber to solve the vanishing gradient problem.

### The Key Idea: Two Separate Memory Lines

```
Regular RNN:  One memory stream  →  h(t)
LSTM:         Two memory streams →  h(t) (short-term) + c(t) (long-term)
```

`c(t)` is the **cell state** — a "conveyor belt" of long-term memory that runs through the entire sequence with minimal modification.

### LSTM Gates (The Control System)

LSTM uses **3 gates** to control information flow:

```
┌──────────────────────────────────────────────────────┐
│                    LSTM CELL                         │
│                                                      │
│  h(t-1), x(t)                                        │
│       │                                              │
│       ├──► [Forget Gate f]  → What to forget from c  │
│       ├──► [Input Gate i]   → What new info to add   │
│       ├──► [Cell Gate g]    → New candidate values   │
│       └──► [Output Gate o]  → What to output         │
│                                                      │
│  c(t) = f * c(t-1) + i * g   ← Updated cell state   │
│  h(t) = o * tanh(c(t))       ← New hidden state      │
└──────────────────────────────────────────────────────┘
```

### Gate Formulas

```python
# All gates take same inputs: previous hidden state + current input
f_t = sigmoid(W_f · [h(t-1), x(t)] + b_f)   # Forget gate: 0=forget, 1=keep
i_t = sigmoid(W_i · [h(t-1), x(t)] + b_i)   # Input gate: what to update
g_t = tanh(W_g · [h(t-1), x(t)] + b_g)      # Candidate: new values
o_t = sigmoid(W_o · [h(t-1), x(t)] + b_o)   # Output gate: what to expose

c_t = f_t * c(t-1) + i_t * g_t              # Update cell state
h_t = o_t * tanh(c_t)                        # Compute hidden state
```

### Intuitive Explanation

Imagine you're summarizing a book chapter by chapter:
- **Forget gate**: "I'll forget minor character names from chapter 1"
- **Input gate**: "This new character in chapter 5 is important, I'll remember them"
- **Cell state**: Your running summary that persists
- **Output gate**: "For this question, I'll only use info about the main plot"

---

## GRU — Gated Recurrent Unit

GRU (2014) is a simplified version of LSTM with only **2 gates** and **no separate cell state**.

```
LSTM: 3 gates + cell state + hidden state  (more parameters, slower)
GRU:  2 gates + hidden state only          (fewer parameters, faster)
```

### GRU Gates

```
Reset Gate (r):  How much of past to forget when computing new memory
Update Gate (z): How much of old memory to keep vs new memory
```

```python
r_t = sigmoid(W_r · [h(t-1), x(t)])          # Reset gate
z_t = sigmoid(W_z · [h(t-1), x(t)])          # Update gate
h̃_t = tanh(W_h · [r_t * h(t-1), x(t)])      # Candidate hidden state
h_t = (1 - z_t) * h(t-1) + z_t * h̃_t        # Final hidden state
```

The update gate `z_t` works like a **blending knob**:
- `z = 0` → use entirely old memory (skip new input)
- `z = 1` → use entirely new candidate (full update)

---

## LSTM vs GRU vs RNN — Comparison

| Feature | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Gates | None | 3 (forget, input, output) | 2 (reset, update) |
| Memory | Hidden state only | Hidden + cell state | Hidden state only |
| Long-term memory | ❌ Poor | ✅ Excellent | ✅ Good |
| Speed | ✅ Fast | ❌ Slow | ✅ Faster than LSTM |
| Parameters | Few | Many | Moderate |
| Vanishing gradient | ❌ Suffers | ✅ Solved | ✅ Mostly solved |
| Best for | Short sequences | Long sequences | Medium sequences |

> 🎯 **Rule of thumb:** Try GRU first. If accuracy isn't enough, switch to LSTM.

---

## Code Examples

### Setup

```bash
pip install torch numpy matplotlib
```

---

### 1. Simple RNN from Scratch (NumPy)

Understanding RNN at its core — no libraries, pure math.

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01   # input → hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden → hidden
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01  # hidden → output
        self.b_h = np.zeros((hidden_size, 1))   # hidden bias
        self.b_y = np.zeros((output_size, 1))   # output bias
        
        self.hidden_size = hidden_size
    
    def forward(self, inputs):
        """
        inputs: list of input vectors, each shape (input_size, 1)
        Returns: list of outputs and hidden states
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state = zeros
        
        self.hidden_states = {-1: h}  # Store for backprop
        self.inputs = inputs
        outputs = []
        
        for t, x in enumerate(inputs):
            # Core RNN equation
            h = np.tanh(
                self.W_xh @ x +        # input contribution
                self.W_hh @ h +        # memory contribution
                self.b_h               # bias
            )
            
            # Compute output
            y = self.W_hy @ h + self.b_y
            
            self.hidden_states[t] = h
            outputs.append(y)
        
        return outputs, h
    
    def predict(self, inputs):
        outputs, _ = self.forward(inputs)
        return outputs


# --- Demo ---
np.random.seed(42)

input_size  = 3   # e.g., 3 features per time step
hidden_size = 5   # memory size
output_size = 2   # e.g., binary classification

rnn = SimpleRNN(input_size, hidden_size, output_size)

# Create a fake sequence of 4 time steps
sequence = [np.random.randn(input_size, 1) for _ in range(4)]

outputs, final_hidden = rnn.predict(sequence)

print("=== Simple RNN from Scratch ===")
print(f"Sequence length: {len(sequence)}")
print(f"Output at each step: {len(outputs)} outputs")
print(f"Output shape at step 0: {outputs[0].shape}")
print(f"Final hidden state shape: {final_hidden.shape}")
print(f"Output at last step:\n{outputs[-1]}")
```

**Output:**
```
=== Simple RNN from Scratch ===
Sequence length: 4
Output at each step: 4 outputs
Output shape at step 0: (2, 1)
Final hidden state shape: (5, 1)
Output at last step:
[[-0.00316]
 [ 0.00124]]
```

---

### 2. RNN with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Model Definition ───────────────────────────────────────────────────────

class RNNClassifier(nn.Module):
    """
    RNN for sequence classification.
    E.g., given a sequence of sensor readings → classify as normal/anomaly
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        # PyTorch's built-in RNN layer
        # batch_first=True means input shape: (batch, seq_len, features)
        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            nonlinearity = 'tanh',   # 'tanh' or 'relu'
            dropout = 0.2 if num_layers > 1 else 0  # dropout between layers
        )
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # RNN forward pass
        # out shape:    (batch_size, seq_len, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        out, hidden = self.rnn(x, h0)
        
        # Use the LAST time step's output for classification
        # out[:, -1, :] → shape: (batch_size, hidden_size)
        last_output = out[:, -1, :]
        
        # Pass through linear layer
        result = self.fc(last_output)
        return result


# ─── Training Example ────────────────────────────────────────────────────────

# Hyperparameters
INPUT_SIZE  = 10   # features per time step
HIDDEN_SIZE = 64   # RNN memory size
NUM_LAYERS  = 2    # stacked RNN layers
NUM_CLASSES = 3    # number of output classes
SEQ_LEN     = 20   # sequence length
BATCH_SIZE  = 32
EPOCHS      = 5

# Create fake dataset
X = torch.randn(100, SEQ_LEN, INPUT_SIZE)          # 100 samples
y = torch.randint(0, NUM_CLASSES, (100,))           # labels 0, 1, 2

# Model, loss, optimizer
model     = RNNClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("=== RNN with PyTorch ===")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    
    # Mini-batch training
    total_loss = 0
    for i in range(0, len(X), BATCH_SIZE):
        X_batch = X[i:i+BATCH_SIZE]
        y_batch = y[i:i+BATCH_SIZE]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / (len(X) / BATCH_SIZE)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

# Inference
model.eval()
with torch.no_grad():
    sample = torch.randn(1, SEQ_LEN, INPUT_SIZE)   # 1 sample
    output = model(sample)
    predicted_class = output.argmax(dim=1).item()
    print(f"\nPredicted class: {predicted_class}")
```

---

### 3. LSTM for Text Generation

Train a character-level LSTM to generate text — a classic RNN task.

```python
import torch
import torch.nn as nn
import numpy as np

# ─── Data Preparation ────────────────────────────────────────────────────────

text = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
""".lower().strip()

# Build character vocabulary
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size   = len(chars)

print(f"Text length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Vocabulary: {''.join(chars)}")

# Convert text to integers
encoded = [char_to_idx[c] for c in text]

# Create input/target sequences
SEQ_LEN = 30

X_data, y_data = [], []
for i in range(0, len(encoded) - SEQ_LEN):
    X_data.append(encoded[i:i+SEQ_LEN])
    y_data.append(encoded[i+SEQ_LEN])    # predict next character

X_tensor = torch.tensor(X_data, dtype=torch.long)
y_tensor = torch.tensor(y_data, dtype=torch.long)

print(f"\nTraining samples: {len(X_data)}")

# ─── LSTM Model ──────────────────────────────────────────────────────────────

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        
        # Embedding: convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size  = embed_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.3 if num_layers > 1 else 0
        )
        
        # Output layer: predict next character
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)        # → (batch, seq_len, embed_size)
        out, hidden = self.lstm(embedded, hidden)  # → (batch, seq_len, hidden)
        logits = self.fc(out[:, -1, :])     # → (batch, vocab_size)
        return logits, hidden
    
    def generate(self, seed_text, length=100, temperature=1.0):
        """Generate text starting from seed_text"""
        self.eval()
        
        generated = seed_text
        
        # Encode seed
        input_seq = [char_to_idx.get(c, 0) for c in seed_text[-SEQ_LEN:]]
        
        with torch.no_grad():
            for _ in range(length):
                # Pad/trim to SEQ_LEN
                padded = input_seq[-SEQ_LEN:]
                x = torch.tensor([padded], dtype=torch.long)
                
                logits, _ = self.forward(x)
                
                # Apply temperature (higher = more random, lower = more predictable)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()
                
                next_char = idx_to_char[next_char_idx]
                generated += next_char
                input_seq.append(next_char_idx)
        
        return generated


# ─── Training ─────────────────────────────────────────────────────────────────

EMBED_SIZE  = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 0.002

model     = CharLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\n=== Character-Level LSTM ===")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batches = 0
    
    for i in range(0, len(X_tensor), BATCH_SIZE):
        X_batch = X_tensor[i:i+BATCH_SIZE]
        y_batch = y_tensor[i:i+BATCH_SIZE]
        
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    if (epoch + 1) % 2 == 0:
        avg_loss = total_loss / batches
        sample = model.generate("to be or", length=50, temperature=0.8)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.3f} | Sample: '{sample}'")

# Generate text after training
print("\n=== Generated Text ===")
print(model.generate("to be", length=100, temperature=0.7))
```

---

### 4. GRU for Sentiment Analysis

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Fake Sentiment Dataset ───────────────────────────────────────────────────

# In real projects, use datasets like IMDB, SST-2
# Here we simulate tokenized sentences (integer sequences)

VOCAB_SIZE   = 5000
MAX_SEQ_LEN  = 50
EMBED_SIZE   = 100
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
NUM_CLASSES  = 2     # positive / negative
BATCH_SIZE   = 32
EPOCHS       = 5

# Fake data: random token sequences, labels 0 (negative) or 1 (positive)
N_SAMPLES = 500
X = torch.randint(0, VOCAB_SIZE, (N_SAMPLES, MAX_SEQ_LEN))
y = torch.randint(0, NUM_CLASSES, (N_SAMPLES,))

# Train/val split
split = int(0.8 * N_SAMPLES)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ─── GRU Model ───────────────────────────────────────────────────────────────

class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(SentimentGRU, self).__init__()
        
        # Embedding layer (learns word representations)
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_size,
            padding_idx    = 0   # index 0 is treated as padding (no gradient)
        )
        
        # GRU layer
        self.gru = nn.GRU(
            input_size  = embed_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = False,  # Set True for Bidirectional GRU
            dropout = 0.3 if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.5)
        
        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch, seq_len, embed)
        
        # GRU: only need the last hidden state for classification
        _, hidden = self.gru(embedded)   # hidden: (num_layers, batch, hidden)
        
        # Take the last layer's hidden state
        last_hidden = self.dropout(hidden[-1])       # (batch, hidden)
        
        return self.fc(last_hidden)                  # (batch, num_classes)


# ─── Training ─────────────────────────────────────────────────────────────────

model     = SentimentGRU(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

print("=== GRU Sentiment Classifier ===")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for i in range(0, len(X_train), BATCH_SIZE):
        X_b = X_train[i:i+BATCH_SIZE]
        y_b = y_train[i:i+BATCH_SIZE]
        
        output = model(X_b)
        loss = criterion(output, y_b)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    val_acc = evaluate(model, X_val, y_val)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Val Acc: {val_acc:.2%}")
```

---

### 5. Bidirectional RNN

A **Bidirectional RNN** processes the sequence in **both directions** — forward and backward — giving the model context from both past AND future.

```
Forward:   "The [cat] sat" → context from left
Backward:  "The [cat] sat" ← context from right
Combined:  Full context around each word!
```

```python
import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    Processes sequence in both directions.
    Great for: NER, POS tagging, sentiment — where full context matters.
    NOT suitable for: text generation (can't look into the future!).
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,   # ← This is the key setting
            dropout       = 0.2 if num_layers > 1 else 0
        )
        
        # Hidden size is doubled because forward + backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # out: (batch, seq_len, hidden*2) — concatenated forward+backward
        out, (hidden, cell) = self.lstm(x)
        
        # Concatenate last forward and last backward hidden states
        # hidden: (num_layers*2, batch, hidden)
        # We want the last layer's forward and backward:
        forward_hidden  = hidden[-2]   # forward direction, last layer
        backward_hidden = hidden[-1]   # backward direction, last layer
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.fc(self.dropout(combined))


# Demo
INPUT_SIZE  = 20
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
NUM_CLASSES = 5
SEQ_LEN     = 15
BATCH_SIZE  = 8

model = BidirectionalLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
output = model(x)

print("=== Bidirectional LSTM ===")
print(f"Input:  {x.shape}      → (batch, seq_len, features)")
print(f"Output: {output.shape}  → (batch, num_classes)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

### 6. Stacked / Deep RNN

Stack multiple RNN layers on top of each other for learning more complex patterns.

```python
import torch
import torch.nn as nn

class DeepLSTM(nn.Module):
    """
    Multiple LSTM layers stacked vertically.
    Each layer learns higher-level abstractions.
    
    Layer 1: learns low-level patterns (character combinations, short phrases)
    Layer 2: learns mid-level patterns (grammar, word groups)
    Layer 3: learns high-level patterns (semantics, topic)
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(DeepLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,   # stacking happens here!
            batch_first = True,
            dropout     = dropout       # applied between layers (not after last)
        )
        
        self.norm = nn.LayerNorm(hidden_size)   # stabilizes training
        self.fc   = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = self.norm(out[:, -1, :])
        return self.fc(last)


# Compare shallow vs deep
for n_layers in [1, 2, 3, 4]:
    model = DeepLSTM(
        input_size=32, hidden_size=128, 
        num_layers=n_layers, output_size=10
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Layers: {n_layers} | Parameters: {params:,}")

# Output:
# Layers: 1 | Parameters:   85,642
# Layers: 2 | Parameters:  151,178
# Layers: 3 | Parameters:  216,714
# Layers: 4 | Parameters:  282,250
```

---

### 7. Sequence-to-Sequence (Encoder-Decoder)

The architecture behind machine translation, summarization, and chatbots.

```
"Je suis étudiant"  →  [Encoder]  →  [Context Vector]  →  [Decoder]  →  "I am a student"
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Reads the input sequence and compresses it into a context vector."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))    # (batch, src_len, embed)
        outputs, (hidden, cell) = self.lstm(embedded)
        # hidden & cell carry the compressed meaning of the input
        return outputs, hidden, cell


class Decoder(nn.Module):
    """Generates the output sequence one token at a time."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg_token, hidden, cell):
        # trg_token: (batch,) → add seq dimension
        trg = trg_token.unsqueeze(1)                    # (batch, 1)
        embedded = self.dropout(self.embedding(trg))    # (batch, 1, embed)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))         # (batch, vocab_size)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Complete encoder-decoder model."""
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: source sequence (e.g., French sentence)
        trg: target sequence (e.g., English sentence)
        teacher_forcing_ratio: how often to use real vs predicted token
        """
        batch_size = src.size(0)
        trg_len    = trg.size(1)
        trg_vocab  = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab)
        
        # Encode the source sequence
        _, hidden, cell = self.encoder(src)
        
        # First decoder input: <start> token (assumed to be index 1)
        dec_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t] = output
            
            # Teacher forcing: sometimes use real target, sometimes use prediction
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = trg[:, t] if use_teacher else output.argmax(1)
        
        return outputs


# ─── Demo ────────────────────────────────────────────────────────────────────

SRC_VOCAB  = 3000
TRG_VOCAB  = 3000
EMBED_SIZE = 128
HIDDEN     = 256
LAYERS     = 2

encoder = Encoder(SRC_VOCAB, EMBED_SIZE, HIDDEN, LAYERS)
decoder = Decoder(TRG_VOCAB, EMBED_SIZE, HIDDEN, LAYERS)
model   = Seq2Seq(encoder, decoder)

total_params = sum(p.numel() for p in model.parameters())

# Fake translation batch
src = torch.randint(1, SRC_VOCAB, (4, 10))   # batch=4, src_len=10
trg = torch.randint(1, TRG_VOCAB, (4, 8))    # batch=4, trg_len=8

output = model(src, trg)

print("=== Sequence-to-Sequence Model ===")
print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
print(f"Total params:   {total_params:,}")
print(f"Input:  {src.shape}  → (batch, src_len)")
print(f"Target: {trg.shape}   → (batch, trg_len)")
print(f"Output: {output.shape} → (batch, trg_len, vocab_size)")
```

---

## Training Tips & Best Practices

### 1. Gradient Clipping (Essential!)
```python
# Always clip gradients to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Do this AFTER loss.backward() and BEFORE optimizer.step()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

### 2. Proper Weight Initialization
```python
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param)   # Good for RNNs
        elif 'bias' in name:
            nn.init.zeros_(param)

model.apply(init_weights)
```

### 3. Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Call after each epoch with validation loss
scheduler.step(val_loss)
```

### 4. Truncated BPTT (for very long sequences)
```python
# Instead of backpropping through the entire sequence
# backprop through only the last K steps

TRUNCATE_LEN = 35   # backprop through last 35 steps

# Detach hidden state to stop gradient flow
hidden = hidden.detach()
```

### 5. Batch Size & Sequence Length
```python
# Larger batch → more stable gradients but more memory
# Longer sequences → more context but slower training

# Good starting values:
BATCH_SIZE = 32
SEQ_LEN    = 50   # for text; adjust for your task
```

### 6. Dropout for Regularization
```python
# Dropout BETWEEN layers (not after the last layer)
self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3,
                    dropout=0.3)   # 30% dropout between layer 1→2, 2→3

# Add explicit dropout on embeddings too
self.embedding_dropout = nn.Dropout(0.2)
```

---

## Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Not clipping gradients | Exploding gradients, NaN loss | `clip_grad_norm_(params, 1.0)` |
| Forgetting `detach()` on hidden state | Memory leak across batches | `hidden = hidden.detach()` |
| Wrong input shape | Shape mismatch errors | Use `batch_first=True` consistently |
| Using RNN for long-range dependencies | Vanishing gradient | Use LSTM or GRU |
| Not shuffling training data | Biased learning | `DataLoader(..., shuffle=True)` |
| Too many layers on small data | Overfitting | Start with 1-2 layers |
| Using vanilla RNN for text generation | Poor quality output | Use LSTM/GRU |
| Same learning rate throughout | Suboptimal convergence | Use LR scheduler |

---

## Real-World Applications

| Application | Architecture | Input | Output |
|---|---|---|---|
| Speech Recognition | Bidirectional LSTM | Audio frames | Text |
| Machine Translation | Seq2Seq + Attention | Text | Text |
| Text Generation | LSTM / GRU | Characters/words | Next character/word |
| Sentiment Analysis | GRU + Embedding | Sentence | Positive/Negative |
| Stock Price Prediction | LSTM | Price history | Future price |
| Music Generation | LSTM | Note sequences | Next note |
| Named Entity Recognition | Bidirectional LSTM | Words | Entity labels |
| Video Captioning | CNN + LSTM | Video frames | Caption |
| Anomaly Detection | LSTM Autoencoder | Sensor data | Normal/Anomaly |
| Handwriting Recognition | RNN + CTC | Pen strokes | Text |

---

## When NOT to Use RNN

Modern alternatives often outperform RNNs:

| Task | Better Alternative | Why |
|---|---|---|
| Long document understanding | **Transformer** | Global attention, no vanishing gradient |
| Image classification | **CNN** | Spatial not sequential |
| Short text classification | **BERT / DistilBERT** | Pre-trained, transfers well |
| Time series forecasting | **Temporal Fusion Transformer** | Handles multiple horizons |
| Real-time inference | **Simpler ML models** | RNNs can be slow |

> 💡 **Note:** Transformers have largely replaced RNNs for NLP tasks (2018+), but RNNs are still useful for: resource-constrained environments, real-time streaming tasks, and when you need an interpretable sequential model.

---

## Resources

### Papers
- [Long Short-Term Memory (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) — Original LSTM paper
- [Empirical Evaluation of Gated Recurrent Neural Networks (2014)](https://arxiv.org/abs/1412.3555) — GRU paper
- [Sequence to Sequence Learning (2014)](https://arxiv.org/abs/1409.3215) — Encoder-Decoder

### Tutorials
- [PyTorch RNN Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [The Unreasonable Effectiveness of RNNs — Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks — Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Books
- *Deep Learning* — Goodfellow, Bengio, Courville (Chapter 10)
- *Hands-On Machine Learning* — Aurélien Géron (Chapter 15)

---

## 📁 Repository Structure

```
rnn-deep-learning/
├── README.md                    ← You are here
├── requirements.txt
├── 01_rnn_from_scratch.py       ← NumPy RNN implementation
├── 02_rnn_pytorch.py            ← PyTorch RNN classifier
├── 03_lstm_text_generation.py   ← Character-level LSTM
├── 04_gru_sentiment.py          ← GRU sentiment analysis
├── 05_bidirectional_rnn.py      ← BiLSTM
├── 06_stacked_rnn.py            ← Deep stacked LSTM
├── 07_seq2seq.py                ← Encoder-Decoder
└── utils/
    ├── data_utils.py
    └── train_utils.py
```

---

-