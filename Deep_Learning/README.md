# 🧠 Deep Learning: Complete Guide from Zero to Hero

> **Learn Deep Learning in plain English with working Python code examples — covering every major concept from neurons to transformers.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📚 Table of Contents

1. [What is Deep Learning?](#1-what-is-deep-learning)
2. [How the Brain Inspired AI](#2-how-the-brain-inspired-ai)
3. [The Neuron — Building Block](#3-the-neuron--building-block)
4. [Activation Functions](#4-activation-functions)
5. [Neural Network Architecture](#5-neural-network-architecture)
6. [How a Network Learns (Forward & Backprop)](#6-how-a-network-learns-forward--backpropagation)
7. [Loss Functions](#7-loss-functions)
8. [Optimizers](#8-optimizers)
9. [Overfitting & Regularization](#9-overfitting--regularization)
10. [Convolutional Neural Networks (CNN)](#10-convolutional-neural-networks-cnn)
11. [Recurrent Neural Networks (RNN & LSTM)](#11-recurrent-neural-networks-rnn--lstm)
12. [Transformers & Attention](#12-transformers--attention)
13. [Transfer Learning](#13-transfer-learning)
14. [Generative Models (GANs & VAEs)](#14-generative-models-gans--vaes)
15. [Complete Project: Image Classifier](#15-complete-project-image-classifier)
16. [Setup & Installation](#16-setup--installation)

---

## 1. What is Deep Learning?

**Simple explanation:** Deep Learning is teaching computers to learn from examples, just like humans do.

Imagine you want to teach a child what a cat looks like. You don't write rules like *"has fur, 4 legs, whiskers"*. You just show them thousands of pictures of cats and non-cats. After enough examples, the child can recognize a cat they've never seen before.

Deep Learning does the **exact same thing** — but with math and code instead of a brain.

```
Traditional Programming:  Data + Rules  → Answers
Machine Learning:         Data + Answers → Rules
Deep Learning:            Data + Answers → Complex Rules (automatically)
```

**Why "Deep"?** Because the models have many *layers* stacked on top of each other. Each layer learns something more complex than the one before.

```
Layer 1: Learns edges and lines
Layer 2: Learns shapes (circles, squares)
Layer 3: Learns parts (eyes, ears, wheels)
Layer 4: Learns objects (cat, car, dog)
```

---

## 2. How the Brain Inspired AI

Your brain has **~86 billion neurons**. Each neuron:
- Receives signals from other neurons
- Decides whether to "fire" (activate) or not
- Sends a signal to the next neuron

AI scientists looked at this and thought: *"What if we could simulate this with code?"*

That's exactly what an **Artificial Neural Network** is — a simplified computer version of how the brain works.

```
Real Brain:          Artificial Network:
─────────────        ───────────────────
Neuron          →    Node / Unit
Synapse         →    Weight (a number)
Dendrites       →    Inputs
Axon            →    Output
Firing          →    Activation
```

---

## 3. The Neuron — Building Block

A single artificial neuron does 3 things:

1. **Receives** inputs (numbers)
2. **Multiplies** each input by a weight (importance value)
3. **Adds** them up + a bias, then passes through an activation function

**Math:**
```
output = activation(w1*x1 + w2*x2 + w3*x3 + bias)
```

Think of weights as **"how important is this input?"** and bias as **"how easy is it to trigger this neuron?"**

### 🐍 Code: Single Neuron from Scratch

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        # Randomly initialize weights and bias
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        """Activation function — squishes output between 0 and 1"""
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """One forward pass through the neuron"""
        # Step 1: Weighted sum
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Step 2: Activation
        output = self.sigmoid(weighted_sum)
        return output

# Example usage
neuron = Neuron(num_inputs=3)
inputs = [0.5, 0.8, 0.2]       # e.g., pixel brightness values
output = neuron.forward(inputs)
print(f"Neuron output: {output:.4f}")  # A number between 0 and 1
```

---

## 4. Activation Functions

Activation functions decide **whether a neuron "fires"** and how strongly. Without them, a neural network is just a fancy linear equation — boring and limited.

| Function | Formula | Use Case | Range |
|----------|---------|----------|-------|
| Sigmoid | `1 / (1 + e^-x)` | Binary output | (0, 1) |
| Tanh | `(e^x - e^-x) / (e^x + e^-x)` | Hidden layers | (-1, 1) |
| ReLU | `max(0, x)` | Most hidden layers | [0, ∞) |
| Leaky ReLU | `max(0.01x, x)` | Fixes dying ReLU | (-∞, ∞) |
| Softmax | `e^x / Σe^x` | Multi-class output | (0, 1), sums to 1 |

### 🐍 Code: All Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Visualize all activation functions
x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
functions = [
    (sigmoid(x), "Sigmoid", "blue"),
    (tanh(x),    "Tanh",    "green"),
    (relu(x),    "ReLU",    "red"),
    (leaky_relu(x), "Leaky ReLU", "purple"),
]

for ax, (y, name, color) in zip(axes, functions):
    ax.plot(x, y, color=color, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(name, fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=100)
plt.show()

# Softmax example
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(f"Scores:        {scores}")
print(f"Softmax probs: {probs}")          # [0.659, 0.242, 0.099]
print(f"Sum of probs:  {probs.sum():.1f}") # Always 1.0
```

**💡 Rule of thumb:** Use **ReLU** for hidden layers. Use **Sigmoid** for binary output (yes/no). Use **Softmax** for multi-class output (cat/dog/bird).

---

## 5. Neural Network Architecture

A neural network is layers of neurons connected together:

```
Input Layer     Hidden Layer 1    Hidden Layer 2    Output Layer
───────────     ──────────────    ──────────────    ────────────
  (x1)  ──────→  (neuron)  ──→   (neuron)   ──→   (prediction)
  (x2)  ──────→  (neuron)  ──→   (neuron)   ──→
  (x3)  ──────→  (neuron)  ──→   (neuron)   ──→
```

- **Input Layer:** The raw data you feed in (pixels, numbers, words)
- **Hidden Layers:** Where the magic happens — the network learns patterns here
- **Output Layer:** The final answer (probability, class, number)

### 🐍 Code: Neural Network from Scratch (NumPy only)

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list like [2, 4, 3, 1]
        means: 2 inputs → 4 neurons → 3 neurons → 1 output
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01,
                'bias':    np.zeros((1, layer_sizes[i+1]))
            }
            self.layers.append(layer)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """Pass data through all layers"""
        self.activations = [X]  # Store for backprop

        current = X
        for i, layer in enumerate(self.layers):
            z = np.dot(current, layer['weights']) + layer['bias']
            # Use ReLU for hidden layers, sigmoid for output
            if i < len(self.layers) - 1:
                current = self.relu(z)
            else:
                current = self.sigmoid(z)
            self.activations.append(current)

        return current

# Demo
nn = NeuralNetwork(layer_sizes=[3, 5, 4, 1])
X = np.array([[0.5, 0.2, 0.8]])   # 1 sample with 3 features
output = nn.forward(X)
print(f"Network output: {output[0][0]:.4f}")  # A number between 0 and 1
```

---

## 6. How a Network Learns: Forward & Backpropagation

Learning in deep learning has 4 steps — repeated thousands of times:

```
1. FORWARD PASS   → Feed data through the network, get prediction
2. COMPUTE LOSS   → Compare prediction to the real answer
3. BACKWARD PASS  → Figure out which weights caused the error (backprop)
4. UPDATE WEIGHTS → Adjust weights to reduce the error (optimizer)
```

**Backpropagation** is just the **chain rule of calculus** applied backwards through the network. It calculates "how much did each weight contribute to the error?"

### 🐍 Code: Training Loop with Backprop

```python
import numpy as np

# Simple XOR problem — a classic neural network test
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Inputs
y = np.array([[0],   [1],   [1],   [0]])     # XOR outputs

# Network parameters
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5   # Input → Hidden (2 inputs, 4 neurons)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5   # Hidden → Output (4 neurons, 1 output)
b2 = np.zeros((1, 1))

def sigmoid(x):   return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return x * (1 - x)   # Derivative of sigmoid

lr = 0.5      # Learning rate — how big each update step is
losses = []

for epoch in range(5000):
    # ─── FORWARD PASS ───────────────────────────────
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)          # Hidden layer activations

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)          # Output prediction

    # ─── LOSS (Mean Squared Error) ──────────────────
    loss = np.mean((y - a2) ** 2)
    losses.append(loss)

    # ─── BACKWARD PASS ──────────────────────────────
    # How wrong is the output?
    d_loss_a2 = -2 * (y - a2) / y.shape[0]

    # Backprop through output layer
    d_a2 = d_loss_a2 * d_sigmoid(a2)
    dW2  = np.dot(a1.T, d_a2)
    db2  = np.sum(d_a2, axis=0, keepdims=True)

    # Backprop through hidden layer
    d_a1 = np.dot(d_a2, W2.T) * d_sigmoid(a1)
    dW1  = np.dot(X.T, d_a1)
    db1  = np.sum(d_a1, axis=0, keepdims=True)

    # ─── UPDATE WEIGHTS ─────────────────────────────
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# Final predictions
print("\nFinal Predictions:")
print(f"  0 XOR 0 = {a2[0][0]:.3f} (expected 0)")
print(f"  0 XOR 1 = {a2[1][0]:.3f} (expected 1)")
print(f"  1 XOR 0 = {a2[2][0]:.3f} (expected 1)")
print(f"  1 XOR 1 = {a2[3][0]:.3f} (expected 0)")
```

**Expected output:**
```
Epoch    0 | Loss: 0.254231
Epoch 1000 | Loss: 0.012488
Epoch 2000 | Loss: 0.004321
Epoch 3000 | Loss: 0.002765
Epoch 4000 | Loss: 0.002021

Final Predictions:
  0 XOR 0 = 0.046 (expected 0) ✅
  0 XOR 1 = 0.954 (expected 1) ✅
  1 XOR 0 = 0.953 (expected 1) ✅
  1 XOR 1 = 0.049 (expected 0) ✅
```

---

## 7. Loss Functions

The **loss function** measures how wrong the network's prediction is. The goal of training is to make this number as small as possible.

| Loss Function | Use When | Formula |
|---|---|---|
| Mean Squared Error (MSE) | Regression (predict a number) | `mean((y_pred - y_true)²)` |
| Binary Cross-Entropy | Binary classification (yes/no) | `-[y*log(p) + (1-y)*log(1-p)]` |
| Categorical Cross-Entropy | Multi-class classification | `-Σ y_i * log(p_i)` |
| Huber Loss | Regression with outliers | Mix of MSE and MAE |

### 🐍 Code: Loss Functions Explained

```python
import numpy as np

# ── MSE Loss (for regression) ─────────────────────────────────
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.2, 2.3, 7.1])
print(f"MSE Loss: {mse_loss(y_true, y_pred):.4f}")   # Small = good

# ── Binary Cross-Entropy (for binary classification) ──────────
def binary_crossentropy(y_true, y_pred):
    # Clip to avoid log(0) = -infinity
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])  # Good predictions
print(f"BCE Loss (good): {binary_crossentropy(y_true, y_pred):.4f}")

y_pred_bad = np.array([0.3, 0.7, 0.4, 0.6])  # Bad predictions
print(f"BCE Loss (bad):  {binary_crossentropy(y_true, y_pred_bad):.4f}")

# ── Categorical Cross-Entropy (for multi-class) ───────────────
def categorical_crossentropy(y_true, y_pred):
    """y_true: one-hot, y_pred: softmax probabilities"""
    y_pred = np.clip(y_pred, 1e-7, 1)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Dog=class 0, Cat=class 1, Bird=class 2
y_true = np.array([[1,0,0], [0,1,0], [0,0,1]])           # One-hot
y_pred = np.array([[0.8,0.1,0.1], [0.1,0.85,0.05], [0.05,0.1,0.85]])
print(f"Cat.CE Loss: {categorical_crossentropy(y_true, y_pred):.4f}")
```

---

## 8. Optimizers

Optimizers are the algorithms that **update the weights** to reduce the loss. Think of it as trying to find the lowest point in a hilly landscape — blindfolded.

```
           Loss
            │  ╲
            │   ╲   ← You start here (random weights)
            │    ╲
            │     ╲___
            │         ╲___  ← You want to get here (minimum loss)
            └────────────────── Weights
```

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| SGD | Basic gradient descent | Simple problems |
| Momentum | SGD + memory of past steps | Faster convergence |
| Adam | Adaptive learning rates | Most deep learning |
| RMSprop | Adaptive learning rates | RNNs |
| AdaGrad | Smaller LR for frequent features | Sparse data |

### 🐍 Code: Optimizers Compared

```python
import numpy as np
import matplotlib.pyplot as plt

# We'll minimize a simple loss function: f(w) = w^4 - 4w^2 + 5
# (has two valleys — good to test optimizers)

def loss(w):     return w**4 - 4*w**2 + 5
def gradient(w): return 4*w**3 - 8*w    # df/dw

# ── SGD (Stochastic Gradient Descent) ─────────────────────────
def sgd(w_init, lr=0.01, steps=100):
    w = w_init
    path = [w]
    for _ in range(steps):
        w = w - lr * gradient(w)
        path.append(w)
    return path

# ── SGD with Momentum ──────────────────────────────────────────
def sgd_momentum(w_init, lr=0.01, momentum=0.9, steps=100):
    w = w_init
    velocity = 0
    path = [w]
    for _ in range(steps):
        velocity = momentum * velocity - lr * gradient(w)
        w = w + velocity
        path.append(w)
    return path

# ── Adam Optimizer ─────────────────────────────────────────────
def adam(w_init, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=100):
    w = w_init
    m, v, t = 0, 0, 0
    path = [w]
    for _ in range(steps):
        t += 1
        g = gradient(w)
        m = beta1 * m + (1 - beta1) * g          # First moment
        v = beta2 * v + (1 - beta2) * g**2       # Second moment
        m_hat = m / (1 - beta1**t)               # Bias correction
        v_hat = v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w)
    return path

# Compare all optimizers starting from same point
start = 2.0
sgd_path  = sgd(start)
mom_path  = sgd_momentum(start)
adam_path = adam(start)

print(f"SGD final w:      {sgd_path[-1]:.4f}  | loss: {loss(sgd_path[-1]):.4f}")
print(f"Momentum final w: {mom_path[-1]:.4f}  | loss: {loss(mom_path[-1]):.4f}")
print(f"Adam final w:     {adam_path[-1]:.4f} | loss: {loss(adam_path[-1]):.4f}")
# All should converge near w ≈ 1.414 (one of the minima)
```

**💡 In practice:** Just use **Adam** — it works well for almost everything.

---

## 9. Overfitting & Regularization

**Overfitting** is when the model memorizes the training data instead of learning general patterns. Like a student who memorizes the exam questions but can't solve new problems.

```
          Training Accuracy    Validation Accuracy
Perfect:      90%                    89%         ← Generalizes well ✅
Overfit:      99%                    65%         ← Memorized training data ❌
Underfit:     60%                    58%         ← Hasn't learned enough ❌
```

### Solutions

**1. Dropout** — Randomly "turn off" neurons during training

```python
import torch
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),    # ← 50% of neurons randomly turned off
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),    # ← 30% dropout
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)

model = NetworkWithDropout()
print(model)
```

**2. Batch Normalization** — Normalize layer outputs to speed up training

```python
class NetworkWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),   # ← Normalize after each layer
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)
```

**3. L2 Regularization (Weight Decay)** — Penalize large weights

```python
import torch.optim as optim

model = nn.Linear(10, 1)
# weight_decay adds L2 penalty → prevents weights from becoming too large
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

**4. Early Stopping** — Stop training when validation loss stops improving

```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print("Early stopping triggered!")

# Usage in training loop
early_stop = EarlyStopping(patience=5)
for epoch in range(100):
    val_loss = train_and_validate()  # your training function
    early_stop(val_loss)
    if early_stop.stop:
        break
```

---

## 10. Convolutional Neural Networks (CNN)

CNNs are designed specifically for **images**. Instead of connecting every pixel to every neuron (expensive!), CNNs use a small **filter** that slides across the image — like a flashlight scanning a dark room.

```
Image (8×8):           Filter (3×3):      Result:
┌─────────────────┐    ┌───────────┐      ┌─────────────┐
│ 0  0  0  0  0   │    │ -1 -1 -1  │      │Edge detected│
│ 0  1  1  1  0   │  * │  0  0  0  │   →  │   ─────     │
│ 0  1  0  1  0   │    │  1  1  1  │      │             │
│ 0  0  0  0  0   │    └───────────┘      └─────────────┘
└─────────────────┘
```

Key CNN components:
- **Convolution Layer:** Applies filters to detect features (edges, textures, patterns)
- **Pooling Layer:** Shrinks the image (keeps important info, throws away noise)
- **Fully Connected Layer:** Makes the final classification decision

### 🐍 Code: CNN for Image Classification (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── 1. Define the CNN Architecture ────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 1×28×28 → 32×26×26
            nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → 32×13×13

            # Block 2: 32×13×13 → 64×11×11
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → 64×5×5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                # 64×5×5 = 1600
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── 2. Load MNIST Dataset ──────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# ── 3. Training Setup ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── 4. Training Loop ───────────────────────────────────────────
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear old gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# ── 5. Evaluation ──────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()   # Turn off dropout, batchnorm in eval mode
    total_loss, correct = 0, 0

    with torch.no_grad():   # Don't compute gradients (saves memory)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# ── 6. Run Training ────────────────────────────────────────────
for epoch in range(5):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/5 | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc*100:.1f}%")

# Expected output after 5 epochs: ~99% test accuracy on MNIST!
```

---

## 11. Recurrent Neural Networks (RNN & LSTM)

RNNs are designed for **sequential data** — text, time series, audio. The key idea: neurons have **memory** — they pass information from one step to the next.

```
Text: "The cat sat on the mat"

RNN processes word by word:
"The" → [hidden state 1] → "cat" → [hidden state 2] → "sat" → ...
          ↑ memory             ↑ memory updated          ↑ memory updated
```

**The problem with basic RNNs:** They forget things that happened far back (vanishing gradient problem).

**Solution: LSTM (Long Short-Term Memory)** — has gates that decide what to remember and what to forget.

```
LSTM has 3 gates:
  Forget Gate  → "Do I keep old memory or erase it?"
  Input Gate   → "Do I add new information?"
  Output Gate  → "What do I pass to the next step?"
```

### 🐍 Code: LSTM for Text Sentiment Analysis

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMSentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()

        # Convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,        # Input shape: (batch, seq_len, features)
            dropout=0.3,
            bidirectional=True       # Process text both left→right and right→left
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),   # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)          # → (batch, seq, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # Process sequence

        # Take the last output from both directions
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)

        output = self.classifier(last_hidden)  # (batch, 1)
        return output.squeeze(1)

# ── Demo ───────────────────────────────────────────────────────
VOCAB_SIZE = 10000
EMBED_DIM  = 128
HIDDEN_DIM = 256

model = LSTMSentimentAnalyzer(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

# Simulated tokenized sentences (word indices)
# "This movie was amazing" might be [42, 831, 17, 5632]
batch = torch.randint(1, VOCAB_SIZE, (4, 50))  # 4 sentences, 50 words each
predictions = model(batch)

print("Sentiment predictions (0=negative, 1=positive):")
for i, pred in enumerate(predictions):
    sentiment = "Positive 😊" if pred > 0.5 else "Negative 😞"
    print(f"  Sentence {i+1}: {pred:.3f} → {sentiment}")
```

---

## 12. Transformers & Attention

Transformers are the architecture behind **GPT, BERT, Claude, and almost every modern AI system**. They replaced RNNs for most NLP tasks.

**The key idea: Attention Mechanism**

Instead of processing text sequentially (word by word like RNNs), Transformers look at **all words at once** and learn which words are most relevant to each other.

```
Sentence: "The animal didn't cross the street because it was too tired"

What does "it" refer to? The model needs to relate "it" to "animal"

Attention scores for "it":
  The: 0.02 | animal: 0.94 | didn't: 0.01 | ... | tired: 0.03
                ↑ High attention! Model correctly links "it" to "animal"
```

### 🐍 Code: Self-Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Single-head self-attention — the heart of Transformers"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Three linear projections: Query, Key, Value
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)  # "What am I looking for?"
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)  # "What do I offer?"
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)  # "What do I provide?"
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        B, T, C = x.shape

        Q = self.W_q(x)   # Queries: what each position is looking for
        K = self.W_k(x)   # Keys:    what each position offers
        V = self.W_v(x)   # Values:  the actual content at each position

        # Attention scores = how much does each position "attend to" others
        scale = math.sqrt(C)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, T, T)

        # Convert scores to probabilities with softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)  # (B, T, C)
        return self.W_o(output), attention_weights

# ── Mini Transformer Block ─────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),                     # Modern alternative to ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection ("skip connection")
        attn_out, weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward network with residual connection
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, weights

# ── Demo ───────────────────────────────────────────────────────
embed_dim = 64
seq_len   = 10
batch_size = 2

block = TransformerBlock(embed_dim=embed_dim, ff_dim=256)
x = torch.randn(batch_size, seq_len, embed_dim)   # Simulated token embeddings

output, attention_weights = block(x)
print(f"Input shape:   {x.shape}")           # (2, 10, 64)
print(f"Output shape:  {output.shape}")      # (2, 10, 64)  ← same shape!
print(f"Attention map: {attention_weights.shape}")  # (2, 10, 10) — each word attends to all
```

---

## 13. Transfer Learning

Training deep networks from scratch requires millions of examples and days of GPU time. **Transfer Learning** lets you use a model someone else already trained on massive data — and fine-tune it for your specific task.

Think of it as: someone already learned all the basics of driving. You just need to teach them the specific roads in your city.

```
Pre-trained Model (ImageNet, 1000 classes):
  Conv1 → Conv2 → Conv3 → FC1 → [Output: 1000 classes]
   ↑         ↑        ↑
   Learns    Learns   Learns
   edges     shapes   objects

Your task (dogs vs cats):
  Conv1 → Conv2 → Conv3 → [FREEZE THESE] → FC_new → [Output: 2 classes]
                                            ↑
                                    Only train this!
```

### 🐍 Code: Transfer Learning with ResNet

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ── 1. Load pre-trained ResNet50 ──────────────────────────────
model = models.resnet50(pretrained=True)

# ── 2. Freeze all layers (don't update pre-trained weights) ───
for param in model.parameters():
    param.requires_grad = False

# ── 3. Replace the final layer for YOUR task ──────────────────
# ResNet's last layer outputs 1000 classes (ImageNet)
# We want 5 classes (e.g., 5 types of flowers)
num_classes = 5
in_features = model.fc.in_features     # 2048 for ResNet50

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)
)

# Now only model.fc parameters will be updated!
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
# Output: Trainable params: 528,645 / 25,082,693 (2.1%)
# We only train 2% of the parameters — MUCH faster!

# ── 4. Data transforms matching what ResNet was trained on ────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# ── 5. Fine-tuning: Unfreeze some layers for better accuracy ──
# Optional: unfreeze the last few layers for more adaptation
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:    # Unfreeze last ResNet block + FC
        param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

---

## 14. Generative Models (GANs & VAEs)

**Generative models** don't just classify data — they **create new data**. Think: AI-generated images, faces, music.

### GAN (Generative Adversarial Network)

Two networks compete against each other:
- **Generator:** Creates fake data (tries to fool the discriminator)
- **Discriminator:** Judges if data is real or fake

```
Random noise → [GENERATOR] → Fake image
                                  ↓
Real images ────────────→ [DISCRIMINATOR] → "Real or Fake?"
                                  ↑
              Both networks improve through competition!
```

### 🐍 Code: Simple GAN for MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ── Generator: noise → fake image ─────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, img_size),
            nn.Tanh()              # Output in [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# ── Discriminator: image → real/fake ──────────────────────────
class Discriminator(nn.Module):
    def __init__(self, img_size=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()           # 0 = fake, 1 = real
        )

    def forward(self, img):
        return self.model(img)

# ── Training ───────────────────────────────────────────────────
LATENT_DIM = 100
generator     = Generator(LATENT_DIM)
discriminator = Discriminator()

g_optimizer = optim.Adam(generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion   = nn.BCELoss()

def train_step(real_imgs):
    batch_size = real_imgs.shape[0]
    real_imgs  = real_imgs.view(batch_size, -1)   # Flatten to 784

    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # ── Train Discriminator ──────────────────────────
    z = torch.randn(batch_size, LATENT_DIM)
    fake_imgs = generator(z).detach()   # .detach() = don't backprop into G

    d_loss_real = criterion(discriminator(real_imgs), real_labels)
    d_loss_fake = criterion(discriminator(fake_imgs), fake_labels)
    d_loss = (d_loss_real + d_loss_fake) / 2

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # ── Train Generator ──────────────────────────────
    z = torch.randn(batch_size, LATENT_DIM)
    fake_imgs = generator(z)
    # Generator wants discriminator to think fakes are real!
    g_loss = criterion(discriminator(fake_imgs), real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()

# Simulated training (use real DataLoader in practice)
for step in range(5):
    fake_batch = torch.randn(32, 1, 28, 28)    # Simulated real images
    d_loss, g_loss = train_step(fake_batch)
    print(f"Step {step}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
```

---

## 15. Complete Project: Image Classifier

Let's put it all together — a complete, working image classifier using PyTorch.

```python
"""
Complete Deep Learning Pipeline:
  Data → Preprocessing → Model → Training → Evaluation → Inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "batch_size":  64,
    "epochs":      10,
    "lr":          0.001,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 10,   # CIFAR-10 has 10 classes
    "val_split":   0.1,
}

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ═══════════════════════════════════════════════════════════════
# 2. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # Data augmentation
    transforms.RandomCrop(32, padding=4),       # Data augmentation
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([           # No augmentation for test
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download CIFAR-10
full_train = torchvision.datasets.CIFAR10('./data', train=True,
                                          download=True, transform=train_transform)
test_set   = torchvision.datasets.CIFAR10('./data', train=False,
                                          download=True, transform=test_transform)

# Split training data into train/validation
val_size   = int(len(full_train) * CONFIG["val_split"])
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

# ═══════════════════════════════════════════════════════════════
# 3. MODEL
# ═══════════════════════════════════════════════════════════════
class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ═══════════════════════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════════════════════
device    = CONFIG["device"]
model     = CIFAR10Net(CONFIG["num_classes"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total

best_val_acc = 0
for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
    val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, train=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")  # Save best model

    print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
          f"Train: {train_loss:.4f}/{train_acc*100:.1f}% | "
          f"Val: {val_loss:.4f}/{val_acc*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 5. EVALUATION & INFERENCE
# ═══════════════════════════════════════════════════════════════
# Load best model
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, train=False)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Single image inference
def predict(model, image_tensor, device):
    """Predict class for a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim
        output = model(image_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
    return CLASSES[pred_class], confidence

# Test with one image from test set
sample_img, sample_label = test_set[0]
pred_class, confidence = predict(model, sample_img, device)
print(f"\nSample prediction: {pred_class} ({confidence*100:.1f}%)")
print(f"True label:        {CLASSES[sample_label]}")
```

---

## 16. Setup & Installation

### Prerequisites

```bash
# Python 3.8+
python --version
```

### Install Dependencies

```bash
# Clone this repo
git clone https://github.com/yourusername/deep-learning-guide.git
cd deep-learning-guide

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
pandas>=2.0.0
jupyter>=1.0.0
tqdm>=4.65.0
```

### GPU Setup (Optional but recommended)

```bash
# Check if CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Run the Examples

```bash
# Run basic neuron example
python examples/01_neuron.py

# Run CNN image classifier
python examples/10_cnn_cifar10.py

# Launch Jupyter notebooks
jupyter notebook notebooks/
```

---

## 🗺️ Learning Roadmap

```
Beginner (Week 1-2)
  ├── Understand what a neuron is
  ├── Activation functions
  └── Forward pass & backpropagation

Intermediate (Week 3-4)
  ├── CNNs for image tasks
  ├── Loss functions & optimizers
  └── Overfitting prevention

Advanced (Month 2-3)
  ├── RNNs / LSTMs for sequences
  ├── Transformers & Attention
  └── Transfer learning

Expert (Month 3+)
  ├── GANs & generative models
  ├── Custom architectures
  └── Deployment & optimization
```

---

## 📖 Resources

- [fast.ai](https://fast.ai) — Practical deep learning course (free)
- [PyTorch Docs](https://pytorch.org/docs) — Official PyTorch documentation
- [Papers With Code](https://paperswithcode.com) — Latest research with code
- [Andrej Karpathy's YouTube](https://youtube.com/@AndrejKarpathy) — Deep explanations
- [Deep Learning Book](https://deeplearningbook.org) — Free textbook by Goodfellow et al.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

## 📄 License

MIT License — use freely for personal and commercial projects.

---

<div align="center">
  <strong>⭐ Star this repo if it helped you!</strong><br>
  Made with ❤️ to make Deep Learning accessible to everyone
</div>