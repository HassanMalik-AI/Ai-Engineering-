# 🧠 Neural Networks in Deep Learning

> A complete, production-ready reference guide covering everything from biological neurons to modern deep learning architectures — with TensorFlow & Keras code examples.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

---

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [The Biological Neuron](#-the-biological-neuron)
3. [Artificial Neuron (Perceptron)](#-artificial-neuron-perceptron)
4. [Weights & Biases](#-weights--biases)
5. [Activation Functions](#-activation-functions)
6. [Neural Network Architecture](#-neural-network-architecture)
7. [Forward Propagation](#-forward-propagation)
8. [Loss Functions](#-loss-functions)
9. [Backpropagation](#-backpropagation)
10. [Gradient Descent & Optimizers](#-gradient-descent--optimizers)
11. [Learning Rate](#-learning-rate)
12. [Regularization Techniques](#-regularization-techniques)
13. [Batch Normalization](#-batch-normalization)
14. [Dropout](#-dropout)
15. [Data Preprocessing & Metrics](#-data-preprocessing--metrics)
16. [Building Models with Keras](#-building-models-with-keras)
17. [Training Pipeline](#-training-pipeline)
18. [Callbacks & Monitoring](#-callbacks--monitoring)
19. [Saving & Loading Models](#-saving--loading-models)
20. [Common Architectures](#-common-architectures)
21. [Hyperparameter Tuning](#-hyperparameter-tuning)
22. [Best Practices](#-best-practices)
23. [Troubleshooting](#-troubleshooting)
24. [Resources](#-resources)

---

## 🌟 Introduction

**Deep Learning** is a subfield of machine learning that uses multi-layered **artificial neural networks** to learn representations from data. Inspired by the structure of the human brain, these networks can automatically discover intricate patterns in large datasets — powering applications like image recognition, natural language processing, speech synthesis, and more.

```
Data → [Input Layer] → [Hidden Layers] → [Output Layer] → Prediction
              ↑              ↑                 ↑
           Neurons        Neurons           Neurons
           Weights        Weights           Weights
           Biases         Biases            Biases
```

### Why Deep Learning?

| Feature | Traditional ML | Deep Learning |
|---------|---------------|---------------|
| Feature Engineering | Manual | Automatic |
| Performance on Big Data | Plateaus | Keeps improving |
| Interpretability | Higher | Lower |
| Compute Required | Low | High |
| Best Use Cases | Tabular data | Images, text, audio |

---

## 🔬 The Biological Neuron

The artificial neural network draws inspiration from the biological neuron in the human brain.

```
                    Dendrites (receive signals)
                         │  │  │
                         ▼  ▼  ▼
               ┌─────────────────────────┐
               │        Cell Body        │
               │   (processes signals)   │
               └──────────┬──────────────┘
                          │
                     Axon (output)
                          │
                          ▼
               ┌─────────────────────────┐
               │   Synapses (weighted    │
               │   connections to next   │
               │   neurons)              │
               └─────────────────────────┘
```

| Biological | Artificial |
|------------|------------|
| Dendrite | Input |
| Synapse strength | Weight |
| Cell body | Weighted sum + activation |
| Axon | Output |
| Firing threshold | Activation function |

---

## ⚡ Artificial Neuron (Perceptron)

The fundamental building block of a neural network.

### Mathematical Definition

```
output = activation( Σ(wᵢ · xᵢ) + b )
```

Where:
- `xᵢ` = inputs
- `wᵢ` = weights
- `b` = bias
- `activation()` = non-linear activation function

```
  x₁ ──(w₁)──┐
  x₂ ──(w₂)──┤
  x₃ ──(w₃)──┼──→ [Σ + b] ──→ [f(z)] ──→ output
  x₄ ──(w₄)──┤
  xₙ ──(wₙ)──┘
                          ↑
                     bias (b)
```

### Code Example — Single Neuron from Scratch

```python
import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        # Initialize weights randomly from a normal distribution
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias    = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias  # weighted sum
        return self.sigmoid(z)                    # activation

# Usage
neuron = Neuron(n_inputs=3)
x = np.array([0.5, -1.2, 0.8])
print(f"Output: {neuron.forward(x):.4f}")
```

---

## ⚖️ Weights & Biases

### Weights

Weights determine the **strength and direction** of the connection between neurons.

- **Positive weight** → activates the next neuron when the input is active
- **Negative weight** → suppresses the next neuron
- **Near-zero weight** → connection has little influence

### Biases

Bias allows the model to **shift the activation function** left or right, giving the network extra flexibility.

```
Without bias: output = f(w·x)        → must pass through origin
With bias:    output = f(w·x + b)     → can shift freely
```

### Weight Initialization Strategies

```python
import tensorflow as tf
from tensorflow import keras

# Xavier / Glorot Initialization — good for tanh/sigmoid
layer_xavier = keras.layers.Dense(
    128,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros'
)

# He Initialization — good for ReLU
layer_he = keras.layers.Dense(
    128,
    kernel_initializer='he_normal',
    bias_initializer='zeros'
)

# LeCun Initialization — good for SELU
layer_lecun = keras.layers.Dense(
    128,
    kernel_initializer='lecun_normal',
    bias_initializer='zeros'
)

# Random Normal — general purpose
layer_rand = keras.layers.Dense(
    128,
    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    bias_initializer='zeros'
)
```

> ⚠️ **Why initialization matters**: Poor initialization can cause **vanishing** or **exploding** gradients, preventing the network from learning.

---

## 🔥 Activation Functions

Activation functions introduce **non-linearity**, allowing networks to learn complex patterns.

### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | `1/(1+e⁻ˣ)` | (0, 1) | Binary output |
| Tanh | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | (-1, 1) | Hidden layers |
| ReLU | `max(0, x)` | [0, ∞) | Hidden layers (default) |
| Leaky ReLU | `max(0.01x, x)` | (-∞, ∞) | Avoid dead neurons |
| ELU | `x if x>0 else α(eˣ-1)` | (-α, ∞) | Smooth negative |
| Softmax | `eˣⁱ/Σeˣʲ` | (0, 1) | Multi-class output |
| GELU | `x·Φ(x)` | (-∞, ∞) | Transformers |
| Swish | `x·sigmoid(x)` | (-∞, ∞) | Deep networks |

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# All activation functions in TensorFlow / Keras
x = tf.linspace(-5.0, 5.0, 200)

activations = {
    "ReLU":       tf.nn.relu(x),
    "Sigmoid":    tf.nn.sigmoid(x),
    "Tanh":       tf.nn.tanh(x),
    "Leaky ReLU": tf.nn.leaky_relu(x, alpha=0.1),
    "ELU":        tf.nn.elu(x),
    "SELU":       tf.nn.selu(x),
    "Swish":      tf.nn.swish(x),
    "GELU":       tf.nn.gelu(x),
}

# Using in layers
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),         # string shortcut
    keras.layers.Dense(64,  activation=tf.nn.leaky_relu),  # function ref
    keras.layers.Dense(32,  activation=keras.activations.tanh),  # keras ref
    keras.layers.Dense(10,  activation='softmax'),
])
```

---

## 🏗️ Neural Network Architecture

### Layer Types

```
┌─────────────────────────────────────────────────────┐
│                  Neural Network                      │
│                                                      │
│  [Input Layer]  [Hidden Layers]     [Output Layer]  │
│   x₁  x₂  x₃ → ○ ○ ○ → ○ ○ ○ ○ → ○  ○  ○  ○     │
│                  Layer 1   Layer 2    Layer 3        │
└─────────────────────────────────────────────────────┘
```

| Layer | Role |
|-------|------|
| **Input Layer** | Receives raw features; no computation |
| **Hidden Layer(s)** | Learns intermediate representations |
| **Output Layer** | Produces predictions; activation depends on task |

### Network Depth vs Width

- **Depth** (more layers) → learns hierarchical, abstract features
- **Width** (more neurons) → learns more diverse features at the same level

```python
from tensorflow import keras

# Shallow but wide
shallow_model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Deep and narrow
deep_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Print architecture summary
deep_model.summary()
```

---

## ➡️ Forward Propagation

Forward propagation is the process of passing input data **through the network layer by layer** to produce an output.

```
Layer 0 (Input):   a⁰ = x
Layer 1:           z¹ = W¹·a⁰ + b¹,   a¹ = f(z¹)
Layer 2:           z² = W²·a¹ + b²,   a² = f(z²)
...
Layer L (Output):  ŷ = aᴸ
```

```python
import numpy as np

def forward_pass(X, weights, biases, activations):
    """
    X:           input matrix (batch_size, n_features)
    weights:     list of weight matrices per layer
    biases:      list of bias vectors per layer
    activations: list of activation functions per layer
    """
    a = X
    cache = [a]  # store activations for backprop

    for W, b, activation in zip(weights, biases, activations):
        z = np.dot(a, W) + b      # linear transformation
        a = activation(z)          # apply activation
        cache.append(a)

    return a, cache  # output and intermediate activations

# --- TensorFlow equivalent (automatic) ---
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])

X_sample = tf.random.normal((5, 20))
output = model(X_sample, training=False)
print("Output shape:", output.shape)   # (5, 1)
```

---

## 📉 Loss Functions

The **loss function** measures how far the model's predictions are from the true values.

### Common Loss Functions

| Task | Loss Function | Keras Name |
|------|---------------|------------|
| Binary Classification | Binary Cross-Entropy | `binary_crossentropy` |
| Multi-class Classification | Categorical Cross-Entropy | `categorical_crossentropy` |
| Multi-class (sparse labels) | Sparse Categorical Cross-Entropy | `sparse_categorical_crossentropy` |
| Regression | Mean Squared Error | `mse` |
| Regression (robust) | Mean Absolute Error | `mae` |
| Regression (smooth) | Huber Loss | `huber` |

```python
import tensorflow as tf
from tensorflow import keras

# 1. Binary Cross-Entropy
bce = keras.losses.BinaryCrossentropy()
y_true = tf.constant([1.0, 0.0, 1.0])
y_pred = tf.constant([0.9, 0.1, 0.8])
print(f"BCE Loss: {bce(y_true, y_pred).numpy():.4f}")

# 2. Categorical Cross-Entropy
cce = keras.losses.CategoricalCrossentropy()
y_true = tf.constant([[1,0,0], [0,1,0], [0,0,1]], dtype=tf.float32)
y_pred = tf.constant([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.05,0.05,0.9]])
print(f"CCE Loss: {cce(y_true, y_pred).numpy():.4f}")

# 3. Mean Squared Error
mse = keras.losses.MeanSquaredError()
y_true = tf.constant([3.0, -0.5, 2.0, 7.0])
y_pred = tf.constant([2.5,  0.0, 2.0, 8.0])
print(f"MSE Loss: {mse(y_true, y_pred).numpy():.4f}")

# 4. Huber Loss (less sensitive to outliers)
huber = keras.losses.Huber(delta=1.0)
print(f"Huber Loss: {huber(y_true, y_pred).numpy():.4f}")

# Custom loss function
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce  = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t  = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = alpha * tf.pow(1 - p_t, gamma) * bce
    return tf.reduce_mean(loss)
```

---

## 🔄 Backpropagation

Backpropagation is the algorithm that **computes gradients** of the loss with respect to every weight in the network using the **chain rule** of calculus. These gradients tell us how to adjust the weights to reduce the loss.

### The Math

```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
          ↑          ↑        ↑
     loss gradient  activation  input
                    gradient
```

### Chain Rule in Action

```
Output Layer:   δᴸ  = ∂L/∂zᴸ = (aᴸ - y) · f'(zᴸ)
Hidden Layers:  δˡ  = (Wˡ⁺¹)ᵀ · δˡ⁺¹ · f'(zˡ)
Weight Update:  ΔWˡ = δˡ · (aˡ⁻¹)ᵀ
Bias Update:    Δbˡ = δˡ
```

```python
import numpy as np

# Manual backprop — simple 2-layer network
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)
def mse_loss(y_pred, y_true): return np.mean((y_pred - y_true) ** 2)

# ----- Forward Pass -----
X = np.array([[0.1, 0.5], [0.3, -0.2]])  # (2, 2)
y = np.array([[1], [0]])                  # (2, 1)

W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

z1 = X @ W1 + b1          # (2, 3)
a1 = sigmoid(z1)           # (2, 3)
z2 = a1 @ W2 + b2         # (2, 1)
a2 = sigmoid(z2)           # (2, 1) — predictions

loss = mse_loss(a2, y)
print(f"Loss: {loss:.4f}")

# ----- Backward Pass -----
lr = 0.01
m  = X.shape[0]

dL_da2 = 2 * (a2 - y) / m           # ∂L/∂a2
dL_dz2 = dL_da2 * sigmoid_grad(z2)  # ∂L/∂z2
dL_dW2 = a1.T @ dL_dz2              # ∂L/∂W2
dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

dL_da1 = dL_dz2 @ W2.T              # propagate back through W2
dL_dz1 = dL_da1 * sigmoid_grad(z1)  # ∂L/∂z1
dL_dW1 = X.T @ dL_dz1               # ∂L/∂W1
dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

# ----- Weight Updates -----
W2 -= lr * dL_dW2
b2 -= lr * dL_db2
W1 -= lr * dL_dW1
b1 -= lr * dL_db1

# TensorFlow handles all of this automatically with GradientTape:
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn   = tf.keras.losses.MeanSquaredError()

X_tf = tf.constant(X, dtype=tf.float32)
y_tf = tf.constant(y, dtype=tf.float32)

with tf.GradientTape() as tape:
    predictions = model(X_tf, training=True)
    loss = loss_fn(y_tf, predictions)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print(f"TF Loss: {loss.numpy():.4f}")
```

---

## 🚀 Gradient Descent & Optimizers

Optimizers use gradients to **update weights** and minimize the loss.

### Gradient Descent Variants

| Variant | Update on | Speed | Memory | Noise |
|---------|-----------|-------|--------|-------|
| **Batch GD** | Entire dataset | Slow | High | Low |
| **Stochastic GD (SGD)** | 1 sample | Fast | Low | High |
| **Mini-batch GD** | Batch of N | Balanced | Medium | Medium |

### Optimizers Comparison

```python
import tensorflow as tf

# 1. SGD — basic, supports momentum & Nesterov
sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

# 2. Adam — adaptive learning rate (most popular)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,       # decay for 1st moment
    beta_2=0.999,     # decay for 2nd moment
    epsilon=1e-7
)

# 3. RMSprop — good for RNNs
rmsprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    epsilon=1e-7
)

# 4. AdaGrad — adapts lr per parameter
adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01)

# 5. AdamW — Adam with weight decay (modern standard)
adamw = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.004
)

# 6. Nadam — Adam + Nesterov momentum
nadam = tf.keras.optimizers.Nadam(learning_rate=0.002)

# Compile with optimizer
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## 📐 Learning Rate

The **learning rate** controls how large a step the optimizer takes when updating weights. It is arguably the most important hyperparameter.

```
Too high:  weights overshoot → training diverges
Too low:   training too slow → stuck in local minima
Just right: fast convergence to a good minimum
```

### Learning Rate Schedules

```python
import tensorflow as tf

# 1. Constant (default)
lr_constant = 0.001

# 2. Exponential Decay
lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True   # discrete steps vs smooth
)

# 3. Cosine Decay (widely used in modern networks)
lr_schedule_cos = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=5000,
    alpha=0.0         # min lr as fraction of initial
)

# 4. Polynomial Decay
lr_schedule_poly = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    end_learning_rate=0.0001,
    power=0.5
)

# 5. Warmup + Cosine Decay (best practice for large models)
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        self.peak_lr      = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def __call__(self, step):
        warmup  = self.peak_lr * (step / self.warmup_steps)
        cosine  = self.peak_lr * 0.5 * (
            1 + tf.cos(np.pi * (step - self.warmup_steps)
                       / (self.total_steps - self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

# 6. Callbacks-based (ReduceLROnPlateau — most practical)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # new_lr = lr * factor
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Use schedule in optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_cos)
```

---

## 🛡️ Regularization Techniques

Regularization prevents **overfitting** by constraining the model's complexity.

### L1 & L2 Regularization

```python
from tensorflow import keras

# L2 (Ridge) — penalizes large weights, smooth solutions
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,),
                       kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.Dense(64,  activation='relu',
                       kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.Dense(1,   activation='sigmoid')
])

# L1 (Lasso) — promotes sparsity (weights go to 0)
model_l1 = keras.Sequential([
    keras.layers.Dense(128, activation='relu',
                       kernel_regularizer=keras.regularizers.L1(0.001)),
    keras.layers.Dense(1,   activation='sigmoid')
])

# L1 + L2 (Elastic Net)
model_elastic = keras.Sequential([
    keras.layers.Dense(128, activation='relu',
                       kernel_regularizer=keras.regularizers.L1L2(l1=0.001, l2=0.001)),
    keras.layers.Dense(1,   activation='sigmoid')
])
```

---

## 📊 Batch Normalization

Batch Normalization normalizes the input to each layer, **stabilizing and accelerating training**.

```
x_norm = (x - μ_batch) / √(σ²_batch + ε)
output = γ · x_norm + β        # learnable scale and shift
```

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(256, input_shape=(784,)),
    keras.layers.BatchNormalization(),          # normalize after linear
    keras.layers.Activation('relu'),            # then activate
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(10, activation='softmax')
])

# For CNNs — axis=-1 is the channel axis
cnn_model = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.Activation('relu'),
])

# Layer Normalization (used in Transformers — normalizes over features)
transformer_layer = keras.layers.LayerNormalization(epsilon=1e-6)
```

---

## 💧 Dropout

Dropout randomly **deactivates neurons during training**, forcing the network to learn redundant representations.

```
During Training:  each neuron active with probability p
During Inference: all neurons active, weights scaled by p
```

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.5),       # 50% neurons dropped during training
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),       # 30% dropped
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),       # 20% dropped
    keras.layers.Dense(10, activation='softmax')
])

# SpatialDropout2D for CNNs — drops entire feature maps
cnn = keras.Sequential([
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.SpatialDropout2D(0.2),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
```

> 💡 **Tip**: Use higher dropout (0.5) for large fully-connected layers; lower dropout (0.1–0.3) for convolutional layers.

---

## 📈 Data Preprocessing & Metrics

### Data Preprocessing

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Load and split data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 784).astype("float32")
X_test  = X_test.reshape(-1, 784).astype("float32")

# 2. Normalize — [0, 255] → [0, 1]
X_train /= 255.0
X_test  /= 255.0

# 3. Standardization — zero mean, unit variance
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     # use train stats on test!

# 4. One-hot encode labels
y_train_ohe = keras.utils.to_categorical(y_train, num_classes=10)
y_test_ohe  = keras.utils.to_categorical(y_test,  num_classes=10)

# 5. Validation split
X_train, X_val, y_train_f, y_val_f = train_test_split(
    X_train, y_train_ohe, test_size=0.1, random_state=42
)

# 6. Data augmentation for images
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.2),
])

# 7. tf.data pipeline (efficient, scalable)
BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_f))
train_ds = (train_ds
            .shuffle(buffer_size=1024)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_f))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
```

### Evaluation Metrics

```python
import tensorflow as tf
from tensorflow import keras

# Classification Metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc'),
    ]
)

# Regression Metrics
model_reg.compile(
    optimizer='adam',
    loss='mse',
    metrics=[
        'mae',
        'mse',
        keras.metrics.RootMeanSquaredError(name='rmse'),
        keras.metrics.MeanAbsolutePercentageError(name='mape'),
    ]
)

# Custom metric — F1 Score
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall    = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
```

---

## 🔨 Building Models with Keras

### 1. Sequential API

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
], name="sequential_mlp")

model.summary()
```

### 2. Functional API (multi-input/output, skip connections)

```python
from tensorflow import keras

inputs = keras.Input(shape=(784,), name="input")

x = keras.layers.Dense(512, activation='relu')(inputs)
x = keras.layers.Dropout(0.3)(x)

# Skip connection
shortcut = keras.layers.Dense(256)(inputs)   # project inputs
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Add()([x, shortcut])         # add skip
x = keras.layers.Activation('relu')(x)

outputs = keras.layers.Dense(10, activation='softmax', name="output")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="functional_mlp")
model.summary()
```

### 3. Model Subclassing (full control)

```python
import tensorflow as tf
from tensorflow import keras

class ResidualBlock(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation='relu')
        self.dense2 = keras.layers.Dense(units)
        self.bn1    = keras.layers.BatchNormalization()
        self.bn2    = keras.layers.BatchNormalization()
        self.proj   = keras.layers.Dense(units)  # for shortcut projection

    def call(self, inputs, training=False):
        x        = self.bn1(self.dense1(inputs), training=training)
        x        = self.bn2(self.dense2(x),      training=training)
        shortcut = self.proj(inputs)
        return tf.nn.relu(x + shortcut)


class DeepResNet(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.block1  = ResidualBlock(256)
        self.block2  = ResidualBlock(128)
        self.block3  = ResidualBlock(64)
        self.dropout = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.block1(inputs, training=training)
        x = self.block2(x,      training=training)
        x = self.block3(x,      training=training)
        x = self.dropout(x,     training=training)
        return self.output_layer(x)


model = DeepResNet(n_classes=10)
model.build(input_shape=(None, 784))
model.summary()
```

---

## 🏋️ Training Pipeline

```python
import tensorflow as tf
from tensorflow import keras

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

# 2. Build model
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,),
                       kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(256, activation='relu',
                       kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

# 3. Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=50,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
    ],
    verbose=1
)

# 5. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")

# 6. Predict
y_pred = model.predict(X_test[:10])
y_pred_classes = y_pred.argmax(axis=1)
print("Predictions:", y_pred_classes)
print("True labels:", y_test[:10])
```

---

## 🔔 Callbacks & Monitoring

```python
import tensorflow as tf
from tensorflow import keras

# 1. EarlyStopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,         # minimum change to qualify as improvement
    restore_best_weights=True,
    verbose=1
)

# 2. ModelCheckpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}_{val_accuracy:.4f}.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 3. TensorBoard
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,       # weight histograms every epoch
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0
)
# Run: tensorboard --logdir=./logs

# 4. ReduceLROnPlateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 5. CSVLogger
csv_logger = keras.callbacks.CSVLogger('training_log.csv', append=False)

# 6. Custom Callback
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        print(f"\nEpoch {epoch+1}: LR = {lr:.2e}")

callbacks = [early_stop, checkpoint, tensorboard, reduce_lr, csv_logger, LearningRateLogger()]
```

---

## 💾 Saving & Loading Models

```python
import tensorflow as tf
from tensorflow import keras

# ----- SAVING -----

# 1. Save full model (recommended — SavedModel format)
model.save('my_model.keras')          # Keras v3 format
model.save('my_model/')               # TensorFlow SavedModel format

# 2. Save weights only
model.save_weights('weights.weights.h5')

# 3. Save architecture only (JSON)
model_json = model.to_json()
with open('architecture.json', 'w') as f:
    f.write(model_json)

# ----- LOADING -----

# 1. Load full model
loaded_model = keras.models.load_model('my_model.keras')

# 2. Load weights into existing architecture
model.load_weights('weights.weights.h5')

# 3. Load architecture from JSON
with open('architecture.json') as f:
    model_json = f.read()
new_model = keras.models.model_from_json(model_json)
new_model.load_weights('weights.weights.h5')

# ----- TF-Lite Export (for mobile/edge) -----
converter = tf.lite.TFLiteConverter.from_saved_model('my_model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 🏛️ Common Architectures

### 1. Multilayer Perceptron (MLP)

```python
def build_mlp(input_dim, hidden_units, n_classes, dropout=0.3):
    model = keras.Sequential(name="MLP")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation='relu',
                                     kernel_initializer='he_normal'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    return model

mlp = build_mlp(784, [512, 256, 128], 10)
```

### 2. Convolutional Neural Network (CNN)

```python
from tensorflow import keras

def build_cnn(input_shape=(32, 32, 3), n_classes=10):
    model = keras.Sequential([
        # Block 1
        keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                            input_shape=input_shape),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),

        # Block 2
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),

        # Block 3
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),

        # Classifier
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(n_classes, activation='softmax')
    ], name="CNN")
    return model

cnn = build_cnn()
cnn.summary()
```

### 3. Recurrent Neural Network (LSTM)

```python
from tensorflow import keras

def build_lstm(vocab_size, embedding_dim, max_len, n_classes):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        keras.layers.SpatialDropout1D(0.2),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.LSTM(64,  return_sequences=False),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ], name="LSTM")
    return model

lstm = build_lstm(vocab_size=10000, embedding_dim=128, max_len=200, n_classes=5)
lstm.summary()
```

### 4. Autoencoder

```python
from tensorflow import keras

# Encoder
encoder_input = keras.Input(shape=(784,))
x = keras.layers.Dense(256, activation='relu')(encoder_input)
x = keras.layers.Dense(128, activation='relu')(x)
bottleneck = keras.layers.Dense(32, activation='relu', name="bottleneck")(x)

# Decoder
x = keras.layers.Dense(128, activation='relu')(bottleneck)
x = keras.layers.Dense(256, activation='relu')(x)
decoder_output = keras.layers.Dense(784, activation='sigmoid')(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="Autoencoder")
encoder     = keras.Model(encoder_input, bottleneck,    name="Encoder")

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
```

### 5. Transfer Learning

```python
import tensorflow as tf
from tensorflow import keras

# Load pre-trained backbone (no top classifier)
base_model = keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze backbone initially
base_model.trainable = False

# Add custom head
inputs = keras.Input(shape=(224, 224, 3))
x      = base_model(inputs, training=False)
x      = keras.layers.GlobalAveragePooling2D()(x)
x      = keras.layers.Dense(256, activation='relu')(x)
x      = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs, name="TransferLearning")

# Phase 1: Train head only
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Phase 2: Fine-tune — unfreeze top layers
base_model.trainable = True
fine_tune_at = 200  # freeze all layers before this index
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # lower lr for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🎛️ Hyperparameter Tuning

```python
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))

    # Tune number of layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
        model.add(keras.layers.Dense(
            units=hp.Choice(f'units_{i}', [64, 128, 256, 512]),
            activation=hp.Choice('activation', ['relu', 'elu', 'selu']),
            kernel_regularizer=keras.regularizers.L2(
                hp.Float('l2', 1e-5, 1e-2, sampling='log')
            )
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(
            hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('lr', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Bayesian Optimization search
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    directory='tuner_results',
    project_name='mnist_tuning'
)

tuner.search(X_train, y_train,
             epochs=20,
             validation_split=0.1,
             callbacks=[keras.callbacks.EarlyStopping(patience=3)])

best_model = tuner.get_best_models(num_models=1)[0]
best_hps   = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:", best_hps.values)
```

---

## ✅ Best Practices

### Data
- [ ] Always normalize/standardize inputs
- [ ] Fit scalers on training data only; transform test data
- [ ] Use stratified splits for imbalanced datasets
- [ ] Augment training data to improve generalization
- [ ] Check for data leakage between train/test sets

### Architecture
- [ ] Start simple; add complexity only if needed
- [ ] Use He initialization with ReLU, Glorot with sigmoid/tanh
- [ ] Always include Batch Normalization in deep networks
- [ ] Use skip/residual connections for networks deeper than 10 layers
- [ ] Match the output activation to the task (sigmoid, softmax, linear)

### Training
- [ ] Start with Adam at `lr=1e-3`; fine-tune with SGD + momentum
- [ ] Use learning rate scheduling (cosine decay or ReduceLROnPlateau)
- [ ] Monitor both training and validation curves for overfitting
- [ ] Use EarlyStopping with `restore_best_weights=True`
- [ ] Clip gradients when training RNNs: `optimizer.clipnorm=1.0`
- [ ] Shuffle training data every epoch

### Debugging
- [ ] Overfit a small batch first — confirms the model can learn
- [ ] Watch for loss = `nan` (exploding gradients / wrong loss function)
- [ ] Use `model.summary()` and `tf.keras.utils.plot_model()` to verify shapes
- [ ] Log learning rate to catch unintended LR decay issues
- [ ] Profile with TensorBoard to spot bottlenecks

---

## 🔧 Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Loss is `NaN` | Exploding gradients / LR too high | Clip gradients, lower LR |
| Loss not decreasing | LR too low / poor init | Increase LR, check init |
| Overfitting | Model too complex | Add Dropout, L2, more data |
| Underfitting | Model too simple / LR too low | Larger model, higher LR |
| Vanishing gradients | Deep network + sigmoid | Use ReLU, Batch Norm, ResNets |
| Slow training | Large batch, no GPU | Smaller batch, tf.data pipeline |
| Class imbalance | Skewed labels | Class weights, oversampling (SMOTE) |
| Unstable training | No Batch Norm | Add BatchNormalization layers |

```python
# Check for NaN in gradients
import tensorflow as tf

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)

    # Check for NaN/Inf
    for g, v in zip(grads, model.trainable_variables):
        if g is not None:
            tf.debugging.check_numerics(g, f"Gradient for {v.name} contains NaN/Inf")

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

---

## 📚 Resources

### Official Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

### Books
- *Deep Learning* — Goodfellow, Bengio, Courville (Free online)
- *Hands-On Machine Learning* — Aurélien Géron (O'Reilly)
- *Deep Learning with Python* — François Chollet (Manning)

### Courses
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) — Andrew Ng (Coursera)
- [fast.ai Practical Deep Learning](https://course.fast.ai/)
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)

### Papers
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy (2015)
- [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al. (2014)
- [Adam Optimizer](https://arxiv.org/abs/1412.6980) — Kingma & Ba (2015)
- [ResNets](https://arxiv.org/abs/1512.03385) — He et al. (2016)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-transformer-section`)
3. Commit your changes (`git commit -m 'Add Transformer architecture section'`)
4. Push to the branch (`git push origin feature/add-transformer-section`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ for the Deep Learning community

⭐ **Star this repo** if you found it helpful!

</div>