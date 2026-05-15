# Mean Shift Clustering — Complete Guide

---

## Table of Contents

1. [What is Mean Shift?](#1-what-is-mean-shift)
2. [Intuition & Core Idea](#2-intuition--core-idea)
3. [How it Works — Step by Step](#3-how-it-works--step-by-step)
4. [The Mathematics](#4-the-mathematics)
5. [Kernel Functions](#5-kernel-functions)
6. [Bandwidth Parameter](#6-bandwidth-parameter)
7. [Algorithm Pseudocode](#7-algorithm-pseudocode)
8. [Python Implementation from Scratch](#8-python-implementation-from-scratch)
9. [Using Scikit-Learn](#9-using-scikit-learn)
10. [Bandwidth Selection Methods](#10-bandwidth-selection-methods)
11. [Convergence & Stopping Criteria](#11-convergence--stopping-criteria)
12. [Advantages & Disadvantages](#12-advantages--disadvantages)
13. [Comparison with Other Algorithms](#13-comparison-with-other-algorithms)
14. [Real-World Applications](#14-real-world-applications)
15. [Hyperparameter Tuning](#15-hyperparameter-tuning)
16. [Time & Space Complexity](#16-time--space-complexity)
17. [Variants of Mean Shift](#17-variants-of-mean-shift)
18. [Full End-to-End Example](#18-full-end-to-end-example)
19. [Common Mistakes & Fixes](#19-common-mistakes--fixes)
20. [Summary Cheatsheet](#20-summary-cheatsheet)

---

## 1. What is Mean Shift?

Mean Shift is a **non-parametric, unsupervised clustering algorithm** that does **not** require you to specify the number of clusters in advance. It was introduced by **Fukunaga & Hostetler (1975)** and later popularized by **Comaniciu & Meer (2002)** for computer vision tasks.

It works by:
- Placing a sliding window (kernel) over each data point
- Shifting the window toward regions of **higher density**
- Repeating until the window converges to a density peak
- Points that converge to the **same peak** belong to the **same cluster**

> **Key insight:** Clusters are natural density peaks in the data distribution. Mean Shift finds them automatically.

---

## 2. Intuition & Core Idea

Imagine you are in a hilly landscape in complete darkness. Your only tool is a flashlight that shows local elevation. Your goal is to reach a hilltop.

The strategy:
1. Look around your current position
2. Figure out which direction goes uphill the most
3. Take a step in that direction
4. Repeat until you reach a peak

That is exactly what Mean Shift does — but in feature/data space rather than physical space. Each data point is like a person walking uphill toward the nearest density peak.

```
Low Density          High Density (Peak = Cluster Center)
    .  .                    :::
  .      .                :::::
.          .            :::::::
                      :::::::::
         → → → → → → →  PEAK
```

---

## 3. How it Works — Step by Step

### Step 1: Initialize
- Every data point is assigned as an initial candidate cluster center (or a subset for efficiency).

### Step 2: Define a Window (Kernel)
- A circular/spherical window of radius `h` (bandwidth) is placed around each point.

### Step 3: Compute the Mean
- Find the **mean of all points** within the window.

### Step 4: Shift
- Move the window center to the computed mean.
- This shift is the **mean shift vector** — it always points toward increasing density.

### Step 5: Repeat
- Repeat Steps 3–4 until the shift is smaller than a threshold (convergence).

### Step 6: Merge Modes
- All trajectories that converge to the same peak are grouped into one cluster.
- Points at the same converged location (within a small tolerance) form a single cluster.

```
Iteration 0:   x  .  .  [window]  .  .  x
Iteration 1:   x  .    [window]   .  .  x
Iteration 2:   x  .      [window] .  .  x
Converged:     x  .        [PEAK] .  .  x
```

---

## 4. The Mathematics

### 4.1 Kernel Density Estimation (KDE)

Mean Shift is built on top of **Kernel Density Estimation**. Given `n` data points `{x₁, x₂, ..., xₙ}`, the KDE at point `x` is:

```
        1     n        x - xᵢ
f(x) = ─── · Σ  K( ──────── )
       n·hᵈ  i=1      h
```

Where:
- `K(·)` is a kernel function
- `h` is the bandwidth (window radius)
- `d` is the dimensionality of the data

### 4.2 The Gradient of KDE

The gradient (direction of steepest ascent in density):

```
          1      n
∇f(x) = ──── · Σ  ∇K( (x - xᵢ) / h )
         n·hᵈ⁺² i=1
```

### 4.3 Mean Shift Vector

The mean shift vector for point `x` is:

```
              Σᵢ xᵢ · g(‖(x - xᵢ)/h‖²)
m(x) =  ─────────────────────────────────  −  x
              Σᵢ g(‖(x - xᵢ)/h‖²)
```

Where `g(t) = -K'(t)` is the derivative of the kernel profile.

This vector is **proportional to the normalized density gradient**, always pointing toward increasing density.

### 4.4 Update Rule

```
x(t+1) = x(t) + m(x(t))
```

Or equivalently:

```
              Σ { xᵢ : xᵢ ∈ S(x) }
x(t+1)  =  ──────────────────────────
              | { xᵢ : xᵢ ∈ S(x) } |
```

Where `S(x)` is the set of points within the window centered at `x`.

---

## 5. Kernel Functions

The choice of kernel affects the shape of the window and smoothness of convergence.

### 5.1 Flat (Uniform) Kernel

```
K(x) = 1   if ‖x‖ ≤ 1
K(x) = 0   otherwise
```

- Simplest kernel
- Counts all points in the window equally
- Can cause oscillations due to sharp boundary

### 5.2 Gaussian Kernel (Most Common)

```
              1          ‖x‖²
K(x) =  ──────────  exp(─────)
         (2π)^(d/2)      2h²
```

- Smooth, infinite support
- Closer points have more influence
- Produces smooth convergence
- **Recommended for most tasks**

### 5.3 Epanechnikov Kernel

```
K(x) = (3/4)(1 - ‖x‖²)   if ‖x‖ ≤ 1
K(x) = 0                  otherwise
```

- Optimal in the MSE sense
- Parabolic shape
- Computationally efficient

### Kernel Comparison Table

| Kernel       | Smoothness | Speed   | Common Use          |
|--------------|------------|---------|---------------------|
| Flat         | Low        | Fastest | Simple demos        |
| Gaussian     | High       | Moderate| General-purpose     |
| Epanechnikov | Medium     | Fast    | Statistical work    |

---

## 6. Bandwidth Parameter

The **bandwidth `h`** is the most important hyperparameter in Mean Shift.

```
Small h  →  Many small, tight clusters (overfitting)
Large h  →  Few large, broad clusters (underfitting)
```

### Visual Effect of Bandwidth

```
Data:    *   *  * * * *   *  *    * * *   *

h=0.5:  [*] [*][*]*[*][*] [*][*] [*][*][*] [*]
         ↑ Many clusters (too fragmented)

h=2.0:  [  *   *  * * * *   *  *  ][  * * *   *  ]
         ↑ Two clusters (just right)

h=5.0:  [         All points                      ]
         ↑ One cluster (too broad)
```

### Rule of Thumb

A commonly used heuristic:

```
h ≈ σ · n^(-1/(d+4))
```

Where `σ` is the standard deviation of the data and `d` is the number of dimensions.

---

## 7. Algorithm Pseudocode

```
Algorithm: MeanShift(X, h, tolerance ε)

Input:
    X         → dataset of n points in d-dimensional space
    h         → bandwidth
    ε         → convergence threshold

Output:
    labels    → cluster assignment for each point
    centers   → final cluster centers (modes)

Steps:

1.  Initialize: set all points as candidate centers
    candidates ← X.copy()

2.  For each candidate point c in candidates:
    a. REPEAT:
         i.  Find all points within distance h from c
             neighbors ← { xᵢ ∈ X : ‖xᵢ - c‖ ≤ h }
         ii. Compute new center as weighted mean
             c_new ← mean(neighbors)           [flat kernel]
                   OR
             c_new ← Σ xᵢ·K(xᵢ-c) / Σ K(xᵢ-c) [Gaussian kernel]
         iii. shift ← ‖c_new - c‖
         iv.  c ← c_new
    b. UNTIL shift < ε

3.  Merge nearby candidates:
    For each pair (cᵢ, cⱼ):
        if ‖cᵢ - cⱼ‖ < h / 2:
            merge into single center

4.  Assign labels:
    For each point xᵢ in X:
        Find nearest converged center
        labels[i] ← index of nearest center

5.  Return labels, centers
```

---

## 8. Python Implementation from Scratch

```python
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class MeanShift:
    """
    Mean Shift Clustering from scratch.
    
    Parameters
    ----------
    bandwidth : float
        Radius of the sliding window (kernel bandwidth).
    max_iter : int
        Maximum iterations per point.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-4):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def gaussian_kernel(self, distance):
        """Compute Gaussian kernel weight."""
        return np.exp(-0.5 * (distance / self.bandwidth) ** 2)

    def shift_point(self, point, X):
        """Shift a single point toward the mean of its neighborhood."""
        for _ in range(self.max_iter):
            # Compute distances from current point to all data points
            distances = np.linalg.norm(X - point, axis=1)

            # Compute kernel weights
            weights = self.gaussian_kernel(distances)

            # Find points within bandwidth
            within_window = distances < self.bandwidth

            if within_window.sum() == 0:
                break  # No points in window, stay put

            # Compute weighted mean (new center)
            new_point = np.average(X[within_window], axis=0,
                                   weights=weights[within_window])

            # Check for convergence
            shift = np.linalg.norm(new_point - point)
            point = new_point

            if shift < self.tol:
                break

        return point

    def fit(self, X):
        """
        Fit Mean Shift clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # Step 1: Shift each point to its mode
        shifted_points = np.array([self.shift_point(X[i].copy(), X)
                                   for i in range(n_samples)])

        # Step 2: Merge nearby modes into cluster centers
        cluster_centers = []
        for point in shifted_points:
            if len(cluster_centers) == 0:
                cluster_centers.append(point)
            else:
                dists = np.linalg.norm(
                    np.array(cluster_centers) - point, axis=1
                )
                if dists.min() > self.bandwidth / 2:
                    cluster_centers.append(point)

        self.cluster_centers_ = np.array(cluster_centers)

        # Step 3: Assign labels
        labels = []
        for point in shifted_points:
            dists = np.linalg.norm(self.cluster_centers_ - point, axis=1)
            labels.append(np.argmin(dists))

        self.labels_ = np.array(labels)
        return self

    def predict(self, X):
        """Assign new points to nearest cluster center."""
        X = np.array(X)
        labels = []
        for point in X:
            dists = np.linalg.norm(self.cluster_centers_ - point, axis=1)
            labels.append(np.argmin(dists))
        return np.array(labels)


# ─── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=200, centers=4,
                           cluster_std=0.6, random_state=42)

    # Fit Mean Shift
    ms = MeanShift(bandwidth=1.5)
    ms.fit(X)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30)
    plt.title("Ground Truth")

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=ms.labels_, cmap='viridis', s=30)
    plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
                c='red', marker='X', s=200, label='Centers')
    plt.title(f"Mean Shift (k={len(ms.cluster_centers_)} found)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("mean_shift_demo.png", dpi=150)
    plt.show()
    print(f"Clusters found: {len(ms.cluster_centers_)}")
```

---

## 9. Using Scikit-Learn

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Generate Data ─────────────────────────────────────────────────────────
X, y_true = make_blobs(n_samples=500, centers=5,
                        cluster_std=0.8, random_state=0)

# ── 2. Normalize (important for bandwidth selection) ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Estimate Bandwidth automatically ──────────────────────────────────────
# quantile: lower = smaller bandwidth = more clusters
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
print(f"Estimated bandwidth: {bandwidth:.3f}")

# ── 4. Fit Mean Shift ─────────────────────────────────────────────────────────
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#   bin_seeding=True  →  speeds up by using a grid of seeds (approximate)
ms.fit(X_scaled)

labels = ms.labels_
centers = ms.cluster_centers_
n_clusters = len(np.unique(labels))
print(f"Number of clusters found: {n_clusters}")

# ── 5. Visualize ──────────────────────────────────────────────────────────────
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

plt.figure(figsize=(8, 6))
for k, color in zip(range(n_clusters), colors):
    mask = labels == k
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                color=color, s=30, label=f"Cluster {k}")

plt.scatter(centers[:, 0], centers[:, 1],
            c='black', marker='*', s=300, zorder=5, label='Centers')
plt.title(f"Mean Shift Clustering — {n_clusters} Clusters Found")
plt.legend()
plt.tight_layout()
plt.show()

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
from sklearn.metrics import silhouette_score, adjusted_rand_score

sil = silhouette_score(X_scaled, labels)
ari = adjusted_rand_score(y_true, labels)
print(f"Silhouette Score : {sil:.4f}  (higher = better, max 1.0)")
print(f"Adjusted Rand    : {ari:.4f}  (1.0 = perfect match)")
```

---

## 10. Bandwidth Selection Methods

### Method 1: `estimate_bandwidth` (Scikit-Learn)

```python
from sklearn.cluster import estimate_bandwidth

bw = estimate_bandwidth(X, quantile=0.2, n_samples=300)
# quantile: fraction of samples used to estimate bandwidth
# smaller quantile → smaller bandwidth → more clusters
```

### Method 2: Rule of Thumb (Silverman's Rule)

```python
def silverman_bandwidth(X):
    n, d = X.shape
    sigma = X.std(axis=0).mean()
    return sigma * (n ** (-1.0 / (d + 4)))

bw = silverman_bandwidth(X)
```

### Method 3: Cross-validation / Silhouette Score

```python
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

bandwidths = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]
scores = []

for bw in bandwidths:
    ms = MeanShift(bandwidth=bw)
    ms.fit(X)
    if len(np.unique(ms.labels_)) > 1:
        score = silhouette_score(X, ms.labels_)
    else:
        score = -1
    scores.append(score)

best_bw = bandwidths[np.argmax(scores)]
print(f"Best bandwidth: {best_bw} (silhouette={max(scores):.4f})")
```

### Method 4: Elbow / Cluster Count Plot

```python
import matplotlib.pyplot as plt

bws = np.linspace(0.3, 3.0, 30)
n_clusters = []

for bw in bws:
    ms = MeanShift(bandwidth=bw)
    ms.fit(X)
    n_clusters.append(len(np.unique(ms.labels_)))

plt.plot(bws, n_clusters, 'b-o')
plt.xlabel("Bandwidth")
plt.ylabel("Number of Clusters")
plt.title("Bandwidth vs. Number of Clusters")
plt.grid(True)
plt.show()
```

---

## 11. Convergence & Stopping Criteria

Mean Shift is **guaranteed to converge** for the Gaussian kernel (Comaniciu & Meer, 2002).

### Why It Converges

Each iteration moves the point uphill in density:
- The mean shift vector always points toward increasing density gradient
- The KDE function is bounded above
- Therefore each step increases `f(x)` monotonically → converges to a local maximum

### Stopping Criteria

```python
# 1. Shift magnitude falls below threshold
shift = ‖x_new - x_old‖
if shift < tol:
    STOP

# 2. Maximum iterations reached
if iteration >= max_iter:
    STOP  # (rare with proper bandwidth)
```

### Practical Convergence Notes

- Flat kernel: may oscillate near boundaries — add small damping
- Gaussian kernel: always smooth convergence
- Typical iterations to converge: 10–50 for well-separated clusters

---

## 12. Advantages & Disadvantages

### Advantages

| Feature | Detail |
|---|---|
| No `k` required | Number of clusters found automatically |
| Arbitrary shapes | Can find non-convex, non-spherical clusters |
| Robust to outliers | Outliers drift to low-density regions |
| One key parameter | Only bandwidth `h` needs tuning |
| Theoretical guarantees | Convergence is mathematically proven |
| Cluster-free points | Points in low-density areas get no cluster |

### Disadvantages

| Limitation | Detail |
|---|---|
| Slow on large data | O(n²) per iteration; slow for n > 10,000 |
| Bandwidth sensitivity | Wrong bandwidth gives wrong clusters |
| High dimensions | Density estimation degrades in high-d (curse of dimensionality) |
| Memory usage | Stores all pairwise distances |
| Non-deterministic size | Can't control the number of clusters directly |

---

## 13. Comparison with Other Algorithms

| Feature | Mean Shift | K-Means | DBSCAN | Agglomerative |
|---|---|---|---|---|
| K required? | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| Cluster shape | Any | Spherical | Any | Any |
| Noise handling | Moderate | Poor | Excellent | Moderate |
| Scalability | Poor (O(n²)) | Good (O(nk)) | Moderate | Poor (O(n²)) |
| Key parameter | bandwidth | k | eps, min_samples | linkage, k |
| Deterministic | ✅ Yes | ❌ (init) | ✅ Yes | ✅ Yes |
| Soft boundaries | ✅ Yes | ❌ Hard | ✅ Yes | ❌ Hard |

### When to Use Mean Shift

- You don't know how many clusters exist
- Clusters have irregular shapes
- You have a small-to-medium dataset (< 10,000 points)
- You need mode-seeking (image segmentation, tracking)

---

## 14. Real-World Applications

### 14.1 Image Segmentation

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import io
import numpy as np

# Load image
img = io.imread("image.jpg")
h, w, c = img.shape

# Flatten to (n_pixels, 3) — RGB features
X = img.reshape(-1, 3).astype(float)

bw = estimate_bandwidth(X, quantile=0.1, n_samples=1000)
ms = MeanShift(bandwidth=bw, bin_seeding=True)
ms.fit(X)

# Replace each pixel with its cluster center color
segmented = ms.cluster_centers_[ms.labels_]
segmented_img = segmented.reshape(h, w, c).astype(np.uint8)
io.imsave("segmented.jpg", segmented_img)
```

### 14.2 Object Tracking (Video)

Mean Shift tracks moving objects by:
1. Modeling the object appearance as a color histogram
2. In each new frame, shifting from the previous position toward the mode of the histogram match
3. The converged position = new object location

This is the basis of the **CamShift algorithm** used in OpenCV.

```python
import cv2

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

# Select initial ROI
x, y, w, h = cv2.selectROI(frame)
roi = frame[y:y+h, x:x+w]

# Build color histogram model
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

track_window = (x, y, w, h)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)

    # Mean Shift tracking
    ret, track_window = cv2.meanShift(dst, track_window, term_criteria)
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) == 27:
        break
```

### 14.3 Geographic / Spatial Clustering

```python
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# GPS coordinates of events
df = pd.read_csv("events.csv")  # columns: lat, lon
coords = df[["lat", "lon"]].values

bw = estimate_bandwidth(coords, quantile=0.15)
ms = MeanShift(bandwidth=bw)
ms.fit(coords)

df["cluster"] = ms.labels_
hotspots = ms.cluster_centers_
print("Hotspot locations:\n", hotspots)
```

### 14.4 Anomaly Detection

Points that converge to a mode with very few members (small cluster) can be flagged:

```python
ms.fit(X)
labels = ms.labels_
cluster_sizes = np.bincount(labels)
threshold = 5  # Flag clusters with fewer than 5 points

anomalies = np.where(cluster_sizes[labels] < threshold)[0]
print(f"Anomalous points: {len(anomalies)}")
```

---

## 15. Hyperparameter Tuning

### The Only Key Parameter: Bandwidth `h`

```python
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Grid search over bandwidth values
bandwidths = np.arange(0.3, 3.0, 0.1)
results = []

for bw in bandwidths:
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(X)
    n = len(np.unique(ms.labels_))

    if n == 1 or n >= len(X):
        results.append({"bw": bw, "n_clusters": n, "sil": np.nan, "db": np.nan})
        continue

    sil = silhouette_score(X, ms.labels_)
    db  = davies_bouldin_score(X, ms.labels_)
    results.append({"bw": bw, "n_clusters": n, "sil": sil, "db": db})

import pandas as pd
df_res = pd.DataFrame(results)
print(df_res)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(df_res.bw, df_res.n_clusters, 'o-')
axes[0].set(xlabel="Bandwidth", ylabel="N Clusters", title="Clusters vs BW")
axes[1].plot(df_res.bw, df_res.sil, 'o-', color='green')
axes[1].set(xlabel="Bandwidth", ylabel="Silhouette", title="Silhouette vs BW")
axes[2].plot(df_res.bw, df_res.db, 'o-', color='red')
axes[2].set(xlabel="Bandwidth", ylabel="Davies-Bouldin", title="DB vs BW")
plt.tight_layout()
plt.show()
```

### `bin_seeding` Parameter (Scikit-Learn)

```python
# bin_seeding=False (default): slower, uses all points as seeds
# bin_seeding=True : faster, bins the space and uses bin centers as seeds
#                    may slightly affect final cluster count

ms_fast = MeanShift(bandwidth=1.5, bin_seeding=True)   # recommended for n > 1000
ms_full = MeanShift(bandwidth=1.5, bin_seeding=False)  # exact, but slower
```

---

## 16. Time & Space Complexity

| Operation | Complexity |
|---|---|
| Naive Mean Shift (per iteration) | O(n²) |
| Total (T iterations) | O(T · n²) |
| With KD-Tree (low-d) | O(T · n · log n) |
| Space | O(n · d) |

### Scaling Strategies

```python
# 1. Use bin_seeding to reduce number of candidates
ms = MeanShift(bandwidth=bw, bin_seeding=True)

# 2. Subsample for bandwidth estimation
bw = estimate_bandwidth(X, quantile=0.2, n_samples=min(1000, len(X)))

# 3. For large datasets, consider Mini-Batch Mean Shift or OPTICS instead

# 4. Reduce dimensionality first
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=10).fit_transform(X)
```

---

## 17. Variants of Mean Shift

### 17.1 Blurring Mean Shift

Instead of using fixed original data points, it updates the data set at each step (blurs the dataset as it iterates). Faster convergence but can merge too aggressively.

### 17.2 CamShift (Continuously Adaptive Mean Shift)

Used in video tracking. Adapts the window size based on the zeroth moment (total weight) of the histogram backprojection. Standard in OpenCV.

### 17.3 Subspace Mean Shift

Projects data onto a lower-dimensional subspace before running Mean Shift. Better for high-dimensional image patches.

### 17.4 Quick Shift

A fast approximate version of Mean Shift. Used in `scikit-image` for superpixels.

```python
from skimage.segmentation import quickshift
segments = quickshift(image, kernel_size=5, max_dist=10, ratio=0.5)
```

### 17.5 Medoid Shift

Uses medoids instead of means for better robustness to outliers in non-Euclidean spaces.

---

## 18. Full End-to-End Example

```python
"""
Full Mean Shift Pipeline:
  - Data generation
  - Preprocessing
  - Bandwidth tuning
  - Fitting
  - Evaluation
  - Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)

# ── 1. Generate Dataset ───────────────────────────────────────────────────────
X, y_true = make_blobs(
    n_samples=400,
    centers=[[-4, -4], [0, 0], [4, 4], [0, 6]],
    cluster_std=[0.8, 0.6, 0.7, 0.9],
    random_state=42
)

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Estimate Bandwidth ─────────────────────────────────────────────────────
bw = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=300, random_state=42)
print(f"Estimated bandwidth: {bw:.4f}")

# ── 4. Fit Mean Shift ─────────────────────────────────────────────────────────
ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
ms.fit(X_scaled)

labels      = ms.labels_
centers     = ms.cluster_centers_
n_clusters  = len(np.unique(labels))
print(f"Clusters discovered: {n_clusters}")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
metrics = {
    "Silhouette Score"         : silhouette_score(X_scaled, labels),
    "Davies-Bouldin Score"     : davies_bouldin_score(X_scaled, labels),
    "Calinski-Harabasz Score"  : calinski_harabasz_score(X_scaled, labels),
    "Adjusted Rand Index"      : adjusted_rand_score(y_true, labels),
}
print("\n── Evaluation Metrics ──")
for k, v in metrics.items():
    print(f"  {k:30s} : {v:.4f}")

# ── 6. Visualize ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ground Truth
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true,
                cmap='tab10', s=25, alpha=0.7)
axes[0].set_title("Ground Truth")

# Mean Shift Result
scatter = axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels,
                          cmap='tab10', s=25, alpha=0.7)
axes[1].scatter(centers[:, 0], centers[:, 1],
                c='black', marker='★', s=300, zorder=10, label='Modes')
axes[1].set_title(f"Mean Shift ({n_clusters} Clusters Found)")
axes[1].legend()

# Bandwidth Effect
bws  = np.linspace(0.2, 2.5, 50)
ncls = []
for b in bws:
    m = MeanShift(bandwidth=b)
    m.fit(X_scaled)
    ncls.append(len(np.unique(m.labels_)))
axes[2].plot(bws, ncls, 'b-')
axes[2].axvline(bw, color='red', linestyle='--', label=f'chosen h={bw:.2f}')
axes[2].set(xlabel="Bandwidth", ylabel="# Clusters",
            title="Bandwidth vs. Cluster Count")
axes[2].legend()

plt.suptitle("Mean Shift Clustering — Full Pipeline", fontsize=14)
plt.tight_layout()
plt.savefig("mean_shift_full_pipeline.png", dpi=150)
plt.show()
```

---

## 19. Common Mistakes & Fixes

### Mistake 1: Not Scaling the Data

```python
# ❌ Wrong — different feature scales distort the kernel window
ms = MeanShift(bandwidth=1.0).fit(X_raw)

# ✅ Correct — always standardize first
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_raw)
ms = MeanShift(bandwidth=1.0).fit(X_scaled)
```

### Mistake 2: Using a Fixed Bandwidth Without Tuning

```python
# ❌ Wrong — guessing bandwidth
ms = MeanShift(bandwidth=1.0)

# ✅ Correct — estimate from data
from sklearn.cluster import estimate_bandwidth
bw = estimate_bandwidth(X, quantile=0.2)
ms = MeanShift(bandwidth=bw)
```

### Mistake 3: Using on High-Dimensional Data Directly

```python
# ❌ Wrong — Mean Shift degrades in high dimensions
ms = MeanShift().fit(X_1000_features)

# ✅ Correct — reduce dimensions first
from sklearn.decomposition import PCA
X_low = PCA(n_components=20).fit_transform(X_1000_features)
ms = MeanShift().fit(X_low)
```

### Mistake 4: Using on Very Large Datasets Without Approximation

```python
# ❌ Wrong — extremely slow for n > 50,000
ms = MeanShift(bin_seeding=False).fit(X_large)

# ✅ Correct — use bin_seeding and subsample for bandwidth estimation
bw = estimate_bandwidth(X_large, quantile=0.2, n_samples=5000)
ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1).fit(X_large)
```

### Mistake 5: Ignoring the "Orphan" Cluster (Label -1 in Some Implementations)

```python
# Check for orphan points if your implementation uses -1 labels
n_noise = np.sum(ms.labels_ == -1)
print(f"Unclustered points: {n_noise}")
```

---

## 20. Summary Cheatsheet

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEAN SHIFT CLUSTERING — CHEATSHEET                   │
├────────────────────────┬────────────────────────────────────────────────┤
│ Type                   │ Non-parametric, unsupervised, density-based    │
│ Core idea              │ Roll uphill in density until mode found        │
│ Key parameter          │ Bandwidth h  (window size)                     │
│ K required?            │ NO — discovered automatically                  │
│ Cluster shape          │ Any shape (non-convex, non-spherical)          │
│ Convergence            │ Guaranteed (Gaussian kernel)                   │
│ Complexity             │ O(T·n²) naive  |  O(T·n·log n) with KD-Tree   │
│ Best for               │ Small-medium datasets, image/video tasks        │
│ Not great for          │ Very large n, very high d, unknown k control   │
├────────────────────────┼────────────────────────────────────────────────┤
│ Main steps             │ 1. Place kernel at each point                  │
│                        │ 2. Compute weighted mean of neighbors          │
│                        │ 3. Shift to new mean                           │
│                        │ 4. Repeat until convergence                    │
│                        │ 5. Merge nearby modes → cluster centers        │
│                        │ 6. Assign each point to nearest center         │
├────────────────────────┼────────────────────────────────────────────────┤
│ Tuning bandwidth       │ estimate_bandwidth(X, quantile=0.2)            │
│                        │ Grid search + Silhouette Score                 │
│                        │ Silverman's rule: h ≈ σ · n^(-1/(d+4))        │
├────────────────────────┼────────────────────────────────────────────────┤
│ Evaluation metrics     │ Silhouette Score (higher = better)             │
│                        │ Davies-Bouldin (lower = better)                │
│                        │ Calinski-Harabasz (higher = better)            │
│                        │ ARI / NMI (when ground truth known)            │
├────────────────────────┼────────────────────────────────────────────────┤
│ Sklearn quick start    │ from sklearn.cluster import MeanShift,         │
│                        │                             estimate_bandwidth │
│                        │ bw = estimate_bandwidth(X, quantile=0.2)       │
│                        │ ms = MeanShift(bandwidth=bw).fit(X)            │
│                        │ labels  = ms.labels_                           │
│                        │ centers = ms.cluster_centers_                  │
└────────────────────────┴────────────────────────────────────────────────┘
```

---

## References

- Fukunaga, K., & Hostetler, L. (1975). *The Estimation of the Gradient of a Density Function, with Applications in Pattern Recognition.* IEEE Trans. Information Theory.
- Comaniciu, D., & Meer, P. (2002). *Mean Shift: A Robust Approach Toward Feature Space Analysis.* IEEE PAMI.
- Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
- Cheng, Y. (1995). *Mean shift, mode seeking, and clustering.* IEEE PAMI.

---

*Generated as a complete educational reference for Mean Shift Clustering.*