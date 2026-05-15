# 🔍 DBSCAN: Density-Based Spatial Clustering of Applications with Noise
### A Complete Guide to Unsupervised Learning with DBSCAN

---

## 📚 Table of Contents

1. [What is DBSCAN?](#what-is-dbscan)
2. [Why DBSCAN? (Motivation)](#why-dbscan-motivation)
3. [Core Concepts & Terminology](#core-concepts--terminology)
4. [How DBSCAN Works (Algorithm)](#how-dbscan-works-algorithm)
5. [Key Parameters: eps & min_samples](#key-parameters-eps--min_samples)
6. [Types of Points in DBSCAN](#types-of-points-in-dbscan)
7. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
8. [DBSCAN vs K-Means vs Hierarchical](#dbscan-vs-k-means-vs-hierarchical)
9. [Advantages of DBSCAN](#advantages-of-dbscan)
10. [Limitations of DBSCAN](#limitations-of-dbscan)
11. [Choosing Parameters (eps & min_samples)](#choosing-parameters-eps--min_samples)
12. [Mathematical Foundation](#mathematical-foundation)
13. [Time & Space Complexity](#time--space-complexity)
14. [Implementation in Python](#implementation-in-python)
15. [Real-World Use Cases](#real-world-use-cases)
16. [Variants of DBSCAN](#variants-of-dbscan)
17. [Evaluation Metrics](#evaluation-metrics)
18. [Common Mistakes & How to Avoid Them](#common-mistakes--how-to-avoid-them)
19. [Practice Questions](#practice-questions)
20. [Summary Cheat Sheet](#summary-cheat-sheet)

---

## 1. What is DBSCAN?

**DBSCAN** stands for **Density-Based Spatial Clustering of Applications with Noise**.

It is an **unsupervised machine learning algorithm** that groups data points into clusters based on **density** — how close points are to each other.

> 💡 **Key Idea**: Points that are closely packed together belong to the same cluster. Points in low-density regions are considered **noise** (outliers).

DBSCAN was proposed by **Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu** in 1996 at the KDD conference.

---

## 2. Why DBSCAN? (Motivation)

### Problems with Traditional Clustering:

| Problem | K-Means Failure | DBSCAN Solution |
|--------|----------------|-----------------|
| Arbitrary shapes | Only finds spherical clusters | Finds clusters of ANY shape |
| Number of clusters | Must specify K beforehand | Discovers K automatically |
| Outliers | Assigns every point to a cluster | Labels outliers as **noise** |
| Unequal density (basic) | Sensitive to initialization | Density-based, not centroid-based |

### When do you NEED DBSCAN?
- Data has **irregular-shaped clusters** (crescents, spirals, rings)
- Data contains **noise/outliers** that should be ignored
- You **don't know** the number of clusters in advance
- Clusters have **varying sizes**

---

## 3. Core Concepts & Terminology

### 🔵 Epsilon (ε) — Neighborhood Radius
A distance threshold. Any point within distance **ε** of another point is considered its **neighbor**.

```
If dist(point_A, point_B) ≤ ε  →  B is a neighbor of A
```

### 🔢 MinPts — Minimum Points
The minimum number of points required in an ε-neighborhood (including the point itself) to form a **dense region** (core point).

### 📍 ε-Neighborhood (N_ε)
The set of all points within distance ε from a point p:

```
N_ε(p) = { q ∈ Dataset | dist(p, q) ≤ ε }
```

---

## 4. Types of Points in DBSCAN

DBSCAN classifies every point into one of three categories:

### 🟢 1. Core Point
A point is a **Core Point** if it has **at least MinPts** neighbors within radius ε (including itself).

```
|N_ε(p)| ≥ MinPts  →  p is a Core Point
```

### 🟡 2. Border Point
A point is a **Border Point** if:
- It is **NOT** a core point (has fewer than MinPts neighbors)
- But it **IS** within the ε-neighborhood of a core point

### 🔴 3. Noise Point (Outlier)
A point is **Noise** if:
- It is **NOT** a core point
- It is **NOT** a border point
- It doesn't belong to any cluster

### Visual Representation:
```
        . . . . .
      .   [C]   .      [C] = Core Point (many neighbors within ε)
        . . . . .      [B] = Border Point (few neighbors, near core)
            [B]        [N] = Noise (isolated, far from dense regions)


                [N]
```

---

## 5. How DBSCAN Works (Algorithm)

### Pseudocode:

```
DBSCAN(Dataset, ε, MinPts):

  cluster_id = 0
  
  FOR each unvisited point P in Dataset:
    Mark P as visited
    
    neighbors = getNeighbors(P, ε)
    
    IF |neighbors| < MinPts:
      Mark P as NOISE
    
    ELSE:
      cluster_id = cluster_id + 1
      expandCluster(P, neighbors, cluster_id, ε, MinPts)

expandCluster(P, neighbors, cluster_id, ε, MinPts):
  Add P to cluster_id
  
  FOR each point Q in neighbors:
    IF Q is not visited:
      Mark Q as visited
      new_neighbors = getNeighbors(Q, ε)
      
      IF |new_neighbors| ≥ MinPts:
        neighbors = neighbors ∪ new_neighbors  (merge)
    
    IF Q is not yet a member of any cluster:
      Add Q to cluster_id
```

### Step-by-step Execution:

```
Step 1: Pick any unvisited point
Step 2: Find all points within ε distance
Step 3: If count ≥ MinPts → CORE POINT → start a new cluster
Step 4: Expand cluster by recursively visiting neighbors
Step 5: If count < MinPts → mark as NOISE (temporarily)
Step 6: Border points get absorbed when a core point visits them
Step 7: Repeat until all points are visited
```

---

## 6. Key Parameters: eps & min_samples

### 📏 eps (ε) — The Radius

| Small ε | Large ε |
|---------|---------|
| More clusters | Fewer clusters |
| More noise points | Less noise |
| Fine-grained clusters | Everything merges |

### 📊 min_samples (MinPts)

| Small MinPts | Large MinPts |
|-------------|-------------|
| More core points | Fewer core points |
| More clusters found | Fewer, denser clusters |
| Sensitive to noise | More robust to noise |

### 🧠 Rule of Thumb for min_samples:
```
min_samples ≥ dimensions + 1
min_samples ≥ 2 × dimensions  (for larger datasets)

For 2D data: min_samples = 4 or 5
For higher dimensions: increase accordingly
```

---

## 7. Step-by-Step Walkthrough

### Example Dataset (2D):
```
Points: A(1,1), B(1,2), C(2,1), D(8,8), E(8,9), F(9,8), G(25,80)

Settings: ε = 2, MinPts = 3
```

### Execution:

```
Step 1: Visit A(1,1)
  Neighbors within ε=2: {A, B, C} → count=3 ≥ MinPts=3
  ✅ A is a CORE POINT → Start Cluster 1

Step 2: Expand from A
  Visit B(1,2): neighbors = {A, B, C} → CORE POINT → add to Cluster 1
  Visit C(2,1): neighbors = {A, B, C} → CORE POINT → add to Cluster 1
  All neighbors of Cluster 1 explored.

Step 3: Visit D(8,8)
  Neighbors within ε=2: {D, E, F} → count=3 ≥ MinPts=3
  ✅ D is a CORE POINT → Start Cluster 2

Step 4: Expand from D
  Visit E(8,9): CORE POINT → Cluster 2
  Visit F(9,8): CORE POINT → Cluster 2

Step 5: Visit G(25,80)
  Neighbors within ε=2: {G} → count=1 < MinPts=3
  ❌ G is NOISE → Label = -1

RESULT:
  Cluster 1: {A, B, C}
  Cluster 2: {D, E, F}
  Noise:     {G}
```

---

## 8. DBSCAN vs K-Means vs Hierarchical

| Feature | DBSCAN | K-Means | Hierarchical |
|---------|--------|---------|--------------|
| **Cluster Shape** | Any shape | Spherical only | Any (with right linkage) |
| **Number of Clusters** | Automatic | Must specify K | Can choose post-hoc |
| **Outlier Handling** | Labels as noise | Forced assignment | Forced assignment |
| **Scalability** | Medium-Large | Large | Small-Medium |
| **Parameters** | ε, MinPts | K | Linkage, cut threshold |
| **Deterministic** | Yes | No (random init) | Yes |
| **Varying Densities** | Struggles | Struggles | Better |
| **High Dimensions** | Degrades | Degrades | Degrades |

### When to Use What:
```
Use K-Means when:
  → You know K
  → Clusters are roughly spherical and equal size
  → Speed is priority

Use DBSCAN when:
  → You don't know K
  → Clusters have complex shapes
  → Data has outliers/noise

Use Hierarchical when:
  → You want a dendrogram
  → Small dataset
  → Need to visualize cluster merging
```

---

## 9. Advantages of DBSCAN

✅ **No need to specify number of clusters** — K is discovered automatically

✅ **Handles arbitrary shapes** — finds spirals, crescents, rings, etc.

✅ **Robust to outliers** — explicitly labels noise points

✅ **Deterministic results** — same output for same input (no random initialization)

✅ **Works with any distance metric** — Euclidean, Manhattan, cosine, etc.

✅ **Scalable** — with spatial indexing (KD-tree), runs in O(n log n)

✅ **Discovers clusters of different sizes** — no assumption of equal cluster size

---

## 10. Limitations of DBSCAN

❌ **Struggles with varying densities** — one ε doesn't fit all densities (HDBSCAN solves this)

❌ **Sensitive to parameter choice** — wrong ε or MinPts gives bad results

❌ **Curse of dimensionality** — distance metrics become meaningless in very high dimensions

❌ **Struggles with high-dimensional data** — density estimation is hard above ~10 dimensions

❌ **Boundary point assignment** — border points are assigned to whichever cluster finds them first (non-deterministic for border points)

❌ **Memory intensive** — needs distance matrix or KD-tree

---

## 11. Choosing Parameters (eps & min_samples)

### 🔑 Method 1: k-Distance Graph (for eps)

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Fit nearest neighbors
k = min_samples  # typically 4-5 for 2D data
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort distances to k-th neighbor
distances = np.sort(distances[:, k-1], axis=0)

# Plot — look for the "elbow"
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-NN Distance')
plt.title('k-Distance Graph — Find the Elbow for eps')
plt.grid(True)
plt.show()

# The eps value is at the "elbow" (knee) of the curve
```

> 📌 **The elbow point** in the k-distance graph gives a good estimate for **ε**.

### 🔑 Method 2: Silhouette Score Grid Search

```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

best_score = -1
best_params = {}

for eps in np.arange(0.1, 2.0, 0.1):
    for min_samples in range(2, 10):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print(f"Best params: {best_params}, Score: {best_score:.3f}")
```

### 🔑 Rule of Thumb Summary:

```
min_samples:
  → Start with: 2 × number_of_features
  → Minimum: 3 (never use 1 or 2)
  → Larger dataset → larger min_samples

eps:
  → Use k-distance graph
  → Choose the "elbow" point
  → Normalize features first (StandardScaler)
```

---

## 12. Mathematical Foundation

### Distance Metric (Euclidean by default):

```
dist(p, q) = √( Σ (pᵢ - qᵢ)² )
             i=1 to d
```

### Density Reachability:

**Directly Density-Reachable**: Point q is directly density-reachable from p if:
```
1. q ∈ N_ε(p)           (q is in ε-neighborhood of p)
2. |N_ε(p)| ≥ MinPts    (p is a core point)
```

**Density-Reachable**: Point q is density-reachable from p if there exists a chain:
```
p → p₁ → p₂ → ... → pₙ → q
where each step is directly density-reachable
```

**Density-Connected**: Points p and q are density-connected if there exists a point o such that:
```
Both p and q are density-reachable from o
```

### Cluster Definition:
A cluster C is a non-empty subset of Dataset satisfying:
```
1. Maximality:  if p ∈ C and q is density-reachable from p → q ∈ C
2. Connectivity: every pair of points in C is density-connected
```

---

## 13. Time & Space Complexity

| Operation | Naive | With KD-Tree/Ball-Tree |
|-----------|-------|------------------------|
| **Time Complexity** | O(n²) | O(n log n) |
| **Space Complexity** | O(n²) | O(n) |

```
n = number of data points

Naive:    compute all pairwise distances → O(n²) time, O(n²) space
KD-Tree:  spatial indexing → O(n log n) time, O(n) space
          (degrades to O(n²) in high dimensions)
```

> ⚠️ For very large datasets (n > 1M), consider **HDBSCAN** or **mini-batch** variants.

---

## 14. Implementation in Python

### 📦 Installation:
```bash
pip install scikit-learn numpy matplotlib seaborn
```

### 🔰 Basic Example:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# 1. Generate sample data (moon shapes — K-Means fails here!)
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. Standardize features (IMPORTANT!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(X_scaled)

# 4. Analyze results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Labels: {set(labels)}")  # -1 = noise

# 5. Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30)
plt.title("True Labels")

plt.subplot(1, 2, 2)
colors = ['red' if l == -1 else plt.cm.tab10(l / max(labels)) for l in labels]
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30)
plt.title(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

plt.tight_layout()
plt.show()
```

### 🔬 Advanced Example with k-Distance Graph:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate data
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.5, random_state=42)
X = StandardScaler().fit_transform(X)

# --- Find optimal eps using k-distance graph ---
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, min_samples - 1])[::-1]

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.axhline(y=0.5, color='r', linestyle='--', label='Chosen eps = 0.5')
plt.xlabel('Points (sorted by distance)')
plt.ylabel(f'{min_samples}-NN Distance')
plt.title('k-Distance Graph — Elbow = Optimal eps')
plt.legend()
plt.grid(True)
plt.show()

# --- Apply DBSCAN with chosen eps ---
eps = 0.5
db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(X)

# --- Identify core, border, noise points ---
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

plt.figure(figsize=(8, 6))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise

    class_member_mask = labels == k

    # Core points (large)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14, label=f'Cluster {k} (core)' if k != -1 else 'Noise')

    # Border points (small)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title(f'DBSCAN | eps={eps} | min_samples={min_samples}')
plt.legend()
plt.show()
```

### 🌐 DBSCAN with Custom Distance Metric:

```python
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import numpy as np

# GPS coordinates (latitude, longitude in radians)
coords = np.radians([[40.7128, -74.0060],   # New York
                     [34.0522, -118.2437],  # Los Angeles
                     [41.8781, -87.6298],   # Chicago
                     [51.5074, -0.1278]])   # London

# Use haversine for geographic distance
db = DBSCAN(eps=0.5, min_samples=2, metric='haversine')
labels = db.fit_predict(coords)
print("Geo Clusters:", labels)
```

---

## 15. Real-World Use Cases

### 🗺️ 1. Geospatial Analysis
```
Cluster GPS locations of:
- Crime incidents in a city
- Store customer locations
- Earthquake epicenters
- Traffic accident hotspots
```

### 🛡️ 2. Anomaly / Fraud Detection
```
- Credit card fraud (noise points = suspicious transactions)
- Network intrusion detection
- Manufacturing defect detection
- Medical outlier detection
```

### 🧬 3. Biology & Medicine
```
- Protein structure clustering
- Gene expression analysis
- Cell type identification (scRNA-seq)
- MRI brain region segmentation
```

### 📸 4. Image Segmentation
```
- Cluster pixels by color/intensity
- Object detection preprocessing
- Satellite image analysis
```

### 📰 5. Document Clustering / NLP
```
- News article grouping (with TF-IDF + cosine distance)
- Social media topic detection
- Customer review clustering
```

### 🛒 6. Customer Segmentation
```
- Group customers by purchase behavior
- Identify niche customer segments
- Detect unusual shopping patterns
```

---

## 16. Variants of DBSCAN

### 🔷 HDBSCAN (Hierarchical DBSCAN)
> **Solves the varying density problem**

- Builds a cluster hierarchy across multiple density levels
- More robust parameter selection
- Better for real-world data

```python
# pip install hdbscan
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
labels = clusterer.fit_predict(X)
```

### 🔷 OPTICS (Ordering Points To Identify Clustering Structure)
> **Generalization of DBSCAN**

- Produces a reachability plot instead of a flat clustering
- Handles varying densities naturally
- More flexible but harder to interpret

```python
from sklearn.cluster import OPTICS
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
labels = optics.fit_predict(X)
```

### 🔷 DBSCAN++ 
- Faster approximation of DBSCAN
- Uses sampling to reduce computation

### 🔷 ST-DBSCAN (Spatio-Temporal DBSCAN)
- Handles data with spatial AND temporal dimensions
- Used for tracking moving objects, events over time

---

## 17. Evaluation Metrics

> ⚠️ Clustering evaluation is harder than classification — we often don't have ground truth.

### 📊 Internal Metrics (No Ground Truth Needed):

#### Silhouette Score
```python
from sklearn.metrics import silhouette_score

# Range: [-1, 1] — higher is better
# -1: wrong cluster, 0: overlapping, 1: perfect
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```

#### Davies-Bouldin Index
```python
from sklearn.metrics import davies_bouldin_score

# Lower is better (0 = perfect)
score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {score:.3f}")
```

#### Calinski-Harabasz Index
```python
from sklearn.metrics import calinski_harabasz_score

# Higher is better
score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {score:.3f}")
```

### 📊 External Metrics (With Ground Truth):

```python
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Adjusted Rand Index: [-1, 1], 1 = perfect
ari = adjusted_rand_score(y_true, labels)

# Adjusted Mutual Information: [0, 1], 1 = perfect
ami = adjusted_mutual_info_score(y_true, labels)

print(f"ARI: {ari:.3f}, AMI: {ami:.3f}")
```

### Handling Noise in Metrics:
```python
# Remove noise points (-1) before computing metrics
mask = labels != -1
score = silhouette_score(X[mask], labels[mask])
```

---

## 18. Common Mistakes & How to Avoid Them

### ❌ Mistake 1: Not Scaling Features
```python
# WRONG — DBSCAN uses distance, scale matters!
db = DBSCAN(eps=0.5).fit(X)

# CORRECT — Always normalize first
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.5).fit(X_scaled)
```

### ❌ Mistake 2: Using Too Small MinPts
```python
# WRONG — MinPts=1 or 2 creates meaningless clusters
db = DBSCAN(eps=0.5, min_samples=1)

# CORRECT — Use at least dimensions+1
min_samples = X.shape[1] + 1  # or 2 * n_features
db = DBSCAN(eps=0.5, min_samples=min_samples)
```

### ❌ Mistake 3: Ignoring the k-Distance Plot
```python
# Don't just guess eps!
# ALWAYS plot the k-distance graph first to find the elbow
```

### ❌ Mistake 4: Using DBSCAN for Very High-Dimensional Data
```python
# For high dimensions (>50 features), reduce first
from sklearn.decomposition import PCA
X_reduced = PCA(n_components=10).fit_transform(X)
# Then apply DBSCAN on X_reduced
```

### ❌ Mistake 5: Evaluating on Noise Points
```python
# WRONG — include noise in silhouette calculation
score = silhouette_score(X, labels)  # labels has -1

# CORRECT — exclude noise points
mask = labels != -1
score = silhouette_score(X[mask], labels[mask])
```

---

## 19. Practice Questions

### 🧠 Conceptual Questions:

1. What does DBSCAN stand for and what year was it introduced?
2. Explain the difference between a core point, border point, and noise point.
3. Why does DBSCAN not require specifying the number of clusters?
4. How does DBSCAN handle outliers differently from K-Means?
5. What happens to border points that are reachable from multiple clusters?
6. Why is feature scaling important before applying DBSCAN?
7. What is the k-distance graph and how is it used to choose eps?
8. Compare DBSCAN and HDBSCAN — when would you choose one over the other?

### 💻 Coding Questions:

9. Apply DBSCAN to the Iris dataset and compare with K-Means results.
10. Use DBSCAN to detect anomalies in a credit card transaction dataset.
11. Implement the k-distance plot and automatically detect the elbow point.
12. Apply DBSCAN with haversine metric to cluster GPS coordinates.
13. Compare Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for DBSCAN vs K-Means on a non-convex dataset.

### 🔬 Analytical Questions:

14. Given ε=1.5 and MinPts=3, manually classify these points as core/border/noise:
    - A(0,0), B(1,0), C(0,1), D(5,5), E(5.5,5), F(100,100)

15. How does increasing MinPts affect:
    - Number of core points?
    - Number of clusters?
    - Number of noise points?

---

## 20. Summary Cheat Sheet

```
╔══════════════════════════════════════════════════════════════╗
║                    DBSCAN CHEAT SHEET                       ║
╠══════════════════════════════════════════════════════════════╣
║ Full Name   │ Density-Based Spatial Clustering of           ║
║             │ Applications with Noise                       ║
╠═════════════╪════════════════════════════════════════════════╣
║ Type        │ Unsupervised, Density-Based Clustering         ║
║ Year        │ 1996 (Ester, Kriegel, Sander, Xu)             ║
╠═════════════╪════════════════════════════════════════════════╣
║ Parameters  │ eps (ε) — neighborhood radius                 ║
║             │ min_samples — minimum points for core         ║
╠═════════════╪════════════════════════════════════════════════╣
║ Point Types │ Core → |N_ε(p)| ≥ MinPts                     ║
║             │ Border → near core, but not core itself       ║
║             │ Noise → label = -1                            ║
╠═════════════╪════════════════════════════════════════════════╣
║ Strengths   │ No K needed, any shape, handles outliers      ║
║ Weaknesses  │ Varying densities, high dimensions, tuning    ║
╠═════════════╪════════════════════════════════════════════════╣
║ Complexity  │ O(n²) naive │ O(n log n) with KD-tree         ║
╠═════════════╪════════════════════════════════════════════════╣
║ Best For    │ Geo data, anomaly detection, irregular shapes ║
║ Avoid When  │ High-D data, very large N, varying density    ║
╠═════════════╪════════════════════════════════════════════════╣
║ sklearn     │ DBSCAN(eps=0.5, min_samples=5)                ║
║             │ .fit_predict(X_scaled)                        ║
╠═════════════╪════════════════════════════════════════════════╣
║ Variants    │ HDBSCAN (varying density)                     ║
║             │ OPTICS (reachability plot)                    ║
║             │ ST-DBSCAN (spatio-temporal)                   ║
╠═════════════╪════════════════════════════════════════════════╣
║ Key Tip     │ ALWAYS scale features before DBSCAN!          ║
║             │ Use k-distance graph to pick eps              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📖 References & Further Reading

- Ester, M., et al. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD-96.
- Scikit-learn DBSCAN documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- HDBSCAN documentation: https://hdbscan.readthedocs.io/
- Campello, R.J.G.B., et al. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates.* PAKDD.

---

*Created as a complete educational guide to DBSCAN clustering for machine learning students and practitioners.*