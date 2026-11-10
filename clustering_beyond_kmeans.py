"""
BEYOND K-MEANS: HIERARCHICAL, DBSCAN + SILHOUETTE + ARI
Quick comparison of clustering algorithms and metrics
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# =============================================================================
# WHAT IS HIERARCHICAL CLUSTERING?
# =============================================================================
# Builds a tree of clusters (dendrogram) by either:
#   - Agglomerative (bottom-up): Start with each point as own cluster,
#     repeatedly merge closest pairs until one cluster remains
#   - Divisive (top-down): Start with all points in one cluster,
#     repeatedly split until each point is its own cluster
#
# Why it exists:
#   - Don't need to specify K upfront — cut dendrogram at any level
#   - Visual interpretability — shows how clusters relate
#   - Handles non-spherical clusters better than K-Means
#
# Where to use:
#   - Small to medium datasets (<10k points — O(nÂ²) or O(nÂ³) complexity)
#   - When you want to explore multiple K without re-running
#   - When cluster hierarchy matters (e.g., biology: species â genus â family)

# =============================================================================
# WHAT IS DBSCAN?
# =============================================================================
# Density-Based Spatial Clustering of Applications with Noise
# Groups together points that are densely packed, marks isolated points as noise
#
# Key parameters:
#   - eps (Îµ): Maximum distance between two points to be considered "neighbors"
#   - min_samples: Minimum number of points to form a dense region (cluster)
#
# How it works:
#   1. Pick random unvisited point
#   2. If it has >= min_samples neighbors within eps, start new cluster
#   3. Expand cluster by adding all density-reachable points
#   4. Repeat until all points visited
#   5. Points not in any cluster = noise/outliers
#
# Why it exists:
#   - Finds arbitrary-shaped clusters (K-Means fails here)
#   - Robust to outliers — automatically identifies noise
#   - Don't need to specify K — algorithm discovers it from density
#
# Where to use:
#   - Spatial data (GPS coordinates, store locations)
#   - Anomaly detection (fraud, network intrusions)
#   - Non-spherical clusters (moon-shaped, ring-shaped)
#   - When you expect outliers/noise in data

# =============================================================================
# WHAT IS SILHOUETTE SCORE?
# =============================================================================
# Measures how similar a point is to its own cluster vs other clusters
#
# Formula for point i:
#   s(i) = (b(i) - a(i)) / max(a(i), b(i))
#   Where:
#     a(i) = avg distance to other points in SAME cluster (cohesion)
#     b(i) = avg distance to points in NEAREST OTHER cluster (separation)
#
# Range: -1 to +1
#   +1: Point is far from neighboring clusters (well-assigned)
#    0: Point is on decision boundary
#   -1: Point might be in wrong cluster
#
# Interpretation:
#   > 0.7: Strong clustering
#   0.5-0.7: Reasonable clustering
#   < 0.5: Weak clustering
#
# When to use:
#   - Compare different K values (pick K with highest silhouette)
#   - Evaluate clustering quality with no ground truth labels
#   - Identify poorly-assigned points (low silhouette = potential outliers)
#
# Limitation: Biased toward convex, compact clusters (favors K-Means)

# =============================================================================
# WHAT IS ADJUSTED RAND INDEX (ARI)?
# =============================================================================
# Measures similarity between two clusterings (e.g., discovered vs true labels),
# adjusted for chance
#
# Why "adjusted": Raw Rand Index can be high by random chance. ARI corrects for this.
#
# Range: -1 to +1
#   +1: Perfect match between two clusterings
#    0: Random labeling (no better than chance)
#   <0: Worse than random (clusterings disagree more than expected)
#
# When to use:
#   - You have GROUND TRUTH LABELS (synthetic data or expert-labeled subset)
#   - Compare different clustering algorithms (which recovers true labels better?)
#   - Tune hyperparameters (pick K that maximizes ARI against known labels)
#
# Key difference from Silhouette:
#   - Silhouette: Internal metric (uses only features, NO labels needed)
#   - ARI: External metric (REQUIRES ground truth labels)

# =============================================================================
# PART 1: CREATE NON-SPHERICAL DATA (MOON-SHAPED) - K-MEANS WILL FAIL
# =============================================================================

print("=" * 80)
print("PART 1: CREATE NON-SPHERICAL DATA (MOON-SHAPED)")
print("=" * 80)
print()

np.random.seed(42)
X_moons, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)

print(f"Dataset: {len(X_moons)} points in moon-shaped pattern")
print(f"True labels (for ARI): {np.bincount(y_true)}")
print()

# =============================================================================
# PART 2: FIT K-MEANS, HIERARCHICAL, AND DBSCAN
# =============================================================================

print("=" * 80)
print("PART 2: FIT THREE CLUSTERING ALGORITHMS")
print("=" * 80)
print()

# K-Means (assumes spherical clusters)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_moons)

# Hierarchical Clustering (Ward linkage)
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_moons)

# DBSCAN (density-based)
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_moons)

print("K-Means:")
print(f"  Clusters found: {len(np.unique(kmeans_labels))}")
print(f"  Points per cluster: {np.bincount(kmeans_labels)}")
print()

print("Hierarchical (Ward):")
print(f"  Clusters found: {len(np.unique(hierarchical_labels))}")
print(f"  Points per cluster: {np.bincount(hierarchical_labels)}")
print()

print("DBSCAN:")
print(f"  Clusters found: {len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
if -1 in dbscan_labels:
    print(f"  Noise points: {(dbscan_labels == -1).sum()}")
    print(f"  Points per cluster: {np.bincount(dbscan_labels[dbscan_labels != -1])}")
else:
    print(f"  Points per cluster: {np.bincount(dbscan_labels)}")
print()

# =============================================================================
# PART 3: EVALUATE WITH SILHOUETTE SCORE AND ADJUSTED RAND INDEX
# =============================================================================

print("=" * 80)
print("PART 3: EVALUATE WITH SILHOUETTE SCORE AND ADJUSTED RAND INDEX")
print("=" * 80)
print()

print(f"{'Algorithm':>15} {'Silhouette':>12} {'ARI (vs true)':>15}")
print("-" * 45)

for name, labels in [('K-Means', kmeans_labels),
                      ('Hierarchical', hierarchical_labels),
                      ('DBSCAN', dbscan_labels)]:
    sil = silhouette_score(X_moons, labels)
    ari = adjusted_rand_score(y_true, labels)
    print(f"{name:>15} {sil:>12.3f} {ari:>15.3f}")

print()
print("Interpretation:")
print("  - Silhouette: Higher = better internal structure")
print("  - ARI: Higher = better agreement with true labels")
print("  - DBSCAN should win on both (moons are non-spherical)")
print()

# =============================================================================
# PART 4: VISUALIZE ALL THREE
# =============================================================================

print("=" * 80)
print("PART 4: VISUALIZE CLUSTERING RESULTS")
print("=" * 80)
print()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# True labels
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
axes[0].set_title('True Labels')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].grid(True, alpha=0.3)

# K-Means
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
axes[1].set_title(f'K-Means\nSilhouette={silhouette_score(X_moons, kmeans_labels):.3f}, ARI={adjusted_rand_score(y_true, kmeans_labels):.3f}')
axes[1].set_xlabel('X1')
axes[1].grid(True, alpha=0.3)

# Hierarchical
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=hierarchical_labels, cmap='viridis', s=50, alpha=0.7)
axes[2].set_title(f'Hierarchical\nSilhouette={silhouette_score(X_moons, hierarchical_labels):.3f}, ARI={adjusted_rand_score(y_true, hierarchical_labels):.3f}')
axes[2].set_xlabel('X1')
axes[2].grid(True, alpha=0.3)

# DBSCAN
axes[3].scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.7)
n_clusters = len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
axes[3].set_title(f'DBSCAN\nClusters={n_clusters}, Silhouette={silhouette_score(X_moons, dbscan_labels):.3f}, ARI={adjusted_rand_score(y_true, dbscan_labels):.3f}')
axes[3].set_xlabel('X1')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/clustering_comparison.png', dpi=150)
plt.show()

print("Visualization saved to output/clustering_comparison.png")
print()
print("Key takeaway:")
print("  - K-Means splits moons in half (wrong â assumes spherical)")
print("  - Hierarchical does better but still struggles")
print("  - DBSCAN correctly identifies two moon-shaped clusters")
print("  - This is why algorithm choice matters!")
print()

# =============================================================================
# PART 5: WHEN TO USE WHICH ALGORITHM
# =============================================================================

print("=" * 80)
print("PART 5: WHEN TO USE WHICH CLUSTERING ALGORITHM")
print("=" * 80)
print()

print("USE K-MEANS WHEN:")
print("  - Clusters are spherical (round blobs)")
print("  - Dataset is large (scales to millions of points)")
print("  - You know K (or can use elbow method)")
print("  - You need speed (fastest of the three)")
print()

print("USE HIERARCHICAL WHEN:")
print("  - Dataset is small (<10k points)")
print("  - You want to explore multiple K without re-running")
print("  - Cluster hierarchy matters (e.g., taxonomy)")
print("  - You need interpretability (dendrogram visualization)")
print()

print("USE DBSCAN WHEN:")
print("  - Clusters are non-spherical (elongated, curved, irregular)")
print("  - You expect outliers/noise in data")
print("  - You don't know K (algorithm discovers it)")
print("  - Dataset is medium size (100 - 50k points)")
print()

print("DON'T USE:")
print("  - K-Means on non-spherical data (will fail like on moons)")
print("  - Hierarchical on large data (too slow: O(nÂ²) or O(nÂ³))")
print("  - DBSCAN when clusters have very different densities (eps is global)")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("SUMMARY: BEYOND K-MEANS")
print("=" * 80)
print()
print("Algorithms:")
print("  - K-Means: Fast, spherical clusters, need K")
print("  - Hierarchical: Tree structure, small data, no need to specify K upfront")
print("  - DBSCAN: Density-based, arbitrary shapes, robust to outliers")
print()
print("Metrics:")
print("  - Silhouette Score: Internal metric (no labels needed), -1 to +1, higher=better")
print("  - Adjusted Rand Index: External metric (needs labels), -1 to +1, 1=perfect match")
print()
print("Rule of thumb:")
print("  1. Try K-Means first (fast baseline)")
print("  2. If clusters look non-spherical, try DBSCAN")
print("  3. If you want to explore K values, try Hierarchical")
print("  4. Always evaluate with Silhouette (internal) or ARI (if labels exist)")