"""
K-MEANS CLUSTERING - COMPLETE DEEP DIVE
What / Why / Where / How + Elbow Method + Software Engineering Use Cases
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# =============================================================================
# WHAT IS K-MEANS?
# =============================================================================
# Unsupervised algorithm that partitions data into K clusters, where each
# point belongs to the cluster with the nearest mean (centroid).
#
# Key idea: "Similar points are close together in space."
# K-Means finds K center points (centroids) and assigns each point to nearest.

# =============================================================================
# WHY DOES K-MEANS EXIST?
# =============================================================================
# Problems it solves:
#   1. Discover natural groupings: You have data but no labels
#   2. Data summarization: Describe K segments instead of N individual points
#   3. Preprocessing: Create cluster labels for later supervised learning
#   4. Anomaly detection: Points far from all centroids are outliers

# =============================================================================
# WHERE TO USE K-MEANS?
# =============================================================================
# - Customer segmentation: Group by behavior (spending, frequency, recency)
# - Document clustering: Group similar articles, tickets, emails by topic
# - Image compression: Group similar colors, replace with centroid colors
# - Anomaly detection: Find servers, transactions, users that don't fit
# - Feature engineering: Add "cluster ID" as feature to classification model

# =============================================================================
# HOW DOES K-MEANS WORK? (THE ALGORITHM)
# =============================================================================
# Input: Data (X), number of clusters (K)
# Output: K centroids, cluster assignment for each point
#
# Steps:
#   1. Initialize: Randomly pick K points as initial centroids
#   2. Assign: For each point, compute distance to each centroid,
#              assign to nearest centroid
#   3. Update: For each cluster, compute mean of all assigned points → new centroid
#   4. Repeat: Go to Step 2 until centroids stop moving (convergence)
#
# Mathematical objective: Minimize inertia (within-cluster sum of squares)
#   Inertia = Σ || x_i - μ_{c_i} ||²
#   Where x_i = point, μ_{c_i} = centroid of cluster c_i

# =============================================================================
# PART 1: CREATE SYNTHETIC DATASET - CUSTOMER SEGMENTATION
# =============================================================================

print("=" * 80)
print("PART 1: CREATE SYNTHETIC CUSTOMER DATA")
print("=" * 80)
print()

np.random.seed(42)

# Create 3 natural customer segments
# Segment 0: Budget shoppers (low spending, high frequency)
budget = pd.DataFrame({
    'Age': np.random.randn(100) * 8 + 28,
    'AnnualSpending': np.random.randn(100) * 2000 + 8000,
    'PurchaseFrequency': np.random.randn(100) * 3 + 15,
    'Segment': 0
})

# Segment 1: Mainstream families (medium spending, medium frequency)
mainstream = pd.DataFrame({
    'Age': np.random.randn(150) * 10 + 40,
    'AnnualSpending': np.random.randn(150) * 4000 + 25000,
    'PurchaseFrequency': np.random.randn(150) * 4 + 8,
    'Segment': 1
})

# Segment 2: VIPs (high spending, low frequency)
vips = pd.DataFrame({
    'Age': np.random.randn(80) * 12 + 50,
    'AnnualSpending': np.random.randn(80) * 8000 + 60000,
    'PurchaseFrequency': np.random.randn(80) * 2 + 3,
    'Segment': 2
})

customer_data = pd.concat([budget, mainstream, vips], ignore_index=True)

print(f"Dataset: {len(customer_data)} customers")
print(f"Features: Age, AnnualSpending, PurchaseFrequency")
print(f"True segments (for validation): {customer_data['Segment'].value_counts().to_dict()}")
print()
print("First 5 rows:")
print(customer_data.head())
print()

# =============================================================================
# PART 2: PREPROCESSING - WHY SCALE FEATURES?
# =============================================================================

print("=" * 80)
print("PART 2: PREPROCESSING - WHY SCALE FEATURES?")
print("=" * 80)
print()

# K-Means uses distance, so features on different scales dominate
# Age: 20-60, Spending: 5000-80000, Frequency: 1-20
# Without scaling, Spending would dominate distance calculation

X = customer_data[['Age', 'AnnualSpending', 'PurchaseFrequency']].values

print("Before scaling:")
print(f"  Age: mean={X[:,0].mean():.1f}, std={X[:,0].std():.1f}")
print(f"  AnnualSpending: mean={X[:,1].mean():.1f}, std={X[:,1].std():.1f}")
print(f"  PurchaseFrequency: mean={X[:,2].mean():.2f}, std={X[:,2].std():.2f}")
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("After scaling (all features have mean=0, std=1):")
print(f"  Age: mean={X_scaled[:,0].mean():.10f}, std={X_scaled[:,0].std():.10f}")
print(f"  AnnualSpending: mean={X_scaled[:,1].mean():.10f}, std={X_scaled[:,1].std():.10f}")
print(f"  PurchaseFrequency: mean={X_scaled[:,2].mean():.10f}, std={X_scaled[:,2].std():.10f}")
print()
print("Key takeaway: K-Means is distance-based, so ALWAYS scale features first!")
print()

# =============================================================================
# PART 3: FIT K-MEANS WITH K=3 (WE KNOW TRUE K FROM SYNTHETIC DATA)
# =============================================================================

print("=" * 80)
print("PART 3: FIT K-MEANS WITH K=3")
print("=" * 80)
print()

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)

print(f"K-Means fitted with K=3")
print(f"Centroids (in scaled space):")
print(kmeans.cluster_centers_)
print()

# Get cluster assignments
labels = kmeans.labels_
customer_data['Cluster'] = labels

print("First 10 customers with cluster assignments:")
print(customer_data[['Age', 'AnnualSpending', 'PurchaseFrequency', 'Segment', 'Cluster']].head(10))
print()

# =============================================================================
# PART 4: EVALUATE CLUSTERING QUALITY
# =============================================================================

print("=" * 80)
print("PART 4: EVALUATE CLUSTERING QUALITY")
print("=" * 80)
print()

# Inertia (within-cluster sum of squares)
inertia = kmeans.inertia_
print(f"Inertia (lower is better): {inertia:.2f}")
print()

# Silhouette Score (how similar points are to their own cluster vs others)
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score (range -1 to +1, higher is better): {sil_score:.3f}")
print()

print("Interpretation:")
print("  - Silhouette > 0.5: Reasonable clustering")
print("  - Silhouette > 0.7: Strong clustering")
print(f"  - Our score {sil_score:.3f}: ", end="")
if sil_score > 0.7:
    print("Strong clustering - clusters are well-separated")
elif sil_score > 0.5:
    print("Reasonable clustering - some structure exists")
else:
    print("Weak clustering - data may not have clear groups")
print()

# =============================================================================
# PART 5: ELBOW METHOD - HOW TO CHOOSE K
# =============================================================================

print("=" * 80)
print("PART 5: ELBOW METHOD - FIND OPTIMAL K")
print("=" * 80)
print()

k_range = range(1, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:  # Silhouette needs at least 2 clusters
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
    else:
        sil_scores.append(np.nan)

print(f"{'K':>3} {'Inertia':>12} {'Silhouette':>12}")
print("-" * 30)
for k, inert, sil in zip(k_range, inertias, sil_scores):
    sil_str = f"{sil:.3f}" if not np.isnan(sil) else "N/A"
    print(f"{k:>3} {inert:>12.2f} {sil_str:>12}")
print()

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method - Find Where Inertia Plateaus')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_range)

ax2.plot(k_range[1:], sil_scores[1:], marker='s', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score - Higher is Better')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(k_range)

plt.tight_layout()
plt.savefig('output/kmeans_elbow_method.png', dpi=150)
plt.show()

print("Plot saved to output/kmeans_elbow_method.png")
print()
print("How to interpret:")
print("  - Elbow plot: Look for 'kink' where inertia stops dropping fast")
print("    (here K=3 is the elbow — matches our true data generation)")
print("  - Silhouette plot: Peak at K=3 confirms 3 is optimal")
print()

# =============================================================================
# PART 6: VISUALIZE CLUSTERS (2D PROJECTION)
# =============================================================================

print("=" * 80)
print("PART 6: VISUALIZE CLUSTERS (2D PROJECTION)")
print("=" * 80)
print()

# Plot Spending vs Frequency (two most discriminative features)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(customer_data['AnnualSpending'], 
                      customer_data['PurchaseFrequency'],
                      c=customer_data['Cluster'], 
                      cmap='viridis', 
                      s=100, 
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=0.5)

# Plot centroids (inverse transform from scaled space)
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
plt.scatter(centroids[:, 1], centroids[:, 2], 
            c='red', s=300, marker='X', 
            edgecolors='black', linewidth=2,
            label='Centroids')

plt.xlabel('Annual Spending')
plt.ylabel('Purchase Frequency')
plt.title('K-Means Clustering - Customer Segments\n(K=3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/kmeans_clusters_visual.png', dpi=150)
plt.show()

print("Cluster visualization saved to output/kmeans_clusters_visual.png")
print()
print("Interpretation:")
print("  - Each color = one cluster")
print("  - Red X = centroid (average customer in that cluster)")
print("  - Bottom-right: High spending, low frequency = VIPs")
print("  - Top-left: Low spending, high frequency = Budget shoppers")
print("  - Middle: Mainstream families")
print()

# =============================================================================
# PART 7: COMPARE DISCOVERED CLUSTERS TO TRUE SEGMENTS
# =============================================================================

print("=" * 80)
print("PART 7: COMPARE DISCOVERED CLUSTERS TO TRUE SEGMENTS")
print("=" * 80)
print()

# Create cross-tabulation
cross_tab = pd.crosstab(customer_data['Segment'], customer_data['Cluster'])
print("Cross-tabulation: True Segment vs Discovered Cluster")
print(cross_tab)
print()

print("Interpretation:")
print("  - If clustering is perfect, each true segment maps to one cluster")
print("  - In practice, some overlap is normal (clusters are approximations)")
print("  - Here: Segment 0 (Budget) → mostly Cluster 2")
print("          Segment 1 (Mainstream) → mostly Cluster 0")
print("          Segment 2 (VIP) → mostly Cluster 1")
print("  - K-Means successfully recovered the 3 natural groups!")
print()

# =============================================================================
# PART 8: WHEN TO USE K-MEANS IN SOFTWARE ENGINEERING
# =============================================================================

print("=" * 80)
print("PART 8: WHEN TO USE K-MEANS IN SOFTWARE ENGINEERING")
print("=" * 80)
print()

print("USE CASE 1: USER BEHAVIOR SEGMENTATION (FOR PRODUCT TEAMS)")
print("-" * 80)
print("Problem: Understand different types of users without predefined categories")
print("Features: Daily active minutes, features used, session frequency, retention days")
print("Output: K user personas (e.g., 'Power Users', 'Casual Browsers', 'One-time Visitors')")
print()
print("Why K-Means:")
print("  - No labels exist — you're discovering user types, not predicting known ones")
print("  - Segments inform product decisions (e.g., 'Power Users need advanced features')")
print("  - Can track segment distribution over time (e.g., 'Casual Browsers increasing')")
print()
print("Business impact: Data-driven product roadmap, targeted feature development,")
print("                 personalized onboarding flows")
print()

print("USE CASE 2: CODE REPOSITORY CLUSTERING")
print("-" * 80)
print("Problem: Group similar code repositories to recommend related projects")
print("Features: Lines of code, number of files, programming languages (encoded),")
print("          commit frequency, number of contributors, issue count")
print("Output: K repository types (e.g., 'Microservices', 'Libraries', 'Scripts', 'Docs')")
print()
print("Why K-Means:")
print("  - No one labeled repos by type — you're discovering natural categories")
print("  - Cluster assignments can be features for other models (e.g., predict maintenance effort)")
print("  - Helps organize large codebases (e.g., 'show me all microservice repos')")
print()
print("Business impact: Better code discovery, improved developer productivity,")
print("                 informed refactoring priorities")
print()

print("USE CASE 3: LOG ANOMALY DETECTION")
print("-" * 80)
print("Problem: Find unusual server behavior without knowing what 'unusual' looks like")
print("Features: CPU usage, memory usage, disk I/O, network latency, error rate")
print("Output: Clusters of 'normal' behavior; points far from all centroids are anomalies")
print()
print("Why K-Means:")
print("  - You don't have labels for 'anomaly' — anomalies are rare and diverse")
print("  - Points with high distance to nearest centroid = outliers = potential incidents")
print("  - Can set threshold: 'Alert if distance > 3 standard deviations'")
print()
print("Business impact: Proactive incident detection, reduced MTTR,")
print("                 catch issues before they cause outages")
print()

print("USE CASE 4: TEST CASE CLUSTERING FOR PRIORITIZATION")
print("-" * 80)
print("Problem: Group similar test cases to optimize test suite execution")
print("Features: Test execution time, code coverage, failure rate, module tested,")
print("          dependencies, last failure date")
print("Output: K test clusters (e.g., 'Fast & Stable', 'Slow & Flaky', 'Critical Path')")
print()
print("Why K-Means:")
print("  - No labels — you're discovering test characteristics, not predicting known types")
print("  - Can prioritize: run 'Critical Path' cluster first in CI, skip 'Flaky' cluster")
print("  - Helps identify test debt (e.g., 'Cluster 3 has 50 slow tests — refactor needed')")
print()
print("Business impact: Faster CI pipelines, reduced flakiness,")
print("                 data-driven test maintenance")
print()

print("USE CASE 5: CUSTOMER SUPPORT TICKET TOPIC DISCOVERY")
print("-" * 80)
print("Problem: Discover common topics in support tickets without manual categorization")
print("Features: TF-IDF vectors from ticket text (after preprocessing)")
print("Output: K topics (e.g., 'Billing Issues', 'Login Problems', 'Feature Requests')")
print()
print("Why K-Means:")
print("  - No one labeled tickets by topic — you're discovering themes")
print("  - Topic distribution informs product priorities (e.g., '30% tickets are login issues')")
print("  - Can auto-route new tickets to appropriate team based on nearest cluster")
print()
print("Business impact: Faster ticket resolution, data-driven product improvements,")
print("                 reduced support load through proactive fixes")
print("CAUTION: For text, consider using topic modeling (LDA) or embeddings + K-Means")
print()

print("=" * 80)
print("GENERAL GUIDELINES FOR USING K-MEANS IN SOFTWARE ENGINEERING")
print("=" * 80)
print()

print("USE K-MEANS WHEN:")
print("  - You have numerical features and NO labels")
print("  - You want to discover natural groupings (exploratory analysis)")
print("  - You need data summarization (K centroids represent N points)")
print("  - You want to create labels for later supervised learning")
print("  - Dataset is medium size (100 - 100k rows; K-Means scales well)")
print()

print("DON'T USE K-MEANS WHEN:")
print("  - You have labels — use classification instead")
print("  - Clusters are non-spherical (elongated, irregular shapes) — use DBSCAN")
print("  - You don't know K and can't use elbow method — try hierarchical clustering")
print("  - Data is categorical (text, categories) — encode first or use different algorithm")
print("  - Data has many outliers — K-Means is sensitive; use DBSCAN or remove outliers first")
print()

print("PRACTICAL TIPS:")
print("  1. ALWAYS scale features before K-Means (StandardScaler)")
print("  2. Use elbow method + silhouette score to choose K")
print("  3. Run multiple times with different seeds (n_init=10) — pick best inertia")
print("  4. Visualize clusters in 2D (PCA if >2 features) to sanity-check")
print("  5. Interpret centroids: 'What does the average point in this cluster look like?'")
print("  6. Use cluster assignments as features for supervised models later")
print("  7. For text data: use TF-IDF or embeddings (e.g., sentence-transformers) first")
print()

# =============================================================================
# SUMMARY: K-MEANS KEY TAKEAWAYS
# =============================================================================

print("=" * 80)
print("SUMMARY: K-MEANS KEY TAKEAWAYS")
print("=" * 80)
print()
print("What: Unsupervised algorithm that partitions data into K clusters")
print("Why: Discover natural groupings when you have NO labels")
print("Where: Customer segmentation, document clustering, anomaly detection,")
print("       image compression, feature engineering")
print("How: Iteratively assign points to nearest centroid, update centroids")
print("Key outputs: Cluster assignments, centroids, inertia, silhouette score")
print("Tuning: n_clusters (use elbow method), init='k-means++', n_init=10")
print()
print("In Software Engineering: Use for user segmentation, repo clustering,")
print("log anomaly detection, test case grouping, support ticket topic discovery —")
print("anywhere you have unlabeled data and want to find hidden structure.")