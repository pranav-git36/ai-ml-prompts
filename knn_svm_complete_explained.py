"""
KNN AND SVM - COMPLETE EXPLAINED CODE
Both algorithms with step-by-step explanations
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# PART 1: KNN - K-NEAREST NEIGHBORS
# =============================================================================

print("=" * 80)
print("PART 1: KNN - K-NEAREST NEIGHBORS")
print("=" * 80)
print()

# --- What ---
# KNN classifies a new point by looking at the K closest training points
# and letting them vote on the class.

# --- Why ---
# - No training phase (just memorizes data)
# - Captures non-linear patterns naturally
# - Simple to understand and explain ("your neighbors say you're class X")

# --- Where to Use ---
# - Recommendation systems ("users like you bought this")
# - Medical diagnosis ("patients with similar symptoms had this disease")
# - Small to medium datasets (prediction is slow on huge datasets)
# - Quick baseline classifier

# --- Key Hyperparameters ---
# - K: Number of neighbors (typical: 3, 5, 7, or sqrt(n))
#   - Small K (e.g., 1): Sensitive to noise, complex boundary
#   - Large K (e.g., 50): Smooth boundary, might oversimplify
# - Distance metric: Euclidean (default), Manhattan, Minkowski

print("STEP 1: CREATE SIMPLE 2D DATASET")
print("-" * 80)

# Create 6 training points with (x, y) coordinates and class labels
train_data = pd.DataFrame({
    'x': [1, 2, 2, 6, 7, 8],
    'y': [1, 2, 3, 6, 7, 8],
    'class': [0, 0, 0, 1, 1, 1]  # Class 0: bottom-left, Class 1: top-right
})

print("Training data:")
print(train_data)
print()

print("STEP 2: NEW POINT TO CLASSIFY")
print("-" * 80)

new_point = np.array([[4, 4]])  # Point at (4, 4) - which class?
print(f"New point coordinates: {new_point[0]}")
print()

print("STEP 3: COMPUTE EUCLIDEAN DISTANCES MANUALLY")
print("-" * 80)

# Compute distance from new point to each training point
distances = []
for i, row in train_data.iterrows():
    x1, y1 = row['x'], row['y']
    # Euclidean distance formula: sqrt((x1-x2)^2 + (y1-y2)^2)
    dist = np.sqrt((x1 - new_point[0][0])**2 + (y1 - new_point[0][1])**2)
    distances.append(dist)
    print(f"Point ({x1},{y1}, class={int(row['class'])}): distance = {dist:.3f}")

train_data['distance'] = distances
train_data_sorted = train_data.sort_values('distance')

print()
print("Sorted by distance (closest first):")
print(train_data_sorted[['x', 'y', 'class', 'distance']].round(3))
print()

print("STEP 4: KNN WITH K=3 - MAJORITY VOTE")
print("-" * 80)

k = 3
k_nearest = train_data_sorted.head(k)
print(f"K = {k} nearest neighbors:")
print(k_nearest[['x', 'y', 'class', 'distance']].round(3))
print()

# Count votes
class_0_votes = (k_nearest['class'] == 0).sum()
class_1_votes = (k_nearest['class'] == 1).sum()
predicted_class = 1 if class_1_votes > class_0_votes else 0

print(f"Votes: Class 0 = {class_0_votes}, Class 1 = {class_1_votes}")
print(f"Prediction: Class {predicted_class} (majority vote)")
print()

print("STEP 5: USING SKLEARN KNN (SAME RESULT)")
print("-" * 80)

X_train = train_data[['x', 'y']].values
y_train = train_data['class'].values

# IMPORTANT: Scale features for KNN (distance-based algorithm)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
new_point_scaled = scaler.transform(new_point)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)  # "Fit" just stores the training data
prediction = knn.predict(new_point_scaled)[0]

print(f"Sklearn KNN prediction: Class {prediction}")
print(f"Matches manual calculation: {prediction == predicted_class}")
print()

print("STEP 6: HOW K AFFECTS PREDICTION")
print("-" * 80)

for k_test in [1, 3, 5]:
    knn_test = KNeighborsClassifier(n_neighbors=k_test)
    knn_test.fit(X_train_scaled, y_train)
    pred_test = knn_test.predict(new_point_scaled)[0]
    print(f"K = {k_test}: Prediction = Class {pred_test}")

print()
print("Key takeaway: Different K values can give different predictions.")
print("  - K=1: Only the single nearest neighbor votes (sensitive to noise)")
print("  - K=3: 3 nearest neighbors vote (balanced)")
print("  - K=5: 5 nearest neighbors vote (smoother, might oversimplify)")
print()

# =============================================================================
# PART 2: SVM - SUPPORT VECTOR MACHINES
# =============================================================================

print("=" * 80)
print("PART 2: SVM - SUPPORT VECTOR MACHINES")
print("=" * 80)
print()

# --- What ---
# SVM finds the decision boundary that maximizes the margin (distance)
# between the boundary and the nearest points from each class (support vectors).

# --- Why ---
# - Maximum margin → better generalization to new data
# - Robust to noise (only support vectors matter, not all points)
# - Works well in high dimensions
# - Kernel trick handles non-linear boundaries

# --- Where to Use ---
# - Text classification (spam detection, sentiment analysis)
# - Image recognition (handwritten digits, object detection)
# - Bioinformatics (gene expression, protein function)
# - When classes are clearly separable with a margin

# --- Key Hyperparameters ---
# - C: Regularization strength
#   - Large C (e.g., 100): Hard margin, narrow, risk overfitting
#   - Small C (e.g., 0.01): Soft margin, wider, better generalization
# - Kernel: 'linear' (fast, for linearly separable), 'rbf' (non-linear, default)
# - gamma (for RBF): How far influence of a single training example reaches

print("STEP 1: CREATE SEPARABLE 2D DATASET")
print("-" * 80)

# Create two clearly separable clusters
np.random.seed(42)
class_0 = pd.DataFrame({
    'x': np.random.randn(20) * 1.5 + 2,  # Mean=2, Std=1.5
    'y': np.random.randn(20) * 1.5 + 2,
    'class': 0
})
class_1 = pd.DataFrame({
    'x': np.random.randn(20) * 1.5 + 6,  # Mean=6, Std=1.5
    'y': np.random.randn(20) * 1.5 + 6,
    'class': 1
})
svm_data = pd.concat([class_0, class_1], ignore_index=True)

print(f"Total samples: {len(svm_data)}")
print(f"Class 0: {len(class_0)}, Class 1: {len(class_1)}")
print()
print("First 5 rows:")
print(svm_data.head())
print()

print("STEP 2: FIT SVM WITH LINEAR KERNEL")
print("-" * 80)

X = svm_data[['x', 'y']].values
y = svm_data['class'].values

# Scale features (important for SVM)
scaler_svm = StandardScaler()
X_scaled = scaler_svm.fit_transform(X)

# Fit SVM with linear kernel
svm_linear = svm.SVC(kernel='linear', C=1.0)
svm_linear.fit(X_scaled, y)

print(f"Model: SVM with linear kernel, C=1.0")
print(f"Number of support vectors: {len(svm_linear.support_)}")
print(f"Support vectors are the critical points that define the boundary.")
print(f"Only these points matter - move any other point and boundary doesn't change.")
print()

print("STEP 3: MAKE PREDICTIONS ON TEST POINTS")
print("-" * 80)

test_points = np.array([[3, 3], [4, 4], [5, 5], [6, 6]])
test_scaled = scaler_svm.transform(test_points)

print(f"{'Test Point':>12} {'Prediction':>12}")
print("-" * 26)
for point, pred in zip(test_points, svm_linear.predict(test_scaled)):
    print(f"{str(point):>12} {pred:>12}")
print()

print("STEP 4: SVM WITH RBF KERNEL (NON-LINEAR)")
print("-" * 80)

svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_scaled, y)

print(f"Model: SVM with RBF kernel, C=1.0")
print(f"RBF kernel can capture non-linear (curved) boundaries.")
print(f"Number of support vectors: {len(svm_rbf.support_)}")
print()

print("Comparison of Linear vs RBF on test points:")
print(f"{'Test Point':>12} {'Linear':>10} {'RBF':>10}")
print("-" * 34)
for point, pred_lin, pred_rbf in zip(test_points,
                                      svm_linear.predict(test_scaled),
                                      svm_rbf.predict(test_scaled)):
    print(f"{str(point):>12} {pred_lin:>10} {pred_rbf:>10}")
print()
print("On this simple linearly-separable data, both kernels agree.")
print("On complex non-linear data (e.g., concentric circles), RBF would capture")
print("curved boundaries that linear SVM cannot.")
print()

print("STEP 5: EFFECT OF C PARAMETER (REGULARIZATION)")
print("-" * 80)

for c_val in [0.01, 1.0, 100]:
    svm_c = svm.SVC(kernel='linear', C=c_val)
    svm_c.fit(X_scaled, y)
    n_sv = len(svm_c.support_)
    acc = svm_c.score(X_scaled, y)
    print(f"C = {c_val:>6}: Support vectors = {n_sv}, Training accuracy = {acc:.1%}")

print()
print("Interpretation:")
print("  - Small C (0.01): Soft margin, wider, more support vectors, better generalization")
print("  - Large C (100): Hard margin, narrow, fewer support vectors, risk overfitting")
print()

print("=" * 80)
print("SUMMARY: KNN vs SVM")
print("=" * 80)
print()
print("KNN:")
print("  - What: Classify by K nearest neighbors' majority vote")
print("  - Pros: Simple, no training, interpretable, handles non-linear patterns")
print("  - Cons: Slow prediction on large data, needs scaling, sensitive to K")
print("  - Use when: Small dataset, need simple baseline, interpretability matters")
print()
print("SVM:")
print("  - What: Find maximum-margin decision boundary")
print("  - Pros: Good generalization, effective in high dimensions, versatile kernels")
print("  - Cons: Slow on huge datasets, sensitive to C/kernel, less interpretable")
print("  - Use when: Text/image classification, high-dimensional data, clear separation")
print()
print("Both need feature scaling (StandardScaler) before training!")