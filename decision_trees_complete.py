"""
DECISION TREES - COMPLETE DEEP DIVE
What / Why / Where / How + Manual Math + Sklearn + Overfitting Demo
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================================================
# WHAT IS A DECISION TREE?
# =============================================================================
# A flowchart-like model that makes predictions by asking yes/no questions
# about features, starting from root and following branches to a leaf node.
#
# Structure:
#   - Root node: First question (e.g., "Is Age < 30?")
#   - Internal nodes: Subsequent questions
#   - Branches: Yes/No outcomes
#   - Leaf nodes: Final predictions (class label or continuous value)
#
# For classification: Leaf predicts majority class of samples in that leaf
# For regression: Leaf predicts average value of samples in that leaf

# =============================================================================
# WHY DO DECISION TREES EXIST?
# =============================================================================
# Problems they solve that linear/logistic regression don't:
#
# 1. Non-linear relationships: Tree can capture complex rules like
#    "if Age < 30 AND Income > 50k then Class=1" — impossible for a straight line
#
# 2. Feature interactions: Automatically models interactions (e.g., "Age matters
#    only if Income is high") without manual interaction terms
#
# 3. No feature scaling needed: Unlike KNN/SVM/logistic regression, trees don't
#    care if features are on different scales
#
# 4. Interpretability: Can draw the tree and show exact rules to stakeholders

# =============================================================================
# WHERE TO USE DECISION TREES?
# =============================================================================
# - Credit scoring: "If Income > 50k AND DebtRatio < 0.4 → approve loan"
# - Medical diagnosis: "If Fever > 38°C AND Cough = Yes → likely viral"
# - Customer segmentation: "If Age < 25 AND PurchaseHistory > 5 → high-value"
# - Regulatory contexts: Banking/healthcare where you must explain every decision
# - Baseline model: Before trying Random Forest or Gradient Boosting

# =============================================================================
# HOW DOES A DECISION TREE LEARN? (THE ALGORITHM)
# =============================================================================
# Built top-down using "recursive binary splitting" (Hunt's algorithm):
#
# Step 1: Start at root — all training samples in one group
# Step 2: Find best split — try every "Is feature X < threshold T?" question,
#         pick the one that best separates classes (maximizes Information Gain)
# Step 3: Split data into two groups (Yes branch, No branch)
# Step 4: Recurse — repeat for each child node
# Step 5: Stop when: all samples pure, max depth reached, min samples per leaf
# Step 6: Predict — for new sample, follow branches to leaf, predict majority class

# =============================================================================
# PART 1: GINI IMPURITY - HOW TREE MEASURES 'GOODNESS' OF A SPLIT
# =============================================================================

print("=" * 80)
print("PART 1: GINI IMPURITY - FORMULA AND MANUAL CALCULATION")
print("=" * 80)
print()

# Formula: Gini = 1 - sum(p_k^2) for each class k
# Where p_k = fraction of samples in class k

print("Example 1: Pure node (all Class 0)")
p0, p1 = 1.0, 0.0
gini = 1 - (p0**2 + p1**2)
print(f"  10 samples: 10 Class 0, 0 Class 1")
print(f"  p_0 = {p0}, p_1 = {p1}")
print(f"  Gini = 1 - ({p0}^2 + {p1}^2) = {gini:.3f}")
print(f"  Interpretation: Gini = 0 means perfectly pure (all one class)")
print()

print("Example 2: Maximally impure node (50/50 split)")
p0, p1 = 0.5, 0.5
gini = 1 - (p0**2 + p1**2)
print(f"  10 samples: 5 Class 0, 5 Class 1")
print(f"  p_0 = {p0}, p_1 = {p1}")
print(f"  Gini = 1 - ({p0}^2 + {p1}^2) = {gini:.3f}")
print(f"  Interpretation: Gini = 0.5 is worst possible for binary classification")
print()

print("Example 3: Moderately impure node")
p0, p1 = 0.7, 0.3
gini = 1 - (p0**2 + p1**2)
print(f"  10 samples: 7 Class 0, 3 Class 1")
print(f"  p_0 = {p0}, p_1 = {p1}")
print(f"  Gini = 1 - ({p0}^2 + {p1}^2) = {gini:.3f}")
print()

# =============================================================================
# PART 2: INFORMATION GAIN - HOW TREE CHOOSES WHICH QUESTION TO ASK
# =============================================================================

print("=" * 80)
print("PART 2: INFORMATION GAIN - MANUAL CALCULATION OF BEST SPLIT")
print("=" * 80)
print()

# Create tiny dataset: predicting Pass/Fail from Hours and Practice
data = pd.DataFrame({
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Practice': [0, 0, 0, 0, 1, 1, 1, 1],  # 0 = No practice, 1 = Did practice
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1]       # Target
})

print("Dataset: Predict Pass/Fail from Hours studied and Practice (Y/N)")
print(data)
print()

# Parent node (all 8 samples): 3 Fail, 5 Pass
parent_p0 = 3/8
parent_p1 = 5/8
parent_gini = 1 - (parent_p0**2 + parent_p1**2)
print(f"Parent node (all 8 samples): 3 Fail, 5 Pass")
print(f"  Parent Gini = 1 - ({parent_p0:.3f}^2 + {parent_p1:.3f}^2) = {parent_gini:.3f}")
print()

# Try split: "Hours < 4?"
print("Try split 1: 'Hours < 4?'")
print("-" * 80)

# Left child (Hours < 4): samples 0,1,2 → 3 Fail, 0 Pass
left_p0, left_p1 = 1.0, 0.0
left_gini = 1 - (left_p0**2 + left_p1**2)
print(f"  Left child (Hours < 4): 3 samples, all Fail")
print(f"  Left Gini = {left_gini:.3f} (pure node)")

# Right child (Hours >= 4): samples 3,4,5,6,7 → 0 Fail, 5 Pass
right_p0, right_p1 = 0.0, 1.0
right_gini = 1 - (right_p0**2 + right_p1**2)
print(f"  Right child (Hours >= 4): 5 samples, all Pass")
print(f"  Right Gini = {right_gini:.3f} (pure node)")

# Weighted average Gini after split
weighted_gini_hours = (3/8) * left_gini + (5/8) * right_gini
info_gain_hours = parent_gini - weighted_gini_hours
print(f"  Weighted Gini = (3/8)*{left_gini} + (5/8)*{right_gini} = {weighted_gini_hours:.3f}")
print(f"  Information Gain = {parent_gini:.3f} - {weighted_gini_hours:.3f} = {info_gain_hours:.3f}")
print()

# Try split: "Practice < 0.5?" (i.e., Practice = No vs Yes)
print("Try split 2: 'Practice = No vs Yes?'")
print("-" * 80)

# Left child (Practice = No): samples 0,1,2,3 → 3 Fail, 1 Pass
left_p0, left_p1 = 3/4, 1/4
left_gini = 1 - (left_p0**2 + left_p1**2)
print(f"  Left child (Practice = No): 4 samples, 3 Fail, 1 Pass")
print(f"  Left Gini = 1 - ({left_p0}^2 + {left_p1}^2) = {left_gini:.3f}")

# Right child (Practice = Yes): samples 4,5,6,7 → 0 Fail, 4 Pass
right_p0, right_p1 = 0.0, 1.0
right_gini = 1 - (right_p0**2 + right_p1**2)
print(f"  Right child (Practice = Yes): 4 samples, all Pass")
print(f"  Right Gini = {right_gini:.3f} (pure node)")

# Weighted average Gini after split
weighted_gini_practice = (4/8) * left_gini + (4/8) * right_gini
info_gain_practice = parent_gini - weighted_gini_practice
print(f"  Weighted Gini = (4/8)*{left_gini:.3f} + (4/8)*{right_gini} = {weighted_gini_practice:.3f}")
print(f"  Information Gain = {parent_gini:.3f} - {weighted_gini_practice:.3f} = {info_gain_practice:.3f}")
print()

print("COMPARISON:")
print(f"  Split on 'Hours < 4': Information Gain = {info_gain_hours:.3f}")
print(f"  Split on 'Practice < 0.5': Information Gain = {info_gain_practice:.3f}")
print()
print(f"Decision: Tree picks 'Hours < 4' as the first split because it has")
print(f"higher Information Gain ({info_gain_hours:.3f} > {info_gain_practice:.3f}).")
print()

# =============================================================================
# PART 3: FIT DECISION TREE IN SKLEARN AND VISUALIZE
# =============================================================================

print("=" * 80)
print("PART 3: FIT DECISION TREE IN SKLEARN")
print("=" * 80)
print()

X = data[['Hours', 'Practice']].values
y = data['Pass'].values

# Fit tree with max_depth=2 (for simplicity)
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)
tree_model.fit(X, y)

print("Fitted Decision Tree:")
print(f"  Criterion: gini")
print(f"  Max depth: 2")
print(f"  Number of nodes: {tree_model.tree_.node_count}")
print()

# Visualize the tree
plt.figure(figsize=(10, 6))
plot_tree(tree_model, 
          feature_names=['Hours', 'Practice'], 
          class_names=['Fail', 'Pass'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree - Predict Pass/Fail from Hours and Practice')
plt.savefig('output/decision_tree_visual.png', dpi=150)
plt.show()

print("Tree visualization saved to output/decision_tree_visual.png")
print()
print("How to read the tree:")
print("  - Each box is a node")
print("  - Top box = root node (first question)")
print("  - Arrows = branches (True/False outcome)")
print("  - Bottom boxes = leaf nodes (final predictions)")
print("  - 'gini' = impurity at that node (0 = pure)")
print("  - 'samples' = number of training samples in that node")
print("  - 'value' = [count Class 0, count Class 1]")
print("  - 'class' = predicted class for that node")
print()

# =============================================================================
# PART 4: MAKE PREDICTIONS - TRACE THROUGH THE TREE
# =============================================================================

print("=" * 80)
print("PART 4: TRACE HOW TREE MAKES A PREDICTION")
print("=" * 80)
print()

test_samples = pd.DataFrame({
    'Hours': [2, 5, 7],
    'Practice': [0, 0, 1]
})

print("Test samples:")
print(test_samples)
print()

for i, row in test_samples.iterrows():
    hours, practice = row['Hours'], row['Practice']
    pred = tree_model.predict([[hours, practice]])[0]
    prob = tree_model.predict_proba([[hours, practice]])[0]
    
    print(f"Sample {i+1}: Hours={hours}, Practice={practice}")
    print(f"  Prediction: {'Pass' if pred==1 else 'Fail'} (Class {pred})")
    print(f"  Probability: {prob[0]:.1%} Fail, {prob[1]:.1%} Pass")
    print()

print("Key takeaway: Tree asks a series of yes/no questions, follows branches,")
print("and predicts the majority class of samples in the final leaf.")
print()

# =============================================================================
# PART 5: OVERFITTING IN DECISION TREES - WHY WE LIMIT DEPTH
# =============================================================================

print("=" * 80)
print("PART 5: OVERFITTING - WHY WE LIMIT TREE DEPTH")
print("=" * 80)
print()

# Create a noisier dataset
np.random.seed(123)
n = 100
hours = np.random.randint(1, 10, n)
practice = np.random.randint(0, 2, n)
# True pattern: Pass if Hours > 5 OR (Hours > 3 AND Practice=1), plus some noise
noise = np.random.rand(n) < 0.15
pass_true = ((hours > 5) | ((hours > 3) & (practice == 1))).astype(int)
pass_noisy = np.where(noise, 1 - pass_true, pass_true)  # Flip 15% labels

tree_data = pd.DataFrame({
    'Hours': hours,
    'Practice': practice,
    'Pass': pass_noisy
})

X_big = tree_data[['Hours', 'Practice']].values
y_big = tree_data['Pass'].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_big, y_big, test_size=0.3, random_state=42)

print(f"Dataset: {n} samples, {y_big.sum()} Pass ({y_big.mean()*100:.1f}%)")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print()

# Tree 1: Unlimited depth (overfits)
tree_unlimited = DecisionTreeClassifier(random_state=42)
tree_unlimited.fit(X_train, y_train)
train_acc_unlimited = tree_unlimited.score(X_train, y_train)
test_acc_unlimited = tree_unlimited.score(X_test, y_test)

# Tree 2: Limited depth (better generalization)
tree_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_limited.fit(X_train, y_train)
train_acc_limited = tree_limited.score(X_train, y_train)
test_acc_limited = tree_limited.score(X_test, y_test)

print("Comparison:")
print(f"{'Model':>25} {'Train Acc':>12} {'Test Acc':>12} {'Gap':>10}")
print("-" * 60)
print(f"{'Unlimited depth':>25} {train_acc_unlimited:>12.1%} {test_acc_unlimited:>12.1%} {train_acc_unlimited - test_acc_unlimited:>10.1%}")
print(f"{'Max depth = 3':>25} {train_acc_limited:>12.1%} {test_acc_limited:>12.1%} {train_acc_limited - test_acc_limited:>10.1%}")
print()

print("Interpretation:")
print("  - Unlimited tree: 100% train accuracy (memorized every sample), but only")
print(f"    {test_acc_unlimited:.1%} test accuracy (overfit - won't generalize)")
print("  - Limited depth (3): Slightly lower train accuracy, but better test accuracy")
print("    (better generalization - this is what we want)")
print()
print("Lesson: Always limit max_depth or use other regularization to prevent overfitting!")
print()

# =============================================================================
# KEY HYPERPARAMETERS TO TUNE
# =============================================================================

print("=" * 80)
print("KEY HYPERPARAMETERS TO TUNE")
print("=" * 80)
print()
print("max_depth: Maximum depth of the tree (typical: 3-10)")
print("  - Prevents overfitting by limiting how many questions can be asked")
print("  - Too small: underfitting (too simple)")
print("  - Too large: overfitting (memorizes noise)")
print()
print("min_samples_split: Minimum samples required to split an internal node (default: 2)")
print("  - Increase to smooth the tree (e.g., 10, 20)")
print()
print("min_samples_leaf: Minimum samples required in a leaf node (default: 1)")
print("  - Increase to prevent leaves with very few samples (e.g., 5, 10)")
print()
print("criterion: 'gini' or 'entropy' (both work well; 'gini' is slightly faster)")
print("  - Gini: 1 - sum(p_k^2)")
print("  - Entropy: -sum(p_k * log2(p_k))")
print()

# =============================================================================
# WHEN TO USE DECISION TREES OVER OTHER ALGORITHMS
# =============================================================================

print("=" * 80)
print("WHEN TO USE DECISION TREES OVER OTHER ALGORITHMS")
print("=" * 80)
print()
print("Use Decision Tree INSTEAD OF Logistic Regression when:")
print("  - Relationship is non-linear (tree captures complex rules)")
print("  - You need interpretability (can show exact rules to stakeholders)")
print("  - Features are on very different scales (no scaling needed)")
print("  - Feature interactions are important (tree models them automatically)")
print()
print("Use Decision Tree INSTEAD OF KNN when:")
print("  - Dataset is large (KNN prediction is slow, tree prediction is fast)")
print("  - You need a compact model (tree is small, KNN stores all training data)")
print("  - Interpretability matters (tree shows rules, KNN just says 'neighbors voted')")
print()
print("Use Decision Tree INSTEAD OF SVM when:")
print("  - You need interpretability (SVM is a black box)")
print("  - Data is tabular with mixed feature types (SVM better for text/images)")
print("  - You need probability outputs (tree gives class probabilities easily)")
print()
print("DON'T use Decision Tree when:")
print("  - Data is very high-dimensional (text, images) — use SVM or neural nets")
print("  - Relationship is truly linear — logistic regression is simpler and more stable")
print("  - You need the absolute best accuracy — use Random Forest or Gradient Boosting")
print("    (ensembles of trees, covered next)")