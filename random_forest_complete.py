"""
RANDOM FOREST - COMPLETE DEEP DIVE
What / Why / Where / How + Feature Importance + OOB Score + When to Use in Software Engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# =============================================================================
# WHAT IS RANDOM FOREST?
# =============================================================================
# An ensemble method that builds MANY decision trees and combines their
# predictions (majority vote for classification, average for regression).
#
# Key idea: "Wisdom of crowds" — many weak learners together make a strong
# learner, as long as they're diverse (not all making the same mistakes).

# =============================================================================
# WHY DOES RANDOM FOREST EXIST?
# =============================================================================
# Problems with a single Decision Tree:
#   1. High variance: Small change in data → completely different tree (unstable)
#   2. Overfitting: Can grow deep and memorize noise, especially with many features
#   3. Greedy splits: Each split is locally optimal, not globally
#
# Random Forest fixes these by:
#   1. Bagging (Bootstrap Aggregating): Train each tree on different random
#      subset of data (sampled with replacement) → creates diversity
#   2. Feature randomness: At each split, consider only random subset of features
#      (e.g., √p features) → decorrelates trees, they don't all rely on same feature
#   3. Averaging predictions: Combining many trees cancels out individual errors

# =============================================================================
# WHERE TO USE RANDOM FOREST?
# =============================================================================
# - Default classifier: Good accuracy fast without much tuning
# - Tabular data: Structured data (rows × columns) — beats SVM and neural nets
#   on most tabular problems
# - Feature importance: Built-in measure of which features matter most
# - When interpretability is secondary: Less interpretable than single tree,
#   but you can still extract feature importance and partial dependence plots
# - Baseline for competitions: Strong, fast baseline before trying XGBoost/LightGBM

# =============================================================================
# HOW DOES RANDOM FOREST WORK? (THE ALGORITHM)
# =============================================================================
# Training phase:
#   1. Decide number of trees (n_estimators, e.g., 100 or 500)
#   2. For each tree:
#      - Bootstrap sample: Randomly sample n samples WITH REPLACEMENT
#        (some appear multiple times, some not at all — "out-of-bag" samples)
#      - Grow tree: At each split, consider only m features (e.g., m = √p)
#      - Grow fully: Don't prune — let each tree grow deep (low bias, high variance)
#   3. Store all trees
#
# Prediction phase:
#   1. Pass new sample through EVERY tree in the forest
#   2. Each tree votes for a class
#   3. Majority vote wins (classification) or average (regression)

# =============================================================================
# PART 1: CREATE SYNTHETIC DATASET - PREDICT LOAN DEFAULT
# =============================================================================

print("=" * 80)
print("PART 1: CREATE SYNTHETIC DATASET")
print("=" * 80)
print()

np.random.seed(42)
n = 500

# Features
income = np.random.randn(n) * 20 + 55          # Income: mean 55k, std 20k
debt_ratio = np.clip(np.random.randn(n) * 0.2 + 0.35, 0.1, 0.8)
credit_score = np.clip(np.random.randn(n) * 100 + 620, 300, 850)
age = np.random.randint(22, 65, n)
loan_amount = np.random.randn(n) * 10 + 50     # Loan amount in lakhs

# True pattern: Default if DebtRatio high OR CreditScore low, plus noise
log_odds = -5 + 8*debt_ratio - 0.01*credit_score + 0.01*income
prob_default = 1 / (1 + np.exp(-log_odds))
noise = np.random.rand(n) < 0.1  # 10% label noise
default = ((np.random.rand(n) < prob_default) ^ noise).astype(int)

rf_data = pd.DataFrame({
    'Income': income,
    'DebtRatio': debt_ratio,
    'CreditScore': credit_score,
    'Age': age,
    'LoanAmount': loan_amount,
    'Default': default
})

print(f"Dataset: {n} samples, {default.sum()} defaults ({default.mean()*100:.1f}%)")
print(f"Features: Income, DebtRatio, CreditScore, Age, LoanAmount")
print()
print("First 5 rows:")
print(rf_data.head())
print()

# =============================================================================
# PART 2: SINGLE DECISION TREE VS RANDOM FOREST - ACCURACY COMPARISON
# =============================================================================

print("=" * 80)
print("PART 2: SINGLE TREE VS RANDOM FOREST - ACCURACY COMPARISON")
print("=" * 80)
print()

X = rf_data.drop('Default', axis=1).values
y = rf_data['Default'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print()

# Single Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_train_acc = tree.score(X_train, y_train)
tree_test_acc = tree.score(X_test, y_test)

# Random Forest (100 trees)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

print("Accuracy Comparison:")
print(f"{'Model':>20} {'Train Acc':>12} {'Test Acc':>12} {'Gap':>10}")
print("-" * 56)
print(f"{'Single Tree':>20} {tree_train_acc:>12.1%} {tree_test_acc:>12.1%} {tree_train_acc - tree_test_acc:>10.1%}")
print(f"{'Random Forest (100)':>20} {rf_train_acc:>12.1%} {rf_test_acc:>12.1%} {rf_train_acc - rf_test_acc:>10.1%}")
print()

print("Interpretation:")
print("  - Single tree: High train accuracy (overfit), lower test accuracy")
print("  - Random Forest: Slightly lower train accuracy, but HIGHER test accuracy")
print("    (better generalization due to averaging many diverse trees)")
print()

# =============================================================================
# PART 3: FEATURE IMPORTANCE - WHICH FEATURES MATTER MOST?
# =============================================================================

print("=" * 80)
print("PART 3: FEATURE IMPORTANCE - WHICH FEATURES MATTER MOST?")
print("=" * 80)
print()

# Get feature importances from Random Forest
feature_names = rf_data.drop('Default', axis=1).columns
importances = rf.feature_importances_

# Create DataFrame for easy viewing
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("Feature Importances from Random Forest:")
print(importance_df.round(4))
print()

# Visualize
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()  # Most important at top
plt.tight_layout()
plt.savefig('output/random_forest_feature_importance.png', dpi=150)
plt.show()

print("Feature importance plot saved to output/random_forest_feature_importance.png")
print()
print("Interpretation:")
print("  - DebtRatio and CreditScore are top 2 features (as expected from true pattern)")
print("  - Age and LoanAmount have low importance (not directly predictive)")
print("  - Use this for feature selection: drop low-importance features to simplify model")
print()

# =============================================================================
# PART 4: OUT-OF-BAG (OOB) SCORE - FREE CROSS-VALIDATION
# =============================================================================

print("=" * 80)
print("PART 4: OUT-OF-BAG (OOB) SCORE - FREE VALIDATION")
print("=" * 80)
print()

# Train Random Forest with OOB score enabled
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.3f}")
print(f"Test Accuracy: {rf_test_acc:.3f}")
print()

print("Interpretation:")
print("  - OOB score is an unbiased estimate of test accuracy")
print("  - Computed using samples that were NOT in the bootstrap sample for each tree")
print("  - No need for separate cross-validation — free validation during training!")
print(f"  - OOB ({rf_oob.oob_score_:.3f}) ≈ Test ({rf_test_acc:.3f}) — good agreement")
print()

# =============================================================================
# PART 5: HOW N_ESTIMATORS AFFECTS ACCURACY
# =============================================================================

print("=" * 80)
print("PART 5: HOW NUMBER OF TREES (N_ESTIMATORS) AFFECTS ACCURACY")
print("=" * 80)
print()

n_estimators_list = [1, 5, 10, 50, 100, 200, 500]
test_accuracies = []

for n_est in n_estimators_list:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    acc = rf_temp.score(X_test, y_test)
    test_accuracies.append(acc)
    print(f"n_estimators = {n_est:>3}: Test accuracy = {acc:.3f}")

print()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(n_estimators_list, test_accuracies, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Test Accuracy')
plt.title('How Number of Trees Affects Random Forest Accuracy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/random_forest_n_estimators.png', dpi=150)
plt.show()

print("Plot saved to output/random_forest_n_estimators.png")
print()
print("Interpretation:")
print("  - n_estimators=1 is just a single tree (lowest accuracy)")
print("  - Accuracy improves as you add more trees (up to a point)")
print("  - Diminishing returns after ~100-200 trees")
print("  - More trees = slower prediction, but training is parallelizable")
print("  - Typical default: 100 trees (good balance)")
print()

# =============================================================================
# PART 6: HOW MAX_FEATURES AFFECTS DIVERSITY
# =============================================================================

print("=" * 80)
print("PART 6: HOW MAX_FEATURES AFFECTS TREE DIVERSITY")
print("=" * 80)
print()

max_features_list = [1, 2, 3, 5]  # Number of features to consider at each split
accuracies_mf = []

for mf in max_features_list:
    rf_mf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42, n_jobs=-1)
    rf_mf.fit(X_train, y_train)
    acc = rf_mf.score(X_test, y_test)
    accuracies_mf.append(acc)
    print(f"max_features = {mf}: Test accuracy = {acc:.3f}")

print()
print("Interpretation:")
print("  - max_features=1: Each split considers only 1 random feature (most diverse trees)")
print("  - max_features=5 (all features): Each split considers all features (trees more correlated)")
print("  - Sweet spot: typically √p for classification (here √5 ≈ 2-3)")
print("  - Smaller max_features = more diversity, but each tree is weaker individually")
print()

# =============================================================================
# PART 7: WHEN TO USE RANDOM FOREST IN SOFTWARE ENGINEERING / PROGRAMMING
# =============================================================================

print("=" * 80)
print("PART 7: WHEN TO USE RANDOM FOREST IN SOFTWARE ENGINEERING")
print("=" * 80)
print()

print("USE CASE 1: BUG PREDICTION")
print("-" * 80)
print("Problem: Which code files are most likely to have bugs in next release?")
print("Features: Lines of code, cyclomatic complexity, number of authors,")
print("          code churn (how often file changed), test coverage %,")
print("          number of dependencies, age of file")
print("Target: Binary (1 = had bug in last 3 months, 0 = no bug)")
print()
print("Why Random Forest:")
print("  - Handles mixed feature types (continuous like LOC, discrete like num_authors)")
print("  - Feature importance tells you which metrics matter most (e.g., 'code churn")
print("    is top predictor' → focus code reviews on frequently-changed files)")
print("  - No need to scale features (LOC is 100-10000, coverage is 0-100)")
print("  - Better accuracy than logistic regression for this non-linear problem")
print()
print("Business impact: Prioritize QA/testing effort on high-risk files,")
print("                 catch bugs before production, reduce incident rate")
print()

print("USE CASE 2: CODE REVIEW AUTOMATION")
print("-" * 80)
print("Problem: Which pull requests need senior engineer review vs can be auto-approved?")
print("Features: Lines changed, files touched, developer experience (tenure, past PRs),")
print("          code complexity of changes, test coverage of changed files,")
print("          time since last review by this developer")
print("Target: Binary (1 = needs senior review, 0 = can auto-approve)")
print()
print("Why Random Forest:")
print("  - Captures interactions (e.g., 'large changes by junior devs need review,")
print("    but same changes by seniors are fine' — tree models this naturally)")
print("  - Feature importance reveals policy gaps (e.g., 'time since last review'")
print("    is important → maybe institute mandatory rotation)")
print("  - Fast prediction (can run on every PR in CI pipeline)")
print()
print("Business impact: Reduce review bottleneck, catch risky changes early,")
print("                 free up senior engineers for high-value reviews")
print()

print("USE CASE 3: INCIDENT / OUTAGE PREDICTION")
print("-" * 80)
print("Problem: Which services are at risk of outage in next 24 hours?")
print("Features: CPU/memory usage trends, error rate, latency percentiles,")
print("          deployment frequency, dependency health scores,")
print("          time since last incident, on-call engineer experience")
print("Target: Binary (1 = outage in next 24h, 0 = no outage)")
print()
print("Why Random Forest:")
print("  - Handles time-series features (trends, not just point-in-time values)")
print("  - Robust to noise (monitoring data is noisy — bagging helps)")
print("  - Feature importance tells you leading indicators (e.g., 'latency p99")
print("    increasing over 2h is top predictor' → set up early warning alerts)")
print("  - OOB score gives free validation (no need to hold out test data)")
print()
print("Business impact: Proactive incident prevention, reduce MTTR,")
print("                 improve SLA compliance, reduce on-call fatigue")
print()

print("USE CASE 4: TECHNICAL DEBT PRIORITIZATION")
print("-" * 80)
print("Problem: Which modules should we refactor first to maximize ROI?")
print("Features: Cyclomatic complexity, code duplication %, test coverage,")
print("          number of bugs in last 6 months, average PR review time,")
print("          developer satisfaction scores (from surveys),")
print("          time since last refactor")
print("Target: Continuous (refactor priority score 0-100) → Random Forest Regression")
print()
print("Why Random Forest:")
print("  - Handles subjective features (developer satisfaction scores)")
print("  - Feature importance reveals what drives technical debt (e.g., 'low")
print("    test coverage correlates with high bug rate' → invest in testing)")
print("  - No linearity assumption (relationship between complexity and debt")
print("    is non-linear — complexity > 50 is exponentially worse)")
print()
print("Business impact: Data-driven tech debt roadmap, justify refactor investment,")
print("                 reduce future bug rate and development velocity drag")
print()

print("USE CASE 5: HIRING / CANDIDATE SCREENING")
print("-" * 80)
print("Problem: Which candidates are likely to pass technical interview?")
print("Features: Years of experience, GitHub activity (commits, repos),")
print("          coding challenge scores, resume keywords match,")
print("          referral source, time to complete challenge")
print("Target: Binary (1 = passed interview, 0 = rejected)")
print()
print("Why Random Forest:")
print("  - Handles sparse features (GitHub activity can be 0 for some candidates)")
print("  - Feature importance reveals what predicts success (e.g., 'coding")
print("    challenge score is top predictor, years of experience is low' →")
print("    adjust screening criteria to weight challenges more)")
print("  - Can detect bias (if 'referral source' has high importance, audit")
print("    for homogeneity in referrals)")
print()
print("Business impact: Improve hiring quality, reduce time-to-hire,")
print("                 reduce unconscious bias in screening")
print("CAUTION: Must audit for fairness — don't encode protected characteristics")
print()

print("USE CASE 6: FEATURE FLAG ROLLOUT DECISIONS")
print("-" * 80)
print("Problem: Should we roll out this feature to 100% of users or roll back?")
print("Features: Error rate in canary (1% users), latency delta, user engagement")
print("          metrics (CTR, time on page), customer support tickets,")
print("          segment-level performance (mobile vs desktop, new vs existing)")
print("Target: Binary (1 = rollout safe, 0 = roll back)")
print()
print("Why Random Forest:")
print("  - Fast prediction (can run every 5 minutes during canary)")
print("  - Handles many features (can monitor 20+ metrics simultaneously)")
print("  - Feature importance tells you which metrics are leading indicators")
print("    (e.g., 'support tickets from mobile users spike first' → monitor that)")
print("  - More robust than threshold-based rules (e.g., 'error rate < 1%'")
print("    is too simplistic — RF learns complex patterns)")
print()
print("Business impact: Safer feature rollouts, faster detection of issues,")
print("                 reduce production incidents from bad releases")
print()

print("USE CASE 7: CUSTOMER CHURN PREDICTION (FOR SAAS / DEV TOOLS)")
print("-" * 80)
print("Problem: Which customers are at risk of canceling subscription?")
print("Features: Usage frequency (logins/day), feature adoption rate,")
print("          support ticket volume, payment history (late payments),")
print("          NPS score, contract age, competitor mentions in support chats")
print("Target: Binary (1 = churn in next 30 days, 0 = retain)")
print()
print("Why Random Forest:")
print("  - Handles imbalanced data (churn is rare — use class_weight='balanced')")
print("  - Feature importance reveals churn drivers (e.g., 'drop in usage")
print("    frequency is top predictor' → trigger engagement campaign)")
print("  - Probability outputs let you prioritize high-risk customers")
print("    (e.g., P(churn) > 0.7 → account manager calls them)")
print()
print("Business impact: Reduce churn rate, increase LTV, proactive retention,")
print("                 optimize customer success team effort")
print()

print("=" * 80)
print("GENERAL GUIDELINES FOR USING RANDOM FOREST IN SOFTWARE ENGINEERING")
print("=" * 80)
print()
print("USE RANDOM FOREST WHEN:")
print("  - You have tabular data (rows × columns) — most common in SE")
print("  - You need good accuracy fast without extensive tuning")
print("  - You want feature importance (to explain to stakeholders)")
print("  - Dataset is medium size (1k - 100k rows) — scales well")
print("  - Features are mixed types (continuous, discrete, counts)")
print("  - Relationship is non-linear (most SE problems are)")
print()
print("DON'T USE RANDOM FOREST WHEN:")
print("  - Data is text (logs, code, tickets) — use NLP (TF-IDF + SVM, or transformers)")
print("  - Data is images (screenshots, UI mockups) — use CNNs")
print("  - Data is time-series with temporal dependencies — use LSTM, Prophet, ARIMA")
print("  - You need the absolute best accuracy — use Gradient Boosting (XGBoost, LightGBM)")
print("  - You need full interpretability — use a single shallow Decision Tree")
print("  - Dataset is tiny (<100 rows) — risk of overfitting even with RF")
print()
print("PRACTICAL TIPS:")
print("  1. Start with default hyperparameters (n_estimators=100, max_features='sqrt')")
print("  2. Check feature importance — drop near-zero importance features")
print("  3. Use OOB score for quick validation (oob_score=True)")
print("  4. For imbalanced data, set class_weight='balanced'")
print("  5. If accuracy plateaus, try Gradient Boosting instead")
print("  6. Always check for data leakage (e.g., future information in features)")
print("  7. Document which features drive predictions — stakeholders care about 'why'")
print()

# =============================================================================
# SUMMARY: RANDOM FOREST KEY TAKEAWAYS
# =============================================================================

print("=" * 80)
print("SUMMARY: RANDOM FOREST KEY TAKEAWAYS")
print("=" * 80)
print()
print("What: Ensemble of many decision trees, combined by majority vote")
print("Why: Reduces overfitting, increases accuracy, stable predictions")
print("Where: Default choice for tabular classification/regression problems")
print("How: Bagging (bootstrap samples) + feature randomness at each split")
print("Key outputs: Feature importances, OOB score (free validation)")
print("Tuning: n_estimators (100-500), max_features (√p), max_depth (None or 10-20)")
print()
print("In Software Engineering: Use for bug prediction, code review automation,")
print("incident prediction, tech debt prioritization, hiring screening,")
print("feature flag decisions, customer churn — anywhere you have tabular")
print("data and need interpretable, accurate predictions fast.")