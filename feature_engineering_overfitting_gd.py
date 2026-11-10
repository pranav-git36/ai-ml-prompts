"""
FEATURE ENGINEERING, OVERFITTING/UNDERFITTING, AND GRADIENT DESCENT
Complete deep dive with practical examples
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =============================================================================
# WHAT IS FEATURE ENGINEERING?
# =============================================================================
# Process of creating new input features from raw data to help ML models
# learn better patterns.
#
# Key insight: "Garbage in, garbage out." Good features often matter more
# than choice of algorithm.

# =============================================================================
# WHY DOES FEATURE ENGINEERING EXIST?
# =============================================================================
# Problems it solves:
#   1. Raw data is messy (timestamps, text, categories can't be consumed directly)
#   2. Models can't learn everything (even deep learning needs structure)
#   3. Domain knowledge matters (you know things model doesn't)

# =============================================================================
# TYPES OF FEATURE ENGINEERING
# =============================================================================
# 1. Handling Missing Values: Drop, mean/median imputation, forward-fill,
#    create "is_missing" flag
# 2. Encoding Categorical: One-hot (nominal), Label (ordinal), Target encoding
# 3. Scaling Numerical: Standardization (Z-score), Min-Max, Robust scaling
# 4. Interaction Features: Ratios, products (e.g., Debt_to_Income)
# 5. Date/Time Features: Hour, dayofweek, is_weekend, month, is_holiday
# 6. Text Features: TF-IDF, embeddings, length, word_count
# 7. Aggregation Features: Groupby stats (mean, sum, count per user/category)

# =============================================================================
# WHAT IS OVERFITTING VS UNDERFITTING?
# =============================================================================
# Overfitting: Model learns training data TOO WELL (including noise),
#              fails to generalize. Symptoms: Train acc=99%, Test acc=70%
# Underfitting: Model is TOO SIMPLE to capture pattern.
#               Symptoms: Train acc=60%, Test acc=58% (both low)

# Bias-Variance Tradeoff:
#   Total Error = BiasÂ² + Variance + Irreducible Error
#   - High bias = underfitting (model too simple)
#   - High variance = overfitting (model too complex)
#   Goal: Find sweet spot where total error is minimized

# How to fix Overfitting:
#   - Get more data
#   - Reduce model complexity (limit tree depth, reduce polynomial degree)
#   - Regularization (L1/L2)
#   - Early stopping
#   - Cross-validation
#   - Ensemble methods (Random Forest, Gradient Boosting)

# How to fix Underfitting:
#   - Increase model complexity
#   - Add more features (polynomial, interactions)
#   - Reduce regularization
#   - Try different algorithm

# =============================================================================
# WHAT IS GRADIENT DESCENT?
# =============================================================================
# Iterative optimization algorithm that finds parameters minimizing cost function.
#
# Intuition: You're on mountain in fog, want to reach valley. Feel which
# direction slopes downward, take step that way, repeat.
#
# Update rule:
#   Îâ¼½ := Îâ¼½ - Î± * (â²¶J / âÎ²â¼½)
#   Where:
#     Îâ¼½ = parameter (e.g., slope or intercept)
#     Î± = learning rate (step size)
#     âJ / âÎ²â¼½ = gradient (derivative of cost w.r.t. parameter)
#
# Variants:
#   - Batch GD: Use ALL training examples for each update (stable, slow)
#   - SGD: Use ONE random example per update (fast, noisy)
#   - Mini-batch GD: Use small batch (32-128 examples) per update (best of both)

# =============================================================================
# PART 1: FEATURE ENGINEERING - PRACTICAL EXAMPLES
# =============================================================================

print("=" * 80)
print("PART 1: FEATURE ENGINEERING - PRACTICAL EXAMPLES")
print("=" * 80)
print()

# Create synthetic e-commerce data
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'Timestamp': pd.date_range('2025-01-01', periods=n, freq='D'),
    'Age': np.random.randint(18, 70, n),
    'Income': np.random.randn(n) * 20000 + 60000,
    'DaysSinceLastPurchase': np.random.exponential(30, n),
    'NumPurchases_Last30Days': np.random.poisson(3, n),
    'TotalSpend_Last90Days': np.random.exponential(500, n),
    'City': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai'], n),
    'IsMember': np.random.choice([0, 1], n, p=[0.6, 0.4]),
})

# Target: Likelihood to purchase in next 7 days
log_odds = -2 + 1.5*df['IsMember'] - 0.02*df['DaysSinceLastPurchase'] + 0.1*df['NumPurchases_Last30Days'] + 0.00001*df['TotalSpend_Last90Days']
prob_purchase = 1 / (1 + np.exp(-log_odds))
df['WillPurchase'] = (np.random.rand(n) < prob_purchase).astype(int)

print("Raw data (before feature engineering):")
print(df[['Timestamp', 'Age', 'Income', 'City', 'IsMember', 'WillPurchase']].head())
print()

# --- STEP 1: Date/Time Features ---
print("STEP 1: Extract Date/Time Features")
print("-" * 80)

df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['Month'] = df['Timestamp'].dt.month
df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)

print(f"Added: DayOfWeek, IsWeekend, Month, IsHolidaySeason")
print(df[['Timestamp', 'DayOfWeek', 'IsWeekend', 'Month', 'IsHolidaySeason']].head())
print()

# --- STEP 2: Interaction Features ---
print("STEP 2: Create Interaction Features")
print("-" * 80)

df['Income_per_Age'] = df['Income'] / df['Age']
df['Engagement_Score'] = df['NumPurchases_Last30Days'] * df['TotalSpend_Last90Days']
df['Recency_Bin'] = pd.cut(df['DaysSinceLastPurchase'], 
                            bins=[0, 7, 30, 90, 999], 
                            labels=['VeryRecent', 'Recent', 'Moderate', 'Old']).astype(str)

print(f"Added: Income_per_Age, Engagement_Score, Recency_Bin")
print(df[['Income', 'Age', 'Income_per_Age', 'Engagement_Score']].head(3))
print()

# --- STEP 3: Encoding Categorical ---
print("STEP 3: Encode Categorical Variables")
print("-" * 80)

df = pd.get_dummies(df, columns=['City'], drop_first=True)
recency_map = {'VeryRecent': 0, 'Recent': 1, 'Moderate': 2, 'Old': 3}
df['Recency_Encoded'] = df['Recency_Bin'].map(recency_map)

print(f"Added: One-hot encoded City columns (drop_first=True)")
print(f"Added: Recency_Encoded (label encoding)")
city_cols = [c for c in df.columns if c.startswith('City_')]
print(f"City columns: {city_cols}")
print(df[city_cols + ['Recency_Bin', 'Recency_Encoded']].head(3))
print()

# --- STEP 4: Aggregation Features ---
print("STEP 4: Aggregation Features")
print("-" * 80)

df['User_ID'] = np.random.randint(1, 100, n)
user_agg = df.groupby('User_ID').agg({
    'TotalSpend_Last90Days': 'mean',
    'NumPurchases_Last30Days': 'sum',
    'WillPurchase': 'mean'
}).reset_index()
user_agg.columns = ['User_ID', 'AvgSpend_PerUser', 'TotalPurchases_PerUser', 'HistoricalPurchaseRate']

df = df.merge(user_agg, on='User_ID', how='left')

print(f"Added: AvgSpend_PerUser, TotalPurchases_PerUser, HistoricalPurchaseRate")
print(df[['User_ID', 'AvgSpend_PerUser', 'TotalPurchases_PerUser']].head(3))
print()

# =============================================================================
# PART 2: OVERFITTING VS UNDERFITTING DEMO
# =============================================================================

print("=" * 80)
print("PART 2: OVERFITTING VS UNDERFITTING - BIAS-VARIANCE TRADEOFF")
print("=" * 80)
print()

np.random.seed(123)
n = 100
X_simple = np.random.randn(n, 1) * 3
y_true = 2 * X_simple[:, 0]**2 - 3 * X_simple[:, 0] + 1
y_noisy = y_true + np.random.randn(n) * 2

X_simple = X_simple.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_simple, y_noisy, test_size=0.3, random_state=42)

print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
print(f"True relationship: y = 2xÂ² - 3x + 1 (quadratic)")
print()

# Model 1: Underfitting
print("Model 1: UNDERFITTING (Linear on quadratic data)")
linear = LinearRegression()
linear.fit(X_train, y_train)
train_mse_linear = mean_squared_error(y_train, linear.predict(X_train))
test_mse_linear = mean_squared_error(y_test, linear.predict(X_test))
print(f"Train MSE: {train_mse_linear:.2f}, Test MSE: {test_mse_linear:.2f}, Gap: {test_mse_linear - train_mse_linear:.2f}")
print(f"Diagnosis: Both HIGH â UNDERFITTING")
print()

# Model 2: Good fit
print("Model 2: GOOD FIT (Polynomial degree=2)")
poly2 = PolynomialFeatures(degree=2)
poly2_model = LinearRegression().fit(poly2.fit_transform(X_train), y_train)
train_mse_poly2 = mean_squared_error(y_train, poly2_model.predict(poly2.transform(X_train)))
test_mse_poly2 = mean_squared_error(y_test, poly2_model.predict(poly2.transform(X_test)))
print(f"Train MSE: {train_mse_poly2:.2f}, Test MSE: {test_mse_poly2:.2f}, Gap: {test_mse_poly2 - train_mse_poly2:.2f}")
print(f"Diagnosis: Both LOW, small gap â GOOD FIT")
print()

# Model 3: Overfitting
print("Model 3: OVERFITTING (Polynomial degree=10)")
poly10 = PolynomialFeatures(degree=10)
poly10_model = LinearRegression().fit(poly10.fit_transform(X_train), y_train)
train_mse_poly10 = mean_squared_error(y_train, poly10_model.predict(poly10.transform(X_train)))
test_mse_poly10 = mean_squared_error(y_test, poly10_model.predict(poly10.transform(X_test)))
print(f"Train MSE: {train_mse_poly10:.2f}, Test MSE: {test_mse_poly10:.2f}, Gap: {test_mse_poly10 - train_mse_poly10:.2f}")
print(f"Diagnosis: Train LOW, Test HIGH â OVERFITTING")
print()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_plot = np.linspace(-4, 4, 100).reshape(-1, 1)

axes[0].scatter(X_train, y_train, alpha=0.5)
axes[0].plot(x_plot, linear.predict(x_plot), 'r-', linewidth=2)
axes[0].set_title(f'UNDERFITTING\nTrain={train_mse_linear:.2f}, Test={test_mse_linear:.2f}')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_train, y_train, alpha=0.5)
axes[1].plot(x_plot, poly2_model.predict(poly2.transform(x_plot)), 'g-', linewidth=2)
axes[1].set_title(f'GOOD FIT\nTrain={train_mse_poly2:.2f}, Test={test_mse_poly2:.2f}')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(X_train, y_train, alpha=0.5)
axes[2].plot(x_plot, poly10_model.predict(poly10.transform(x_plot)), 'm-', linewidth=1)
axes[2].set_title(f'OVERFITTING\nTrain={train_mse_poly10:.2f}, Test={test_mse_poly10:.2f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/overfitting_underfitting_demo.png', dpi=150)
plt.show()
print("Saved to output/overfitting_underfitting_demo.png\n")

# =============================================================================
# PART 3: GRADIENT DESCENT - MANUAL IMPLEMENTATION
# =============================================================================

print("=" * 80)
print("PART 3: GRADIENT DESCENT - MANUAL IMPLEMENTATION")
print("=" * 80)
print()

np.random.seed(42)
n_gd = 100
x_gd = np.random.randn(n_gd) * 3
y_gd = 2 * x_gd + 1 + np.random.randn(n_gd) * 0.5

beta0, beta1 = 0.0, 0.0
learning_rate = 0.01
n_iterations = 1000

beta0_hist, beta1_hist, cost_hist = [beta0], [beta1], []

for i in range(n_iterations):
    y_pred = beta0 + beta1 * x_gd
    grad_b0 = (-2/n_gd) * np.sum(y_gd - y_pred)
    grad_b1 = (-2/n_gd) * np.sum(x_gd * (y_gd - y_pred))
    beta0 -= learning_rate * grad_b0
    beta1 -= learning_rate * grad_b1
    cost = np.mean((y_gd - y_pred)**2)
    cost_hist.append(cost)
    beta0_hist.append(beta0)
    beta1_hist.append(beta1)
    
    if i % 200 == 0:
        print(f"Iter {i:4d}: Îâ¼¡={beta0:.4f}, Îâ¼¡={beta1:.4f}, Cost={cost:.4f}")

print(f"\nFinal: Îâ¼¡={beta0:.4f}, Îâ¼¡={beta1:.4f}")
print(f"True: Îâ¼¡=1.0, Îâ¼¡=2.0")
print()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(beta0_hist, linewidth=2)
axes[0].axhline(y=1.0, color='r', linestyle='--')
axes[0].set_title('Î²â¼¡ Convergence')
axes[0].grid(True, alpha=0.3)

axes[1].plot(beta1_hist, linewidth=2)
axes[1].axhline(y=2.0, color='r', linestyle='--')
axes[1].set_title('Î²â¼¡ Convergence')
axes[1].grid(True, alpha=0.3)

axes[2].plot(cost_hist, linewidth=2)
axes[2].set_title('Cost Convergence')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/gradient_descent_convergence.png', dpi=150)
plt.show()
print("Saved to output/gradient_descent_convergence.png\n")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Feature Engineering:")
print("  - Date/time extraction (hour, dayofweek, is_weekend, month)")
print("  - Interaction features (ratios like Income/Age, products)")
print("  - Encoding (one-hot for nominal, label for ordinal)")
print("  - Aggregation (groupby stats: mean, sum, count)")
print()
print("Overfitting vs Underfitting:")
print("  - Overfitting: High train acc, low test acc")
print("    Fix: Regularize, simplify model, get more data, early stopping")
print("  - Underfitting: Low both train and test acc")
print("    Fix: Increase complexity, add features, reduce regularization")
print("  - Good fit: Both high accuracy, small gap")
print()
print("Gradient Descent:")
print("  - Iteratively update parameters in direction opposite to gradient")
print("  - Update rule: Î := Î - Î± * (â²¶J / âÎ²)")
print("  - Learning rate Î± controls step size")
print("  - Variants: Batch (all data), SGD (one example), Mini-batch (small batch)")
print("  - Used by all modern ML/DL frameworks for optimization")