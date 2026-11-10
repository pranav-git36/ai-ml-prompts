"""
Linear Regression — Full Explanation with Both Approaches
Closed-form solution + Gradient Descent + Visual Plot
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. RAW DATA
# =====================
# 5 houses: Area (sq.ft) and Price (₹ Lakhs)
x = np.array([500, 750, 1000, 1250, 1500], dtype=float)  # Area
y = np.array([1.8, 2.6, 3.4, 4.2, 5.1], dtype=float)     # Price
n = len(x)

print("=" * 60)
print("RAW DATA")
print("=" * 60)
for i in range(n):
    print(f"House {i+1}: Area={x[i]:.0f} sq.ft, Price=₹{y[i]:.1f} Lakhs")
print()

# =====================
# 2. CLOSED-FORM SOLUTION (Least Squares)
# =====================
print("=" * 60)
print("APPROACH 1: CLOSED-FORM LEAST SQUARES")
print("=" * 60)

x_mean = np.mean(x)
y_mean = np.mean(y)
print(f"Step 1: Means")
print(f"  x̄ (mean Area) = {x_mean}")
print(f"  ȳ (mean Price) = {y_mean:.4f}\n")

x_dev = x - x_mean
y_dev = y - y_mean
print(f"Step 2: Deviations from mean")
print(f"{'House':>6} {'x-x̄':>10} {'y-ȳ':>10} {'(x-x̄)(y-ȳ)':>14} {'(x-x̄)Â²':>12}")
for i in range(n):
    prod = x_dev[i] * y_dev[i]
    sq = x_dev[i] ** 2
    print(f"{i+1:>6} {x_dev[i]:>10.1f} {y_dev[i]:>10.2f} {prod:>14.1f} {sq:>12.1f}")

numerator = np.sum(x_dev * y_dev)
denominator = np.sum(x_dev ** 2)
print(f"\nStep 3: Sums")
print(f"  Numerator Σ(x-x̄)(y-ȳ)   = {numerator}")
print(f"  Denominator Σ(x-x̄)Â²     = {denominator}\n")

beta1_cf = numerator / denominator
beta0_cf = y_mean - beta1_cf * x_mean
print(f"Step 4: Coefficients")
print(f"  β₁ (slope)     = {numerator} / {denominator} = {beta1_cf:.6f}")
print(f"  β₀ (intercept) = ȳ - β₁x̄ = {y_mean:.4f} - ({beta1_cf:.6f})×¥{x_mean} = {beta0_cf:.6f}\n")

print(f"Fitted line (closed-form): Price = {beta0_cf:.4f} + {beta1_cf:.6f} × Area\n")

# =====================
# 3. GRADIENT DESCENT SOLUTION (Iterative)
# =====================
print("=" * 60)
print("APPROACH 2: GRADIENT DESCENT")
print("=" * 60)

# Hyperparameters
learning_rate = 0.00000001  # very small because x values are large (500-1500)
n_iterations = 10000

# Initialize with random guesses
beta0_gd = 0.0
beta1_gd = 0.0

print(f"Learning rate = {learning_rate}")
print(f"Iterations = {n_iterations}\n")

for i in range(n_iterations):
    # Predictions with current guesses
    y_pred_gd = beta0_gd + beta1_gd * x
    
    # Compute gradients (derivatives of cost function)
    error = y - y_pred_gd
    grad_beta0 = (-2/n) * np.sum(error)
    grad_beta1 = (-2/n) * np.sum(x * error)
    
    # Update parameters (move opposite to gradient)
    beta0_gd = beta0_gd - learning_rate * grad_beta0
    beta1_gd = beta1_gd - learning_rate * grad_beta1

print(f"After {n_iterations} iterations:")
print(f"  β₀ (intercept) = {beta0_gd:.6f}")
print(f"  β₁ (slope)     = {beta1_gd:.6f}")
print(f"\nFitted line (gradient descent): Price = {beta0_gd:.4f} + {beta1_gd:.6f} × Area\n")

# Compare both approaches
print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Closed-form:    β₀={beta0_cf:.6f}, β₁={beta1_cf:.6f}")
print(f"Gradient Descent: β₀={beta0_gd:.6f}, β₁={beta1_gd:.6f}")
print(f"Difference in β₀: {abs(beta0_cf - beta0_gd):.8f}")
print(f"Difference in β₁: {abs(beta1_cf - beta1_gd):.8f}")
print()

# =====================
# 4. VISUALIZATION
# =====================
# Use closed-form coefficients for the plot (they're exact)
y_pred = beta0_cf + beta1_cf * x
residuals = y - y_pred
rss = np.sum(residuals ** 2)
tss = np.sum((y - y_mean) ** 2)
r2 = 1 - (rss / tss)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', s=100, label='Actual data points', zorder=3, edgecolors='black')

# Plot fitted line
x_line = np.array([x.min(), x.max()])
y_line = beta0_cf + beta1_cf * x_line
plt.plot(x_line, y_line, color='red', linewidth=2.5, 
         label=f'Fitted line: y = {beta0_cf:.2f} + {beta1_cf:.5f}x\n(RÂ² = {r2:.4f})', zorder=2)

# Draw residual lines (vertical from point to line)
for xi, yi, ypi in zip(x, y, y_pred):
    plt.plot([xi, xi], [ypi, yi], 'g--', linewidth=1.5, alpha=0.7, zorder=1)

plt.xlabel('Area (sq.ft)', fontsize=12, fontweight='bold')
plt.ylabel('Price (₹ Lakhs)', fontsize=12, fontweight='bold')
plt.title('Linear Regression: Best-Fit Line with Residuals\n(Green dashed = errors)', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('output/linear_regression_detailed.png', dpi=150)
plt.show()

print("=" * 60)
print("PLOT SAVED TO: output/linear_regression_detailed.png")
print("=" * 60)
print("\nInterpretation:")
print(f"  - Blue dots: your 5 actual houses")
print(f"  - Red line: predictions from Price = {beta0_cf:.2f} + {beta1_cf:.5f}×¥Area")
print(f"  - Green dashed lines: residuals (errors) for each house")
print(f"  - RÂ² = {r2:.4f} means {r2*100:.2f}% of price variation is explained by area")
print()
print(f"Coefficient meaning:")
print(f"  - Each additional 1 sq.ft adds ₹{beta1_cf*100000:.0f} to predicted price")
print(f"  - Each additional 100 sq.ft adds ₹{beta1_cf*100*100000:.0f} to predicted price")