"""
LOGISTIC REGRESSION - SIMPLIFIED WITH ALL CONCEPTS EXPLAINED
Predicts Pass/Fail from Hours Studied
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# =====================
# 1. CREATE SIMPLE DATASET
# =====================
# 7 students: hours studied and whether they passed (1) or failed (0)
data = pd.DataFrame({
    'Hours': [1, 2, 3, 4, 5, 6, 7],
    'Pass':  [0, 0, 0, 0, 1, 1, 1]
})

print("=" * 70)
print("STEP 1: RAW DATA")
print("=" * 70)
print(data)
print()

# =====================
# 2. FIT LOGISTIC REGRESSION MODEL
# =====================
X = data[['Hours']]  # Feature (must be 2D array)
y = data['Pass']     # Target

model = LogisticRegression()
model.fit(X, y)

# Extract coefficients
beta0 = model.intercept_[0]      # Intercept (β₀)
beta1 = model.coef_[0][0]        # Slope for Hours (β₁)

print("=" * 70)
print("STEP 2: FITTED MODEL")
print("=" * 70)
print(f"Model equation: z = {beta0:.3f} + {beta1:.3f} × Hours")
print(f"Then: P(Pass) = 1 / (1 + e^(-z))")
print()

# =====================
# 3. SIGMOID FUNCTION - TURN SCORE INTO PROBABILITY
# =====================
def sigmoid(z):
    """
    Sigmoid function: takes any number z and squeezes it into 0 to 1.
    This is the "probability mapping" step.
    """
    return 1 / (1 + np.exp(-z))

print("=" * 70)
print("STEP 3: SIGMOID FUNCTION IN ACTION")
print("=" * 70)
print("For each student, compute z, then P(Pass) = sigmoid(z):")
print()

print(f"{'Hours':>6} {'z = β₀+β₁×¥Hours':>18} {'P(Pass)':>10} {'Prediction':>12}")
print("-" * 52)

for hours in range(1, 8):
    # Step 3a: Compute the "score" z
    z = beta0 + beta1 * hours
    
    # Step 3b: Apply sigmoid to get probability
    prob = sigmoid(z)
    
    # Step 3c: Apply threshold (0.5) to get class prediction
    prediction = 1 if prob >= 0.5 else 0
    
    print(f"{hours:>6} {z:>18.3f} {prob:>10.3f} {prediction:>12}")

print()
print("Notice: At Hours=4, P(Pass) crosses 0.5 - this is the decision boundary!")
print()

# =====================
# 4. DECISION BOUNDARY - WHERE DOES PREDICTION SWITCH?
# =====================
print("=" * 70)
print("STEP 4: DECISION BOUNDARY")
print("=" * 70)

# Decision boundary is where z = 0 (because sigmoid(0) = 0.5)
# Solve: 0 = β₀ + β₁ × Hours  →  Hours = -β₀ / β₁
boundary_hours = -beta0 / beta1

print(f"Decision boundary formula: z = 0 = β₀ + β₁ × Hours")
print(f"Solving for Hours: Hours = -β₀ / β₁ = -({beta0:.3f}) / {beta1:.3f} = {boundary_hours:.2f} hours")
print()
print(f"Interpretation:")
print(f"  - Students studying < {boundary_hours:.2f} hours → predicted Fail")
print(f"  - Students studying ≥ {boundary_hours:.2f} hours → predicted Pass")
print()

# =====================
# 5. ODDS RATIO INTERPRETATION
# =====================
print("=" * 70)
print("STEP 5: ODDS RATIO - HOW TO INTERPRET β₁")
print("=" * 70)

# Odds ratio = e^β₁
odds_ratio = np.exp(beta1)

print(f"β₁ (coefficient for Hours) = {beta1:.3f}")
print(f"Odds Ratio = e^β₁ = e^{beta1:.3f} = {odds_ratio:.3f}")
print()
print(f"Interpretation:")
print(f"  'Each additional hour of study multiplies the ODDS of passing by {odds_ratio:.2f}×»")
print()

# Show this with actual numbers
print("Verification with actual odds:")
for hours in [3, 4, 5]:
    z = beta0 + beta1 * hours
    prob = sigmoid(z)
    odds = prob / (1 - prob)
    print(f"  At {hours} hours: P(Pass)={prob:.3f}, Odds={odds:.3f}")

print()
print(f"Check: Odds at 4 hours / Odds at 3 hours = ", end="")
odds_3 = sigmoid(beta0 + beta1*3) / (1 - sigmoid(beta0 + beta1*3))
odds_4 = sigmoid(beta0 + beta1*4) / (1 - sigmoid(beta0 + beta1*4))
print(f"{odds_4/odds_3:.3f} (should equal odds ratio {odds_ratio:.3f})")
print()

# =====================
# 6. THRESHOLD TUNING - WHAT IF WE CHANGE THE THRESHOLD?
# =====================
print("=" * 70)
print("STEP 6: THRESHOLD TUNING")
print("=" * 70)

print("Default threshold = 0.5:")
thresholds = [0.3, 0.5, 0.7]

for thresh in thresholds:
    predictions = []
    for hours in range(1, 8):
        z = beta0 + beta1 * hours
        prob = sigmoid(z)
        pred = 1 if prob >= thresh else 0
        predictions.append(pred)
    
    print(f"  Threshold = {thresh}: Predictions for Hours 1-7 = {predictions}")
    print(f"    → Decision boundary shifts: predict Pass starting at Hours = ", end="")
    
    # Find first hour where prediction is 1
    for h in range(1, 8):
        z = beta0 + beta1 * h
        prob = sigmoid(z)
        if prob >= thresh:
            print(f"{h} hours")
            break
    print()

print("Lower threshold (0.3) → more lenient (pass students with fewer hours)")
print("Higher threshold (0.7) → stricter (only pass students with more hours)")
print()

# =====================
# 7. MODEL EVALUATION - ACCURACY
# =====================
print("=" * 70)
print("STEP 7: MODEL EVALUATION")
print("=" * 70)

# Get predictions for all training data
y_pred = model.predict(X)
accuracy = (y_pred == y).mean()

print(f"Predictions: {y_pred}")
print(f"Actual:      {y.values}")
print(f"Correct:     {(y_pred == y).astype(int)}")
print(f"Accuracy = {accuracy:.1%} ({(y_pred == y).sum()}/{len(y)} correct)")
print()

print("=" * 70)
print("SUMMARY OF CONCEPTS")
print("=" * 70)
print("1. Sigmoid: P(Pass) = 1/(1+e^(-z)) - turns any z into probability 0 to 1")
print("2. Decision boundary: where P(Pass) = 0.5, i.e., z = 0")
print("3. Threshold tuning: change 0.5 to 0.3 or 0.7 to adjust leniency")
print("4. Odds ratio: e^β₁ tells how much odds multiply per unit increase in feature")
print("5. MLE (done internally): finds β₀, β₁ that make predictions match actual data best")