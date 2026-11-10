# Basic Machine Learning - II: Practical Notebook

Hands-on companion notebook for the **Basic Machine Learning - II** session (Course 9, upGrad BITS
Pilani GenAI/AI-ML Engineering Program). Covers Decision Trees, Random Forest, Clustering, and
Optimization Basics, each taught end to end with real, runnable code.

## Contents

The notebook is organized into four fully self-contained parts. Decision Tree and Random Forest are
kept in **separate sections** — neither trains, evaluates, or visualizes the other model — so each
algorithm can be followed on its own.

| Part | Topic | What It Covers |
|---|---|---|
| Setup | Imports | Libraries, random seed, `models/` output folder |
| Part 1 | Decision Tree | Demo 1: Play Tennis (14-row toy dataset from the slides) — encode, train with `criterion="entropy"`, visualize, predict a new day. Demo 2: Wine dataset — feature engineering, train/test split, training, evaluation (accuracy, classification report, confusion matrix, macro/weighted precision-recall-F1), feature importance, effect of feature engineering, effect of pruning (`max_depth`), overfitting/underfitting depth sweep, 5-fold cross-validation, model saved |
| Part 2 | Random Forest | Demo 1: Play Tennis — forest of trees, two individual trees visualized side by side to show diversity, vote tally vs. `predict_proba`. Demo 2: Wine dataset (reuses the Part 1 split) — training, evaluation, feature importance, effect of removing low-importance features, effect of the number of trees (`n_estimators`), 5-fold cross-validation, model saved |
| Part 3 | Clustering (K-Means) | Demo 1: toy 2D example. Demo 2: Iris dataset (species label withheld during training) — feature engineering, scaling, Elbow Method for choosing K, fitting, visualization, Silhouette Score, Adjusted Rand Index, model + scaler saved |
| Part 4 | Optimization Basics | Gradient Descent implemented from scratch on a simple quadratic cost function, with the descent path visualized against the cost curve |

## Datasets

All datasets ship with scikit-learn or are defined inline — no downloads required:

- **Play Tennis** — the exact 14-row weather dataset from the slides, defined directly in the
  notebook (`Outlook`, `Temperature`, `Humidity`, `Windy` → `Play`). Used to build intuition for
  Decision Trees and Random Forest before moving to real data.
- **Wine** (`sklearn.datasets.load_wine`) — 178 wines, 3 cultivars, 13 chemical measurements. Used
  for the real Decision Tree and Random Forest practicals.
- **Iris** (`sklearn.datasets.load_iris`) — 150 flowers, 3 species. Used for Clustering, with the
  species label deliberately withheld during training.

## Design Notes

- **No `Pipeline`** is used in the main flow. Every step (scale, fit, predict) is written out
  separately so it's easy to see exactly what's happening at each stage. Two `Pipeline` examples are
  included in Part 3 purely as commented-out reference code, clearly marked as not executed.
- **No feature scaling** for Decision Tree or Random Forest — tree-based models split on raw values,
  so scale doesn't affect them. Scaling **is** used for K-Means, since it's distance-based.
- Every trained model is saved to `models/` with `joblib` for reuse:
  `decision_tree_model.joblib`, `random_forest_model.joblib`, `kmeans_model.joblib`,
  `iris_scaler.joblib`.

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## Running the Notebook

Run all cells top to bottom — later cells in Part 2 and Part 4 reuse data and variables defined
earlier (e.g., Part 2 reuses the Wine train/test split from Part 1). A `models/` folder is created
automatically on first run to hold the saved model files.

```bash
jupyter notebook Basic_Machine_Learning_II_Practical.ipynb
```
