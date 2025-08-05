# ModelInterpreter

`ModelInterpreter` provides a unified interface for interpreting fitted machine learning models using a variety of techniques including SHAP, feature importance, and partial dependence plots (PDP). Compatible with scikit-learn pipelines.

---

## Overview

The `ModelInterpreter` class supports:

- Tree-based models (e.g., RandomForest, XGBoost, LightGBM)
- Linear models (e.g., LogisticRegression, LinearRegression)
- Any model compatible with SHAP or scikit-learn’s `PartialDependenceDisplay`

---

## Features

### SHAP-Based Interpretability
- `shap_summary_plot()`: Global interpretability with SHAP values
- `shap_dependence_plot(feature)`: SHAP value vs feature distribution
- Works with pipelines; automatically extracts preprocessed input

### Feature Importances
- For tree-based models: native feature importances
- For linear models: coefficients (with option to standardize)

### Partial Dependence Plots (PDP)
- `plot_partial_dependence(features)`: Visualize marginal effect of a feature
- Supports interaction terms

### Permutation Importance
- `plot_permutation_importance()`: Measures decrease in model score when a feature’s values are randomly shuffled

---

## Initialization

```python
from glazzbocks import ModelInterpreter

interpreter = ModelInterpreter(model, X_train, y_train)
```

- `model`: Trained scikit-learn model or pipeline
- `X_train`, `y_train`: Training data used during fitting (required for SHAP, PDP, and permutation)

---

## Methods

### `shap_summary_plot()`
> Displays a global SHAP summary plot

### `shap_dependence_plot(feature)`
> SHAP scatter plot for a specific feature

### `plot_feature_importance()`
> Native importance (for trees) or coefficients (for linear models)

### `plot_partial_dependence(features)`
> PDP plots (accepts string or list of feature names)

### `plot_permutation_importance()`
> Visualize permutation-based importance

---

## Notes

- SHAP plots require `shap` to be installed
- PDP and permutation require scikit-learn 0.24+
- Automatically handles preprocessed input from pipelines