# ModelDiagnostics – Model Evaluation & Visualization Module

## Overview
`ModelDiagnostics` is a diagnostic utility designed to visualize and evaluate the performance of trained machine learning models. It supports both **classification** and **regression** tasks and works with any scikit-learn pipeline after fitting.

---

## Key Features

### Classification Diagnostics
- ROC Curve (with AUC)
- F1 Score vs. Threshold
- Confusion Matrix
- Lift Chart
- Cumulative Gain Chart

### Regression Diagnostics
- Predicted vs. Actual Plot
- Residuals vs. Predicted
- Residual Distribution Histogram
- QQ Plot (normality check)
- D'Agostino's Normality Test

### Auto Diagnostics
- `auto_plot_diagnostics()` – automatically determines and renders all applicable plots for the given model type.

---

## Initialization

```python
from glazzbocks.diagnostics import ModelDiagnostics

diagnostics = ModelDiagnostics(pipeline)
```

- `pipeline` must be a fitted `sklearn.Pipeline` with a final estimator (classifier or regressor).

---

## API Reference

### `plot_roc_curve(X_test, y_test)`
Plot the ROC curve with the AUC score (binary classification only).

---

### `plot_f1_threshold(X_test, y_test)`
Plot F1 score against decision thresholds and highlight the optimal point (classification only).

---

### `plot_confusion_matrix(X_test, y_test, normalize='true')`
Show the confusion matrix with optional normalization (classification).

---

### `plot_lift_chart(X_test, y_test, bins=10)`
Visualize model lift across deciles of predicted probabilities (classification).

---

### `plot_cumulative_gain_chart(X_test, y_test)`
Cumulative gain curve showing proportion of positives captured (classification).

---

### `plot_predicted_vs_actual(X_test, y_test)`
Plot actual vs. predicted values with regression line (regression).

---

### `plot_residuals(X_test, y_test, check_normality=True)`
Plot residuals vs. predicted values. Also prints the result of D'Agostino normality test if applicable (regression).

---

### `plot_error_distribution(X_test, y_test)`
Histogram of residuals showing error distribution (regression).

---

### `plot_qq(X_test, y_test)`
Quantile-Quantile plot of residuals to assess normality visually (regression).

---

### `auto_plot_diagnostics(X_test, y_test)`
Automatically executes all appropriate diagnostic plots based on whether the final estimator is a classifier or regressor.

---

## Notes

- All visualizations are based on `matplotlib` and `seaborn`.
- Only binary classification is currently supported for threshold plots.
- Assumes pipeline is already trained/fitted.

---

## Example Usage

```python
from glazzbocks.diagnostics import ModelDiagnostics

# Assume pipeline is already fitted
diagnostics = ModelDiagnostics(pipeline)

# Generate all relevant diagnostic plots
diagnostics.auto_plot_diagnostics(X_test, y_test)
```