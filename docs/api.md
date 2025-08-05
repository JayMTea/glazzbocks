# API Reference

This section documents the core Glazzbocks components. Each module provides modular, interpretable functionality across the machine learning pipeline.

---

## Modules

### [`DataExplorer`](./dataexplorer.md)
Performs Exploratory Data Analysis (EDA) including:
- Missing value visualizations
- Target distribution plots
- Correlation heatmaps
- Skewness & normality testing
- Categorical and numeric summaries

### [`MLPipeline`](./mlpipeline.md)
Builds and trains a modeling pipeline:
- Handles preprocessing
- Supports sklearn-compatible estimators
- Fits and evaluates the model
- Generates performance summaries

### [`ModelDiagnostics`](./diagnostics.md)
Visualizes model performance:
- ROC, Confusion Matrix, F1 vs Threshold (classification)
- Residuals, QQ plot, predicted vs actual (regression)
- `auto_plot_diagnostics()` for smart diagnostics

### [`ModelInterpreter`](./interpreter.md)
Provides post-hoc model interpretation:
- SHAP summary and dependence plots
- Partial Dependence Plots
- Coefficient interpretation (linear models)
- Permutation importance

---

