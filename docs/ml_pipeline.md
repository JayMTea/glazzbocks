# MLPipeline

`MLPipeline` provides an end-to-end framework to build, train, and evaluate machine learning models using interpretable and auditable workflows. It is compatible with any `scikit-learn`-compatible classifier or regressor.

## Key Features

- Automatic detection of task type (classification or regression)
- Modular preprocessing for numeric and categorical data
- Built-in support for imputation and scaling
- Seamless integration with `scikit-learn` models
- Cross-validation with fold-wise metrics
- Feature importance extraction (if model supports it)
- Model persistence (save/load pipelines)

## Initialization

```python
from glazzbocks.pipeline import MLPipeline

pipeline = MLPipeline(df, target_column='target', model=LogisticRegression())
```

### Parameters

- `df`: A pandas DataFrame containing the features and target
- `target_column`: The name of the target column
- `model`: A scikit-learn compatible estimator (e.g., `RandomForestClassifier`, `Ridge`)

---

## API Reference

### `build_pipeline()`
> Constructs a `Pipeline` with preprocessing steps for numeric and categorical features

### `train_model(test_size=0.2, random_state=42)`
> Splits the data, trains the pipeline, and stores the trained model

### `evaluate_model(cv=5)`
> Performs cross-validation and outputs performance metrics based on task type

### `predict(X)`
> Returns model predictions on new data

### `get_feature_importance()`
> Extracts and displays feature importances if supported by the model

### `save_pipeline(path)`
> Saves the trained pipeline to disk using `joblib`

### `load_pipeline(path)`
> Loads a saved pipeline from disk

---

## Example Usage

```python
from sklearn.ensemble import RandomForestClassifier
from glazzbocks.pipeline import MLPipeline

pipeline = MLPipeline(df, target_column='churn', model=RandomForestClassifier())
pipeline.build_pipeline()
pipeline.train_model()
pipeline.evaluate_model()
```

---

## Notes

- Automatically handles categorical vs numeric preprocessing
- Logs pipeline components for transparency
- Easily extendable to support hyperparameter tuning or custom transformers