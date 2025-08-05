# DataExplorer

The `DataExplorer` class provides a structured way to explore your dataset before modeling. It includes descriptive summaries, statistical checks, and exportable visuals—tailored for regression or classification tasks.

## Basic Usage

```python
from glazzbocks import DataExplorer

explorer = DataExplorer(df, target_col='target')

# Summaries
explorer.numeric_summary()
explorer.categorical_summary()

# Visuals
explorer.plot_target()
explorer.correlation_heatmap()
explorer.plot_missing_matrix()

# Statistical Checks
vif_df = explorer.calculate_vif()
normality_df = explorer.test_normality()
