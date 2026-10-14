# mgt3500

Python package containing supervised and unsupervised data mining functions from the function cells in the MGT 3500 Topic 5, Topic 6, and Topic 7 course scripts.

## Install from GitHub

```bash
python -m pip install git+https://github.com/YvonneGao8/mgt3500_function.git
```

In a Jupyter notebook, use `%pip install git+https://github.com/YvonneGao8/mgt3500_function.git`.

## Use

```python
from mgt3500 import (
    class_eval_num, class_eval_fig, prd_evl,
    linear_reg, lpm, logit, naive_bayes, KNN,
    classification_tree, regression_tree,
    cluster_kmeans, association_rule,
)
```

The model functions expect the pandas DataFrame inputs used in the course script. For example, `linear_reg(X, y)` fits and evaluates a model, while `linear_reg(X, y, new_X)` returns predictions for new data. See the course script for the full arguments and workflow for each function.

## Dependencies

The package installs these dependencies automatically, including the clustering and association-rule dependencies `kneed`, `scipy`, and `apyori`:

- `pandas>=2.1`
- `numpy`
- `matplotlib`
- `statsmodels`
- `scikit-learn`
- `scipy`
- `kneed`
- `apyori`

These are also declared in `pyproject.toml`.

## Included functions

- Evaluation: `class_eval_num`, `class_eval_fig`, `prd_evl`
- Supervised models: `linear_reg`, `lpm`, `logit`, `naive_bayes`, `KNN`, `classification_tree`, `regression_tree`
- Unsupervised models: `cluster_kmeans`, `association_rule`
