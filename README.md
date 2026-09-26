# ML Experiment Baseline: Decision Tree on Digits

A small, reproducible scikit-learn experiment comparing Decision Tree accuracy and runtime on the built-in Digits dataset. This repository is a learning baseline; it is not a GPU benchmark or a benchmark across hardware devices.

## Experiment

`sklearn.datasets.load_digits` provides 1,797 handwritten digit samples represented by 64 pixel features. The script creates one stratified 80/20 train/test split with seed `42` (1,437 training and 360 test samples), then evaluates:

| Configuration | Model | `max_depth` | `random_state` |
| --- | --- | ---: | ---: |
| `tree_depth_5` | `DecisionTreeClassifier` | 5 | 42 |
| `tree_depth_10` | `DecisionTreeClassifier` | 10 | 42 |

For each configuration, the experiment records test accuracy and measures fit and prediction time separately.

## Requirements and setup

Tested with Python `3.12.14`. Dependency versions are pinned in `requirements.txt`.

From the repository root on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Run the experiment from the repository root:

```bash
python src/train.py
```

The script writes or overwrites `results/results.csv` and `results/results.json`. Each file contains one record per configuration, including configuration name, model, depth, seed, accuracy, fit/prediction seconds, and train/test sample counts.

Run the smoke test from the repository root:

```bash
python -m pytest -q
```

The test uses small data subsets and a temporary output directory. It checks the experiment record and verifies that CSV and JSON results can be written and read back without modifying the project's `results/` files.

## Limitations

- The built-in Digits dataset is small and does not represent real-world image recognition workloads.
- Results use one train/test split; there is no cross-validation or confidence interval.
- Aggregate accuracy does not show errors by digit class.
- Each runtime is measured once and depends on the CPU and system load; runtime is not expected to match exactly across runs or machines.
- Comparing `max_depth` values 5 and 10 is a demonstration, not an exhaustive hyperparameter search.
- The test set is used to compare configurations in this demo. A rigorous model-selection workflow should use a separate validation set and reserve the test set for final evaluation.

## Reproducibility

For this v0 baseline, the same code, pinned dependencies, seed, and scikit-learn bundled dataset are expected to produce the same split and accuracy. Exact runtime values are not expected to be reproducible.

