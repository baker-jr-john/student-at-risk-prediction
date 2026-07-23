# Student At-Risk Prediction

Mid-semester early-warning model for student academic outcomes, built on the Kaggle [Student Performance & Behavior Dataset](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset), plus a data-quality/bias audit of the dataset's "biased" variant.

Start with [`WRITEUP.md`](WRITEUP.md) for what was found and built and why. Everything below is how to reproduce it.

All commands below are run from the project root — the folder containing this README and [`src/`](src).

## Setup

Requires Python 3.10+. From this directory, create a virtual environment and install the project (this creates the `.venv` used in the Run step):

```bash
# with uv
uv venv .venv && uv pip install --python .venv/bin/python -e .

# or with pip
python -m venv .venv && .venv/bin/pip install -e .
```

## Data

Put the two CSVs from the Kaggle package in the [`data/`](data) folder:

- `Students Performance Dataset.csv` (the "primary" file)
- `Students_Grading_Dataset_Biased.csv` (the "biased" file)

A `data/` folder next to the project root (`../data`) is also auto-detected, or pass `--data-dir` to run against data anywhere else.

## Run

```bash
.venv/bin/python -m src.run_all            # or: --data-dir /path/to/data
```

Runs in a few minutes (the feature-group ablation alone retrains the model eleven times) and writes everything to [`outputs/`](outputs):

| Output | What it is |
|---|---|
| [`validation_report.md`](outputs/validation_report.md) | Data checked against its own documentation; the evidence for every modeling decision |
| [`model_report.md`](outputs/model_report.md) | Model comparison, held-out metrics, subgroup check, coefficients |
| [`audit_report.md`](outputs/audit_report.md) | Why the biased file must not be used for outcome modeling |
| `fig_*.png` | ROC, threshold choice, feature importance, audit figures |

The run is deterministic (fixed seed); re-running reproduces the committed numbers.

## Layout

Source lives in [`src/`](src):

```
src/config.py    # feature roles, constants — the mid-semester scenario is defined here
src/data.py      # loading + validation report
src/features.py  # preprocessing pipelines
src/model.py     # selection, group ablation, threshold choice, evaluation, subgroup check
src/audit.py     # biased-file audit
src/run_all.py   # entry point
```
