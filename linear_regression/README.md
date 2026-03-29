# Linear Regression Module (Member 1)

This folder contains the supervised learning implementation for rating prediction using linear-regression-based models.

## Files

- linear_regression_mal.ipynb: main notebook
- requirements.txt: Python dependencies
- train_and_save.py: script entry point for training and saving artifacts
- src/data_utils.py: dataset loading utilities
- src/preprocessing.py: feature engineering and preprocessing utilities
- src/training.py: model training, evaluation, and artifact saving
- artifacts/: output folder for trained model and results

## Dataset

Kaggle dataset link:
https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews

Download and place these files in one of the following:

1. linear_regression/data/
2. data/ (project root)
3. project root

Required CSV files:

- animes.csv
- profiles.csv
- reviews.csv

## What the notebook does

1. Loads and inspects all three CSV files.
2. Chooses target as anime score.
3. Performs preprocessing:
   - missing value imputation
   - categorical encoding
   - leakage prevention (rank removed)
   - feature engineering (duration minutes, aired year, genre count, text lengths)
   - optional review aggregates
4. Applies dimensionality reduction with TruncatedSVD.
5. Trains and compares:
   - LinearRegression
   - Ridge
   - SGDRegressor (gradient descent)
6. Evaluates with MAE, RMSE, R2, and cross-validated R2.
7. Produces plots and discussion points for viva.

## Setup

### 1. Open project in VS Code

1. Open VS Code.
2. Open the project folder `Entertainment-Rating-Prediction-ML`.
3. Open a terminal in VS Code in the project root.

### 2. Create and activate virtual environment (recommended)

Windows PowerShell:

```powershell
cd linear_regression/
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
cd linear_regression/
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install requirements

From the project root run:

```powershell
pip install -r requirements.txt
```

If `pip` is not recognized, use:

```powershell
python -m pip install -r requirements.txt
```

## Run the notebook in VS Code

1. Install the VS Code extensions:
   - Python (Microsoft)
   - Jupyter (Microsoft)
2. Open `linear_regression/linear_regression_mal.ipynb`.
3. Click `Select Kernel` and choose your `.venv` Python interpreter.
4. Run cells from top to bottom using `Run All`.
5. Run the final "Save Trained Model and Artifacts" section.
6. Confirm files are generated in `linear_regression/artifacts/`.

## Run the notebook in classic Jupyter (browser)

From project root:

```powershell
jupyter notebook
```

Then in the browser:

1. Navigate to `linear_regression/`.
2. Open `linear_regression_mal.ipynb`.
3. Run all cells.

## Run as Python script (separate files workflow)

From project root:

```powershell
python linear_regression/train_and_save.py
```

This will train models and save outputs to `linear_regression/artifacts/`.

## Saved outputs after training

When training completes, the following files are saved:

- `artifacts/best_model.joblib` (best trained pipeline)
- `artifacts/results.csv` (metrics for all compared models)
- `artifacts/metadata.json` (best model and run summary)

If you run from notebook, it also saves:

- `artifacts/best_model_from_notebook.joblib`
- `artifacts/results_from_notebook.csv`
- `artifacts/metadata_from_notebook.json`

## Model performance

From current run:

- Ridge R2 test is about 0.614.
- MAE is about 0.466 on a 1-10 style score scale.

This is a solid baseline result for a linear model on this dataset. It explains about 61.4% of score variance while remaining interpretable.

Ridge is selected as best because it gave the highest R2 in this run. The margin vs LinearRegression is very small, but Ridge regularization usually improves stability when features are correlated.

## Testing with input data

### Step 1: Export prediction input template

From project root:

```powershell
python linear_regression/export_prediction_template.py
```

This creates:

- `linear_regression/artifacts/prediction_input_template.csv`

### Step 2: Edit the template

Open the template CSV and edit values for the anime you want to test.

Use these rules:

- Keep all original column names unchanged.
- Add one row per anime you want to score.
- Numeric columns should stay numeric.
- Categorical columns should use text categories similar to training data.

### Step 3: Run prediction

From project root:

```powershell
python linear_regression/predict_with_saved_model.py
```

This creates:

- `linear_regression/artifacts/prediction_output.csv` with a new `predicted_score` column.

## Quick troubleshooting

- `FileNotFoundError` for CSV files:
  - Ensure `animes.csv`, `profiles.csv`, and `reviews.csv` are placed in one of the dataset paths listed above.
- Kernel not showing in VS Code:
  - Re-activate virtual environment and reload VS Code window.
- Package import error:
  - Re-run dependency installation command and restart the notebook kernel.
