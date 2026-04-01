# K-Means Clustering Module (Member 4)

This folder contains the unsupervised clustering implementation for anime audience/content segmentation.

## Files

- kmeans_mal.ipynb: main notebook
- requirements.txt: Python dependencies
- train_and_save.py: script entry point for training and saving artifacts
- export_cluster_input_template.py: export CSV template for custom input testing
- predict_cluster.py: predict cluster labels for custom inputs
- src/data_utils.py: dataset loading utilities
- src/preprocessing.py: feature engineering and preprocessing utilities
- src/clustering.py: k-means training, k selection, evaluation, and artifact saving
- artifacts/: output folder for trained model and results

## Dataset

Kaggle dataset link:
https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews

Place these files in one of the following paths:

1. kmeans_clustering/data/
2. data/ (project root)
3. project root

Required CSV files:

- animes.csv
- profiles.csv
- reviews.csv

## Setup

### 1. Open project in VS Code

1. Open VS Code.
2. Open the project folder `Entertainment-Rating-Prediction-ML`.
3. Open terminal in `kmeans_clustering/`.

### 2. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install requirements

```powershell
pip install -r requirements.txt
```

If `pip` is not recognized:

```powershell
python -m pip install -r requirements.txt
```

## Run in Jupyter Notebook (VS Code)

1. Install extensions: Python + Jupyter.
2. Open `kmeans_clustering/kmeans_mal.ipynb`.
3. Select the `.venv` kernel.
4. Run all cells.

## Run as Python script

From project root:

```powershell
python kmeans_clustering/train_and_save.py
```

## Saved outputs

After training, these files are generated in `kmeans_clustering/artifacts/`:

- kmeans_pipeline.joblib
- k_evaluation.csv
- cluster_assignments.csv
- cluster_profile.csv
- metadata.json

## Test with custom inputs

### Step 1: Export template

```powershell
python kmeans_clustering/export_cluster_input_template.py
```

### Step 2: Edit template

Edit `kmeans_clustering/artifacts/cluster_input_template.csv`.
Keep column names unchanged.

### Step 3: Predict cluster

```powershell
python kmeans_clustering/predict_cluster.py
```

Output file:

- `kmeans_clustering/artifacts/cluster_predictions.csv`

## Modeling details

- Clustering algorithm: K-Means
- Candidate K range: 2 to 10
- K selection metrics: silhouette score, Calinski-Harabasz, Davies-Bouldin, inertia
- Dimensionality reduction: TruncatedSVD on preprocessed feature matrix
- Leakage control: `score`, `rank`, and `ranked` excluded from clustering features
- Post-hoc analysis: compare resulting clusters against actual `score` values in report only
