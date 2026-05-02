# Results and Discussion

This section summarizes the observed performance and insights from all implemented models in the repository. Metrics reported here are taken from saved artifacts and/or recorded notebook outputs.

---

## 1. Linear Regression (Ridge / LinearRegression / SGDRegressor) — Rating Prediction

### Results

The linear regression module trains multiple linear baselines on engineered anime metadata (with optional review aggregates) and selects the best pipeline. From the saved notebook run artifacts:

- **Best model:** Ridge Regression
- **Test set performance:**
  - **MAE:** 0.4662
  - **RMSE:** 0.6186
  - **R²:** 0.6145
- **Cross-validated performance (training set):**
  - **Mean R²:** 0.6131
  - **Std R²:** 0.0128

### Discussion

- An **R² ≈ 0.61** indicates the model explains a meaningful share of the variance in community ratings using tabular features, but a large portion remains unexplained. This is expected because rating outcomes depend on latent factors (narrative quality, studio reputation, seasonal effects, audience trends) that are not fully captured by the available metadata.
- Ridge and plain Linear Regression perform nearly identically, suggesting the preprocessing + dimensionality reduction step already stabilizes the feature space and reduces multicollinearity effects.
- The pipeline uses **TruncatedSVD** after one-hot encoding. This is effective for handling sparse, high-dimensional categorical features, but it reduces direct interpretability at the original-feature level (coefficients live in the reduced embedding space).

### Limitations

- **Linearity assumption:** relationships between engagement proxies (e.g., members) and rating can be non-linear and saturating.
- **Feature availability bias:** some predictors may reflect post-release popularity rather than purely intrinsic quality.
- **Information loss:** excluding high-cardinality categoricals and dropping long text fields (e.g., synopsis) can remove useful signal.

### Future Work

- Evaluate stronger non-linear regressors (e.g., gradient boosting) under the same leakage controls.
- Add richer features (text embeddings from synopsis/title, studio/producer signals, time-aware features like aired year) and compare performance.
- Use evaluation splits that better reflect deployment (e.g., **time-based split** by aired year) to estimate real-world generalization.

---

## 2. Random Forest Regressor — Rating Prediction

### Results

The random forest notebook implements a RandomForestRegressor on **animes.csv** with genre multi-hot encoding. From the recorded notebook output:

- **MSE:** 0.53245
- **R²:** 0.54233

The feature-importance plot indicates the strongest predictors are typically:
- **members** and **popularity** (dominant)
- **episodes** (smaller contribution)
- individual **genre** indicators (modest contributions)

### Discussion

- Random forests can capture non-linearities and feature interactions; however, the observed **R² ≈ 0.54** is lower than the Ridge baseline recorded in this repository.
- A likely reason is that the notebook uses aggressive row filtering (**dropna()**) and a relatively constrained feature set (mostly engagement proxies + genres), which can reduce dataset size and shift the distribution.
- The dominance of members/popularity is intuitive: engagement proxies correlate with visibility and community rating behavior.

### Limitations

- **Potential leakage risk:** the notebook includes/excludes columns by name; if a score-derived ranking field (e.g., ranked/rank-like) is included, it can indirectly encode the target.
- **No cross-validation/hyperparameter search** is reported, so the metric may be sensitive to one train/test split.
- **Reproducibility:** Colab-style file paths and notebook-only execution make it harder to rerun consistently across environments.

### Future Work

- Replace dropna-based filtering with imputation and an sklearn **Pipeline** for repeatability.
- Add cross-validation and hyperparameter tuning; report uncertainty (mean/std across folds).
- Save the model + metadata artifacts (joblib + results CSV) similar to the linear_regression module.

---

## 3. SVM Classification (Hit vs Standard) — Classification

### Results

The SVM module defines **Hit** as **score > 8.0** and trains kernels using PR-AUC for selection. From the recorded notebook output:

- **Kernel PR-AUC (selection metric):**
  - **RBF:** 0.8575 (best)
  - **Poly:** 0.7826
- **Best model (RBF) test accuracy:** 0.9429
- **Best model (RBF) PR-AUC:** 0.8575

### Discussion

- While accuracy is high, **PR-AUC is the more informative metric** under class imbalance (Hits are typically fewer than Standard releases). PR-AUC reflects the ability to identify Hits without being dominated by majority-class correctness.
- The RBF kernel outperforming polynomial suggests the decision boundary is **non-linear** in the feature space.
- The feature set (members, popularity, episodes, ranked, and multi-label genres) appears highly predictive for this task; however, some of these predictors may represent post-release popularity signals.

### Limitations

- **Threshold definition:** “Hit” = score > 8.0 is a chosen cutoff; results may shift if the threshold changes.
- **Potential leakage / timing bias:** features like ranked (and in some cases popularity) can be strongly linked to score and/or only known after release.
- **Generalization:** evaluation appears to be based on a single split; performance can vary across different splits or time periods.

### Future Work

- Re-evaluate with a **leakage-safe** feature set (exclude ranked/rank-like fields; consider pre-release features only).
- Tune the decision threshold for application goals (maximize recall for Hits vs maximize precision).
- Add stratified cross-validation and/or a time-based split to assess stability.

---

## 4. K-Means Clustering — Audience/Content Segmentation

### Results

The clustering module trains a K-Means model on engineered features (with TruncatedSVD embedding). It explicitly excludes score/rank/ranked from clustering features and uses score only for post-hoc interpretation.

From saved artifacts:

- **Selected k:** 2
- **Silhouette (train):** 0.6764
- **Silhouette (test):** 0.6648
- **Silhouette gap:** 0.0116 (small)

Cluster profiling indicates two clear segments:

- **Cluster 0 — Long-tail / Lower Engagement**
  - size: 18,913
  - mean score: 6.40
  - mean members: ~22k
  - mean popularity rank: ~7,880
- **Cluster 1 — Mainstream / High Engagement**
  - size: 398
  - mean score: 8.23
  - mean members: ~638k
  - mean popularity rank: ~176

### Discussion

- The result separates a small high-engagement cluster from a large long-tail cluster, consistent with real-world popularity distributions.
- High silhouette values suggest strong separation in the embedding space, and the small train–test silhouette gap suggests the segmentation is fairly stable under the current pipeline.
- The post-hoc score difference between clusters supports interpretability: high engagement tends to align with higher ratings, even though score was not used as a clustering feature.

### Limitations

- **Coarse segmentation:** k=2 is simple and may not capture mid-tier or niche sub-segments.
- **Model assumptions:** K-Means favors spherical clusters in Euclidean space and can struggle with irregular shapes or varying densities.
- **Pipeline sensitivity:** results depend on scaling, SVD dimension choice, and feature engineering decisions.

### Future Work

- Explore k>2 and evaluate **cluster stability** across random seeds and resamples.
- Try alternative clustering (Gaussian Mixtures, HDBSCAN) and/or alternative embeddings (UMAP) to capture non-spherical structure.
- Improve interpretability with per-cluster feature summaries and representative title sampling.

---

## Cross-Model Summary (Optional wrap-up)

- Engagement proxies (members/popularity) consistently dominate predictive signal across models, highlighting that visibility strongly correlates with community ratings and hit status.
- Score prediction performance in saved outputs ranges roughly from **R² ≈ 0.54 (Random Forest)** to **R² ≈ 0.61 (Ridge)**.
- A key report caveat is **leakage/timing bias**: some variables may be derived from, or only observable after, the target outcome.
