# Titanic Survival Modeling Notes

## Task Understanding
- **Objective:** Predict passenger survival on the Titanic (`Survived` binary target).
- **Dataset:** Kaggle Titanic tabular data with features such as passenger class, sex, age, family size, and fare.
- **Metric:** Accuracy / weighted F1 (per Green Agent pipeline).

## Baseline Exploration
- Started with simple logistic regression due to small dataset and predominantly categorical/numeric features.
- Compared with decision tree and random forest baselines; logistic regression with proper preprocessing produced competitive accuracy while remaining lightweight for container inference.

## Feature Engineering
- Dropped high-cardinality text fields (`Name`, `Ticket`, `Cabin`) that add noise without strong signals for linear models.
- One-hot encoded categorical variables (`Sex`, `Embarked`).
- Scaled numeric columns (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`) and imputed missing values (median for numeric, most frequent for categorical).
- Considered engineered features (family size, deck extraction) but logistic regression already met baseline requirements without additional complexity.

## Training Procedure
- Used stratified 80/20 train/validation split to monitor generalization.
- Applied class weighting to address class imbalance (62% did not survive).
- Tuned the regularization strength implicitly via default settings; max iterations increased to ensure convergence.
- Validation F1 ≈ 0.78; accuracy ≈ 0.80 on the held-out split.

## Inference & Packaging
- Saved the sklearn pipeline via `joblib` so preprocessing and model weights are encapsulated together.
- `evaluate.py` loads the pipeline, consumes the test CSV exposed by the evaluator, and writes predictions in the required format.
- Docker image is based on `python:3.10-slim`; runtime footprint is minimal (<150 MB) and cold start well under evaluation limits.

## Future Improvements
- Incorporate cross-validation with hyperparameter tuning (e.g., elastic net penalty).
- Engineer additional features (title extraction from names, cabin deck) to boost accuracy.
- Ensemble with gradient boosting or histogram-based models if latency budgets allow.
