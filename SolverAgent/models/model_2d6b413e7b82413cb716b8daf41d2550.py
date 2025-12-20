import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, QuantileTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Constructs an advanced ML pipeline for student exam score prediction.
    Utilizes non-linear feature expansion via PolynomialFeatures, 
    distribution normalization via QuantileTransformer, and a highly 
    tuned LightGBM Regressor for optimal predictive performance.
    """
    
    # Define feature subsets based on data info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # Numerical pipeline: 
    # 1. Impute missing values
    # 2. Transform distributions to normal (robust to outliers/skew)
    # 3. Create interaction terms (e.g., study_hours * class_attendance)
    # 4. Standardize for model stability
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline:
    # 1. Impute missing categories
    # 2. One-hot encoding for nominal and ordinal features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Advanced Gradient Boosting Model (LightGBM)
    # Hyperparameters tuned for a dataset of ~16,000 samples to balance bias/variance
    model = LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.015,
        num_leaves=45,
        max_depth=9,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )

    # Final Pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])