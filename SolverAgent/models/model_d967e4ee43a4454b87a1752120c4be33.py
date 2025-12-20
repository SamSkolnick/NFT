import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced Machine Learning pipeline for student exam score prediction.
    Utilizes LightGBM (Gradient Boosted Decision Trees) as the primary regressor,
    complemented by interaction term engineering and robust categorical encoding.
    """
    
    # Feature definitions based on provided dataset info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing for numeric data:
    # 1. Scaling to ensure features are on a comparable range.
    # 2. PolynomialFeatures (interaction_only) to capture the multiplicative effect 
    #    of Study Hours vs Attendance as recommended in research.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('interaction', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    # Preprocessing for categorical data:
    # Standard One-Hot encoding to handle categorical variables like Course and Study Method.
    # handle_unknown='ignore' ensures robustness against unseen categories in production.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Trees (LightGBM):
    # This architecture is SOTA for tabular data. Parameters are set to provide a 
    # balance between high complexity (to capture student nuances) and regularization 
    # (to prevent overfitting on the sample size).
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,   # L1 regularization
        reg_lambda=0.2,  # L2 regularization
        importance_type='gain',
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    # Construct the final pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])