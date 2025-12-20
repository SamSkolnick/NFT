import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, RobustScaler
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced machine learning pipeline for predicting student exam scores.
    Uses LightGBM as the core regressor due to its superior performance on tabular data,
    and applies Quantile Transformation to handle skewed numerical features as recommended 
    in student performance research.
    """
    
    # Define features based on metadata
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # Preprocessing for numerical data:
    # Use QuantileTransformer to normalize skewed distributions like study_hours and attendance.
    # RobustScaler is added to handle potential outliers in study or sleep hours.
    numeric_transformer = Pipeline(steps=[
        ('robust', RobustScaler()),
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42))
    ])

    # Preprocessing for categorical data:
    # Using OneHotEncoder for categorical features. 
    # handle_unknown='ignore' ensures the pipeline doesn't break on new categories in production.
    categorical_transformer = Pipeline(steps=[
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

    # Gradient Boosted Decision Trees (LightGBM)
    # Hyperparameters tuned for a dataset of 16k rows to prevent overfitting while capturing complexity.
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )

    # Create the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline