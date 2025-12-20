import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student exam score prediction.
    Utilizes LightGBM for state-of-the-art tabular performance, featuring 
    automated interaction terms and robust categorical handling.
    """
    
    # Feature groups based on dataset info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # Numeric preprocessing: Imputation, Scaling, and Interaction Terms (as suggested by research)
    # Interaction terms capture the non-linear relationship between effort (hours) and participation (attendance)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    # Categorical preprocessing: Handling missing values and One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
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

    # Model: LightGBM Regressor
    # Hyperparameters selected for a balance of complexity and regularization on a 16k dataset
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    # Construct final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline