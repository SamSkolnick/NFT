import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    # Feature lists based on data schema
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # Custom feature engineering function to handle non-linear interactions and scaling
    def transform_student_data(X):
        # X is a numpy array passed from ColumnTransformer containing numeric_features
        # Index mapping: 0:age, 1:study_hours, 2:class_attendance, 3:sleep_hours
        study_h = X[:, [1]]
        sleep_h = X[:, [3]]
        
        # Interaction Feature: Study_Hours * Sleep_Hours (synergistic effect)
        interaction = study_h * sleep_h
        
        # Non-linear Scaling: Logarithmic transform on study_hours (diminishing returns)
        log_study = np.log1p(study_h)
        
        # Return concatenated original features + new features
        return np.hstack([X, interaction, log_study])

    # Numeric pipeline with custom engineering and RobustScaler to handle behavioral outliers
    numeric_transformer = Pipeline(steps=[
        ('engineer', FunctionTransformer(transform_student_data)),
        ('scaler', RobustScaler())
    ])

    # Categorical pipeline using OneHotEncoding
    # Note: TfidfVectorizer is imported but only used if a high-cardinality text column existed.
    # Here, categorical features are standard labels, so OneHot is most appropriate.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Gradient Boosted Decision Tree (XGBoost) for regression
    # Optimizing for RMSE and R2 through robust non-linear modeling
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror'
    )

    # Build and return the final pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])