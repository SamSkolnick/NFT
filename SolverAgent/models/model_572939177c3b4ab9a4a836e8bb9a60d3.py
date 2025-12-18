from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

def build_pipeline() -> Pipeline:
    # Feature selection based on provided Data Info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # Numeric pipeline: Imputation for missing values and robust scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Frequent imputation and One-Hot Encoding
    # Note: Using sparse_output=False or a FunctionTransformer to ensure dense output for XGBoost
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Function to ensure the output is dense before passing to the regressor
    def ensure_dense(x):
        return x.toarray() if hasattr(x, "toarray") else x

    # Regression model: XGBoost Regressor
    # Optimized for tabular behavioral data to capture non-linear interactions between study and sleep
    regressor = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror'
    )

    # Construct the final runnable pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('to_dense', FunctionTransformer(ensure_dense)),
        ('regressor', regressor)
    ])