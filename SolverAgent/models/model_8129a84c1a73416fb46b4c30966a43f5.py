import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to implement domain-specific feature engineering:
    1. Log-transformation of study hours to handle diminishing marginal utility.
    2. Interaction term between Study Hours and Sleep Quality to capture fatigue effects.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # Non-linear scaling: Log-transform study time to handle outliers and diminishing returns
        X_out['study_hours_log'] = np.log1p(X_out['study_hours'])
        
        # Interaction Terms: Study_Hours × Sleep_Quality
        # We convert sleep_quality categories to an ordinal scale for the interaction calculation
        mapping = {
            'poor': 1, 
            'fair': 2, 
            'medium': 2, 
            'average': 3, 
            'good': 4, 
            'excellent': 5
        }
        # Safely map the quality labels, defaulting to a median value (3) if unseen
        sq_numeric = X_out['sleep_quality'].astype(str).str.lower().map(mapping).fillna(3)
        X_out['study_sleep_interaction'] = X_out['study_hours'] * sq_numeric
        
        return X_out

def build_pipeline() -> Pipeline:
    """
    Builds a complete machine learning pipeline for exam score regression.
    Target: exam_score
    Features: Mixed numeric and categorical.
    """
    
    # Feature lists based on the dataset description
    # These include original columns and engineered columns created in the 'engineer' step
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'study_hours_log', 'study_sleep_interaction']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # Preprocessing: StandardScaler for numeric, OneHotEncoder for categorical
    # Using sparse_output=False ensures compatibility with some downstream transformers/models
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols if 'categorical_cols' in locals() else categorical_features)
        ]
    )

    # The full pipeline:
    # 1. Domain Engineering (Log transforms and Interactions)
    # 2. ColumnTransformer (Scaling and Encoding)
    # 3. XGBoost Regressor (State-of-the-art for tabular data non-linear interactions)
    return Pipeline([
        ('engineer', StudentFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            max_depth=7,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            objective='reg:squarederror'
        ))
    ])