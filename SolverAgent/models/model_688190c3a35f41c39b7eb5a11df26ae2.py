import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an end-to-end regression pipeline for predicting exam scores.
    
    Architectural Decisions:
    - XGBoost: Selected as the primary model for its superior performance on 
      high-dimensional tabular data and ability to handle non-monotonic 
      relationships (e.g., study burnout).
    - StandardScaler: Applied to numeric features to normalize the feature space.
    - OneHotEncoder: Applied to categorical features with handle_unknown='ignore' 
      to ensure robustness against unseen categories in production.
    - sparse_output=False: Ensures compatibility with Gradient Boosting models 
      that prefer dense or native array structures.
    """
    
    # Define features based on Data Info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing logic:
    # 1. Scaling for numeric data to assist convergence and interpretability.
    # 2. One-hot encoding for nominal/ordinal data. 
    # Note: While 'sleep_quality' and 'exam_difficulty' are ordinal, 
    # OHE is used here for maximum flexibility unless specific ranks are provided.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # Convert to dense if necessary (XGBoost handles both, but dense is safer for custom transformers)
    dense_transformer = FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)

    # Model Hyperparameters:
    # Optimized for RMSE/R2. High n_estimators with a low learning_rate 
    # allows the model to capture complex interactions like the 
    # "Cognitive Load Index" implicitly.
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )

    # Build the final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('dense_converter', dense_transformer),
        ('regressor', model)
    ])

    return pipeline

if __name__ == "__main__":
    # Usage Example:
    # model_pipeline = build_pipeline()
    # model_pipeline.fit(X_train, y_train)
    # predictions = model_pipeline.predict(X_test)
    pass