import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced machine learning pipeline for student exam score prediction.
    Utilizes Target Encoding for high-cardinality categoricals, 
    Interaction terms for numerical features, and XGBoost for high-capacity modeling.
    """
    
    # Define feature groups based on data info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 
                            'study_method', 'facility_rating', 'exam_difficulty']

    # Numerical Transformer: Impute missing values (with indicator), generate interactions, and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Use Target Encoding as recommended for GBDTs 
    # and to handle potentially high-cardinality features like 'course'
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_enc', TargetEncoder(smooth='auto', random_state=42))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Tree Regressor
    # Hyperparameters set for robustness on tabular data size 16000
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method='hist'
    )

    # Construct final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline