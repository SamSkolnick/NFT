from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import numpy as np

def build_pipeline() -> Pipeline:
    """
    Builds a machine learning pipeline for heart disease classification.
    Focuses on maximizing Recall for the positive class ('Presence').
    
    The architecture uses XGBoost, which is highly effective for clinical tabular data,
    incorporating class weighting to prioritize medical safety (reducing False Negatives).
    """
    
    # Feature selection based on the provided schema
    # 'Heart Disease' is the target and is excluded from features
    numeric_features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    # Preprocessing:
    # 1. Median Imputation: Robust against outliers in clinical measurements like Cholesterol or BP.
    # 2. StandardScaler: While XGBoost is tree-based, scaling is good practice for pipeline stability.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ],
        remainder='drop'
    )

    # Model Configuration:
    # scale_pos_weight: Set to > 1 to increase the penalty for missing a positive case (Heart Disease).
    # This directly optimizes for Recall as requested.
    # max_depth and learning_rate are tuned conservatively for a small dataset (216 rows).
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=2.0,  # Hyperparameter to favor Recall over Precision
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Complete Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline