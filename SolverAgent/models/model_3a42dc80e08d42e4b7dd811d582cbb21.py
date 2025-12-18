import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    """
    Builds a machine learning pipeline for heart disease classification.
    Focuses on maximizing Recall for the 'Presence' class using state-of-the-art 
    XGBoost and robust preprocessing for clinical data.
    """
    
    # Feature selection based on provided clinical data
    numeric_features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    
    # Preprocessing: 
    # 1. MICE (IterativeImputer) for missing clinical values
    # 2. RobustScaler to handle medical outliers in measurements like BP and Cholesterol
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', IterativeImputer(max_iter=10, random_state=42)),
                ('scaler', RobustScaler())
            ]), numeric_features)
        ]
    )
    
    # Model: XGBoost Classifier
    # scale_pos_weight is set to > 1 to prioritize Recall (Sensitivity) for the 'Presence' class.
    # Given the small dataset size (216 rows), we use a conservative max_depth and learning_rate.
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=2.5,  # Cost-sensitive learning for Recall optimization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        gamma=0.1,             # Minimum loss reduction for further partition
        subsample=0.8          # Stochastic gradient boosting to prevent overfitting
    )
    
    # Return the complete scikit-learn Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model)
    ])