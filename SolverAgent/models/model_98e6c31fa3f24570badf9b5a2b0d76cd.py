import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    """
    Builds a machine learning pipeline for heart disease classification.
    Uses Quantile Transformation to handle skewed clinical distributions and 
    XGBoost with weighted classes to maximize recall for the 'Presence' class.
    """
    
    # Feature columns as specified in the data info (excluding target 'Heart Disease')
    feature_cols = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    
    # Preprocessing: QuantileTransformer is used to normalize clinical features 
    # like Cholesterol and BP, which often have outliers or skewed distributions.
    # n_quantiles is tuned for the small dataset size (216 samples).
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_scaling', QuantileTransformer(n_quantiles=100, 
                                                   output_distribution='normal', 
                                                   random_state=42), feature_cols)
        ]
    )
    
    # Model: XGBoost is chosen for its superior performance on tabular clinical data.
    # To maximize RECALL for the 'Presence' class, we use scale_pos_weight.
    # A value > 1.0 encourages the model to avoid False Negatives (missing heart disease).
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=3.0,  # Focus on maximizing recall for the positive class
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Complete Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline