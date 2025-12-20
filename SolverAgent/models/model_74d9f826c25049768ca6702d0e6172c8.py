from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def build_pipeline() -> Pipeline:
    """
    Builds a robust classification pipeline for structured data with text features.
    Features used:
    - Numeric: ['0', '3', '22', '1.1', '0.1', '7.25'] (Imputed)
    - Categorical: ['male', 'S'] (One-Hot Encoded)
    - Text: ['Braund, Mr. Owen Harris'] (TF-IDF Vectorized)
    """
    
    # Feature groups based on data info
    num_features = ['0', '3', '22', '1.1', '0.1', '7.25']
    cat_features = ['male', 'S']
    text_feature = 'Braund, Mr. Owen Harris'

    # Transformer for numeric columns: Median imputation is robust to outliers (e.g., Age/Fare)
    numeric_transformer = SimpleImputer(strategy='median')

    # Transformer for categorical columns: Most frequent imputation followed by One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Transformer for name/text: Captures titles (Mr, Miss, Master) which are highly predictive
    # Passing the column name as a string (not a list) ensures the transformer receives a 1D Series
    text_transformer = TfidfVectorizer(max_features=100, lowercase=True, analyzer='word')

    # Composite preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Trees (XGBoost) for high accuracy and speed
    # We use a lower learning rate and moderate depth for better generalization
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    # Final Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline