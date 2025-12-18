import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

def extract_meta_features(x):
    # Convert input to a pandas Series to use string accessor
    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0]
    elif isinstance(x, np.ndarray):
        s = pd.Series(x.flatten())
    else:
        s = pd.Series(x)
    
    s = s.astype(str)
    length = s.str.len().fillna(0)
    caps_count = s.str.count(r'[A-Z]').fillna(0)
    digit_count = s.str.count(r'\d').fillna(0)
    special_char_count = s.str.count(r'[^a-zA-Z0-9\s]').fillna(0)
    currency_symbols = s.str.count(r'[\$£€]').fillna(0)
    
    # Return as a 2D array for the pipeline
    return np.column_stack([
        length, 
        caps_count, 
        digit_count, 
        special_char_count, 
        currency_symbols,
        caps_count / (length + 1)
    ])

def build_pipeline() -> Pipeline:
    """
    Builds a robust SMS spam classification pipeline using TF-IDF with character n-grams 
    to handle 'leetspeak' and meta-features to capture high-risk patterns.
    """
    
    # 1. Feature Engineering via ColumnTransformer
    # We use character n-grams (2-5) as recommended to neutralize leetspeak (e.g., v1@gra)
    # and word-level TF-IDF for semantic context.
    preprocessor = ColumnTransformer([
        ('tfidf_word', TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=2500, 
            stop_words='english'
        ), 'text'),
        
        ('tfidf_char', TfidfVectorizer(
            analyzer='char', 
            ngram_range=(2, 5), 
            max_features=3000
        ), 'text'),
        
        ('meta_features', FunctionTransformer(extract_meta_features), 'text')
    ])

    # 2. Classifier
    # XGBoost is used here as a high-performance alternative to a stacking ensemble.
    # We set scale_pos_weight to address typical SMS spam class imbalance and 
    # optimize for AUPRC/F1-score as requested.
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,  # Common heuristic for spam:ham ratio
        random_state=42,
        eval_metric='aucpr',
        use_label_encoder=False
    )

    # 3. Complete Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        # Convert sparse output from Tfidf to dense if the classifier or subsequent steps require it
        ('dense_converter', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),
        ('clf', clf)
    ])