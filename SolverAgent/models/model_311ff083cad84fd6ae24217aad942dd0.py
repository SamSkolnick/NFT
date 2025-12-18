import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

def extract_metadata(text_data):
    # text_data will be a 1D array-like from ColumnTransformer
    s = pd.Series(text_data.ravel().astype(str))
    
    # Feature 1: Message length
    length = s.str.len().fillna(0)
    
    # Feature 2: Ratio of capital letters (signal for URGENT/FREE)
    caps = s.str.findall(r'[A-Z]').str.len().fillna(0) / (length + 1)
    
    # Feature 3: Count of digits (signal for phone numbers/codes)
    nums = s.str.findall(r'[0-9]').str.len().fillna(0)
    
    # Feature 4: Presence of URLs or common spam triggers
    urls = s.str.contains(r'http|www|\.com|txt|stop|free', case=False).astype(int)
    
    return np.column_stack([length, caps, nums, urls])

def build_pipeline() -> Pipeline:
    # Text Processing: Use both word-level and char-level n-grams
    # Word n-grams capture context, char n-grams (2-5) capture l33tspeak and typos
    text_features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=4000, 
            lowercase=False,
            token_pattern=r'\b\w+\b|[^\w\s]'
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char', 
            ngram_range=(2, 5), 
            max_features=4000, 
            lowercase=False
        )),
        ('metadata', FunctionTransformer(extract_metadata))
    ])

    # Select the 'text' column for processing
    preprocessor = ColumnTransformer([
        ('text_pipeline', text_features, 'text')
    ])

    # XGBoost Classifier
    # scale_pos_weight addresses the typical 85/15 imbalance in SMS datasets
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])