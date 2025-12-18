from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

class TextMetadataExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts metadata features from SMS text: length, capital letter count, 
    digit count, and specific punctuation indicators.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure X is treated as a 1D sequence of strings
        if hasattr(X, 'values'):
            X = X.values
        X = X.flatten() if hasattr(X, 'flatten') else X
        
        features = []
        for text in X:
            text_str = str(text)
            length = len(text_str)
            capitals = sum(1 for c in text_str if c.isupper())
            digits = sum(1 for c in text_str if c.isdigit())
            # Count spam-heavy punctuation and symbols
            special_chars = sum(1 for c in text_str if c in ('$', '£', '€', '!', '*', '#'))
            features.append([length, capitals, digits, special_chars])
        
        return np.array(features)

def build_pipeline() -> Pipeline:
    # 1. Text Feature Engineering
    # Word-level TF-IDF: captures semantic keywords. Keeping lowercase=False to preserve emphasis.
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=2500, 
        lowercase=False, 
        stop_words='english'
    )
    
    # Character-level TF-IDF: handles "leetspeak", obfuscation, and SMS slang (e.g., "v1agra", "freeee")
    char_tfidf = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(3, 5), 
        max_features=2000
    )
    
    # Metadata features: length, digits, and specific symbols as recommended
    metadata = TextMetadataExtractor()

    # Combine all textual features into a single feature space
    feature_union = FeatureUnion([
        ('word_tfidf', word_tfidf),
        ('char_tfidf', char_tfidf),
        ('metadata', metadata)
    ])

    # 2. Pipeline Architecture
    # Apply the feature union specifically to the 'text' column of the input DataFrame
    preprocessor = ColumnTransformer([
        ('text_processing', feature_union, 'text')
    ])

    # 3. Model selection
    # XGBoost is selected for its ability to handle sparse feature sets and non-linear relationships.
    # Parameters are tuned to provide high precision and robust generalization.
    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])