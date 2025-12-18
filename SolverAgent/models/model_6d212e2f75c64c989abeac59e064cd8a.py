import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract metadata features from SMS text.
    Captures signal from punctuation density, capitalization, and URL presence
    which are high-signal features for spam detection.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert input to pandas Series for string manipulation
        s = pd.Series(np.array(X).ravel()).astype(str)
        
        # Calculate numeric features
        length = s.str.len().values.reshape(-1, 1)
        punc_count = s.str.count(r'[^\w\s]').values.reshape(-1, 1)
        digit_count = s.str.count(r'\d').values.reshape(-1, 1)
        upper_count = s.str.count(r'[A-Z]').values.reshape(-1, 1)
        # Spam often contains links; detect common URL patterns
        url_count = s.str.count(r'http|www|https|\.com|\.ly|\.co').values.reshape(-1, 1)
        
        return np.hstack([length, punc_count, digit_count, upper_count, url_count])

def build_pipeline() -> Pipeline:
    """
    Builds a robust SMS Spam classification pipeline.
    Combines word and character n-grams to handle 'leetspeak' and brevity.
    Uses Linear SVM with cost-sensitive learning (class_weight='balanced')
    to maximize F1-score on imbalanced SMS datasets.
    """
    
    # Feature Extraction: Word + Character level TF-IDF
    # Char n-grams (3-5) help identify malicious 'motifs' and obfuscated words.
    text_extraction = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2), 
            min_df=2, 
            stop_words=None, # Stop words are high-signal in SMS context
            token_pattern=r'\b\w\w+\b|(?<!\w)[\!\?]' # Capture important punctuation
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char', 
            ngram_range=(3, 5), 
            min_df=2
        ))
    ])

    # Metadata pipeline: Length, Punctuation, Digits, and Scaler
    meta_pipeline = Pipeline([
        ('stats', TextStatsTransformer()),
        ('scaler', StandardScaler())
    ])

    # Preprocessor using ColumnTransformer to target the 'text' column
    preprocessor = ColumnTransformer([
        ('text_features', text_extraction, 'text'),
        ('meta_features', meta_pipeline, 'text')
    ])

    # Model: LinearSVC is highly effective for high-dimensional sparse text data.
    # class_weight='balanced' addresses the inherent class imbalance in SMS spam.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(
            class_weight='balanced', 
            dual=False, 
            max_iter=3000, 
            C=0.5, 
            random_state=42
        ))
    ])

    return pipeline