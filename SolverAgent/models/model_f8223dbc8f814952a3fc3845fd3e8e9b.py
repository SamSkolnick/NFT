import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

def extract_meta_features(text_col):
    """
    Extracts structural metadata from SMS text which are strong indicators of spam:
    1. Message length: Spam tends to be longer or very short/concise.
    2. Capitalization ratio: Urgency often uses ALL CAPS.
    3. Digit density: Presence of phone numbers, prize amounts, or verification codes.
    """
    # Ensure input is treated as a Series for string operations
    s = pd.Series(np.array(text_col).ravel()).astype(str)
    
    # Calculate features
    length = s.str.len().values.reshape(-1, 1)
    cap_ratio = s.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)).values.reshape(-1, 1)
    digit_count = s.apply(lambda x: sum(1 for c in x if c.isdigit())).values.reshape(-1, 1)
    
    return np.hstack([length, cap_ratio, digit_count])

def build_pipeline() -> Pipeline:
    """
    Builds a robust NLP pipeline for SMS spam classification.
    Uses a hybrid approach:
    - Word-level TF-IDF (1-2 n-grams) for semantic context.
    - Character-level TF-IDF (2-5 n-grams) to handle obfuscation/typos (e.g., 'v1agra').
    - Metadata features to capture structural 'urgency' signals.
    - Logistic Regression with balanced class weights to maximize F1-score on imbalanced data.
    """
    
    # Textual Feature extraction
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        sublinear_tf=True
    )
    
    tfidf_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=5000,
        sublinear_tf=True
    )
    
    # Metadata extraction
    metadata_transformer = FunctionTransformer(extract_meta_features)

    # Combine all feature extractors
    preprocessor = ColumnTransformer([
        ('word_tfidf', tfidf_word, 'text'),
        ('char_tfidf', tfidf_char, 'text'),
        ('metadata', metadata_transformer, 'text')
    ])

    # Logistic Regression is highly effective for high-dimensional text data.
    # class_weight='balanced' is critical for maximizing F1-score by adjusting for the 
    # low frequency of spam messages (usually ~13-15%).
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(
            class_weight='balanced', 
            solver='liblinear', 
            penalty='l2', 
            C=1.0, 
            random_state=42
        ))
    ])
    
    return pipeline