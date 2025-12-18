import pandas as pd
import numpy as np
import unicodedata
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from xgboost import XGBClassifier

def _unicode_normalize(x):
    """
    Normalizes text to combat character substitution obfuscation (e.g., 'v1agra').
    """
    return np.array([unicodedata.normalize('NFKD', str(s)) for s in x.ravel()])

def _extract_meta_features(x):
    """
    Extracts domain-specific features: length, digit density, and capitalization ratio.
    """
    s = pd.Series(x.ravel().astype(str))
    length = s.str.len()
    # Handle division by zero with +1
    digit_ratio = s.str.count(r'\d') / (length + 1)
    caps_ratio = s.str.count(r'[A-Z]') / (length + 1)
    special_ratio = s.str.count(r'[^\w\s]') / (length + 1)
    return np.column_stack([length, digit_ratio, caps_ratio, special_ratio])

def build_pipeline() -> Pipeline:
    """
    Builds an advanced SMS Spam classification pipeline using a Stacking Ensemble,
    combining Word/Char Tfidf and domain-specific meta-features.
    """
    
    # Text Processing Branch 1: Word-level N-grams with Unicode Normalization
    word_pipe = Pipeline([
        ('norm', FunctionTransformer(_unicode_normalize)),
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english',
            sublinear_tf=True
        ))
    ])

    # Text Processing Branch 2: Character-level N-grams to handle obfuscation
    char_pipe = Pipeline([
        ('norm', FunctionTransformer(_unicode_normalize)),
        ('tfidf_char', TfidfVectorizer(
            ngram_range=(2, 5),
            max_features=5000,
            analyzer='char_wb',
            sublinear_tf=True
        ))
    ])

    # Combine text features and engineered numeric features
    preprocessor = ColumnTransformer([
        ('word_vectorizer', word_pipe, 'text'),
        ('char_vectorizer', char_pipe, 'text'),
        ('meta_features', FunctionTransformer(_extract_meta_features), 'text')
    ])

    # Base Estimators for Stacking:
    # 1. ComplementNB: Optimized for imbalanced text classification.
    # 2. LogisticRegression: Robust baseline with cost-sensitive weighting.
    base_estimators = [
        ('cnb', ComplementNB()),
        ('lr', LogisticRegression(
            class_weight='balanced', 
            max_iter=2000, 
            solver='liblinear', 
            penalty='l2'
        ))
    ]

    # Meta-Learner: XGBoost captures non-linear interactions between predictions and meta-features
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=42
        ),
        passthrough=True  # Allows the meta-learner to see original features + base predictions
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stack)
    ])