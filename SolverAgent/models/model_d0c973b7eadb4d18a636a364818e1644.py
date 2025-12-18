import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def build_pipeline() -> Pipeline:
    """
    Expert-level SMS Spam classification pipeline.
    
    Strategy:
    1. Feature Engineering: Uses a combination of character-level and word-level TF-IDF.
       - Character n-grams (3-5) are used to capture 'l33t-speak', typos, and motifs 
         common in obfuscated spam messages (as per 'k-mer frequency' insights).
       - Word n-grams (1-2) capture the semantic context of common spam phrases.
    2. Imbalance Handling: LinearSVC with 'class_weight=balanced' is used to maximize 
       F1-score by penalizing misclassifications of the minority (spam) class.
    3. Scalability: LinearSVC is chosen for its high-throughput and efficiency 
       on sparse high-dimensional text data.
    """
    
    # Character-level vectorizer for robustness against intentional obfuscation
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        sublinear_tf=True,
        strip_accents='unicode'
    )
    
    # Word-level vectorizer for semantic intent
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english'
    )
    
    # Combine feature extractors targeting the 'text' column
    preprocessor = ColumnTransformer([
        ('tfidf_char', char_vectorizer, 'text'),
        ('tfidf_word', word_vectorizer, 'text')
    ], remainder='drop')
    
    # Final Pipeline with Linear Support Vector Classifier
    # dual=False is preferred when n_samples < n_features (typical with 5-char ngrams)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LinearSVC(
            class_weight='balanced',
            random_state=42,
            dual=False,
            max_iter=3000,
            C=0.8 # Slight regularization to improve generalization
        ))
    ])
    
    return pipeline