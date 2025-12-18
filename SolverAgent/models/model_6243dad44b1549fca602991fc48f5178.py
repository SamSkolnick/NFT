import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """
    Builds a robust SMS Spam classification pipeline.
    Uses character-level TF-IDF (n-grams 2-5) to counter obfuscation (e.g., 'F.r.e.e') 
    and non-alphanumeric evasion tactics.
    Employs Logistic Regression with class weighting to handle imbalance and maximize F1-score.
    """
    
    # Character-level n-grams are used to capture 'malicious motifs' regardless of whitespace.
    # sublinear_tf is applied to scale term frequencies logarithmically, 
    # which is often beneficial in short-text classification.
    text_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True
    )

    # Preprocessor targets the specific 'text' column as defined in the data info.
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', text_vectorizer, 'text')
        ],
        remainder='drop'
    )

    # Logistic Regression with 'balanced' class weights implements Cost-Sensitive Learning.
    # This prioritizes the minority (spam) class to optimize the Precision-Recall balance (F1-score).
    # liblinear is efficient for high-dimensional sparse data produced by char n-grams.
    classifier = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        solver='liblinear',
        max_iter=1000
    )

    # Combine into a single Pipeline object.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline