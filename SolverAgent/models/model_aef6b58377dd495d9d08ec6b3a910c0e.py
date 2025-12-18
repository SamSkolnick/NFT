from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_pipeline() -> Pipeline:
    """
    Builds a fast and efficient pipeline for spam/ham classification.
    Uses TfidfVectorizer for text processing and MultinomialNB for classification.
    """
    # Define the text transformation step
    # We use 'text' as the feature column based on the dataset description
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2)), 'text')
    ])

    # Combine preprocessing and the classifier into a pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    return pipeline