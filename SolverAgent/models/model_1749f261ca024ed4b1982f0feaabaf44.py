from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_pipeline() -> Pipeline:
    """
    Builds a fast and efficient pipeline for spam/ham classification using 
    TF-IDF vectorization and Multinomial Naive Bayes.
    """
    # Preprocessing: Apply TfidfVectorizer to the 'text' column
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True), 'text')
    ])

    # Combine preprocessing and the classifier into a pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB())
    ])

    return pipeline