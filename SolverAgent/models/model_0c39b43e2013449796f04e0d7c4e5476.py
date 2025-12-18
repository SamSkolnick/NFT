from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_pipeline() -> Pipeline:
    """
    Builds a simple and fast pipeline for spam classification using TF-IDF and Naive Bayes.
    This combination typically yields high accuracy on text-based spam datasets.
    """
    
    # Text processing using TF-IDF Vectorizer on the 'text' column
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', TfidfVectorizer(stop_words='english', lowercase=True), 'text')
        ],
        remainder='drop'
    )

    # Pipeline combining preprocessing and the Multinomial Naive Bayes classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB())
    ])

    return pipeline