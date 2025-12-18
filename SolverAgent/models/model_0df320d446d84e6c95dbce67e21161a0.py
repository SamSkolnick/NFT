from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_pipeline() -> Pipeline:
    """
    Builds a simple and fast pipeline for email spam classification.
    Uses TF-IDF vectorization with bigrams and Multinomial Naive Bayes 
    to balance speed and accuracy.
    """
    # Define the text preprocessing step
    # ngram_range=(1, 2) captures both individual words and word pairs
    # min_df=2 helps ignore extremely rare words/typos to improve generalization
    preprocessor = ColumnTransformer([
        ('text_tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2), 'text')
    ])

    # Multinomial Naive Bayes is a strong, fast baseline for text classification
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    return pipeline