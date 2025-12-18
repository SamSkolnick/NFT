from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """
    Builds a simple and fast pipeline for spam/ham classification.
    Uses TF-IDF with bigrams to capture semantic context and Logistic Regression 
    with balanced class weights to maximize validation accuracy.
    """
    
    # Feature Engineering: TF-IDF with 1-2 n-grams to capture phrases
    # min_df=2 helps reduce noise and prevent overfitting
    preprocessor = ColumnTransformer([
        ('text_vec', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=2), 'text')
    ])

    # Model: Logistic Regression is fast and generally outperforms Naive Bayes on 
    # text classification datasets of this size.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])

    return pipeline