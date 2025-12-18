from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_pipeline() -> Pipeline:
    # ColumnTransformer to apply TfidfVectorizer to the 'text' column
    preprocessor = ColumnTransformer([
        ('text_vec', TfidfVectorizer(ngram_range=(1, 2)), 'text')
    ])
    
    # Simple and fast pipeline using Multinomial Naive Bayes
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB())
    ])
    
    return pipeline