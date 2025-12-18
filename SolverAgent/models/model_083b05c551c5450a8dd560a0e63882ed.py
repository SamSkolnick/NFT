from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_pipeline() -> Pipeline:
    # Assuming Data Info says text column is 'text'
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(), 'text')
    ])
    return Pipeline([('pre', preprocessor), ('clf', MultinomialNB())])