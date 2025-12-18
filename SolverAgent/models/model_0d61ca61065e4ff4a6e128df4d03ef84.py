from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def build_pipeline() -> Pipeline:
    # Based on the data info, 'text' is the feature column and 'label' is the target.
    # LinearSVC is fast and typically provides high accuracy for text classification tasks.
    
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2)), 'text')
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=42))
    ])
    
    return pipeline