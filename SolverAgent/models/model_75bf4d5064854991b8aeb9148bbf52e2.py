from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """
    Builds a fast and accurate pipeline for spam detection.
    Uses TF-IDF with character n-grams (2-5) to handle misspellings and 
    Logistic Regression with balanced class weights for robust classification.
    """
    
    # Feature Engineering: Character n-grams are used as per the research roadmap
    # to handle intentional misspellings and provide high accuracy for spam patterns.
    text_processor = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        max_features=20000,
        sublinear_tf=True
    )

    # ColumnTransformer to select the 'text' column specifically
    preprocessor = ColumnTransformer([
        ('text_features', text_processor, 'text')
    ])

    # Classifier: LogisticRegression is fast, efficient on high-dimensional 
    # text data, and supports cost-sensitive learning via class_weight.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced', 
            solver='liblinear', 
            C=1.0, 
            random_state=42
        ))
    ])

    return pipeline