from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """
    Builds a robust and fast spam-ham classifier pipeline.
    
    Using TfidfVectorizer with ngram_range(1, 2) captures both individual high-risk keywords
    and common spam phrases. LogisticRegression is chosen for its speed and high performance 
    on high-dimensional sparse data typical of text classification.
    """
    # Preprocessing: Apply TF-IDF vectorization to the 'text' column.
    # We use a broad ngram range and avoid stripping punctuation to preserve 
    # signals like 'FREE!!!' which are highly indicative of spam.
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), 
            min_df=2, 
            max_df=0.9, 
            sublinear_tf=True
        ), 'text')
    ])

    # Classifier: Logistic Regression with balanced class weights.
    # This handles any inherent class imbalance in the spam/ham labels.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced', 
            solver='liblinear', 
            random_state=42, 
            C=1.0
        ))
    ])

    return pipeline