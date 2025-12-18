from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """
    Builds a pipeline for SMS Spam classification.
    
    Uses TfidfVectorizer to capture word and character n-gram patterns.
    Employs Logistic Regression with cost-sensitive learning (class_weight='balanced')
    to maximize F1-score given the inherent class imbalance in SMS datasets.
    """
    
    # Preprocessor: Apply TF-IDF to the 'text' column.
    # We use a mix of unigrams and bigrams. sublinear_tf=True helps scale 
    # the impact of high-frequency words often found in SMS.
    preprocessor = ColumnTransformer([
        ('text_tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents='unicode'
        ), 'text')
    ])

    # Model: Logistic Regression is a robust baseline that performs well 
    # on sparse TF-IDF vectors. 'balanced' class weights are critical 
    # for optimizing F1-score in imbalanced spam detection tasks.
    classifier = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        solver='liblinear',
        max_iter=1000
    )

    # Construct the final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline