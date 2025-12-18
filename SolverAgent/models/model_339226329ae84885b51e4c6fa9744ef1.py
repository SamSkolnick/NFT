from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance SMS spam classification pipeline.
    Uses character-level TF-IDF to handle 'leet-speak' and informal syntax,
    and Logistic Regression with balanced class weights to maximize F1-score 
    on imbalanced datasets.
    """
    # Character n-grams (2-5) provide robustness against adversarial spelling (e.g., 'w1n')
    # and are effective for the short-form nature of SMS.
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        sublinear_tf=True,
        strip_accents='unicode'
    )

    # ColumnTransformer targets the 'text' column from the input DataFrame
    preprocessor = ColumnTransformer([
        ('text_features', tfidf, 'text')
    ], remainder='drop')

    # Logistic Regression with class_weight='balanced' optimizes for F1-score 
    # by penalizing the model more for misclassifying the minority (Spam) class.
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model)
    ])

    return pipeline