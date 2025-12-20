from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    """
    Builds a state-of-the-art classification pipeline for the provided dataset.
    Features are handled based on their nature: 
    - Numeric: Imputed and scaled.
    - Categorical: Imputed and One-Hot Encoded.
    - Text: Vectorized using TF-IDF.
    """
    
    # Feature groups based on data description
    numeric_features = ['0', '3', '22', '1.1', '0.1', '7.25']
    categorical_features = ['male', 'S']
    text_feature = 'Braund, Mr. Owen Harris'
    # 'A/5 21171' (Ticket) and 'Unnamed: 10' (Cabin) are excluded to maintain 
    # simplicity and prevent overfitting on high-cardinality noise.

    # 1. Processing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Processing for standard categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Processing for text data (Names/Descriptions)
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Trees are the gold standard for this scale of structured data
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])