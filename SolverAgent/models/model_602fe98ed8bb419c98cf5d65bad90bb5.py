from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    # Feature identification based on the provided dataset structure
    numeric_features = ['0', '3', '22', '1.1', '0.1', '7.25']
    categorical_features = ['male', 'Unnamed: 10', 'S']
    # Text features to be processed via TF-IDF
    name_feature = 'Braund, Mr. Owen Harris'
    ticket_feature = 'A/5 21171'

    # Transformer for numerical columns: Impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Transformer for categorical columns: Impute missing as a new category and encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combining all transformations
    # Note: TfidfVectorizer expects a 1D input, so we pass column names as strings
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('name_tfidf', TfidfVectorizer(max_features=500), name_feature),
            ('ticket_tfidf', TfidfVectorizer(max_features=100), ticket_feature)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Tree Classifier (XGBoost)
    # Optimized for classification as per task description, despite the regression note
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Create and return the full pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])