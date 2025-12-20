from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    # Identify feature groups based on Data Info
    # Numeric: ['0' (PassengerId/Index), '3' (Pclass), '22' (Age), '1.1' (SibSp), '0.1' (Parch), '7.25' (Fare)]
    numeric_features = ['0', '3', '22', '1.1', '0.1', '7.25']
    
    # Categorical: Low-cardinality nominal features
    categorical_features = ['male', 'S', 'Unnamed: 10']
    
    # Text: High-cardinality/string features (Name)
    text_feature = 'Braund, Mr. Owen Harris'

    # 1. Pipeline for numeric data: Imputation + Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline for categorical data: Imputation + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Text transformer: TF-IDF to extract title/family name signals
    # ColumnTransformer passes a 1D Series to TfidfVectorizer if a single string is used for selection.
    text_transformer = TfidfVectorizer(max_features=100)

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='drop'
    )

    # Define high-performance Gradient Boosted Decision Tree model
    # Optimized for structured data with potential class imbalance handling
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    # Return the unified pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])