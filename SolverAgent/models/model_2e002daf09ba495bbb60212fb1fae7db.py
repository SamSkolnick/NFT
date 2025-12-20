from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    # Numeric Features: PassengerId ('0'), Pclass ('3'), Age ('22'), SibSp ('1.1'), Parch ('0.1'), Fare ('7.25')
    num_features = ['0', '3', '22', '1.1', '0.1', '7.25']
    
    # Categorical Features: Sex ('male'), Embarked ('S'), Ticket ('A/5 21171'), Cabin ('Unnamed: 10')
    cat_features = ['male', 'S', 'A/5 21171', 'Unnamed: 10']
    
    # Text Feature: Name ('Braund, Mr. Owen Harris')
    text_feature = 'Braund, Mr. Owen Harris'

    # Preprocessing for numerical data: Impute missing values with median and scale
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: Impute missing with a constant and One-Hot Encode
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessing for text data: TF-IDF vectorization to extract title/name patterns
    # Note: ColumnTransformer passes a 1D Series when the column name is a string
    text_transformer = TfidfVectorizer(max_features=100)

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features),
            ('text', text_transformer, text_feature)
        ]
    )

    # Gradient Boosted Decision Trees for high-performance tabular classification
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Build the final pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', model)
    ])