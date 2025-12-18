from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier

def build_pipeline() -> Pipeline:
    # Defining feature groups based on the provided dataset structure
    # Numeric features: Pclass is treated as numeric/ordinal here for the GBDT
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    
    # Categorical features with potentially missing values (Cabin, Embarked)
    # Sex is categorical but binary
    categorical_features = ['Sex', 'Embarked', 'Cabin', 'Ticket']
    
    # Text feature: Name contains titles like 'Mr', 'Mrs', 'Master' which are high-value signals
    text_feature = 'Name'

    # 1. Numeric Transformer: Handle missing Age/Fare using Median and Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Transformer: Treat missing Cabin/Embarked as a separate category
    # Handle unknown categories for the test set to avoid crashes
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Text Transformer: Use Tfidf on Name
    # Extracting character n-grams or words helps capture titles (Mr, Master) and family names
    text_transformer = TfidfVectorizer(max_features=100, analyzer='word', stop_words='english')

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ]
    )

    # 4. Model: HistGradientBoostingClassifier
    # As per research summary, GBDTs are superior for this low-N tabular dataset.
    # HistGradientBoosting is sklearn's high-performance implementation similar to LightGBM.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_depth=4,
            l2_regularization=1.0,
            random_state=42
        ))
    ])

    return pipeline