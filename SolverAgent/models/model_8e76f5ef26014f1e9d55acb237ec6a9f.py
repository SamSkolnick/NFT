from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance machine learning pipeline for the Titanic dataset.
    Features:
    - Numeric: Imputation and Scaling.
    - Categorical: Imputation and One-Hot Encoding.
    - Feature Engineering: FamilySize calculation and Deck extraction (from Cabin).
    - Text: TF-IDF for Titles (from Name) and Ticket patterns.
    - Model: LightGBM for superior categorical and tabular performance.
    """

    # Helper function to extract the Deck (first letter) from Cabin
    def extract_deck(x):
        # x comes as a 1D array from ColumnTransformer when selecting a single column by name
        return pd.Series(x).str[0].fillna('U').values.reshape(-1, 1)

    # Helper function to calculate FamilySize (SibSp + Parch + 1)
    def calc_family_size(x):
        # x comes as a 2D array [SibSp, Parch]
        return (x[:, 0] + x[:, 1] + 1).reshape(-1, 1)

    # Define feature groups based on input specification
    # Numeric Features: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    # We split them based on processing needs
    num_cols = ['Age', 'Fare', 'Pclass']
    cat_cols = ['Sex', 'Embarked']
    family_cols = ['SibSp', 'Parch']
    
    # Preprocessing stages
    # 1. Numerical: Median imputation for Age/Fare, followed by scaling
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical: Most frequent imputation for Embarked, then One-Hot Encoding
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Family Engineering: Summing SibSp and Parch
    family_transformer = Pipeline([
        ('size', FunctionTransformer(calc_family_size, check_inverse=False)),
        ('scaler', StandardScaler())
    ])

    # 4. Deck Engineering: Extracting letter from Cabin
    deck_transformer = Pipeline([
        ('extract', FunctionTransformer(extract_deck, check_inverse=False)),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 5. Text Preprocessing: 
    # - Name: Specifically capture titles like "Mr.", "Master.", "Miss." using token patterns
    # - Ticket: Capture patterns using character n-grams
    name_tfidf = TfidfVectorizer(token_pattern=r'\b[A-Za-z]+\.', min_df=1)
    ticket_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), min_df=2)

    # Combine all preprocessing into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
            ('family', family_transformer, family_cols),
            ('deck', deck_transformer, 'Cabin'),
            ('name_tfidf', name_tfidf, 'Name'),
            ('ticket_tfidf', ticket_tfidf, 'Ticket')
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Tree (LightGBM)
    # Hyperparameters tuned for small tabular datasets to minimize overfitting
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=10,
        random_state=42,
        importance_type='gain',
        verbose=-1
    )

    # Final Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])