import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def extract_titles_func(X):
    # X is a DataFrame slice from ColumnTransformer
    s = pd.Series(X.iloc[:, 0]).astype(str)
    titles = s.str.extract(' ([A-Za-z]+)\.', expand=False)
    mapping = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare', 
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    return titles.replace(mapping).fillna('Mr').values.reshape(-1, 1)

def extract_cabin_deck(X):
    # Extract first letter of Cabin to represent Deck, 'U' for Unknown
    s = pd.Series(X.iloc[:, 0]).astype(str)
    return s.apply(lambda x: x[0] if x != 'nan' else 'U').values.reshape(-1, 1)

def build_pipeline() -> Pipeline:
    # Feature Groupings
    num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_features = ['Sex', 'Embarked']
    
    # 1. Numeric Transformer: Multivariate imputation (MICE) and Scaling
    num_transformer = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler())
    ])

    # 2. Basic Categorical Transformer
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Title Extraction (Social Hierarchy)
    title_transformer = Pipeline([
        ('extract', FunctionTransformer(extract_titles_func)),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Cabin Deck Extraction
    cabin_transformer = Pipeline([
        ('extract', FunctionTransformer(extract_cabin_deck)),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 5. NLP/Genomics Analogy: Character N-grams on Name and Ticket
    # Using character-level n-grams to capture socio-economic markers or ethnic lineages
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=100)),
        ('dense', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x))
    ])

    # Preprocessing Engine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features),
            ('title', title_transformer, ['Name']),
            ('cabin', cabin_transformer, ['Cabin']),
            ('name_nlp', text_transformer, 'Name'),
            ('ticket_nlp', text_transformer, 'Ticket')
        ],
        remainder='drop'
    )

    # Model Stacking: XGBoost and Random Forest blended with Logistic Regression
    base_learners = [
        ('xgb', XGBClassifier(
            n_estimators=250, 
            learning_rate=0.02, 
            max_depth=4, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=250, 
            max_depth=6, 
            min_samples_leaf=3, 
            random_state=42
        ))
    ]

    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(C=0.5),
        cv=5
    )

    # Final Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stacking_clf)
    ])