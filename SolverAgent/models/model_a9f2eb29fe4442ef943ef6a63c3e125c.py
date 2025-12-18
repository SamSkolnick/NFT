import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Expert Feature Engineering: 
    - Social Proxying: Extracts Titles from Names to proxy age/status.
    - Relational Mapping: Identifies Cabin presence to handle systematic missingness.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Social Proxying: Extract Title
        # Names are in format: 'Surname, Title. Name'
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Consolidation of rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')
        X['Title'] = X['Title'].fillna('Mr')
        
        # 2. Cabin Presence (Binary): Presence is often more predictive than the specific cabin.
        X['HasCabin'] = X['Cabin'].apply(lambda x: 1 if pd.notna(x) else 0)
        
        return X

def build_pipeline() -> Pipeline:
    """
    Builds a high-bias/low-variance Stacked Generalizer for small tabular data.
    Implements MICE imputation, Title extraction, and GBDT/RF ensembling.
    """
    # Feature Groups
    # Numeric: Handled via MICE (IterativeImputer)
    num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    # Categorical: Handled via OHE
    cat_cols = ['Sex', 'Embarked', 'Title', 'HasCabin']
    # Text: Handled via Tfidf
    name_col = 'Name'
    ticket_col = 'Ticket'

    # Preprocessing for numeric data (MICE + Scaling)
    num_transformer = Pipeline([
        ('mice_imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combined ColumnTransformer
    # sparse_threshold=0 ensures the output is dense for the GBDT learners
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
        ('name_tfidf', TfidfVectorizer(max_features=100), name_col),
        ('ticket_tfidf', TfidfVectorizer(max_features=50), ticket_col)
    ], sparse_threshold=0)

    # Base Learners: GBDT (HistGradientBoosting) and Random Forest
    # Tuned for small dataset stability (N < 1000)
    base_learners = [
        ('gbdt', HistGradientBoostingClassifier(
            max_iter=300, 
            learning_rate=0.02, 
            max_depth=5, 
            l2_regularization=1.5, 
            early_stopping=True, 
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300, 
            max_depth=7, 
            min_samples_leaf=3, 
            random_state=42
        ))
    ]

    # Meta-Classifier: Logistic Regression for robust ensembling
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(C=0.1),
        cv=5, 
        n_jobs=-1
    )

    # Final Pipeline
    return Pipeline([
        ('feat_eng', TitanicFeatureExtractor()),
        ('pre', preprocessor),
        ('clf', stack)
    ])