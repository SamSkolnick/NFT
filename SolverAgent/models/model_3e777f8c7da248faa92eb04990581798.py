import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def extract_features(X):
    """
    Custom transformer for NLP-inspired title extraction and relational features.
    """
    X = X.copy()
    
    # 1. NLP-Inspired Extraction: Extract Titles from Name
    X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    X['Title'] = X['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    
    # 2. Relational Aggregation proxies: Family Size and Alone status
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    
    # 3. Cabin handling: Extract Deck (proxy for survival niche/vulnerability)
    X['Deck'] = X['Cabin'].str[0].fillna('U')
    
    # 4. Ticket Feature: Ticket length as a proxy for ticket type/group
    X['TicketLen'] = X['Ticket'].apply(lambda x: len(str(x)))
    
    return X

def to_dense(x):
    """Convert sparse TF-IDF output to dense for tree-based models."""
    return x.toarray() if hasattr(x, 'toarray') else x

def build_pipeline() -> Pipeline:
    # Define columns based on the feature engineering output
    num_cols = ['Age', 'Fare', 'FamilySize', 'TicketLen']
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck']
    text_col = 'Name'

    # Preprocessing for numeric data: Impute missing values then scale
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: Impute then One-Hot Encode
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Text transformation for 'Name' column
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100)),
        ('dense', FunctionTransformer(to_dense))
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
        ('name_nlp', text_transformer, text_col)
    ])

    # Base estimators for the Stacked Ensemble
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, 
                               use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42)),
        ('cat', CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03, verbose=0, random_state=42))
    ]

    # Meta-learner: Logistic Regression to combine predictions
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0),
        cv=5
    )

    # Final full pipeline
    return Pipeline([
        ('feature_engineering', FunctionTransformer(extract_features)),
        ('preprocessing', preprocessor),
        ('classifier', stacking_clf)
    ])