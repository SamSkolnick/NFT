import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform feature extraction from Titanic raw data.
    Extracts Titles, Decks, Family Size, and Ticket groupings.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Title Extraction: Capture social hierarchy from Names
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')
        X['Title'] = X['Title'].fillna('Mr')
        
        # 2. Deck Extraction: Capture socioeconomic status from Cabin
        # Filling missing cabins with 'U' for Unknown
        X['Deck'] = X['Cabin'].str.slice(0, 1).fillna('U')
        
        # 3. Family Size: Modeling "Linked Fates"
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        
        # 4. Ticket Prefix: Identifying groups/travel classes
        X['TicketPrefix'] = X['Ticket'].astype(str).str.extract(r'([A-Za-z/.]+)', expand=False).fillna('NUM')
        
        return X

class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Converts sparse matrices (from TfidfVectorizer) to dense format 
    to ensure compatibility with HistGradientBoosting and other dense-only estimators.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

def build_pipeline() -> Pipeline:
    """
    Builds a complete sklearn Pipeline for the Titanic dataset featuring
    custom extraction, Tfidf for names, and an Ensemble Stacking Classifier.
    """
    
    # Feature columns based on TitanicFeatureEngineer output
    num_cols = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
    text_col = 'Name'  # Used for Tfidf feature extraction

    # Preprocessing pipelines for different data types
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100)),
        ('dense', DenseTransformer())
    ])

    # Combine all feature processors
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('text', text_transformer, text_col)
    ])

    # Stacked Ensemble: Combining GBDT and Random Forest with Logistic Meta-learner
    # for better generalization on small sample sizes (N=712).
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=250, max_depth=6, 
                                      min_samples_leaf=2, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, 
                                               learning_rate=0.05, random_state=42))
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1),
        cv=5,
        passthrough=False
    )

    # Final construction
    return Pipeline([
        ('engineer', TitanicFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('clf', stacking_clf)
    ])