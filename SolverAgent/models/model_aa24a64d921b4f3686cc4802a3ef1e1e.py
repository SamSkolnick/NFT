from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import re

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to implement high-signal feature engineering:
    - Title extraction from Name
    - FamilySize and IsAlone indicators
    - Cabin Deck extraction
    - Ticket group frequency
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Parse Title
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Miss",
            "Mme": "Mrs", "Don": "Rare", "Lady": "Rare", "Countess": "Rare",
            "Jonkheer": "Rare", "Sir": "Rare", "Capt": "Rare", "Ms": "Miss", "Dona": "Rare"
        }
        X['Title'] = X['Title'].map(title_mapping).fillna("Rare")
        
        # Family Features
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        
        # Cabin Handling - Extract Deck
        X['Deck'] = X['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
        
        # Ticket Grouping (Frequency count)
        ticket_counts = X['Ticket'].value_counts()
        X['TicketGroup'] = X['Ticket'].map(ticket_counts)
        
        return X

def build_pipeline() -> Pipeline:
    """
    Builds a robust ML pipeline for the Titanic dataset using an ensemble of 
    GBDTs, Random Forest, and Logistic Regression with advanced feature engineering.
    """
    
    # Define feature groups after transformation
    num_features = ['Age', 'Fare', 'FamilySize', 'Pclass', 'SibSp', 'Parch', 'TicketGroup', 'IsAlone']
    cat_features = ['Sex', 'Embarked', 'Title', 'Deck']
    text_feature = 'Name'

    # 1. Preprocessing for Numeric: Iterative Imputation for Age based on context
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler())
    ])

    # 2. Preprocessing for Categorical: One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
            ('name_tfidf', TfidfVectorizer(max_features=150, analyzer='char_wb', ngram_range=(2,3)), text_feature)
        ]
    )

    # 4. Ensemble Model: HistGradientBoosting (GBDT), RandomForest, and LogisticRegression
    # HistGradientBoosting handles NaNs internally and is similar to LightGBM/CatBoost
    ensemble = VotingClassifier(
        estimators=[
            ('gbdt', HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_depth=5, 
                l2_regularization=1.5, random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=250, max_depth=8, min_samples_leaf=4, 
                random_state=42
            )),
            ('lr', LogisticRegression(
                penalty='l2', C=0.1, solver='liblinear', random_state=42
            ))
        ],
        voting='soft'
    )

    # Final Pipeline
    pipeline = Pipeline(steps=[
        ('engineer', TitanicFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])

    return pipeline