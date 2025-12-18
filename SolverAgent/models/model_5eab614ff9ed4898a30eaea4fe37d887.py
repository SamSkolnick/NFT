import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ticket_counts_ = {}

    def fit(self, X, y=None):
        # Calculate Ticket counts to estimate GroupSize (shared tickets indicate cliques)
        # This is performed during fit to prevent leakage from the test set
        self.ticket_counts_ = X['Ticket'].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Social Titles (Status extraction)
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
        X['Title'] = X['Title'].fillna('Unknown')

        # 2. Deck (Socio-economics from Cabin)
        X['Deck'] = X['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

        # 3. GroupSize (Aggregating shared Ticket numbers)
        X['GroupSize'] = X['Ticket'].map(self.ticket_counts_).fillna(1)

        # 4. FamilySize (Standard engineering)
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        
        # Keep engineered and raw features for the ColumnTransformer
        return X[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Deck', 'GroupSize', 'FamilySize', 'Name']]

def build_pipeline() -> Pipeline:
    # Feature columns defined after extraction in TitanicFeatureEngineer
    numeric_features = ['Pclass', 'Age', 'Fare', 'GroupSize', 'FamilySize']
    categorical_features = ['Sex', 'Embarked', 'Title', 'Deck']
    text_feature = 'Name'

    # Numeric pipeline: Impute missing Age/Fare and scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing Embarked/Title/Deck and encode
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # NLP marker pipeline: Use character n-grams on names for lineage markers
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=100)),
        ('dense', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt', text_transformer, text_feature)
    ])

    # Stacked Ensemble of GBDTs with a Logistic Regression meta-learner
    # Using 5-fold cross-validation internally to minimize overfitting
    estimators = [
        ('xgb', XGBClassifier(
            n_estimators=200, 
            max_depth=4, 
            learning_rate=0.01, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200, 
            max_depth=3, 
            learning_rate=0.01,
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=5, 
            random_state=42
        ))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=10  # 10-fold Stratified CV as requested for small datasets
    )

    return Pipeline([
        ('engineer', TitanicFeatureEngineer()),
        ('pre', preprocessor),
        ('clf', stack)
    ])