import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier

def engineer_features(X):
    """
    Custom feature engineering for Titanic survival prediction.
    Extracts Title from Name, Deck from Cabin, and calculates FamilySize.
    """
    X_out = X.copy()
    
    # 1. Title Extraction: Proxy for social status and age
    X_out['Title'] = X_out['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    X_out['Title'] = X_out['Title'].replace(rare_titles, 'Rare')
    X_out['Title'] = X_out['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
    X_out['Title'] = X_out['Title'].fillna('Mr')
    
    # 2. Deck Mapping: Capture socio-economic stratification from Cabin
    X_out['Deck'] = X_out['Cabin'].str[0].fillna('U')
    
    # 3. Family Dynamics: Combine SibSp and Parch
    X_out['FamilySize'] = X_out['SibSp'] + X_out['Parch'] + 1
    X_out['IsAlone'] = (X_out['FamilySize'] == 1).astype(int)
    
    # 4. Socio-economic stratification: Log transform Fare to handle skew
    X_out['Fare'] = X_out['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    
    return X_out

def build_pipeline() -> Pipeline:
    """
    Builds a robust ML pipeline using CatBoost and advanced feature engineering.
    """
    # Define feature groups for the transformer
    # Note: These columns must exist AFTER the engineer_features step
    num_cols = ['Age', 'Fare', 'Pclass', 'FamilySize', 'IsAlone']
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
    text_col = 'Name'  # Text column for Tfidf
    
    # Transformer for numerical features: Median imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Transformer for categorical features: Mode imputation and OneHot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Processor to combine feature transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
            ('text', TfidfVectorizer(max_features=50), text_col)
        ]
    )
    
    # The complete pipeline: Engineering -> Preprocessing -> CatBoost
    return Pipeline(steps=[
        ('engineer', FunctionTransformer(engineer_features)),
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5,
            bootstrap_type='Bayesian',
            verbose=0,
            random_seed=42
        ))
    ])