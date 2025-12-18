from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def build_pipeline() -> Pipeline:
    """
    Expert-level Titanic survival prediction pipeline using a Stacked Ensemble 
    of GBDTs and Random Forest with NLP-inspired feature engineering.
    """
    
    # 1. Feature Engineering Helper Functions
    def extract_title(df):
        # Extracts title from Name column (e.g., Master, Mr, Royal)
        s = df.iloc[:, 0].str.extract(' ([A-Za-z]+)\.', expand=False)
        mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
        rare = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        s = s.replace(mapping).replace(rare, 'Rare')
        return s.fillna('Mr').values.reshape(-1, 1)

    def extract_deck(df):
        # Extracts Deck from Cabin column (e.g., C85 -> C)
        return df.iloc[:, 0].str[0].fillna('U').values.reshape(-1, 1)

    def compute_family_size(df):
        # Combines SibSp and Parch for family social graphing
        return (df.iloc[:, 0] + df.iloc[:, 1] + 1).values.reshape(-1, 1)

    # 2. Individual Feature Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    title_pipeline = Pipeline([
        ('title_ext', FunctionTransformer(extract_title)),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    deck_pipeline = Pipeline([
        ('deck_ext', FunctionTransformer(extract_deck)),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    family_pipeline = Pipeline([
        ('fsize_ext', FunctionTransformer(compute_family_size)),
        ('scaler', StandardScaler())
    ])

    # 3. ColumnTransformer for specialized feature extraction
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, ['Age', 'Fare', 'Pclass']),
        ('cat', cat_pipeline, ['Sex', 'Embarked']),
        ('family', family_pipeline, ['SibSp', 'Parch']),
        ('title', title_pipeline, ['Name']),
        ('deck', deck_pipeline, ['Cabin']),
        ('ticket', TfidfVectorizer(analyzer='char', ngram_range=(2, 3)), 'Ticket')
    ])

    # 4. Meta-Learner and Base Models for Stacking
    # Base models: XGBoost (GBDT) and RandomForest
    base_models = [
        ('xgb', XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=2,
            random_state=42
        ))
    ]

    # Meta-learner: Regularized Logistic Regression to reduce variance
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=0.1, penalty='l2', solver='lbfgs'),
        cv=10,
        n_jobs=-1
    )

    # 5. Full Pipeline with Sparse-to-Dense conversion for robustness
    return Pipeline([
        ('preprocessor', preprocessor),
        ('densify', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),
        ('classifier', stack)
    ])
呈现完毕