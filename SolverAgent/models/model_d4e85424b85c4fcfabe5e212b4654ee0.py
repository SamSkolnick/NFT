import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class TitleExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def extract_title(name):
            if not isinstance(name, str): return 'Rare'
            search = re.search(' ([A-Za-z]+)\.', name)
            if search:
                title = search.group(1)
                if title in ['Mlle', 'Ms']: return 'Miss'
                elif title == 'Mme': return 'Mrs'
                elif title in ['Mr', 'Mrs', 'Miss', 'Master']: return title
                else: return 'Rare'
            return 'Rare'
        return pd.DataFrame(X.iloc[:, 0].apply(extract_title))

class DeckExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        decks = X.iloc[:, 0].fillna('U').apply(lambda x: str(x)[0])
        return pd.DataFrame(decks)

class FamilySizeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # SibSp is col 0, Parch is col 1
        return pd.DataFrame(X.iloc[:, 0] + X.iloc[:, 1] + 1)

def build_pipeline() -> Pipeline:
    # 1. Feature Engineering: Title from Name
    name_transformer = Pipeline([
        ('title_ext', TitleExtractor()),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 2. Feature Engineering: Deck from Cabin
    cabin_transformer = Pipeline([
        ('deck_ext', DeckExtractor()),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Numeric: Age and Fare Binning (Non-linear relationships)
    age_fare_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bin', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'))
    ])

    # 4. Feature Engineering: FamilySize
    family_transformer = Pipeline([
        ('size_ext', FamilySizeExtractor()),
        ('scaler', StandardScaler())
    ])

    # 5. Standard Categoricals
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer to orchestrate features
    preprocessor = ColumnTransformer([
        ('name_title', name_transformer, ['Name']),
        ('cabin_deck', cabin_transformer, ['Cabin']),
        ('age_fare_bin', age_fare_transformer, ['Age', 'Fare']),
        ('family_size', family_transformer, ['SibSp', 'Parch']),
        ('categorical', cat_transformer, ['Sex', 'Pclass', 'Embarked']),
        ('ticket_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=50), 'Ticket')
    ])

    # Base Estimators for Stacking (GBDTs, RF, and MLP)
    base_estimators = [
        ('xgb', XGBClassifier(
            n_estimators=150, 
            max_depth=3, 
            learning_rate=0.05, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=150, 
            max_depth=3, 
            learning_rate=0.05, 
            random_state=42, 
            verbosity=-1
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=6, 
            min_samples_leaf=2, 
            random_state=42
        )),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(32, 16), 
            activation='relu', 
            solver='adam', 
            max_iter=1000, 
            random_state=42
        ))
    ]

    # Meta-learner: Logistic Regression (Clinical/Risk Modeling Standard)
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0),
        cv=10,
        passthrough=False
    )

    # Final Pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stacking_clf)
    ])