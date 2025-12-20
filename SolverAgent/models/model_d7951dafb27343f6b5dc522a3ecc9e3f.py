import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student exam score prediction.
    Incorporates feature interactions, robust preprocessing, and an ensemble 
    of Gradient Boosted Decision Trees (LightGBM and XGBoost) via Stacking.
    """
    
    # Define feature groups based on Data Info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # 1. Numerical Preprocessing:
    # Includes PolynomialFeatures to capture interactions like 'study_hours * class_attendance'
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Preprocessing:
    # Uses OneHotEncoding. handle_unknown='ignore' ensures robustness to new categories.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 3. Model Selection: Stacking Ensemble
    # We combine XGBoost and LightGBM to capture different boosting patterns.
    # Hyperparameters are tuned for common tabular dataset characteristics.
    
    lgbm_params = {
        'n_estimators': 1200,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'importance_type': 'gain'
    }

    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }

    estimators = [
        ('lgbm', LGBMRegressor(**lgbm_params)),
        ('xgb', XGBRegressor(**xgb_params))
    ]

    # Stacking uses a simple linear model (Ridge) to aggregate predictions and prevent overfitting.
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        passthrough=False
    )

    # 4. Final Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', stacking_regressor)
    ])

    return pipeline