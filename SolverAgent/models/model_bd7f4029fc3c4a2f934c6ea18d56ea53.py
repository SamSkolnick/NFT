import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, PolynomialFeatures
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student exam score prediction.
    
    Strategies implemented:
    1. Quantile Transformation: Normalizes feature distributions as recommended by research for non-normal tabular data.
    2. Polynomial Interactions: Captures non-linear relationships between numeric drivers (e.g., study_hours * attendance).
    3. Hybrid Ensemble: Combines LightGBM (leaf-wise growth) and XGBoost (level-wise growth) to capture diverse structural patterns.
    4. GBDT Optimization: Tuned learning rates and regularization for a dataset of this scale (~16k rows).
    """
    
    # Feature selection based on Data Info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 
                           'study_method', 'facility_rating', 'exam_difficulty']
    
    # 1. Numerical Preprocessing
    # QuantileTransformer mapped to normal distribution handles outliers and skewed features.
    # PolynomialFeatures (interaction_only) creates proxy variables like study-efficiency (hours * attendance).
    num_transformer = Pipeline(steps=[
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical Preprocessing
    # OneHotEncoder handles nominal and ordinal categories. 
    # GBDTs effectively partition the high-dimensional space created by OHE.
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine transformations into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # 3. Model Definition: Gradient Boosted Decision Tree (GBDT) Ensemble
    # LightGBM: Optimized for categorical features and large feature spaces.
    lgbm_reg = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=45,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # XGBoost: Provides robust level-wise splitting to balance the LightGBM predictions.
    xgb_reg = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Ensemble Strategy
    # Using a VotingRegressor (weighted average) to reduce variance and improve R2.
    # LightGBM is given slightly higher weight as it typically handles tabular benchmarks better.
    ensemble = VotingRegressor(
        estimators=[
            ('lgbm', lgbm_reg),
            ('xgb', xgb_reg)
        ],
        weights=[0.55, 0.45]
    )
    
    # Construct final pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ensemble)
    ])