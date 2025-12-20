from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, QuantileTransformer
from lightgbm import LGBMRegressor
import numpy as np

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student performance prediction.
    
    Strategy:
    1. Numerical Features: Uses QuantileTransformer (Normal) to handle potential non-Gaussian 
       distributions in study hours/attendance, followed by PolynomialFeatures to capture 
       critical interactions (e.g., the synergy between attendance and study hours).
    2. Categorical Features: Uses OneHotEncoder for non-ordinal features and handles unseen values.
    3. Model: Employs LightGBM, a GBDT model, with hyperparameters tuned for a 16k row dataset.
    4. Target Transformation: Uses TransformedTargetRegressor with StandardScaler to normalize 
       the exam_score, mitigating 'Target Shift' and improving gradient convergence.
    """
    
    # Feature Selection from provided Data Info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 
                            'study_method', 'facility_rating', 'exam_difficulty']

    # Numerical preprocessing: Quantile normalization and second-order interactions
    numeric_transformer = Pipeline(steps=[
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])

    # Categorical preprocessing: Robust One-Hot Encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Advanced Gradient Boosting (LightGBM)
    # n_estimators and learning_rate are balanced for a dataset of 16,000 samples.
    # num_leaves and max_depth are set to capture high-order interactions without over-fitting.
    lgbm_regressor = LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        n_estimators=2500,
        learning_rate=0.015,
        num_leaves=63,
        max_depth=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        importance_type='gain',
        verbosity=-1
    )

    # Wrapping the model to apply Z-score normalization to the target (exam_score)
    # This aligns with the Research Summary regarding Z-score normalization of Score.
    model = TransformedTargetRegressor(
        regressor=lgbm_regressor,
        transformer=StandardScaler()
    )

    # Final Pipeline construction
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])