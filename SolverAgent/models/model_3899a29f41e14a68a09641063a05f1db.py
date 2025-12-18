import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance regression pipeline for student exam score prediction.
    Utilizes Quantile Transformation for numeric features as per research recommendations
    and LightGBM for robust gradient boosting.
    """
    
    # Define feature groups based on provided metadata
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing for numeric data: 
    # QuantileTransformer normalizes distributions (effective for socioeconomic/attendance proxies)
    numeric_transformer = Pipeline(steps=[
        ('quantile', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data:
    # OneHotEncoder handles categories like course and study_method
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Trees (LightGBM)
    # Optimized for tabular data with non-linear relationships
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=45,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )

    # Construct the final pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])