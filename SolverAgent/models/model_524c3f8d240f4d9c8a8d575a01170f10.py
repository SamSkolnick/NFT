from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, FunctionTransformer
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

def build_pipeline() -> Pipeline:
    """
    Builds a regression pipeline using XGBoost to predict exam scores.
    Incorporates best practices: 
    - PowerTransformer (Yeo-Johnson) for 'study_hours' to model non-linear "dosage".
    - StandardScaler for remaining numeric features.
    - OneHotEncoding for categorical features.
    - XGBoost to capture complex interactions like 'Cognitive Load' (Study x Sleep).
    """
    
    # 1. Feature Selection from Data Info
    numeric_features = ['age', 'class_attendance', 'sleep_hours']
    # 'study_hours' is handled separately for Box-Cox/Yeo-Johnson normalization
    skewed_features = ['study_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # 2. Preprocessing Components
    # PowerTransformer(method='yeo-johnson') is used to normalize study durations 
    # and handle the "dosage" scaling analogized from PK/PD modeling.
    skew_transformer = Pipeline(steps=[
        ('yeo_johnson', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    # Standard scaling for other numeric metrics
    numeric_transformer = StandardScaler()

    # OneHotEncoding for categorical features. 
    # sparse_output=False ensures compatibility with the XGBoost input requirements.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 3. Column Transformer Assembly
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('skew', skew_transformer, skewed_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 4. Model Definition: XGBoost Regressor
    # Gradient Boosting is chosen to capture the non-linear interactions 
    # between sleep quality and study intensity.
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror',
        importance_type='gain'
    )

    # 5. Pipeline Construction
    # FunctionTransformer is included as a safety layer to ensure data is dense 
    # and properly formatted for the regressor after transformations.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('dense_converter', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),
        ('regressor', model)
    ])

    return pipeline