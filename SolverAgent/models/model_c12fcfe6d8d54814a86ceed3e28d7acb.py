import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    # Feature selection based on data info
    # Numeric features for standard scaling
    numeric_features = ['age', 'class_attendance', 'sleep_hours']
    # Study hours specifically targeted for log-transformation to handle diminishing returns
    log_features = ['study_hours']
    # Categorical features for one-hot encoding
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('log', Pipeline([
                ('log_transform', FunctionTransformer(np.log1p)),
                ('log_scale', StandardScaler())
            ]), log_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # XGBRegressor is used to capture non-linearities and burnout inflection points
    # Parameter selection optimized for RMSE and R2 on tabular datasets of this scale
    regressor = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror'
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])