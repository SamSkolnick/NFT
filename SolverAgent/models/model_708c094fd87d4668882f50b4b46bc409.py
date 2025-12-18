import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Expert ML Pipeline for predicting student exam scores.
    Incorporates non-linear transforms (log scaling), interaction terms (Study x Sleep),
    and Gradient Boosted Trees for optimal RMSE/R2 performance.
    """
    
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

    def engineer_features(X):
        # Convert to DataFrame if it's a numpy array to use column names or indices
        df = pd.DataFrame(X, columns=numeric_features)
        
        # Log-scaling study hours to model diminishing returns
        df['log_study_hours'] = np.log1p(df['study_hours'])
        
        # Interaction Term: Study Habits x Sleep Quality (Recovery)
        # Treating as 'Training Load' x 'Recovery' analogy
        df['study_sleep_interaction'] = df['study_hours'] * df['sleep_hours']
        
        # Acute workload proxy (study intensity relative to age/attendance)
        df['study_intensity'] = df['study_hours'] / (df['class_attendance'] + 1e-5)
        
        return df.values

    # Pipeline for numeric features including non-linear transforms and interactions
    numeric_transformer = Pipeline(steps=[
        ('engineer', FunctionTransformer(engineer_features)),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Ensure all data becomes dense if required by subsequent steps
    # (Though XGBoost handles sparse, the prompt suggests a FunctionTransformer for density)
    to_dense = FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Gradient Boosted Trees (XGBoost) for capturing complex interactions
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror'
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('dense_conv', to_dense),
        ('regressor', model)
    ])