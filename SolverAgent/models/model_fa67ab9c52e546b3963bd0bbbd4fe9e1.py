from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced machine learning pipeline for student exam score prediction.
    Utilizes Target Encoding for categorical features as per best practices for 
    high-cardinality/tabular data and LightGBM for non-linear interaction capture.
    """
    
    # Define feature groups based on provided metadata
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 
        'course', 
        'internet_access', 
        'sleep_quality', 
        'study_method', 
        'facility_rating', 
        'exam_difficulty'
    ]
    
    # Preprocessing: 
    # 1. Standardize numerical features to assist gradient-based optimization.
    # 2. Use TargetEncoder for categorical features to map categories to their 
    #    impact on the target variable (exam_score), handling non-linear relationships.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', TargetEncoder(target_type='continuous', random_state=42), categorical_features)
        ],
        remainder='drop'
    )
    
    # Model: LightGBM Regressor
    # Optimized for tabular data with a focus on generalization via subsampling and tree complexity.
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )
    
    # Construct final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    return pipeline