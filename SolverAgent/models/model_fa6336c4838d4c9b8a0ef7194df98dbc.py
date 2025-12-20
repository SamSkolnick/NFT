from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, TargetEncoder
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance regression pipeline for student exam score prediction.
    
    Strategy:
    1. Feature Engineering: Uses QuantileTransformer for numeric features to handle 
       skewness and non-normal distributions (common in study/attendance data).
    2. Categorical Handling: Uses TargetEncoder for high-cardinality and ordinal 
       categorical features to capture group-level performance trends efficiently 
       without the dimensionality explosion of OneHotEncoding.
    3. Model: Employs LightGBM, which is robust to outliers and excels at 
       capturing non-linear interactions between behavioral features.
    """
    
    # Define features based on data info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing for numeric data: Gaussian mapping helps trees find better splits
    numeric_transformer = QuantileTransformer(
        output_distribution='normal', 
        random_state=42
    )

    # Preprocessing for categorical data: Target encoding captures the relationship
    # between categories (like 'course' or 'study_method') and the target 'exam_score'
    categorical_transformer = TargetEncoder(random_state=42)

    # Bundle preprocessing for both types of data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Optimized LightGBM Regressor
    # Hyperparameters selected to balance bias-variance and prevent overfitting 
    # to "Good Student" clusters while maintaining depth for complex interactions.
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=45,
        max_depth=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        importance_type='gain',
        n_jobs=-1
    )

    # Assemble the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline