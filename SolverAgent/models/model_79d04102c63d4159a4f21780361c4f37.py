from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student exam score prediction.
    Utilizes RobustScaler to handle potential outliers in study habits and 
    XGBoost with optimized hyperparameters for tabular data regression.
    """
    
    # Define feature groups based on the provided dataset info
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

    # Preprocessing logic:
    # 1. RobustScaler is used for numeric features to minimize the impact of extreme study patterns.
    # 2. OneHotEncoder handles categorical variables, with handle_unknown='ignore' to ensure 
    #    robustness during inference on unseen categories.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Model Selection: XGBoost Regressor
    # Optimized for the 16,000 record dataset size. 
    # 'hist' tree_method is used for faster histogram-based training.
    # Learning rate and depth are set to capture complex interactions (e.g., attendance vs. sleep quality).
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    # Construct the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline