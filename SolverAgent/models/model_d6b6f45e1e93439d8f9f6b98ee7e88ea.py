from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an optimized regression pipeline for predicting student exam scores.
    Uses LightGBM as the primary regressor, as recommended for tabular data, 
    combined with robust scaling and encoding for categorical features.
    """
    
    # Define feature groups based on Data Info
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
    # 1. Standardize numeric features for stable gradient descent.
    # 2. One-hot encode categorical features (handle_unknown='ignore' ensures robustness).
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # Model: LightGBM Regressor
    # Parameters are selected to handle non-linear relationships and prevent overfitting 
    # on a dataset of 16,000 samples.
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        importance_type='gain',
        n_jobs=-1,
        verbosity=-1
    )

    # Final Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline