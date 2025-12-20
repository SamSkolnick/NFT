from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student exam score prediction.
    Utilizes interaction features between study habits and XGBoost for high-performance modeling.
    """
    
    # Feature definitions based on the provided dataset info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]
    
    # Preprocessing for numerical data: 
    # 1. Standardize features to ensure stable convergence.
    # 2. Generate interaction terms (e.g., study_hours * class_attendance) 
    #    as these are often highly predictive in behavioral tasks.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    # Preprocessing for categorical data:
    # Use OneHotEncoding. handle_unknown='ignore' ensures robustness for deployment.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Optimized XGBoost parameters for the dataset size (~16,000 rows)
    # n_estimators and learning_rate are balanced to prevent overfitting while capturing complexity.
    # max_depth=7 allows the model to learn complex non-linear interactions.
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    # Final Pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])