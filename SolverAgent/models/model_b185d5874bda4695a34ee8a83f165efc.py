from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Expert Machine Learning Pipeline for Student Exam Score Prediction.
    
    Strategies implemented:
    1. Interaction Features: Captures synergistic effects like 'Study Hours' x 'Attendance'.
    2. GBDT Architecture: Uses XGBoost with regularization to handle potential overfitting.
    3. Robust Encoding: OneHot encoding for categorical variables to handle varied levels.
    4. Numeric Scaling: Standardization to stabilize gradient-based learning.
    """
    
    # Define feature groups based on the dataset schema
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing for numeric data:
    # Scale first, then generate interaction terms (e.g., Attendance * Study Hours)
    # This helps the model capture "Engagement x Ability" dynamics.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    # Preprocessing for categorical data:
    # Use OneHotEncoder with handle_unknown='ignore' to prevent crashes on new categories.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Model: XGBRegressor
    # Tuned for a dataset size of 16,000 to maximize generalization (R2 improvement focus).
    # We include regularization (alpha/lambda) to prevent overfitting on smaller cohort patterns.
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    # Construct the full pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])