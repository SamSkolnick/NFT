from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a state-of-the-art regression pipeline for student performance prediction.
    Incorporates feature interactions to capture 'Learning Engagement' proxies and
    uses LightGBM for robust gradient boosting.
    """
    # Feature Categorization
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Numeric transformation: 
    # 1. Create interactions (e.g., study_hours * class_attendance) to proxy engagement depth.
    # 2. Scale features to stabilize gradient descent.
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # Categorical transformation:
    # Use OneHotEncoding. While some features are ordinal, OHE captures non-linear 
    # jumps in impact (e.g., the gap between 'medium' and 'hard' difficulty).
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Trees (LightGBM)
    # Tuned hyperparameters based on the iterative target (R2 > 0.70):
    # - learning_rate: Lower rate with more estimators allows the model to find subtle patterns.
    # - num_leaves: Higher leaf count captures complex non-linear interactions.
    # - reg_alpha/lambda: L1/L2 regularization to prevent overfitting on the 16k dataset.
    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.007,
        num_leaves=63,
        max_depth=10,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=5,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )

    # Return the unified pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])