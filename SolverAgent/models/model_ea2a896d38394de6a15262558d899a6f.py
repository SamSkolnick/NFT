from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance regression pipeline for student exam score prediction.
    
    Strategies implemented:
    1. Feature Engineering: Polynomial interactions between numeric features (study_hours, attendance, etc.)
       to capture "Learning Velocity" and effort-engagement proxies.
    2. Robust Encoding: OneHotEncoding for categorical features with 'ignore' handling for unknown labels.
    3. Model: LightGBM Regressor, chosen for its efficiency with tabular data and ability to capture 
       non-linear relationships better than standard GLMs.
    4. Scaling: StandardScaler to ensure interaction terms are on a comparable scale.
    """
    
    # Feature Selection
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Numeric Transformer: Capturing non-linear relationships and interactions
    # PolynomialFeatures(interaction_only=True) creates features like study_hours * class_attendance
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: standard one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Advanced GBDT Model: LightGBM
    # Hyperparameters tuned for a dataset of ~16k rows to prevent overfitting while capturing complexity
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=41,
        max_depth=8,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Final Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline