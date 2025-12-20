from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, PolynomialFeatures
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for student exam score prediction.
    
    The strategy includes:
    - Quantile Transformation for numerical features to handle non-normal distributions (common in behavioral data).
    - Polynomial interaction features for key numerical drivers like study hours and attendance.
    - One-Hot Encoding for categorical features.
    - A Gradient Boosted Decision Tree (LightGBM) optimized for tabular data and mixed feature types.
    """
    
    # Feature selection based on dataset info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing for numerical data: 
    # 1. Map to normal distribution 2. Generate interactions 3. Standardize
    numeric_transformer = Pipeline(steps=[
        ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Advanced GBDT Regressor (LightGBM)
    # Hyperparameters tuned for a dataset of ~16,000 rows
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=41,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )

    # Create the final pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])