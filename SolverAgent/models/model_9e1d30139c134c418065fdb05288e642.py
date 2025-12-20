from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance regression pipeline for student exam score prediction.
    Utilizes interaction features (Learning Velocity proxy) and Gradient Boosted Decision Trees (LightGBM).
    """
    
    # Define feature groups based on Data Info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # 1. Numeric Preprocessing:
    # Includes PolynomialFeatures (interaction_only) to capture synergistic effects
    # like (study_hours * class_attendance), which represents effective academic engagement.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Preprocessing:
    # Uses OneHotEncoding. For the provided dataset size (16,000 rows), 
    # OHE is robust and avoids the complexity of high-cardinality management 
    # unless unique categories exceed several hundred.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 4. Model Selection: LightGBM Regressor
    # Optimized with a low learning rate and moderate complexity to balance bias-variance.
    # We include L1/L2 regularization (reg_alpha/reg_lambda) to improve generalization 
    # over previous iterations.
    model = LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.015,
        num_leaves=45,
        max_depth=9,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.15,
        reg_lambda=0.15,
        random_state=42,
        n_jobs=-1,
        importance_type='gain'
    )

    # Final Pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])