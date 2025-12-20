from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance regression pipeline for student exam score prediction.
    
    Strategies implemented:
    1. Robust Preprocessing: Median imputation for numericals to handle outliers, 
       and most_frequent for categoricals.
    2. Interaction Engineering: Uses PolynomialFeatures on numeric columns to capture 
       non-linear relationships (e.g., study_hours * class_attendance) as recommended 
       in educational research.
    3. Categorical Encoding: One-hot encoding for all categorical variables, 
       ensuring the model captures high-cardinality relationships like 'course'.
    4. Gradient Boosting: Utilizes LightGBM, a state-of-the-art GBDT framework, 
       with optimized hyperparameters for regularization (lambda_l1/l2) and 
       generalization (learning_rate/subsample).
    """
    
    # Define feature groups
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = [
        'gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]

    # Preprocessing for numeric data: Impute -> Scale -> Interactions
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])

    # Preprocessing for categorical data: Impute -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Final Pipeline with LightGBM Regressor
    # Parameters tuned for a dataset of ~16,000 samples to prevent overfitting 
    # while maximizing R2.
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.015,
            num_leaves=63,
            max_depth=12,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1,
            boosting_type='gbdt',
            importance_type='gain'
        ))
    ])