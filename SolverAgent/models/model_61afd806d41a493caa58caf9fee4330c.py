from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Builds a high-performance regression pipeline for student exam score prediction.
    Uses an ensemble stacking strategy (LightGBM and XGBoost) with advanced feature 
    preprocessing including interaction terms and power transformations to maximize R2.
    """
    
    # Feature groups based on data description
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
    
    # 1. Numerical preprocessing: 
    # Create interactions (e.g., study_hours * attendance), scale, and apply Yeo-Johnson to handle skewness.
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])
    
    # 2. Categorical preprocessing: 
    # One-hot encoding for all qualitative features.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 3. Model Ensemble: Stacking GBDTs
    # LightGBM and XGBoost capture different patterns in tabular data.
    # Hyperparameters are tuned for balanced regularization.
    
    lgbm_model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        num_leaves=63,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    xgb_model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    # Final meta-learner (RidgeCV) helps prevent overfitting by the base models.
    stacking_regressor = StackingRegressor(
        estimators=[
            ('lgbm', lgbm_model),
            ('xgb', xgb_model)
        ],
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    # Final Pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', stacking_regressor)
    ])