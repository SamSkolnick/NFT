from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from xgboost import XGBRegressor

def build_pipeline() -> Pipeline:
    """
    Builds an advanced regression pipeline for predicting student exam scores.
    Uses XGBoost with engineered interaction terms and robust preprocessing.
    """
    
    # Feature categorization based on dataset info
    numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
    
    # Interaction features: Synergy between study efforts and attendance
    interaction_features = ['study_hours', 'class_attendance']

    # Preprocessing logic:
    # - StandardScaler for continuous variables to aid convergence.
    # - OneHotEncoder for categoricals (handles high-cardinality like 'course').
    # - PolynomialFeatures to capture non-linear interactions (Velocity/Synergy).
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('interact', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), interaction_features)
        ],
        remainder='drop'
    )

    # Gradient Boosted Decision Tree (XGBoost)
    # Configured with regularization (alpha/lambda) to prevent overfitting on small cohorts
    # and subsampling to handle potential noise/variance in student data.
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        reg_alpha=0.5,
        reg_lambda=1.0
    )

    # Final Pipeline construction
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])