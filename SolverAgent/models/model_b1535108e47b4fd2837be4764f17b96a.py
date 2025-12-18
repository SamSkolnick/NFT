from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    """
    Builds an optimized machine learning pipeline for heart disease classification.
    Given the small dataset size (216 samples), we focus on robust scaling, 
    imputation, and a gradient boosting model with regularization to prevent overfitting.
    """
    
    # Feature categories based on the provided data info
    # All features are provided as numeric in the task description
    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    # Preprocessing for numeric data:
    # 1. Median Imputation (more robust to outliers than mean)
    # 2. Standard Scaling (essential for clinical physiological metrics)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ],
        remainder='drop'
    )

    # XGBClassifier is chosen for its performance on structured tabular data.
    # Hyperparameters are constrained for the small dataset size (216 rows):
    # - low n_estimators and low max_depth to prevent overfitting.
    # - learning_rate set to a conservative value.
    # - reg_lambda/alpha added for L1/L2 regularization.
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_lambda=1.5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Construct the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline