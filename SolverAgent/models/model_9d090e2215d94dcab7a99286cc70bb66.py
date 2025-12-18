from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

def build_pipeline() -> Pipeline:
    """
    Builds a machine learning pipeline for heart disease classification.
    Uses RobustScaler to handle clinical outliers and XGBoost for high-performance non-linear modeling.
    To maximize Recall for the 'Presence' class, scale_pos_weight is utilized.
    """
    
    # Define features based on the provided dataset information
    numeric_features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    
    # Preprocessing: Apply RobustScaler to all numeric features as per best practices for medical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features)
        ],
        remainder='drop'
    )
    
    # Model: XGBoost Classifier
    # scale_pos_weight is set to a value > 1 to increase the sensitivity (Recall) for the positive class.
    # Given the constraint to maximize recall for 'Presence', we prioritize the minority class.
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.01,
        scale_pos_weight=2.0,  # Hyperparameter to boost Recall for the positive class
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Create the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
    
    return pipeline