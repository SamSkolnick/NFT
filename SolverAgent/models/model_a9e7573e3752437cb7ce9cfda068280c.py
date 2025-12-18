import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBClassifier

def engineer_cardiac_features(X):
    """
    Expert feature engineering for heart disease prediction.
    Includes interaction terms and physiological ratios.
    """
    # Ensure working with a DataFrame
    if not isinstance(X, pd.DataFrame):
        # Column names based on provided info
        cols = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
                'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 
                'Slope of ST', 'Number of vessels fluro', 'Thallium']
        X = pd.DataFrame(X, columns=cols)
    
    X = X.copy()
    
    # 1. Age-stratified Blood Pressure Interaction
    # Physiological reality: High BP is more critical as age increases
    X['BP_Age_Product'] = X['BP'] * X['Age']
    
    # 2. Metabolic Proxy
    # Since HDL is missing, we use Age as a denominator for Cholesterol 
    # as a proxy for cumulative metabolic exposure.
    X['Chol_Age_Ratio'] = X['Cholesterol'] / (X['Age'] + 1)
    
    # 3. Cardiac Reserve Proxy
    # Relationship between Max Heart Rate and Blood Pressure during stress
    X['HR_BP_Ratio'] = X['Max HR'] / (X['BP'] + 1)
    
    # 4. Age-stratified BP buckets (Binned)
    X['BP_Category'] = np.digitize(X['BP'], bins=[120, 130, 140, 180])
    
    return X

def build_pipeline() -> Pipeline:
    """
    Builds a robust ML pipeline for heart disease classification.
    Optimized for high recall using cost-sensitive learning (XGBoost scale_pos_weight).
    """
    
    # Define the 13 feature columns provided in the data info
    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
        'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 
        'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    # Preprocessing Pipeline:
    # 1. Select the specific columns to prevent leakage or noise.
    # 2. Apply custom physiological feature engineering.
    # 3. Standardize features (assists with interpretation and model convergence).
    preprocessing = Pipeline([
        ('selection', FunctionTransformer(lambda x: x[features])),
        ('engineering', FunctionTransformer(engineer_cardiac_features)),
        ('scaler', StandardScaler())
    ])

    # Model Selection: XGBoost
    # Reasons: Handles non-linearities, missing values, and allows cost-sensitive tuning.
    # scale_pos_weight is set > 1 to prioritize Recall for the 'Presence' class.
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,            # Shallow trees to prevent overfitting on small N=216
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.5,   # Focus on reducing False Negatives (Maximize Recall)
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Combine into final Pipeline object
    pipeline = Pipeline([
        ('preprocessor', preprocessing),
        ('clf', model)
    ])

    return pipeline