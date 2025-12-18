import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

class HeartDiseaseEngineer(BaseEstimator, TransformerMixin):
    """
    Expert feature engineering for heart disease risk stratification.
    Captures non-linear interactions between physiological indicators.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        # Interaction between Age and Blood Pressure as a combined risk factor
        X['Age_BP_Risk'] = X['Age'] * X['BP']
        # Heart rate reserve ratio (Cardiovascular efficiency proxy)
        X['HR_Efficiency'] = X['Max HR'] / (X['Age'] + 1)
        # Interaction between structural vessel damage and electrical ST depression
        X['Vessel_ST_Risk'] = X['Number of vessels fluro'] * X['ST depression']
        # Log cholesterol to normalize the typical right-skew in lipid profiles
        X['Log_Chol'] = np.log1p(X['Cholesterol'])
        return X

def build_pipeline() -> Pipeline:
    """
    Constructs an advanced ML pipeline for heart disease prediction.
    Features: 
    - Custom clinical feature engineering
    - Robust scaling to handle physiological outliers
    - Target encoding for categorical clinical markers
    - Cost-sensitive gradient boosting for data imbalance
    """
    
    # Feature selection based on provided clinical data info
    numeric_cols = [
        'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 
        'Age_BP_Risk', 'HR_Efficiency', 'Vessel_ST_Risk', 'Log_Chol'
    ]
    
    categorical_cols = [
        'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
        'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    
    # Preprocessing block
    preprocessor = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numeric_cols),
        ('cat', TargetEncoder(random_state=42), categorical_cols)
    ])
    
    # Classifier: XGBoost tuned for clinical tabular data (small dataset constraints)
    # scale_pos_weight is utilized for cost-sensitive learning as requested.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        scale_pos_weight=1.3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Assembly of the expert pipeline
    return Pipeline(steps=[
        ('engineer', HeartDiseaseEngineer()),
        ('preprocessor', preprocessor),
        ('clf', model)
    ])