from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, FunctionTransformer
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def build_pipeline() -> Pipeline:
    """
    Builds an advanced ML pipeline for heart disease prediction.
    Prioritizes Recall/Sensitivity using XGBoost and clinical feature engineering.
    """
    
    # Feature names provided in the dataset info
    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 
        'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 
        'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]
    
    # Identify continuous clinical vitals for PowerTransformation
    vitals = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

    def create_clinical_interactions(X):
        """
        Creates domain-driven features:
        1. BP-Age Interaction: Reflects cumulative cardiovascular strain.
        2. ST-Slope Interaction: Combines related markers of myocardial ischemia.
        3. HR-Age Ratio: Proxy for cardiac reserve relative to age-expected maximums.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features)
        
        X = X.copy()
        X['BP_Age'] = X['BP'] * X['Age']
        X['ST_Slope'] = X['ST depression'] * X['Slope of ST']
        X['HR_Age_Ratio'] = X['Max HR'] / (X['Age'] + 1)
        return X

    # Update processed features to include new interaction terms
    extended_vitals = vitals + ['BP_Age', 'ST_Slope', 'HR_Age_Ratio']

    # Preprocessing: PowerTransformer handles non-normal distributions in clinical data
    preprocessor = ColumnTransformer([
        ('clinical_vitals', Pipeline([
            ('power', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ]), extended_vitals)
    ], remainder='passthrough')

    # Classifier: XGBoost (GBDT) optimized for small, imbalanced tabular datasets.
    # scale_pos_weight is used to prioritize Recall (Sensitivity) over Accuracy,
    # minimizing False Negatives in a clinical context.
    # High regularization (alpha, lambda) and shallow trees prevent overfitting on N=216.
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=1.5,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    return Pipeline([
        ('feature_engineering', FunctionTransformer(create_clinical_interactions)),
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
呈现完毕