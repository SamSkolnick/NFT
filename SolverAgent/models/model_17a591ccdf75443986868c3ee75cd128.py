import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

def build_pipeline() -> Pipeline:
    # Column names in the specific order provided
    numeric_features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    def engineer_clinical_features(X):
        """
        Creates domain-specific ratios to improve model performance.
        X is expected to be a numpy array after passing through the Imputer.
        Indices based on column order: Age=0, BP=3, Cholesterol=4, Max HR=7
        """
        # Ensure input is a numpy array
        X_arr = np.array(X)
        age = X_arr[:, 0]
        bp = X_arr[:, 3]
        chol = X_arr[:, 4]
        max_hr = X_arr[:, 7]
        
        # Domain-driven ratios (avoid division by zero)
        bp_age_ratio = (bp / (age + 1)).reshape(-1, 1)
        chol_age_ratio = (chol / (age + 1)).reshape(-1, 1)
        hr_age_ratio = (max_hr / (age + 1)).reshape(-1, 1)
        
        return np.hstack([X_arr, bp_age_ratio, chol_age_ratio, hr_age_ratio])

    # Preprocessing pipeline
    # 1. Select numeric features
    # 2. Impute missing values using MICE (IterativeImputer)
    # 3. Add engineered domain ratios
    # 4. Standardize features
    preprocessor = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('feature_eng', FunctionTransformer(engineer_clinical_features)),
        ('scaler', StandardScaler())
    ])

    # Wrap the selection in a ColumnTransformer
    col_transformer = ColumnTransformer([
        ('numeric_processing', preprocessor, numeric_features)
    ])

    # XGBoost Classifier
    # scale_pos_weight is used to maximize Recall for the positive class (Heart Disease Presence)
    # A value > 1 increases the penalty for misclassifying the minority (presence) class.
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.5,  # Prioritizing Recall
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Probability Calibration (Platt Scaling) to provide clinicians with actionable risk scores
    calibrated_clf = CalibratedClassifierCV(
        estimator=xgb_model,
        method='sigmoid',
        cv=5
    )

    # Final Pipeline
    return Pipeline([
        ('pre', col_transformer),
        ('clf', calibrated_clf)
    ])