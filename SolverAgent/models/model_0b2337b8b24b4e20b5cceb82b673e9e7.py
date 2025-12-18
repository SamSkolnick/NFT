from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def build_pipeline() -> Pipeline:
    """
    Builds a machine learning pipeline for heart disease classification.
    Focuses on maximizing recall for the 'Presence' class using balanced class weights 
    and Quantile Transformation for physiological metrics as per best practices.
    """
    # Define features based on the provided data info
    # Although some are logically categorical, the task specifies them as Numeric Features.
    numeric_features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    # Preprocessing: Use QuantileTransformer to handle non-Gaussian physiological distributions
    # and map them to a normal distribution, which helps tree-based models and linear components alike.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', QuantileTransformer(output_distribution='normal', random_state=42), numeric_features)
        ]
    )

    # Model Selection: RandomForestClassifier with 'balanced' class weights is highly effective 
    # for maximizing recall in small medical datasets (216 samples) as it penalizes misclassifying 
    # the minority 'Presence' class.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=5,
            random_state=42
        ))
    ])

    return pipeline