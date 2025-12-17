import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    # standard Green Agent paths
    data_dir = os.environ.get("EVAL_DATA_DIR", "/data")
    output_dir = os.environ.get("EVAL_OUTPUT_DIR", "/output")
    predictions_file = os.environ.get("EVAL_PREDICTIONS_FILE", os.path.join(output_dir, "predictions.csv"))

    print("White Agent (Real ML) starting...")
    
    # Load Data
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: train.csv or test.csv not found in {data_dir}")
        return

    print("Loading data...")
    # Both train.csv and test.csv appear to lack headers and have the target column.
    
    col_names = [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", 
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    ]
    
    train_df = pd.read_csv(train_path, header=None, names=col_names)
    test_df = pd.read_csv(test_path, header=None, names=col_names)
    
    # Feature Selection
    # Drop columns that are hard to use without nlp/complex logic: Name, Ticket, Cabin
    # Target: Survived
    target = "Survived"
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    passenger_ids = test_df["PassengerId"]
    
    # Preprocessing Pipeline
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"] # Pclass is technically ordinal/categorical
    
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Model Pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
    ])
    
    # Train
    print("Training Random Forest model...")
    clf.fit(X_train, y_train)
    
    # Predict
    print("Generating predictions...")
    preds = clf.predict(X_test)
    
    # Save output
    output_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds,
        "prediction": preds # Required by Green Agent
    })
    
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    output_df.to_csv(predictions_file, index=False)
    print(f"Saved {len(output_df)} predictions to {predictions_file}")

if __name__ == "__main__":
    main()
