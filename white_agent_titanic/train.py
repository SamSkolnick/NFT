"""
Train a simple Titanic survival classifier for the sample white agent.

This script reads the pre-split training data located in ./data/train.csv,
fits a preprocessing + logistic regression pipeline, performs a quick
validation report, and saves the trained pipeline to ./model/model.pkl.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"
COLUMN_NAMES = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(
        DATA_DIR / "train.csv",
        header=None,
        names=COLUMN_NAMES,
    )
    y = df["Survived"]
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
    return X, y


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", clf)])


def main() -> None:
    X, y = load_data()

    cat_cols = ["Sex", "Embarked"]
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    pipeline = build_pipeline(cat_cols, num_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, digits=3)
    print("Validation report:\n", report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
