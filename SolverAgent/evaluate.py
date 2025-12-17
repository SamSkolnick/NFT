"""
Evaluation entrypoint for the sample white agent.

Expected environment variables (defaults in brackets):
    - EVAL_DATA_DIR: directory containing the evaluation split (./data)
    - EVAL_OUTPUT_DIR: directory where predictions.csv will be written (./output)
    - EVAL_PREDICTIONS_FILE: filename for predictions (predictions.csv)

The script loads the pre-trained pipeline saved by train.py and produces
a CSV with a single `prediction` column.
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.pkl"
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


def get_paths() -> tuple[Path, Path]:
    data_dir = Path(os.environ.get("EVAL_DATA_DIR", ROOT / "data"))
    output_dir = Path(os.environ.get("EVAL_OUTPUT_DIR", ROOT / "output"))
    pred_filename = os.environ.get("EVAL_PREDICTIONS_FILE", "predictions.csv")
    return data_dir, output_dir / pred_filename


def load_features(data_dir: Path) -> pd.DataFrame:
    test_path = data_dir / "test" / "test.csv" if (data_dir / "test").exists() else data_dir / "test.csv"
    df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES, na_values=[""])
    if "Survived" in df.columns:
        df = df.drop(columns=["Survived"])
    passenger_ids = df.get("PassengerId")
    X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    return X, passenger_ids


def main() -> None:
    data_dir, predictions_path = get_paths()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    X_test, passenger_ids = load_features(data_dir)
    preds = model.predict(X_test)

    result = pd.DataFrame({"prediction": preds})
    if passenger_ids is not None:
        result.insert(0, "PassengerId", passenger_ids)

    result.to_csv(predictions_path, index=False)
    (predictions_path.parent / "metrics.json").write_text(
        json.dumps({"total_predictions": len(result)}, indent=2)
    )
    print(f"Wrote predictions to {predictions_path}")


if __name__ == "__main__":
    main()
