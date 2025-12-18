"""
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
import os
import litellm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import importlib.util
import time
import uuid
import json
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"


def generate_model_code(task_description: str, constraints: str, data_info: str, llm_model: str) -> str:
    """
    Asks the LLM to write a Python function `build_pipeline` that returns a Scikit-Learn pipeline.
    """
    prompt = f"""
    You are an expert Machine Learning Engineer.

    Task: {task_description}
    Constraints: {constraints}
    Data Info (label of each column of the dataset in order): {data_info}
    
    Write a Python function named `build_pipeline` that returns a sklearn.pipeline.Pipeline.
    
    Requirements:
    1. The function signature MUST be: `def build_pipeline() -> Pipeline:`
    2. Input `X` will be a pandas DataFrame. You MUST select the correct columns!
    3. If text data, use ColumnTransformer to apply TfidfVectorizer to the specific text column.
    4. Return ONLY the Python code. No markdown backticks.
    5. Include ALL imports (including ColumnTransformer, TfidfVectorizer, etc.).
    
    Example Output:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    
    def build_pipeline():
        # Assuming Data Info says text column is 'text'
        preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(), 'text')
        ])
        return Pipeline([('pre', preprocessor), ('clf', MultinomialNB())])
    """
    
    try:
        print(f"Generating model code with {llm_model}...")
        response = litellm.completion(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        code = response.choices[0].message.content.strip()
        # Strip markdown if present
        if code.startswith("```python"):
            code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("\n", 1)[0]
        return code
    except Exception as e:
        print(f"Code generation failed: {e}")
        # Fallback to a simple default
        return """
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
def build_pipeline():
    return Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler()), ('clf', LogisticRegression())])
"""

def save_and_load_code(code: str) -> Any:
    """
    Saves code to a unique file and loads it as a module.
    """
    filename = f"model_{uuid.uuid4().hex}.py"
    models_dir = DATA_DIR.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    filepath = models_dir / filename
    filepath.write_text(code)
    
    print(f"Saved generated model code to {filepath}")
    
    spec = importlib.util.spec_from_file_location("generated_model", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module, filepath

# Dynamic Data Loading - No hardcoded columns


def load_data(data_path: Path = None, target_col: str = None, type: str = 'tsv', extract_labels: bool = True) -> tuple[pd.DataFrame, pd.Series, list]:
    # we expect the first row to always have the column titles and the first column to be the answers
    path = data_path or (DATA_DIR / "train.csv")
    if not path.exists():
        raise FileNotFoundError(f"Data not found at {path}")
    
    # Infer type from extension if not explicitly tsv but path ends in .tsv
    if str(path).endswith('.tsv'):
        type = 'tsv'

    if type == 'tsv':
        df = pd.read_csv(path, sep='\t')
    else:
        df = pd.read_csv(path)
    
    column_titles = df.columns
    
    labels = None
    data = df
    
    if extract_labels:
        # Identify target column
        # If target_col provided, use it. Else use the first column.
        if target_col:
            labels_name = target_col
        else:
            labels_name = df.columns[0]
            
        labels = df[labels_name]
        data = df.drop(columns=[labels_name])
            
    return data, column_titles, labels

def train_model(task_desc: str, constraints: str, llm_model: str = "gpt-4o", data_path: Path = None, valdata_path: Path = None, target_col: str = None) -> dict:
    X, column_titles, labels = load_data(data_path, target_col, extract_labels=True)
    
    # Allow user to override labels name if desired, or just use the name from the series
    y = labels

    # 1. Analyze Data for LLM
    start_time = time.time()
    
    # Helper to safe-list columns
    numeric_cols = list(X.select_dtypes(include=['number']).columns)
    cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
    
    data_info = f"""
    Columns: {list(column_titles)}
    Target: {y.name}
    Numeric Features: {numeric_cols}
    Categorical Features: {cat_cols}
    Shape: {X.shape}
    Example row: {X.iloc[0].to_dict()}
    """
    
    # 2. Generate Code
    code = generate_model_code(task_desc, constraints, data_info, llm_model)
    
    # 3. Load & Execute
    module, code_path = save_and_load_code(code)
    
    # 4. Get Pipeline
    if not hasattr(module, "build_pipeline"):
        raise ValueError("Generated code does not contain `build_pipeline` function.")
        
    pipeline = module.build_pipeline()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, digits=3, output_dict=True)
    
    # Save Pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # 5. Validation Predictions (if valdata_path provided)
    val_preds_list = []
    if valdata_path and valdata_path.exists():
        print(f"Generating predictions for validation set: {valdata_path}")
        X_test, _, _ = load_data(valdata_path, type='tsv' if str(valdata_path).endswith('.tsv') else 'csv', extract_labels=False)
        
        # Ensure columns match - naive approach, assuming same schema
        # Align columns if possible (add missing as 0/nan, drop extra) - For now just predict
        try:
             # Just protect against shape mismatch if possible, but sklearn pipelines usually handle by name if pandas
             val_preds = pipeline.predict(X_test)
             val_preds_list = val_preds.tolist()
        except Exception as e:
            print(f"Validation prediction failed: {e}")
            
    
    return {
        "selected_model": "Custom Generated Pipeline",
        "validation_report": report,
        "model_path": str(MODEL_PATH.resolve()),
        "code_path": str(code_path),
        "val_predictions": val_preds_list
    }

def main() -> None:
    task_desc = "Predict survival."
    constraints = "Fast"
    result = train_model(task_desc, constraints)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
