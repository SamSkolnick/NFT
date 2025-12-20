"""
This script reads the pre-split training data located in ./data/train.csv,
fits a preprocessing + logistic regression pipeline, performs a quick
validation report, and saves the trained pipeline to ./model/model.pkl.
"""

from pathlib import Path

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
from google import genai
from google.genai import types
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import psutil

import importlib.util
import time
import uuid
import json
import subprocess
import sys
from typing import Any

class UnrecoverableModuleError(Exception):
    """Raised when a module cannot be installed after multiple attempts."""
    def __init__(self, module_name):
        self.module_name = module_name
        super().__init__(f"Module {module_name} could not be installed.")

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"


def generate_model_enrichment(task_desc: str, llm_model: str = "gemini-3-flash-preview") -> str:
    """
    Conducts research to enrich the model creation using Google Search grounding.
    """
    prompt = f"""
    You are a world-class Data Scientist and Machine Learning Researcher.
    
    Task Description: {task_desc}
    
    Your goal is to provide a brief high-level research summary to guide a Machine Learning Engineer in building the best possible model for this task.
    
    Please provide:
    1. Recommended Model Architectures (e.g., Logistic Regression vs Random Forest vs XGBoost vs Transformer-based).
    2. Key Feature Engineering ideas (e.g., TF-IDF parameters, handling imbalanced data, specific scaling).
    3. Common pitfalls to avoid for this specific domain.
    
    Consider problems that have been solved and that have similar structure to this task across various domains. Explicitly state the cross domain-transfer learning aspect of the mode. 

    Keep it concise (max 500 words).
    """
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set for enrichment.")
        return ""
    
    client = genai.Client(api_key=api_key)
    
    try:
        print(f"Conducting model enrichment research with Gemini ({llm_model}) and Google Search...")
        response = client.models.generate_content(
            model=llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Enrichment research failed: {e}")
        return "Focus on standard Scikit-Learn pipelines with robust preprocessing."

def generate_model_code(task_description: str, constraints: str, data_info: str, enrichment_info: str, llm_model: str = "gemini-3-flash-preview", problem_type: str = "classification", forbidden_libraries: list[str] = None, iteration_history: list[dict] = None) -> str:
    """
    Asks the LLM to write a Python function `build_pipeline` that returns a Scikit-Learn pipeline.
    """
    forbidden_clause = ""
    if forbidden_libraries:
        forbidden_clause = f"\n    IMPORTANT: DO NOT USE the following libraries as they are not available: {', '.join(forbidden_libraries)}."

    history_clause = ""
    if iteration_history:
        history_clause = "\n    Previous Iteration Results (for reference and improvement):"
        for i, entry in enumerate(iteration_history):
            history_clause += f"\n    Iteration {i+1}:"
            history_clause += f"\n    - Results: {json.dumps(entry.get('results', {}))}"
            history_clause += f"\n    - Strategy Used: {entry.get('strategy', 'N/A')}"
            history_clause += f"\n    - Performance: Learn from what worked or didn't work here.\n"

    prompt = f"""
    You are an expert Machine Learning Engineer.
    
    Background Research & Best Practices:
    {enrichment_info}

    Task: {task_description}
    Constraints: {constraints}{forbidden_clause}{history_clause}
    Data Info (label of each column of the dataset in order): {data_info}
    
    Available Libraries: scikit-learn, xgboost, lightgbm, pytorch, pandas, numpy, scipy, statsmodels, transformers, nltk, spacy, matplotlib, seaborn.
    
    1. The function signature MUST be: `def build_pipeline() -> Pipeline:`
    2. Input `X` will be a pandas DataFrame. You MUST select the correct columns!
    3. If text data, use ColumnTransformer to apply TfidfVectorizer to the specific text column.
    4. Return ONLY the Python code. No markdown backticks.
    5. Include ALL imports (including ColumnTransformer, TfidfVectorizer, etc.).
    6. Ensure the model is appropriate for the problem type ({problem_type}).
    7. ONLY use models from scikit-learn, xgboost, or lightgbm.
    8. AVOID defining custom functions or classes if possible. If you MUST use a custom transformer, ensure it is a simple class defined within the script (not a lambda or a local function within `build_pipeline`), as it needs to be picklable by `dill`.
    
    Example Output:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from xgboost import XGBClassifier
    
    def build_pipeline():
        # Assuming Data Info says text column is 'text'
        preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(), 'text')
        ])
        return Pipeline([('pre', preprocessor), ('clf', XGBClassifier())])
    """
    
    # Configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set. Using default pipeline.")
        return """
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=['number'])),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])
"""
    
    client = genai.Client(api_key=api_key)
    
    try:
        print(f"Generating model code with Gemini ({llm_model}) and Google Search...")
        response = client.models.generate_content(
            model=llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        code = response.text.strip()
        
        # Strip markdown if present
        if "```" in code:
            # Try to extract content between backticks
            import re
            match = re.search(r"```(?:python)?\n?(.*?)\n?```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                # Fallback: just strip the start/end backticks if they are there
                code = code.strip("`").strip()
                if code.startswith("python"):
                    code = code[6:].strip()
                    
        return code
    except Exception as e:
        print(f"Code generation failed: {e}")
        # Fallback to a simple default
        return """
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=['number'])),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])
"""

def install_missing_packages(packages: list[str]):
    """Attempts to install missing packages via pip."""
    try:
        print(f"Detected missing packages: {packages}. Attempting to install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        print("Installation successful.")
    except Exception as e:
        print(f"Failed to install packages {packages}: {e}")
        raise

def save_and_load_code(code: str, retry_count: int = 0) -> Any:
    """
    Saves code to a unique file and loads it as a module.
    Automatically retries with pip install if ModuleNotFoundError occurs.
    """
    filename = f"model_{uuid.uuid4().hex}.py"
    models_dir = DATA_DIR.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    filepath = models_dir / filename
    filepath.write_text(code)
    
    print(f"Saved generated model code to {filepath}")
    
    spec = importlib.util.spec_from_file_location("generated_model", filepath)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        return module, filepath
    except ModuleNotFoundError as e:
        missing_module = e.name
        if retry_count < 2 and missing_module:
            print(f"Module '{missing_module}' not found. Retry {retry_count + 1}/2...")
            try:
                # Handle common mapping if needed
                pkg_to_install = missing_module
                if pkg_to_install == "sklearn":
                    pkg_to_install = "scikit-learn"
                
                install_missing_packages([pkg_to_install])
                # Reload by calling itself recursively
                return save_and_load_code(code, retry_count + 1)
            except Exception:
                print(f"Double failure on {missing_module}. Giving up on this library.")
                raise UnrecoverableModuleError(missing_module)
        else:
            raise UnrecoverableModuleError(missing_module or "Unknown")

def save_and_load_code_legacy(code: str) -> Any:
    """Legacy version without auto-install for reference."""
    filename = f"model_{uuid.uuid4().hex}.py"
    models_dir = DATA_DIR.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    filepath = models_dir / filename
    filepath.write_text(code)
    
    spec = importlib.util.spec_from_file_location("generated_model", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, filepath

# Dynamic Data Loading - No hardcoded columns


def load_data(data_path: Path = None, target_col: Any = None, type: str = 'csv', extract_labels: bool = True) -> tuple[pd.DataFrame, pd.Series, list]:
    # we expect the first row to always have the column titles and the first column to be the answers
    path = data_path

    
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
        if target_col is not None:
            if isinstance(target_col, int):
                labels_name = df.columns[target_col]
            else:
                labels_name = target_col
        else:
            labels_name = df.columns[0]
            
        labels = df[labels_name]
        data = df.drop(columns=[labels_name])
            
    return data, column_titles, labels

def train_model(task_desc: str, constraints: str, llm_model: str = "gemini-3-flash-preview", data_path: Path = None, valdata_path: Path = None, target_col: Any = None, do_improvement_loop: bool = True) -> dict:
    data_path = Path(data_path) if data_path else None
    valdata_path = Path(valdata_path) if valdata_path else None
    
    # Resource Tracking Init
    process = psutil.Process()
    start_time = time.time()
    start_cpu_time = process.cpu_times()
    start_mem = process.memory_info().rss
    
    X, column_titles, labels = load_data(data_path, target_col, extract_labels=True)
    
    # Analyze Problem Type early
    unique_ratio = len(labels.unique()) / len(labels)
    if pd.api.types.is_numeric_dtype(labels) and (len(labels.unique()) > 20 or unique_ratio > 0.1):
        problem_type = "regression"
    else:
        problem_type = "classification"
    print(f"Detected problem type: {problem_type}")

    # Allow user to override labels name if desired, or just use the name from the series
    y = labels
    
    # Clean data: drop rows where target or features are completely missing
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(y) == 0:
        raise ValueError("No valid data points after dropping NaNs.")

    # 1. Analyze Data for LLM
    start_time = time.time()
    
    # Enrichment Phase
    enrichment_info = generate_model_enrichment(task_desc, llm_model)

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

    def get_score(report, p_type):
        if p_type == "classification":
            return report.get("macro avg", {}).get("f1-score", 0)
        else:
            return report.get("r2_score", -1.0)

    iteration_history = []
    best_results = None
    best_pipeline = None
    best_score = -float('inf')
    forbidden_libs = []
    
    max_iterations = 5 if do_improvement_loop else 1
    for iteration in range(max_iterations):
        print(f"\n--- Improvement Loop: Iteration {iteration + 1}/{max_iterations} ---")
        
        # 1. Generate & Load Code with Retry for Modules
        module = None
        code_path = None
        current_code = None
        
        max_gen_retries = 3
        for attempt in range(max_gen_retries):
            try:
                current_code = generate_model_code(
                    task_desc, constraints, data_info, enrichment_info, llm_model, 
                    problem_type=problem_type, forbidden_libraries=forbidden_libs,
                    iteration_history=iteration_history
                )
                module, code_path = save_and_load_code(current_code)
                break
            except UnrecoverableModuleError as e:
                print(f"Attempt {attempt + 1}: Could not load model due to missing library '{e.module_name}'.")
                forbidden_libs.append(e.module_name)
                if attempt == max_gen_retries - 1:
                    if iteration == 0: raise # Fail if we can't even get one model
                    break # Just use whatever we have so far
                print(f"Retrying code generation without {forbidden_libs}...")
        
        if not module: continue # Skip if generation failed completely

        try:
            # 2. Train and Evaluate
            if not hasattr(module, "build_pipeline"):
                raise ValueError("Generated code does not contain `build_pipeline` function.")
            pipeline = module.build_pipeline()
            
            if problem_type == "classification":
                # Ensure labels are clean for training
                y_encoded = y
                if not pd.api.types.is_integer_dtype(y):
                    le = LabelEncoder()
                    y_encoded = pd.Series(le.fit_transform(y), name=y.name, index=y.index)
                
                X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded)
            else:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=None)
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            if problem_type == "classification":
                val_report = classification_report(y_val, y_pred, digits=3, output_dict=True)
            else:
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                val_report = {"mean_squared_error": mse, "r2_score": r2, "problem_type": "regression"}
            
            current_score = get_score(val_report, problem_type)
            print(f"Iteration {iteration + 1} Score ({'F1' if problem_type == 'classification' else 'R2'}): {current_score:.4f}")
            
            # Resource Tracking Final for this iteration
            iter_end_mem = process.memory_info().rss
            iter_end_time = time.time()
            
            # 3. Best Model Tracking
            if current_score > best_score:
                best_score = current_score
                best_pipeline = pipeline
                
                model_size = 0
                temp_model_path = DATA_DIR.parent / "model" / "temp_best.pkl"
                temp_model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_model_path, "wb") as f:
                    dill.dump(pipeline, f)
                model_size = temp_model_path.stat().st_size
                
                best_results = {
                    "selected_model": "Iterative Best Pipeline",
                    "iteration": iteration + 1,
                    "research": enrichment_info,
                    "validation_report": val_report,
                    "model_path": str(MODEL_PATH.resolve()),
                    "code_path": str(code_path),
                    "score": best_score,
                    "resource_usage": {
                        "elapsed_time_seconds": round(iter_end_time - start_time, 2),
                        "memory_rss_delta_mb": round((iter_end_mem - start_mem) / (1024 * 1024), 2),
                        "model_size_kb": round(model_size / 1024, 2)
                    }
                }
            
            # Add to history for next iteration LLM feedback
            iteration_history.append({
                "iteration": iteration + 1,
                "results": val_report,
                "code": current_code,
                "score": current_score
            })
            
            if best_score > 0.999: # Early exit for near-perfect models
                print("Achieved near-perfect performance. Stopping improvement loop.")
                break
                
        except Exception as e:
            sys.stderr.write(f"Error during iteration {iteration + 1}: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue

    if not best_pipeline:
        raise ValueError("Could not build a valid model in 5 iterations.")

    # 4. Final Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        dill.dump(best_pipeline, f)
        
    # 5. Final Validation (if requested)
    val_preds_list = []
    if valdata_path and valdata_path.exists():
        print(f"Generating best-model predictions for validation set: {valdata_path}")
        X_test, _, _ = load_data(valdata_path, type='tsv' if str(valdata_path).endswith('.tsv') else 'csv', extract_labels=False)
        
        # Ensure columns match - naive approach, assuming same schema
        # Align columns if possible (add missing as 0/nan, drop extra) - For now just predict
        try:
             # Just protect against shape mismatch if possible, but sklearn pipelines usually handle by name if pandas
             val_preds = pipeline.predict(X_test)
             val_preds_list = val_preds.tolist()
        except Exception as e:
            print(f"Validation prediction failed: {e}")
            
    
    # Resource Tracking Final
    end_time = time.time()
    end_cpu_time = process.cpu_times()
    end_mem = process.memory_info().rss
    
    # Model size
    model_size = 0
    if MODEL_PATH.exists():
        model_size = MODEL_PATH.stat().st_size

    resource_usage = {
        "elapsed_time_seconds": round(end_time - start_time, 2),
        "cpu_time_user": round(end_cpu_time.user - start_cpu_time.user, 2),
        "cpu_time_system": round(end_cpu_time.system - start_cpu_time.system, 2),
        "memory_rss_delta_mb": round((end_mem - start_mem) / (1024 * 1024), 2),
        "model_size_kb": round(model_size / 1024, 2)
    }
    
    # Results extraction
    best_results["val_predictions"] = val_preds_list
    best_results["iteration_count"] = iteration + 1
    
    return best_results

def main() -> None:
    task_desc = "Predict survival."
    constraints = "Fast"
    result = train_model(task_desc, constraints)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
