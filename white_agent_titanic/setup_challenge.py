
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prompt(question, default=None, options=None):
    d_text = f" [{default}]" if default else ""
    op_text = f" ({'/'.join(options)})" if options else ""
    
    while True:
        try:
            val = input(f"{question}{op_text}{d_text}: ").strip()
        except EOFError:
            return default if default else ""
            
        if not val and default:
            return default
        if not val and not default:
            continue
            
        if options and val not in options:
            print(f"Invalid option. Please choose from: {options}")
            continue
        return val

def detect_id_column(df):
    # Heuristic: Look for 'id' case insensitive
    for col in df.columns:
        if str(col).lower() in ['id', 'passengerid', 'uid', 'index']:
            return col
    # Check for unique sequential integers
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) and df[col].is_unique:
            return col
    return None

def detect_target_column(df):
    # Heuristic: Look for common target names
    candidates = ['target', 'label', 'class', 'survived', 'y', 'outcome', 'spam']
    for col in df.columns:
        if str(col).lower() in candidates:
            return col
    # Heuristic: Last column is often target
    return df.columns[-1]

def main():
    print("Welcome to the Smart White Agent Challenge Setup Wizard!")
    print("------------------------------------------------------")
    
    # 1. Load Data
    data_path = prompt("Path to your dataset (CSV)", "./data/raw_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    try:
        df = pd.read_csv(data_path)
        print(f"\n✓ Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")
        print(f"Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Heuristics for configuration
    suggested_id = detect_id_column(df)
    suggested_target = detect_target_column(df)
    
    challenge_name = prompt("Challenge Name", "My Challenge")
    
    # Problem Type Detection
    suggested_problem = "classification"
    if suggested_target:
        n_unique = df[suggested_target].nunique()
        if n_unique > 20 and pd.api.types.is_numeric_dtype(df[suggested_target]):
            suggested_problem = "regression"
            
    problem_type = prompt("Problem Type", suggested_problem, options=["classification", "regression"])
    
    # Select Columns
    print("\n--- Column Mapping ---")
    id_col = prompt(f"Select ID Column", suggested_id)
    if id_col not in df.columns:
        print(f"Warning: '{id_col}' not found in CSV. Creating synthetic ID.")
        df['id'] = range(len(df))
        id_col = 'id'
        
    target_col = prompt(f"Select Target Column", suggested_target)
    if target_col not in df.columns:
        print("Error: Target column must exist.")
        return

    # Metrics
    default_metrics = "accuracy,f1_score" if problem_type == "classification" else "rmse,r2_score"
    metrics = prompt("Metrics (comma separated)", default_metrics).split(",")
    metrics = [m.strip() for m in metrics]

    # 3. Dataset Splitting
    print("\n--- Dataset Splitting ---")
    should_split = prompt("Do you want to split this into Train/Validation/Test sets automatically?", "yes", options=["yes", "no"])
    
    final_gt_file = os.path.basename(data_path)
    base_dir = os.path.dirname(os.path.abspath(data_path))
    
    if should_split == "yes":
        test_size = float(prompt("Test set size (0.1 - 0.5)", "0.2"))
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        train_path = os.path.join(base_dir, "train.csv")
        test_path = os.path.join(base_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n✓ Created split datasets:")
        print(f"  - Training Data (for you/User): {train_path} ({len(train_df)} rows)")
        print(f"  - Ground Truth (for Agent):    {test_path} ({len(test_df)} rows)")
        
        final_gt_file = "test.csv"
        print("Using 'test.csv' as the Ground Truth for the White Agent.")
    else:
        print(f"Using original file '{final_gt_file}' as Ground Truth.")

    # 4. Save Config
    config = {
        "challenge_name": challenge_name,
        "problem_type": problem_type,
        "metrics": metrics,
        "data_path": base_dir,
        "ground_truth_file": final_gt_file,
        "id_column": id_col,
        "target_column": target_col,
        "prediction_column": target_col,
        "constraints": {
            "max_runtime_seconds": 600,
            "max_memory_mb": 4096,
            "allowed_extensions": [".csv"]
        }
    }
    
    
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    config_name = challenge_name.lower().replace(" ", "_").replace("-", "_") + "_config.json"
    save_path = os.path.join(configs_dir, config_name)
    
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"\n✓ Configuration saved to {save_path}")
    print("\n[Next Steps]")
    print(f"1. Run the White Agent Manager:")
    print(f"   python WhiteAgentController.py")
    print(f"   (It will automatically load all configs in 'configs/')")

if __name__ == "__main__":
    main()
