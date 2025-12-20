import os
import sys
import json
import shutil
from pathlib import Path

# Add SolverAgent to path so we can import train
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train_model

def run_batch_test():
    datasets_root = Path("/Users/samuelskolnick/MLEngineer/Datasets")
    
    # Mapping for inconsistent naming
    dataset_configs = [
        {
            "name": "ExamScores",
            "folder": datasets_root / "ExamScores",
            "train": "1_train_80_percent.csv",
            "test": "2_test_20_features.csv",
            "target": "exam_score",
            "desc": "Regression task to predict student exam scores."
        }]       
    '''{
            "name": "HeartDisease",
            "folder": datasets_root / "HeartDisease",
            "train": "heart_train_80.csv",
            "test": "heart_test_20_features.csv",
            "target": "Heart Disease",
            "desc": "Classification task to predict heart disease presence."
        },
        {
            "name": "Titanic",
            "folder": datasets_root / "Titanic",
            "train": "titanic_train_80.csv",
            "test": "titanic_test_20_features.csv",
            "target": "Survived",
            "desc": "Titanic survival prediction."
        },
        {
            "name": "SpamHam",
            "folder": datasets_root / "spamham",
            "train": "train.csv",
            "test": "test.csv",
            "target": "label",
            "desc": "SMS Spam classification."
        }
    ]'''
    
    print(f"Starting batch evaluation for {len(dataset_configs)} datasets...")
    
    for config in dataset_configs:
        folder = config["folder"]
        if not folder.exists():
            print(f"Warning: Folder {folder} does not exist. Skipping.")
            continue
            
        train_path = folder / config["train"]
        test_path = folder / config["test"]
        
        if not train_path.exists():
            print(f"Warning: Train file {train_path} not found. Skipping.")
            continue

        print(f"\n--- Processing: {config['name']} ---")
        
        try:
            results = train_model(
                task_desc=config["desc"],
                constraints="Maximize performance metrics. Use advanced ML techniques.",
                data_path=train_path,
                valdata_path=test_path,
                target_col=config["target"]
            )
            
            # Save files directly to the dataset directory
            
            # 1. Save Research
            research_path = folder / "research_report.md"
            research_path.write_text(f"# Research Report: {config['name']}\n\n{results.get('research', 'N/A')}")
            
            # 2. Save Code
            code_path = folder / "generated_model.py"
            original_code_path = Path(results["code_path"])
            shutil.copy(original_code_path, code_path)
            
            # 3. Save Results JSON
            results_path = folder / "evaluation_results.json"
            clean_results = results.copy()
            if "val_predictions" in clean_results:
                del clean_results["val_predictions"]
            with open(results_path, "w") as f:
                json.dump(clean_results, f, indent=2)
                
            print(f"Success! Files saved to {folder}")
            
        except Exception as e:
            print(f"Error processing {config['name']}: {e}")

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("GEMINI_API_KEY is required.")
        sys.exit(1)
    run_batch_test()
