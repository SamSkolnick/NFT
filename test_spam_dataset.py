
import sys
import pandas as pd
from pathlib import Path
import json

# Add SolverAgent to path
sys.path.append(str(Path(__file__).parent / "SolverAgent"))
from train import train_model

def main():
    raw_path = Path("/Users/samuelskolnick/MLEngineer/spamhamdata.csv")
    print(f"Loading raw data from {raw_path}...")
    
    # Parse TSV, assign names
    # Inspecting head: "ham\tGo until..."
    try:
        df = pd.read_csv(raw_path, sep='\t', header=None, names=["label", "text"])
        print(f"Data Loaded. Shape: {df.shape}")
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Split
    print("Splitting into Train/Val...")
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_path = Path("train_spam.csv")
    val_path = Path("val_spam.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"Saved {train_path} and {val_path}")
    
    # Run Agent
    print("\n--- Invoking White Agent ---")
    task = "Classify emails as spam or ham based on text content."
    constraints = "Use a lightweight model suitable for text classification (e.g. TF-IDF + Naive Bayes or Logistic Regression). Handle text data."
    
    # We expect the agent to:
    # 1. Detect 'label' as target.
    # 2. See 'text' column.
    # 3. Ask LLM for code (which should include TfidfVectorizer).
    
    try:
        result = train_model(
            task_desc=task, 
            constraints=constraints, 
            llm_model="gpt-4o", 
            data_path=train_path.resolve()
        )
        
        print("\n--- Training Successful ---")
        print(f"Selected Model: {result['selected_model']}")
        print(f"Model Saved To: {result['model_path']}")
        print(f"Generated Code: {result['code_path']}")
        
        print("\nVal Report from Agent (Internal Split):")
        print(json.dumps(result["validation_report"], indent=2))
        
    except Exception as e:
        print(f"Agent Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
