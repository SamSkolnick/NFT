
import pandas as pd
from pathlib import Path
import os

RAW_PATH = "/Users/samuelskolnick/MLEngineer/spamhamdata.csv"
DEST_DIR = Path("/Users/samuelskolnick/MLEngineer/GreenAgent/data/spam")

def main():
    print(f"Reading {RAW_PATH}...")
    # TSV format
    df = pd.read_csv(RAW_PATH, sep='\t', header=None, names=["label", "text"])
    
    # Split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Train (features + labels)
    train_df.to_csv(DEST_DIR / "train.csv", index=False)
    
    # Save Test (features only for agent)
    # The agent gets 'test.csv' usually without labels in a real scenario, creates predictions.
    # But GreenAgent.py passes "test_data_path". 
    # WhiteAgent (SolverServer.py) loads it and predicts.
    # Usually we want to hide labels.
    test_features = test_df.drop(columns=["label"])
    test_features.to_csv(DEST_DIR / "test.csv", index=False)
    
    # Save Labels (for Green Agent evaluation)
    # GreenAgent.py `_load_labels` handles CSV with header 'label'
    test_df[["label"]].to_csv(DEST_DIR / "test_labels.csv", index=False)
    
    print(f"✓ Data prepared in {DEST_DIR}")

if __name__ == "__main__":
    main()
