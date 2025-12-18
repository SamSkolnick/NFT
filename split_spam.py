import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_spam_data(input_path: str, output_dir: str):
    # Read TSV
    df = pd.read_csv(input_path, sep='\t')
    
    # Split 80/20
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Create output dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save train set (with labels)
    train_df.to_csv(out_path / "train.csv", index=False)
    
    # Save test set (without labels for the solver to predict)
    test_features = test_df.drop(columns=['label'])
    test_features.to_csv(out_path / "test.csv", index=False)
    
    # Save test labels (for the green agent to evaluate)
    test_labels = test_df[['label']]
    test_labels.to_csv(out_path / "test_labels.csv", index=False)
    
    print(f"Split complete. Files saved in {output_dir}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

if __name__ == "__main__":
    split_spam_data("/Users/samuelskolnick/MLEngineer/spamhamdata.tsv", "/Users/samuelskolnick/MLEngineer/data/spam")
