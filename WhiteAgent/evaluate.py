import os
import pandas as pd
import glob
import random

def main():
    # standard Green Agent paths
    data_dir = os.environ.get("EVAL_DATA_DIR", "/data")
    output_dir = os.environ.get("EVAL_OUTPUT_DIR", "/output")
    predictions_file = os.environ.get("EVAL_PREDICTIONS_FILE", os.path.join(output_dir, "predictions.csv"))

    print(f"White Agent starting...")
    print(f"Reading data from: {data_dir}")
    print(f"Writing predictions to: {predictions_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)

    # Updated Logic: Specifically look for test.csv or similar to get the IDs
    # This ensures we match the ground truth (dummy_test_labels.csv) count.
    
    test_file = os.path.join(data_dir, "test.csv")
    ids = []
    results = [] # Initialize results list
    
    if os.path.exists(test_file):
        print(f"Reading test data from {test_file}")
        try:
            df = pd.read_csv(test_file)
            if "PassengerId" in df.columns:
                ids = df["PassengerId"].tolist()
            else:
                print("Warning: PassengerId column not found in test.csv")
        except Exception as e:
            print(f"Error reading test.csv: {e}")
    
    # Fallback: If test.csv doesn't exist or is invalid, try to discover from metadata or other files
    if not ids:
        print("Fallback: Attempting to infer IDs from file list or dummy labels...")
        # Check for the labels file itself (often provided in eval container for reference, or by mistake)
        labels_file = os.path.join(data_dir, "dummy_test_labels.csv")
        if os.path.exists(labels_file):
             print(f"Reading IDs from {labels_file}")
             try:
                 df = pd.read_csv(labels_file)
                 if "PassengerId" in df.columns:
                     ids = df["PassengerId"].tolist()
             except Exception:
                 pass

    # If still no IDs, fallback to original behavior (file listing)
    if not ids:
        print("Fallback: Using filenames as IDs")
        input_files = glob.glob(os.path.join(data_dir, "*"))
        ids = [os.path.basename(f) for f in input_files]

    print(f"Generating predictions for {len(ids)} items...")

    for item_id in ids:
        # Mock prediction logic (Titanic: 0 or 1)
        prediction = random.choice([0, 1])
        # Note: GreenAgent expects 'prediction' column, but let's stick to 'Survived' if it matches the label file
        # Actually GreenAgent.py evaluate_performance likely blindly matches, but let's check.
        # The labels file has 'Survived'.
        results.append({"PassengerId": item_id, "Survived": prediction})

    # Write to CSV
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(predictions_file, index=False)
        print(f"Successfully wrote {len(results)} predictions to {predictions_file}")
    else:
        print("Warning: No inputs found to generate predictions for.")
        # Create an empty file just so we don't crash the Green Agent
        pd.DataFrame(columns=["PassengerId", "Survived"]).to_csv(predictions_file, index=False)

if __name__ == "__main__":
    main()
