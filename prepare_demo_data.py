
import pandas as pd
import numpy as np
import os

def prepare_simulated_data():
    base_path = "/Users/samuelskolnick/MLEngineer/Datasets/Titanic"
    labels_df = pd.read_csv(os.path.join(base_path, "titanic_test_20_labels.csv"))
    y_true = labels_df.iloc[:, -1].values
    
    # 1. Baseline Agent (Low accuracy)
    # Just predicts majority class (died = 0)
    baseline_preds = np.zeros_like(y_true)
    pd.DataFrame({"prediction": baseline_preds}).to_csv("demo_baseline_preds.csv", index=False)
    
    # 2. Optimized Agent (High accuracy)
    # Matches y_true but with a few errors
    optimized_preds = y_true.copy()
    # Flip exactly 2 values to make it 90% accurate (if len is 20)
    optimized_preds[0] = 1 - optimized_preds[0]
    optimized_preds[5] = 1 - optimized_preds[5]
    pd.DataFrame({"prediction": optimized_preds}).to_csv("demo_optimized_preds.csv", index=False)
    
    # 3. Hallucinating Agent (Random/Garbage)
    garbage_preds = np.random.choice([0, 1], size=len(y_true))
    pd.DataFrame({"prediction": garbage_preds}).to_csv("demo_garbage_preds.csv", index=False)

    print("Simulated prediction files created.")

if __name__ == "__main__":
    prepare_simulated_data()
