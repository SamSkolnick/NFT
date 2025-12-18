# How to Set Up Agents with Different Datasets

The framework is designed to be **dataset-agnostic**. You can evaluate any CSV-based machine learning task by following these steps.

---

## 1. Prepare Your Data
The easiest way to set up a new dataset is using the provided wizard script.

### Using `setup_challenge.py`
Run the following command:
```bash
python setup_challenge.py
```
**What this script does:**
1.  **Loads your CSV**: Analyzes columns and rows.
2.  **Suggests Target**: Asks you which column is the target (label).
3.  **Splits Data**: Automatically creates a `train.csv` (features + labels) and a `test.csv` (features only) to maintain a fair evaluation.
4.  **Generates Labels**: Saves the true labels in `test_labels.csv` for the Green Agent to use during scoring.
5.  **Creates Config**: Generates a JSON file in `configs/` (e.g., `configs/mydata_config.json`).

---

## 2. Configure the Green Agent
The **Green Agent** (Evaluator) uses the generated config to know where the data is and how to score it.

**Example Config (`configs/spam_config.json`):**
```json
{
  "challenge_name": "Spam Detection",
  "train_data_path": "data/spam/train.csv",
  "test_data_path": "data/spam/test.csv",
  "test_labels": "data/spam/test_labels.csv",
  "metrics": ["accuracy", "f1_score"]
}
```

---

## 3. Launch the Agents
To run the evaluation on your specific dataset:

1.  **Set the Task environment variable**:
    ```bash
    export TASK_CONFIG=configs/mydata_config.json
    ```

2.  **Start the Green Agent**:
    ```bash
    python manage_agents.py start-green
    ```

3.  **Start the Solver Agent**:
    ```bash
    python manage_agents.py start-solver
    ```

---

## 4. How it Works (Under the Hood)
1.  **Request**: The Green Agent reads the `train_data_path` and `test_data_path` from the config.
2.  **Payload**: It sends these paths to the **Solver Agent** via an A2A message.
3.  **Training**: The Solver Agent (using `train.py`) loads the `train.csv`, researches best practices, generates code, and trains the model.
4.  **Prediction**: The Solver makes predictions on `test.csv` and returns them as an artifact (`predictions.csv`).
5.  **Scoring**: The Green Agent compares these predictions against the hidden `test_labels.csv` and reports the final score.
