# White Agent Demo Script & Framework Overview

This document provides the structured answers required for the White Agent demonstration video, covering the task introduction, agent framework, and demonstration results.

---

## 1. Task Introduction

### What is the task?
The primary task of the **White Agent Framework** is to act as a **Generic Machine Learning Evaluator and Solver**. 
- The **Green Agent** (Assessor) defines a challenge (e.g., "Predict if an SMS is spam") and provides a dataset.
- The **Solver Agent** (Participant) must autonomously design, train, and deploy a machine learning model to solve that specific challenge without human intervention.

### What does the environment look like?
The environment is a **distributed multi-agent system** over the **A2A (Agent-to-Agent) Protocol**.
- It consists of networked services (Servers and Controllers) that communicate via JSON-RPC.
- Public accessibility is achieved through **Cloudflare Tunnels**, allowing agents to collaborate across different machines.
- The "local" environment for the task is a file system containing datasets (CSV/TSV) and configuration metadata.

### What actions can each agent take?
- **Assessor (Green Agent)**:
    - **Register Skill**: Defines the benchmark requirements.
    - **Request Solution**: Sends task details and data paths to the Solver.
    - **Evaluate**: Compares Solver predictions against ground truth and calculates scores (Accuracy, F1, RMSE).
- **Solver Agent**:
    - **Research**: Uses the Gemini API to find best practices and pitfalls for the specific domain.
    - **Generate Code**: Dynamically writes Python code to build a Scikit-Learn pipeline.
    - **Train**: Executes the generated code on the training data.
    - **Predict**: Generates results for the held-out validation set.

---

## 2. Agent Framework

### What is the overall framework design of the white agent?
The architecture follows a **Decoupled AutoML Design**. 
- **Controller-Agent Pattern**: Each agent has a controller for lifecycle management and a server for task execution.
- **A2A Compliance**: Both agents implement standard `AgentCard` and `AgentSkill` discovery, allowing them to be plugged into the broader AgentBeats ecosystem.
- **Dynamic Execution**: The solver doesn't use a fixed model; it builds a new one for every new task.

### What is the decision making pipeline of the white agent?
1.  **Task Ingestion**: The agent parses the task description and inspects the data schema (features vs. target).
2.  **Enrichment Research**: The agent queries Gemini to identify domain-specific strategies (e.g., "Use bigrams for spam" or "Handle imbalanced tabular data with XGBoost weighting").
3.  **Code Synthesis**: The LLM generates a complete, runnable Python script containing a `build_pipeline()` function.
4.  **Auto-Training**: The agent loads this script as a module, fits the pipeline on the dataset, and performs stratified splits for validation.
5.  **Output Generation**: The agent returns predictions along with a "Research Report" justifying its architectural choices.

### What are the inputs and outputs of the agent at each step?
| Step | Input | Output |
| :--- | :--- | :--- |
| **Enrichment** | Task Description | Research Report (Concise best practices) |
| **Code Generation** | Task + Research + Data Schema | `model_uuid.py` (Python Source Code) |
| **Training** | Training Data (CSV) + Python Code | `model.pkl` (Trained Pipeline) |
| **Inference** | Validation Data (CSV) | `val_predictions` (List of labels) |

---

## 3. Demonstration

### Task Completion: Spam/Ham Classification
In the demonstration, the White Agent is tasked with classifying SMS messages as "spam" or "ham".

1.  **Step 1 (Input)**: "Classify emails as spam or ham."
2.  **Step 2 (Action)**: The agent researches NLP best practices, identifying that **Multinomial Naive Bayes** with **TF-IDF Bigrams** is optimal for fast SMS classification.
3.  **Step 3 (Output)**: It generates 25 lines of Python code using `ColumnTransformer` and `TfidfVectorizer`.
4.  **Step 4 (Execution)**: The agent trains the model in seconds, handling the imbalanced dataset (3865 ham vs 593 spam).

### Quantitative Results
On the **Spam/Ham Benchmark**, the White Agent achieved:
- **Overall Accuracy**: **98.2%**
- **Ham F1-Score**: **0.990**
- **Spam F1-Score**: **0.932**
- **Precision (Spam)**: **94.8%**

This demonstrates that the agent's "Decision Making Pipeline"—combining reasoning (research) with execution (code generation)—far outperforms static, pre-defined models.
