# Green Agent Evaluator - Complete User Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage Modes](#usage-modes)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

---

## Overview

The **Green Agent** is an evaluation system designed to assess AI agents (white agents) on machine learning tasks. It evaluates:
- **Execution**: Does the agent produce valid predictions?
- **Performance**: How accurate are the predictions?
- **Constraints**: Does the agent respect time/memory limits?
- **Research Quality**: Does the agent demonstrate novel, cross-domain approaches?
- **Data Transfer**: Sends training and validation datasets directly as A2A artifacts (supports files up to 10MB).

### Key Features
- **RAG-Enhanced Evaluation**: Uses ChromaDB to retrieve past evaluation contexts
- **Cross-Domain Analysis**: Identifies and rewards agents that leverage knowledge across domains
- **AgentBeats Integration**: Compatible with the AgentBeats platform for A2A orchestration
- **Interactive Testing**: Built-in CLI tool for manual testing

---

## Installation

### Prerequisites
- Python 3.10+
- ChromaDB
- Anthropic API key (optional, for research evaluation)
- Gemini API key (optional, for research evaluation)

### Setup

1. **Clone and navigate to the repository**:
```bash
cd ./MLEngineer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set environment variables**:
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...  # Optional, for research eval
export ANTHROPIC_MODEL=claude-sonnet-4-5   # Optional, default model
export OPENROUTER_API_KEY=sk-or-v1-...     # For LLM verification
export OPENROUTER_MODEL=anthropic/claude-3.5-sonnet  # Optional
```

4. **Verify installation**:
```bash
python -c "from GreenAgent import GreenAgent; print('Green Agent installed successfully')"
```

---

## Configuration

### Task Configuration

Create a `task_config.json` file:

```json
{
  "data_path": "/path/to/your/data",
  "test_labels": "/path/to/test_labels.csv",
  "constraints": {
    "max_time_seconds": 1200,
    "max_memory_mb": 4096,
    "max_cpus": 2.0
  }
}
```

**Fields**:
- `data_path`: Directory containing training/test data (will be mounted to `/data` in container)
- `test_labels`: CSV file with ground truth labels (must match test set size)
- `constraints.max_time_seconds`: Maximum execution time (default: 3600)
- `constraints.max_memory_mb`: Maximum memory in MB (default: 8192)
- `constraints.max_cpus`: CPU limit (default: 2.0, currently not enforced)

### Data Format

Your `data_path` directory should contain your dataset files (e.g., `train.csv`, `test.csv`).

The `test_labels` CSV should have at minimum:
- A unique identifier column (e.g., `PassengerId`)
- A label column (e.g., `Survived`)

---

## Usage Modes

### 1. Command-Line Interface (CLI)

**Basic evaluation**:
```bash
python -m green_agent_cli evaluate \
  --config task_config.json \
  --agent-url http://localhost:8005
```

**Full options**:
```bash
python -m green_agent_cli evaluate \
  --config task_config.json \
  --agent-url http://localhost:8005 \
  --output results.json
```

**CLI Arguments**:
- `--config`: Path to task configuration JSON
- `--docker-image`: Name of the Docker image to evaluate
- `--research-artifacts`: Path to research documentation/notes
- `--no-pull-image`: Skip pulling image (use local copy)
- `--eval-command`: Override container command (default: `python evaluate.py`)
- `--output`: Save results to JSON file

**Expected Output**:
```json
{
  "execution": {
    "success": true,
    "predictions": "/tmp/outputs_.../predictions.csv",
    "time_seconds": 0.8,
    "memory_used_mb": 0.0,
    "logs": "..."
  },
  "research": {
    "score": 0.75,
    "impact": {...},
    "novelty": {...}
  },
  "constraints": {
    "passed": true,
    "violations": []
  },
  "performance": {
    "accuracy": 0.77,
    "f1_score": 0.78
  }
}
```

### 2. AgentBeats Controller

**Start the controller**:
```bash
agentbeats run_ctrl
```

This will:
1. Start the AgentBeats controller UI (typically at `http://localhost:8080`)
2. Launch the Green Agent server using `run.sh`
3. Provide a management interface for starting/stopping the agent

**Deployment**:
For production deployment, use the `Procfile`:
```
web: agentbeats run_ctrl
```

### 3. A2A Server Mode

**Start as an A2A server**:
```bash
python -m green_agent_cli serve \
  --config task_config.json \
  --host 0.0.0.0 \
  --port 9999 \
  --public-url https://your-public-url.com
```

**A2A Server Options**:
- `--host`: Host interface (default: `0.0.0.0`)
- `--port`: Port number (default: `9999`)
- `--public-url`: Public URL advertised in agent card
- `--agent-name`: Display name (default: "Green Agent Evaluator")
- `--agent-description`: Optional description

### 4. Interactive Test Runner

**Launch interactive tests**:
```bash
python3 interactive_runner.py
```

**Features**:
- Select from predefined test scenarios
- Inspect RAG-retrieved past contexts
- Manually edit context before evaluation
- View detailed evaluation breakdowns

**Interactive Flow**:
1. Choose scenario (e.g., "Cross-Domain: NLP to Biology")
2. Review retrieved past evaluations
3. Optionally edit the context
4. Execute evaluation
5. View detailed results (scores, reasoning, metrics)

---

## Evaluation Metrics

### Execution Metrics
- **Success**: Whether the container ran without errors
- **Time**: Wall-clock execution time in seconds
- **Memory**: Peak memory usage in MB (currently not tracked)
- **Predictions**: Path to generated predictions CSV

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Harmonic mean of precision and recall

### Research Metrics (Requires `ANTHROPIC_API_KEY`)

#### 1. **Research Impact** (40% weight)
- Performance improvement over baseline
- Research-to-code traceability
- Cross-domain transfer quality
- Approach novelty

#### 2. **Cross-Domain Transfer** (30% weight)
- **Domain Distance**: How far the agent transferred knowledge (0.0-1.0)
- **Analogy Quality**: Strength of cross-domain connections (0.0-1.0)
- **Bonus**: +0.2 if distance > 0.7 and analogy > 0.8

#### 3. **Novelty** (10% weight)
- **Standard** (0.0-0.3): Random Forest, vanilla CNN
- **Common** (0.4-0.6): ResNet, LSTM
- **Creative** (0.7-0.9): Less common combinations
- **Novel** (0.9-1.0): Unexpected or original approaches

#### 4. **RAG Context**
- Past evaluations are retrieved from ChromaDB
- Used to assess novelty (penalize repeated approaches)
- Informs cross-domain validation

---

## Examples

### Example 1: Evaluate Titanic Agent

**Setup**:
```bash
# 1. Create task config
cat > task_config.json << EOF
{
  "data_path": "/Users/samuelskolnick/MLEngineer/white_agent_titanic/data",
  "test_labels": "/Users/samuelskolnick/MLEngineer/white_agent_titanic/data/dummy_test_labels.csv",
  "constraints": {
    "max_time_seconds": 1200,
    "max_memory_mb": 4096,
    "max_cpus": 2.0
  }
}
EOF

# 2. Run evaluation
python -m green_agent_cli evaluate \
  --config task_config.json \
  --agent-url http://localhost:8005
```

**Expected Output**:
```json
{
  "execution": {"success": true, "time_seconds": 0.8},
  "constraints": {"passed": true},
  "performance": {"accuracy": 0.77, "f1_score": 0.78}
}
```

### Example 2: Custom White Agent

**White Agent Requirements**:
1. **Dockerfile** that includes your ML code
2. **evaluate.py** script (or specify custom with `--eval-command`)
3. Reads data from `/data` (mounted from `data_path`)
4. Writes predictions to `/output/predictions.csv`

**Sample `evaluate.py`**:
```python
import pandas as pd
import pickle

# Load model
with open('/app/model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
df_test = pd.read_csv('/data/test.csv')

# Generate predictions
predictions = model.predict(df_test)

# Save to output
output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})
output.to_csv('/output/predictions.csv', index=False)
print("Wrote predictions to /output/predictions.csv")
```

---

## Troubleshooting

### Common Issues

#### 1. `FileNotFoundError: Config file not found`
**Solution**: Ensure `task_config.json` exists in the current directory
```bash
ls -la task_config.json  # Check if file exists
```

#### 2. `No such file or directory: '/data/test.csv'`
**Cause**: Volume mounting issue or incorrect `data_path`
**Solution**: 
- Verify `data_path` in `task_config.json` points to correct directory
- Ensure Docker Desktop has file sharing enabled for the directory
- Check that `test.csv` exists in the `data_path`

#### 3. `ValueError: Found input variables with inconsistent numbers of samples`
**Cause**: Mismatch between predictions and test labels
**Solution**: 
- Ensure `test_labels` CSV has same number of rows as test data
- Verify predictions CSV has correct format

#### 4. `ResearchEvaluator not initialized (missing API key)`
**Cause**: `ANTHROPIC_API_KEY` not set
**Solution**: 
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```
Or accept that research evaluation will be skipped (execution/performance still work)

#### 5. `TypeError: ... takes 3 positional arguments but 4 were given`
**Cause**: Outdated code after RAG integration
**Solution**: This should be fixed. If you encounter it, ensure you're using the latest version:
```bash
git pull origin main
```

#### 6. Docker image not found
**Solution**:
```bash
# Build your Docker image first
docker build -t your-agent:latest .

# Or use --no-pull-image if image is local
python -m green_agent_cli evaluate ... --no-pull-image
```

---

## API Reference

### `GreenAgent` Class

#### Constructor
```python
from GreenAgent import GreenAgent

agent = GreenAgent(task_config: dict)
```

**Parameters**:
- `task_config`: Dictionary with keys `data_path`, `test_labels`, `constraints`

#### Methods

##### `evaluate(submission: dict) -> dict`
Evaluates a white agent submission.

**Parameters**:
```python
submission = {
    "docker_image": "your-agent:latest",
    "research_artifacts": "/path/to/research",
    "storage_method": "local",  # or "s3", "gcs"
    "pull_image": True,
    "eval_command": "python evaluate.py",  # optional
    "docker_credentials": {...}  # optional
}
```

**Returns**:
```python
{
    "execution": {...},
    "research": {...},
    "constraints": {...},
    "performance": {...}
}
```

##### `run_white_agent(docker_image, task_path, command, auth_config, pull_image) -> dict`
Runs a Docker container and returns execution metadata.

##### `evaluate_performance(predictions_path, test_labels) -> dict`
Computes accuracy and F1 score.

##### `check_constraints(execution, constraints) -> dict`
Validates that execution meets constraints.

---

### `ResearchEvaluator` Class

#### Constructor
```python
from ResearchEval import ResearchEvaluator

evaluator = ResearchEvaluator(anthropic_api_key: str)
```

#### Methods

##### `evaluate_research(task, code_path, research_path, performance, past_context) -> dict`
Evaluates research quality.

**Parameters**:
- `task`: Task metadata (domain, baseline, etc.)
- `code_path`: Path to white agent code directory
- `research_path`: Path to research artifacts
- `performance`: Performance metrics dict
- `past_context`: RAG-retrieved context string

**Returns**:
```python
{
    "score": 0.75,
    "impact": {...},
    "cross_domain": {...},
    "novelty": {...},
    "originality": {...}
}
```

---

## Advanced Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Claude API for research eval | None (required for research) |
| `ANTHROPIC_MODEL` | Claude model name | `claude-sonnet-4-5` |
| `OPENROUTER_API_KEY` | OpenRouter API for LLM calls | None |
| `OPENROUTER_MODEL` | Model via OpenRouter | `anthropic/claude-3.5-sonnet` |
| `TOKENIZERS_PARALLELISM` | HuggingFace tokenizer setting | `false` (to avoid warnings) |

### ChromaDB Configuration

The Green Agent uses ChromaDB to store and retrieve past evaluations.

**Default storage**: `/Users/samuelskolnick/MLEngineer/agent_memory_db`

**Collection**: `evaluation_results`

**Query**: Retrieves top-5 similar past runs based on task description

---

## Best Practices

1. **Always use `--no-pull-image`** if testing locally to save bandwidth
2. **Set realistic constraints** to avoid false positives
3. **Include research artifacts** for better evaluation quality
4. **Use AgentBeats** for production deployments
5. **Monitor `/tmp/outputs_*`** directories if debugging predictions
6. **Check logs** in execution results for container errors

---

## Contributing

To contribute to the Green Agent:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## License

[Your license here]

---

## Support

For issues or questions:
- GitHub Issues: [Your repo URL]
- Documentation: This file
- Interactive Testing: `python3 interactive_runner.py`
