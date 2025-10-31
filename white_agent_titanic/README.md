# Sample Titanic White Agent

This directory provides a minimal reference submission that the Green Agent can evaluate. It trains a logistic regression model on the Titanic dataset and packages the prediction entrypoint in a Docker image compatible with the evaluation harness.

## Contents

- `data/`: pre-split data (`train.csv`, `test.csv`) created from the Kaggle Titanic dataset.
- `model/`: serialized sklearn pipeline (`model.pkl`) produced by `train.py`.
- `train.py`: offline training script.
- `evaluate.py`: container entrypoint that reads the hidden test set, runs inference, and writes `predictions.csv`.
- `Dockerfile`: builds a runnable image for the white agent.
- `requirements.txt`: runtime dependencies for the container.
- `research/research.md`: brief notes documenting the modeling approach.

## Training the model

```bash
python train.py
```

The script will print a validation report and write the trained pipeline to `model/model.pkl`. Re-run it whenever you update the training data or features.

## Building the Docker image

```bash
docker build -t titanic-white-agent:latest .
```

## Local evaluation dry-run

To mimic the Green Agent’s execution environment, mount the evaluation data and an output directory:

```bash
mkdir -p /tmp/titanic_output
docker run --rm \
  -e EVAL_DATA_DIR=/data \
  -e EVAL_OUTPUT_DIR=/output \
  -v "$(pwd)/data":/data:ro \
  -v /tmp/titanic_output:/output \
  titanic-white-agent:latest

cat /tmp/titanic_output/predictions.csv | head
```

The container writes `predictions.csv` (with a required `prediction` column) to the specified output directory, matching the expectations of the Green Agent.
