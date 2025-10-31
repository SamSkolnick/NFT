This is the planning so far

Green Agent Components:
    1. Database with Chromadb
    2. ResearchEval
    3. Run input model on test data
    4. Output results 
    

Planed workflow:

1. Initate Green Agent with task:
    Intake:
    1. Task definition
    2. Constraints
    3. Holdout data for evaluation with labels
2. Input docker container with model and defined entyrpoint where you input the validation data, and method for getting predictions, I guess it also eneeds a way to iddentify when done. 
    a. White agents docker container must have a script called evaluate.py which has the following useage:
        Oh we could imput the path to da as an enviroment vairbael when runnign and then we need to get and evaluate 
        python solve.py /path/to/eval/data
3. While running on validation data, evaluate research
4. Combine for final scores

### Persistent Memory (ChromaDB)
- The Green Agent and research components share a Chroma database located at `./agent_memory_db` by default; override with `AGENT_MEMORY_DB_PATH`.
- Install dependencies with `pip install -r requirements.txt` (ensures `chromadb` is present) before starting the server; otherwise the agent aborts with guidance.
- Evaluation summaries are stored in the `evaluation_results` collection and mirrored to the shared `research_and_development` collection for downstream retrieval.

## Running the Green Agent as an A2A server

1. Install dependencies with Python 3.10 (the server re-execs into python3.10 when available):
   ```
   python3.10 -m pip install -r requirements.txt
   ```
2. Prepare a task configuration (either JSON file or CLI flags). A minimal JSON file:
   ```json
   {
     "data_path": "./tasks/sample_task",
     "test_labels": "./tasks/sample_task/test_labels.csv",
     "constraints": { # optional
       "max_time_seconds": 3600,
       "max_memory_mb": 8192,
       "max_cpus": 2.0
     }
   }
   ```
3. Start the server (if you run `python RunEval.py ...` from an older interpreter, it will automatically restart under python3.10 when installed):
   ```
   python RunEval.py --config task_config.json --port 9999 --public-url http://localhost:9999
   ```
4. The exposed A2A skill expects a JSON payload in the message body:
   ```json
   {
     "docker_image": "white_agent:latest",
     "research_artifacts": "/path/to/research",
     "storage_method": "local",
     "eval_command": "python evaluate.py",
     "pull_image": true,
     "docker_credentials": {
       "username": "my-docker-user",
       "password": "my-docker-password",
       "registry": "https://index.docker.io/v1/"
     }
   }
   ```
   If `eval_command` is omitted, the server invokes `python evaluate.py` by default with environment variables (`EVAL_DATA_DIR`, `EVAL_OUTPUT_DIR`, `EVAL_PREDICTIONS_FILE`) set for data and result locations.
   - `pull_image` (default `true`) controls whether the Green Agent tries to pull the image before execution. The agent skips the pull when the image already exists locally; set it to `false` to always use the local cache.
   - `docker_credentials` (optional) forwards Docker registry credentials. You can also use `docker_auth` or `registry_auth` fields with the same shape; the evaluator filters for `username`, `password`, `email`, `registry`, and `identitytoken`.
5. You can exercise the server with the official [A2A CLI samples](https://github.com/a2aproject/a2a-samples) or another A2A-compliant client. The agent streams status updates, emits structured evaluation summaries, and returns container logs as artifacts.

## Sample White Agent (Titanic)

- Reference implementation lives in `white_agent_titanic/`.
- Train the bundled logistic-regression model with `python white_agent_titanic/train.py`.
- Build the container via `docker build -t titanic-white-agent white_agent_titanic`.
- Submit using the A2A client (example payload):
  ```json
  {
    "docker_image": "titanic-white-agent:latest",
    "research_artifacts": "white_agent_titanic/research",
    "storage_method": "local",
    "pull_image": false
  }
  ```
  - The container reads the evaluation CSV from `/data/test/test.csv` (or `/data/test.csv`) and writes `/output/predictions.csv`, matching the Green Agent contract.
