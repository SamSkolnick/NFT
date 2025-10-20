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
     "constraints": {
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
     "eval_command": "python evaluate.py"
   }
   ```
   If `eval_command` is omitted, the server invokes `python evaluate.py` by default with environment variables (`EVAL_DATA_DIR`, `EVAL_OUTPUT_DIR`, `EVAL_PREDICTIONS_FILE`) set for data and result locations.
5. You can exercise the server with the official [A2A CLI samples](https://github.com/a2aproject/a2a-samples) or another A2A-compliant client. The agent streams status updates, emits structured evaluation summaries, and returns container logs as artifacts.
