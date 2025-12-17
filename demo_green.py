
import subprocess
import time
import requests
import json
import os
import signal
import sys
import shutil

WHITE_AGENT_DIR = "white_agent_titanic"

def run_command(command, cwd=None):
    print(f"\n[RUNNING] {command} (cwd={cwd or os.getcwd()})")
    subprocess.check_call(command, shell=True, cwd=cwd)

def step_1_train_model():
    print("\n--- [Step 1] Training White Agent Model ---")
    # Clean old model
    model_path = os.path.join(WHITE_AGENT_DIR, "model", "model.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Run training script
    run_command("python train.py", cwd=WHITE_AGENT_DIR)
    
    if os.path.exists(model_path):
        print("✓ Model successfully trained and saved.")
    else:
        raise Exception("Model training failed: model.pkl not found.")

def step_2_build_docker():
    print("\n--- [Step 2] Building White Agent Docker Container ---")
    run_command("docker build -t white_agent_demo_submission:latest .", cwd=WHITE_AGENT_DIR)
    print("✓ Docker image built successfully.")

def start_green_controller():
    print("\n--- [Step 3] Starting Green Agent Controller ---")
    # Kill any existing controller on port 8000
    try:
        subprocess.run("lsof -ti:8000 | xargs kill -9", shell=True, stderr=subprocess.DEVNULL)
    except: pass

    proc = subprocess.Popen(
        ["python", "GreenAgentController.py"], 
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    return proc

def wait_for_agent_ready(url, timeout=30):
    print("Waiting for Green Agent to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("✓ Agent is online!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    print("\nTimed out waiting for agent.")
    return False

def step_4_submit_to_green_agent(challenge_name=None):
    print(f"\n--- [Step 4] Submitting to Green Agent (Challenge: {challenge_name or 'Default'}) ---")
    
    url = "http://localhost:8000/"
    
    submission_data = {
        "docker_image": "white_agent_demo_submission:latest", 
        "research_artifacts": "/tmp/dummy_research", 
        "pull_image": False
    }
    
    if challenge_name:
        submission_data["challenge"] = challenge_name
    
    payload = {
        "context_id": "full_demo_1",
        "task_id": "full_demo_task_1",
        "metadata": {
            "submission": submission_data
        }
    }
    
    requests.post(url, json=payload)
    print("Request sent. Streaming results...")
    
    # We will assume Streamed output for successful connection
    # Re-connect to read stream (conceptually, A2A might return stream immediately or use SSE endpoint)
    # The current server implementation returns StreamingResponse immediately.
    
    with requests.post(url, json=payload, stream=True) as response:
         if response.status_code != 200:
            print(f"Error: {response.text}")
            return

         for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        event = json.loads(data_str)
                        event_type = event.get("type")
                        if event_type == "status_update":
                            msg = event.get("status", {}).get("message", {}).get("body", "")
                            print(f"[STATUS] {msg}")
                        elif event_type == "artifact_update":
                            artifact = event.get("artifact", {})
                            name = artifact.get("name")
                            if name == "evaluation_summary":
                                print("\n✓ SUCCESS: Evaluation Complete!")
                                print(json.dumps(artifact.get("data"), indent=2))
                    except: pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", help="Name of the challenge to run (e.g. titanic, spamham)", default=None)
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    args = parser.parse_args()

    try:
        # Check if we are in right dir
        if not os.path.exists("GreenAgentController.py"):
            print("Error: Please run this from the MLEngineer root directory.")
            return

        if not args.skip_train:
            step_1_train_model()
            step_2_build_docker()
        else:
            print("\n[Skipping Training & Build steps]")
        
        controller_proc = start_green_controller()
        
        try:
            if wait_for_agent_ready("http://localhost:8000/.well-known/agent-card.json"):
                step_4_submit_to_green_agent(args.challenge)
            else:
                print("Failed to start agent.")
        finally:
            print("\nShutting down controller...")
            os.killpg(os.getpgid(controller_proc.pid), signal.SIGTERM)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
