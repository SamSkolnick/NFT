
import subprocess
import time
import requests
import json
import os
import signal
import sys

def run_command(command, cwd=None):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True, cwd=cwd)

def build_submission_image():
    print("\n[1/4] Building Reference Submission Image...")
    # Build the 'white_agent_titanic' image as 'white_agent_demo_submission'
    run_command("docker build -t white_agent_demo_submission:latest .", cwd="white_agent_titanic")
    print("✓ Image built: white_agent_demo_submission:latest")

def start_assessor_controller():
    print("\n[2/4] Starting Assessor Agent Controller...")
    # Start the controller in the background
    # We use setsid to make it easy to kill the whole process group later
    proc = subprocess.Popen(
        ["python", "WhiteAgentController.py"], 
        cwd="white_agent_titanic",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    return proc

def wait_for_agent_ready(url, timeout=30):
    print("Waiting for agent to be ready...")
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

def run_evaluation():
    print("\n[3/4] Sending Evaluation Request...")
    # The agent card url is http://localhost:8000/.well-known/agent-card.json
    # The capabilities typically expose a 'default' endpoint or we use the A2A RPC format.
    # But for this simple HttpHandler, it usually accepts POST to root or a specific task endpoint.
    # Based on A2AStarletteApplication default handler:
    # It listens on POST /.
    
    url = "http://localhost:8000/"
    
    # A2A Task Request Format
    payload = {
        "context_id": "demo_context_1",
        "task_id": "demo_task_1",
        "metadata": {
            "submission": {
                "docker_image": "white_agent_demo_submission:latest",
                "pull_image": False # Local image
            }
        }
    }
    
    response = requests.post(url, json=payload)
    print(f"Initial Response Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    # In a real streaming scenario (SSE), we would listen for events.
    # The standard A2A python client handles this. 
    # For this simple demo, if streaming=True in capability, it might return a stream.
    # Let's see if we can just read the stream.
    
    print("\n[4/4] Streaming Execution Logs & Results:")
    try:
        cur_line = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # A2A SSE format usually starts with "data: "
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        event = json.loads(data_str)
                        event_type = event.get("type")
                        
                        if event_type == "status_update":
                            status = event.get("status", {})
                            state = status.get("state")
                            msg = status.get("message", {}).get("body", "")
                            print(f"[STATUS: {state}] {msg}")
                            
                        elif event_type == "artifact_update":
                            artifact = event.get("artifact", {})
                            name = artifact.get("name")
                            if name == "evaluation_summary":
                                print("\n--- EVALUATION RESULT ---")
                                print(json.dumps(artifact.get("data"), indent=2))
                                print("-------------------------")
                    except json.JSONDecodeError:
                        print(f"Raw: {decoded_line}")
    except Exception as e:
        print(f"Error reading stream: {e}")

def main():
    try:
        cwd = os.getcwd()
        if not cwd.endswith("MLEngineer"):
             os.chdir("..") # Ensure we are in root or close to it
        
        build_submission_image()
        
        controller_proc = start_assessor_controller()
        
        try:
            if wait_for_agent_ready("http://localhost:8000/.well-known/agent-card.json"):
                run_evaluation()
            else:
                print("Failed to start agent.")
        finally:
            print("\nShutting down controller...")
            os.killpg(os.getpgid(controller_proc.pid), signal.SIGTERM)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
