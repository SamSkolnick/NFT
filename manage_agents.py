import argparse
import subprocess
import os
import signal
import sys
import time
import json
import uuid
import httpx
import asyncio
from a2a.client import ClientFactory, ClientConfig
from a2a.types import AgentCard, Message, Role, TextPart, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

# Configuration
GREEN_AGENT_PORT = 8000
SOLVER_AGENT_PORT = 8005

def run_process(command, cwd, name, port=None):
    print(f"[{name}] Starting...")
    if port:
        print(f"[{name}] Port: {port}")
        
    env = os.environ.copy()
    if port:
        env["PORT"] = str(port)
        env["AGENT_URL"] = f"http://127.0.0.1:{port}"
        
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        preexec_fn=os.setsid
    )
    return proc

def wait_for_agent(port, timeout=30):
    url = f"http://localhost:{port}/.well-known/agent-card.json"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url)
            if resp.status_code == 200:
                print(f"✓ Agent on port {port} is ready!")
                return True
        except:
            pass
        time.sleep(1)
    return False

def start_green():
    return run_process(["python", "GreenAgentController.py"], "GreenAgent", "Green Agent", GREEN_AGENT_PORT)

def start_solver():
    return run_process(["python", "SolverController.py"], "SolverAgent", "Solver Agent", SOLVER_AGENT_PORT)

def stop_all():
    # Kill ports
    print("Stopping all agents...")
    subprocess.run(f"lsof -t -i:{GREEN_AGENT_PORT} -i:{SOLVER_AGENT_PORT} | xargs kill -9", shell=True)
    print("✓ Agents stopped.")

async def run_a2a_demo():
    print("\n=== Running A2A Demo (Remote Solver) ===\n")
    stop_all()
    
    green_proc = start_green()
    solver_proc = start_solver()
    
    try:
        if not wait_for_agent(GREEN_AGENT_PORT) or not wait_for_agent(SOLVER_AGENT_PORT):
             print("Failed to start agents.")
             return

        # Prepare Submission
        submission_data = {
            "agent_url": f"http://localhost:{SOLVER_AGENT_PORT}",
            "research_artifacts": "/path/to/research", 
            "docker_image": "placeholder"
        }
        
        print("\n--- Submitting A2A Request ---")
        
        # Connect to Green Agent
        config = ClientConfig()
        factory = ClientFactory(config=config)
        
        async with httpx.AsyncClient() as http:
             card_data = (await http.get(f"http://localhost:{GREEN_AGENT_PORT}/.well-known/agent-card.json")).json()
             if "0.0.0.0" in card_data.get("url", ""):
                 card_data["url"] = card_data["url"].replace("0.0.0.0", "127.0.0.1")
             agent_card = AgentCard(**card_data)
        
        client = factory.create(agent_card)
        
        submission_json = json.dumps(submission_data)
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text=submission_json)],
            context_id=f"demo_a2a_{uuid.uuid4().hex}",
            metadata={"submission": submission_data}
        )
        
        async for item in client.send_message(request=message):
             if isinstance(item, tuple):
                 task, event = item
                 if event:
                     if isinstance(event, TaskStatusUpdateEvent):
                         msg = ""
                         if event.status and event.status.message and event.status.message.parts:
                              part = event.status.message.parts[0]
                              p = part.root if hasattr(part, 'root') else part
                              if hasattr(p, 'text'): msg = p.text
                         if msg:
                             print(f"[GREEN STATUS] {msg}")
                     
                     elif isinstance(event, TaskArtifactUpdateEvent):
                         if event.artifact.name == "evaluation_summary":
                             print("\n✓ A2A Evaluation Complete!")
                             data = None
                             if event.artifact.parts:
                                  part = event.artifact.parts[0]
                                  p = part.root if hasattr(part, 'root') else part
                                  if hasattr(p, 'data'): data = p.data
                             
                             if data:
                                 print(json.dumps(data, indent=2))
    finally:
        stop_all()

async def submit_task(green_url, solver_url):
    print(f"\n=== Submitting Task to Green Agent ({green_url}) ===")
    print(f"Solver URL: {solver_url}")
    
    # 1. Connect to Green Agent
    config = ClientConfig()
    factory = ClientFactory(config=config)
    
    try:
        async with httpx.AsyncClient() as http:
             resp = await http.get(f"{green_url.rstrip('/')}/.well-known/agent-card.json")
             if resp.status_code != 200:
                 print(f"Error: Failed to fetch card from {green_url}. Status: {resp.status_code}")
                 return
             card_data = resp.json()
             # Normalize localhost IP if needed
             if "0.0.0.0" in card_data.get("url", ""):
                 card_data["url"] = card_data["url"].replace("0.0.0.0", "127.0.0.1")
             agent_card = AgentCard(**card_data)
        
        client = factory.create(agent_card)
        
        # 2. Prepare Submission
        submission_data = {
            "agent_url": solver_url,
            "research_artifacts": "/path/to/research", 
            "docker_image": "placeholder"
        }
        submission_json = json.dumps(submission_data)
        
        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text=submission_json)],
            context_id=f"manual_sub_{uuid.uuid4().hex}",
            metadata={"submission": submission_data}
        )
        
        print("\nSending request...")
        async for item in client.send_message(request=message):
             if isinstance(item, tuple):
                 task, event = item
                 if event:
                     if isinstance(event, TaskStatusUpdateEvent):
                         msg = ""
                         if event.status and event.status.message and event.status.message.parts:
                              part = event.status.message.parts[0]
                              p = part.root if hasattr(part, 'root') else part
                              if hasattr(p, 'text'): msg = p.text
                         if msg:
                             print(f"[STATUS] {msg}")
                     
                     elif isinstance(event, TaskArtifactUpdateEvent):
                         if event.artifact.name == "evaluation_summary":
                             print("\n✓ Evaluation Complete!")
                             data = None
                             if event.artifact.parts:
                                  part = event.artifact.parts[0]
                                  p = part.root if hasattr(part, 'root') else part
                                  if hasattr(p, 'data'): data = p.data
                             
                             if data:
                                 print(json.dumps(data, indent=2))
        
    except httpx.ConnectError:
        print(f"Error: Could not connect to Green Agent at {green_url}. Is it running?")
    except Exception as e:
        print(f"Error during submission: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage AgentBeats Agents")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("start-green", help="Start Green Agent (Port 8000)")
    subparsers.add_parser("start-solver", help="Start Solver Agent (Port 8005)")
    subparsers.add_parser("stop-all", help="Stop all running agents")
    subparsers.add_parser("demo-a2a", help="Run end-to-end A2A demo")
    
    submit_parser = subparsers.add_parser("submit-task", help="Submit task to running agents")
    submit_parser.add_argument("--green-url", default=f"http://localhost:{GREEN_AGENT_PORT}", help="URL of Green Agent")
    submit_parser.add_argument("--solver-url", default=f"http://localhost:{SOLVER_AGENT_PORT}", help="URL of Solver Agent")
    
    args = parser.parse_args()
    
    if args.command == "start-green":
        start_green()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            stop_all()
            
    elif args.command == "start-solver":
        start_solver()
        try:
             while True: time.sleep(1)
        except KeyboardInterrupt:
             stop_all()
             
    elif args.command == "stop-all":
        stop_all()
        
    elif args.command == "demo-a2a":
        asyncio.run(run_a2a_demo())
        
    elif args.command == "submit-task":
        asyncio.run(submit_task(args.green_url, args.solver_url))

if __name__ == "__main__":
    main()
