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

def start_tunnel(port):
    print(f"[Tunnel] Starting Cloudflare tunnel for port {port}...")
    # cloudflared logs URL to stderr
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid
    )
    
    public_url = None
    # Read stderr line by line to find URL
    # We need to do this non-blocking or with a thread usually, but here we just wait a bit
    # Actually, we need to read it continuously until we find the URL
    start = time.time()
    while time.time() - start < 20: 
        line = proc.stderr.readline()
        if not line: break
        if "trycloudflare.com" in line:
            # Extract URL
            import re
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match:
                public_url = match.group(0)
                print(f"[Tunnel] Verified URL: {public_url}")
                break
    
    if not public_url:
        print("[Tunnel] Failed to find public URL.")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return None, None
        
    return proc, public_url

def start_agent_with_tunnel(command, cwd, name, port):
    # 1. Start Tunnel first to get URL
    tunnel_proc, public_url = start_tunnel(port)
    if not public_url: return None
    
    # 2. Start Agent with AGENT_URL set
    print(f"[{name}] Starting with AGENT_URL={public_url}")
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["AGENT_URL"] = public_url
    
    agent_proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr, # Maybe redirect to file to keep console clean?
        env=env,
        preexec_fn=os.setsid
    )
    return agent_proc, tunnel_proc, public_url

def start_green(tunnel=False):
    if tunnel:
        agent, tun, url = start_agent_with_tunnel(["python", "GreenAgentController.py"], "GreenAgent", "Green Agent", GREEN_AGENT_PORT)
        return agent
    else:
        return run_process(["python", "GreenAgentController.py"], "GreenAgent", "Green Agent", GREEN_AGENT_PORT)

def start_solver(tunnel=False):
    if tunnel:
        agent, tun, url = start_agent_with_tunnel(["python", "SolverController.py"], "SolverAgent", "Solver Agent", SOLVER_AGENT_PORT)
        return agent
    else:
        return run_process(["python", "SolverController.py"], "SolverAgent", "Solver Agent", SOLVER_AGENT_PORT)

def stop_all():
    print("Stopping all agents and tunnels...")
    # Kill ports and cloudflared
    subprocess.run(f"lsof -t -i:{GREEN_AGENT_PORT} -i:{SOLVER_AGENT_PORT} | xargs kill -9", shell=True)
    subprocess.run("pkill -f cloudflared", shell=True)
    print("✓ Stopped.")

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
    
    green_parser = subparsers.add_parser("start-green", help="Start Green Agent (Port 8000)")
    green_parser.add_argument("--tunnel", action="store_true", help="Expose via Cloudflare Tunnel")
    
    solver_parser = subparsers.add_parser("start-solver", help="Start Solver Agent (Port 8005)")
    solver_parser.add_argument("--tunnel", action="store_true", help="Expose via Cloudflare Tunnel")
    
    subparsers.add_parser("stop-all", help="Stop all running agents")
    subparsers.add_parser("demo-a2a", help="Run end-to-end A2A demo")
    
    submit_parser = subparsers.add_parser("submit-task", help="Submit task to running agents")
    submit_parser.add_argument("--green-url", default=f"http://localhost:{GREEN_AGENT_PORT}", help="URL of Green Agent")
    submit_parser.add_argument("--solver-url", default=f"http://localhost:{SOLVER_AGENT_PORT}", help="URL of Solver Agent")
    
    tb_parser = subparsers.add_parser("start-taubench-solver", help="Start Tau Bench Solver Agent")
    tb_parser.add_argument("--tunnel", action="store_true", help="Expose via Cloudflare Tunnel")
    
    subparsers.add_parser("demo-taubench", help="Run Tau Bench E2E Demo (Local)")
    
    args = parser.parse_args()
    
    if args.command == "start-green":
        start_green(tunnel=args.tunnel)
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            stop_all()
            
    elif args.command == "start-solver":
        start_solver(tunnel=args.tunnel)
        try:
             while True: time.sleep(1)
        except KeyboardInterrupt:
             stop_all()
             
    elif args.command == "stop-all":
        stop_all()
        
    elif args.command == "demo-a2a":
        # Default local demo
        asyncio.run(run_a2a_demo())

    elif args.command == "demo-a2a-public":
        print("\n=== Running A2A Demo (Public Cloudflare Tunnels) ===\n")
        stop_all()
        
        # 1. Start Agents with Tunnels
        green_proc, green_tun, green_url = start_agent_with_tunnel(["python", "GreenAgentController.py"], "GreenAgent", "Green Agent", GREEN_AGENT_PORT)
        solver_proc, solver_tun, solver_url = start_agent_with_tunnel(["python", "SolverController.py"], "SolverAgent", "Solver Agent", SOLVER_AGENT_PORT)
        
        if not green_url or not solver_url:
            print("Failed to start tunnels.")
            stop_all()
            sys.exit(1)
            
        print(f"\n[Green Agent] Public URL: {green_url}")
        print(f"[Solver Agent] Public URL: {solver_url}")
        
        # 2. Submit Task
        print("\nWaiting for agents to stabilize...")
        time.sleep(10)
        
        try:
             asyncio.run(submit_task(green_url, solver_url))
        except KeyboardInterrupt:
             pass
        finally:
             stop_all()

    elif args.command == "submit-task":
        asyncio.run(submit_task(args.green_url, args.solver_url))

    elif args.command == "start-taubench-solver":
        # Launch TauBenchSolverServer with tunnel option
        if args.tunnel:
             # Just like start-solver but different file
             start_agent_with_tunnel([sys.executable, "TauBenchSolverServer.py"], "SolverAgent", "Tau Bench Solver", SOLVER_AGENT_PORT)
        else:
             run_process([sys.executable, "TauBenchSolverServer.py"], "SolverAgent", "Tau Bench Solver", SOLVER_AGENT_PORT)
        try:
             while True: time.sleep(1)
        except KeyboardInterrupt:
             stop_all()

    elif args.command == "demo-taubench":
        # Orchestrate end-to-end Tau Bench Demo
        stop_all()
        green_proc = start_green()
        # Start Tau Bench Solver locally
        solver_proc = run_process([sys.executable, "TauBenchSolverServer.py"], "SolverAgent", "Tau Bench Solver", SOLVER_AGENT_PORT)
        
        time.sleep(5) # Wait for startup
        
        # Submit Task
        print("\n=== Submitting Tau Bench Task ===")
        # We need to target the Solver's A2A endpoint. 
        # The TauBenchSolverExecutor will trigger the loop.
        # It needs to know Green URL.
        # HACK: The Solver will default to localhost:8000 if not specified, which works for this local demo.
        # We pass Solver URL as "target" and Green URL as "payload" (though payload is ignored by current implementation)
        
        asyncio.run(submit_task("http://localhost:8005", "http://localhost:8000"))

if __name__ == "__main__":
    main()
