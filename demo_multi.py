
import subprocess
import time
import requests
import json
import os
import signal
import sys

def start_white_controller():
    print("\nStarting White Agent Controller...")
    try:
        subprocess.run("lsof -ti:8000 | xargs kill -9", shell=True, stderr=subprocess.DEVNULL)
    except: pass

    # Ensure we are in the right dir
    cwd = os.path.join(os.getcwd(), "white_agent_titanic")
    
    proc = subprocess.Popen(
        ["python", "WhiteAgentController.py"], 
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    return proc

def wait_for_agent(url):
    print("Waiting for agent...")
    for _ in range(30):
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("✓ Agent online")
                return True
        except: pass
        time.sleep(1)
    return False

def main():
    proc = start_white_controller()
    try:
        if wait_for_agent("http://localhost:8000/.well-known/agent-card.json"):
            resp = requests.get("http://localhost:8000/.well-known/agent-card.json")
            card = resp.json()
            print("\n--- Discovered Skills ---\n")
            for skill in card.get("skills", []):
                print(f"Skill ID: {skill['id']}")
                print(f"Name:     {skill['name']}")
                print(f"Description: {skill['description']}")
                print("-" * 20)
            
            p_titanic = [s for s in card['skills'] if 'titanic' in s['id']]
            if p_titanic:
                print("\n✓ SUCCESS: Found Titanic skill.")
            else:
                print("\nX FAILURE: Titanic skill not found.")
                
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

if __name__ == "__main__":
    main()
