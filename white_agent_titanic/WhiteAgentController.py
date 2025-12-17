import os
import uuid
import time
import datetime
import socket
import random
import asyncio
import shutil
from multiprocessing import Process
import subprocess
import httpx
from importlib.resources import files
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response, HTMLResponse
import uvicorn
from a2a.client import A2ACardResolver
from a2a.types import AgentCard

from pydantic_settings import BaseSettings

class ControllerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    https_enabled: bool = False
    cloudrun_host: str | None = None
    agent_maintainer_sleep_n_seconds: float = 1.0

settings = ControllerSettings()
app = FastAPI()

# --- Identical Controller Logic as Green Agent ---
# We can import it if it were a lib, but for standalone robustness we duplicate.
# This controller manages the lifecycle of `run.sh` -> `WhiteAgentServer:app`.

@app.get("/info", response_class=HTMLResponse)
async def get_info_page():
    try:
        html_content = files("agentbeats.frontend").joinpath("ctrl_info.html").read_text()
    except Exception:
        html_content = "<html><body><h1>AgentBeats Controller</h1><p>Frontend not available.</p></body></html>"
    return html_content

@app.get("/status")
def get_status():
    with open("run.sh", "r") as f:
        starting_command = f.read().strip()
    agents_folder = os.path.join(".ab", "agents")
    maintained_agents = len(os.listdir(agents_folder)) if os.path.exists(agents_folder) else 0
    running_agents = 0
    if os.path.exists(agents_folder):
        for agent_id in os.listdir(agents_folder):
            agent_folder = os.path.join(agents_folder, agent_id)
            if os.path.exists(os.path.join(agent_folder, "state")):
                with open(os.path.join(agent_folder, "state"), "r") as f:
                    if f.read().strip() == "running":
                        running_agents += 1
    return {
        "maintained_agents": maintained_agents,
        "running_agents": running_agents,
        "starting_command": starting_command,
    }

@app.get("/agents")
def list_agents():
    agents_folder = os.path.join(".ab", "agents")
    agents = {}
    if os.path.exists(agents_folder):
        for agent_id in os.listdir(agents_folder):
            agent_folder = os.path.join(agents_folder, agent_id)
            with open(os.path.join(agent_folder, "port"), "r") as f:
                agent_port = int(f.read().strip())
            with open(os.path.join(agent_folder, "state"), "r") as f:
                state = f.read().strip()
            
            protocol = "https" if settings.https_enabled else "http"
            host = settings.cloudrun_host if settings.cloudrun_host else settings.host
            port_str = f":{settings.port}" if not settings.cloudrun_host else ""
            url = f"{protocol}://{host}{port_str}/to_agent/{agent_id}"

            agents[agent_id] = {
                "url": url,
                "internal_port": agent_port,
                "state": state,
            }
    return agents

@app.api_route("/to_agent/{agent_id}/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_to_agent(agent_id: str, full_path: str, request: Request):
    agent_folder = os.path.join(".ab", "agents", agent_id)
    with open(os.path.join(agent_folder, "port"), "r") as f:
        agent_port = int(f.read().strip())
    agent_url = f"http://localhost:{agent_port}/{full_path}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        response = await client.request(
            method=request.method,
            url=agent_url,
            content=await request.body(),
            headers=request.headers,
            params=request.query_params,
        )
        return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

@app.get("/.well-known/agent-card.json")
async def get_root_agent_card(request: Request):
    agents_dir = os.path.join(".ab", "agents")
    if not os.path.exists(agents_dir): return Response(status_code=404)
    agent_ids = [d for d in os.listdir(agents_dir) if not d.startswith("archived_")]
    if not agent_ids: return Response(status_code=404)
    return await proxy_to_agent(agent_ids[0], ".well-known/agent-card.json", request)

def find_unoccupied_port():
    while True:
        port = random.randint(10000, 60000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError: continue

async def get_agent_card(agent_port: int) -> AgentCard | None:
    try:
        async with httpx.AsyncClient() as client:
            resolver = A2ACardResolver(httpx_client=client, base_url=f"http://localhost:{agent_port}")
            return await resolver.get_agent_card()
    except Exception: return None

def maintain_agent_process(agent_id: str):
    agent_folder = os.path.join(".ab/agents", agent_id)
    agent_p = None
    agent_port = None
    
    while True:
        try:
            with open(os.path.join(agent_folder, "state"), "r") as f:
                state = f.read().strip()
        except FileNotFoundError:
            state = "pending"

        if state == "pending":
            agent_port = find_unoccupied_port()
            with open(os.path.join(agent_folder, "port"), "w") as f: f.write(str(agent_port))
            
            env = os.environ.copy()
            env["AGENT_PORT"] = str(agent_port)
            
            # Start run.sh (which runs uvicorn WhiteAgentServer:app)
            with open(os.path.join(agent_folder, "stdout.log"), "w") as fout, open(os.path.join(agent_folder, "stderr.log"), "w") as ferr:
                agent_p = subprocess.Popen(["./run.sh"], cwd=os.getcwd(), shell=True, stdout=fout, stderr=ferr, env=env)
            
            with open(os.path.join(agent_folder, "state"), "w") as f: f.write("starting")
            
        elif state == "starting":
            card = asyncio.run(get_agent_card(agent_port))
            if card:
                with open(os.path.join(agent_folder, "agent_card"), "w") as f: f.write(card.model_dump_json(indent=2))
                with open(os.path.join(agent_folder, "state"), "w") as f: f.write("running")
                
        elif state == "running":
            if agent_p and agent_p.poll() is not None:
                with open(os.path.join(agent_folder, "state"), "w") as f: f.write(f"finished({agent_p.poll()})")
                
        elif state == "reset_requested":
            if agent_p: agent_p.terminate(); agent_p.wait()
            # Archive logic omitted for brevity, just restarting state
            with open(os.path.join(agent_folder, "state"), "w") as f: f.write("pending")

        time.sleep(1)

def main():
    if not os.path.exists("run.sh"):
        # Auto-create run.sh if missing
        with open("run.sh", "w") as f:
            f.write("#!/bin/bash\nuvicorn WhiteAgentServer:app --host 0.0.0.0 --port $AGENT_PORT\n")
        os.chmod("run.sh", 0o755)

    os.makedirs(".ab", exist_ok=True)
    if os.path.exists(".ab/agents"): shutil.rmtree(".ab/agents")
    os.makedirs(".ab/agents", exist_ok=True)
    
    agent_id = uuid.uuid4().hex
    agent_folder = os.path.join(".ab/agents", agent_id)
    os.makedirs(agent_folder, exist_ok=True)
    
    with open(os.path.join(agent_folder, "state"), "w") as f: f.write("pending")
    
    p = Process(target=maintain_agent_process, args=(agent_id,))
    p.start()
    uvicorn.run(app, host=settings.host, port=settings.port)

if __name__ == "__main__":
    main()
