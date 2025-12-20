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
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response, HTMLResponse, StreamingResponse
import uvicorn
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from pydantic_settings import BaseSettings

class ControllerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8005  # Default to Solver Port
    https_enabled: bool = False
    cloudrun_host: str | None = None
    agent_maintainer_sleep_n_seconds: float = 1.0

settings = ControllerSettings()
app = FastAPI()

@app.get("/status")
def get_status():
    # Simplification: Assume running if process is alive
    agents_folder = os.path.join(".ab", "agents")
    # For Solver, we just check if any agent is active
    if not os.path.exists(agents_folder):
         return {"maintained_agents": 0, "running_agents": 0, "starting_command": "./run.sh"}

    maintained_agents = len(os.listdir(agents_folder))
    running_agents = 0
    for agent_id in os.listdir(agents_folder):
        agent_folder = os.path.join(agents_folder, agent_id)
        if os.path.exists(os.path.join(agent_folder, "state")):
            with open(os.path.join(agent_folder, "state"), "r") as f:
                state = f.read().strip()
            if state == "running":
                running_agents += 1
    return {
        "maintained_agents": maintained_agents,
        "running_agents": running_agents,
        "starting_command": "./run.sh",
    }

@app.get("/agents")
def list_agents():
    agents_folder = os.path.join(".ab", "agents")
    agents = {}
    if not os.path.exists(agents_folder):
        return {}

    for agent_id in os.listdir(agents_folder):
        agent_folder = os.path.join(agents_folder, agent_id)
        if not os.path.exists(os.path.join(agent_folder, "port")):
            continue
            
        with open(os.path.join(agent_folder, "port"), "r") as f:
            try:
                agent_port = int(f.read().strip())
            except ValueError:
                continue
                
        with open(os.path.join(agent_folder, "state"), "r") as f:
            state = f.read().strip()
            
        protocol = "https" if settings.https_enabled else "http"
        public_url_base = os.environ.get("AGENT_URL")
        
        if public_url_base:
             public_url_base = public_url_base.rstrip("/")
             url = f"{public_url_base}/to_agent/{agent_id}"
        elif settings.cloudrun_host is not None:
            host = settings.cloudrun_host
            url = f"{protocol}://{host}/to_agent/{agent_id}"
        else:
            host = settings.host
            port = settings.port
            url = f"{protocol}://{host}:{port}/to_agent/{agent_id}"
            
        agents[agent_id] = {
            "url": url,
            "internal_port": agent_port,
            "state": state,
        }
    return agents

@app.api_route("/to_agent/{agent_id}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_to_agent_root(agent_id: str, request: Request):
    return await proxy_to_agent(agent_id, "", request)

@app.api_route("/to_agent/{agent_id}/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_to_agent(agent_id: str, full_path: str, request: Request):
    agent_folder = os.path.join(".ab", "agents", agent_id)
    if not os.path.exists(agent_folder):
        return Response(content="Agent not found", status_code=404)
        
    with open(os.path.join(agent_folder, "port"), "r") as f:
        agent_port = int(f.read().strip())
    
    agent_url = f"http://127.0.0.1:{agent_port}/{full_path}"
    
    client = httpx.AsyncClient(follow_redirects=True, timeout=600)
    
    req = client.build_request(
        method=request.method,
        url=agent_url,
        content=await request.body(),
        headers=request.headers.raw,
        params=request.query_params,
    )

    try:
        response = await client.send(req, stream=True)
        
        async def stream_response():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        exclude_headers = ["content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"]
        headers = {k: v for k, v in response.headers.items() if k.lower() not in exclude_headers}

        return StreamingResponse(
            stream_response(),
            status_code=response.status_code,
            headers=headers,
        )
    except Exception as e:
        await client.aclose()
        return Response(content=f"Proxy error: {str(e)}", status_code=502)

@app.get("/.well-known/agent-card.json")
async def get_root_agent_card(request: Request):
    agents_dir = os.path.join(".ab", "agents")
    if not os.path.exists(agents_dir):
         return Response(status_code=404)
    
    # Filter for valid agent directories (not archives)
    agent_ids = [d for d in os.listdir(agents_dir) 
                 if not d.startswith("archived_") and os.path.isdir(os.path.join(agents_dir, d))]
    
    if not agent_ids:
         return Response(status_code=404)
    # Just take the first one
    return await proxy_to_agent(agent_ids[0], ".well-known/agent-card.json", request)


def find_unoccupied_port(preferred_port: int = None):
    if preferred_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", preferred_port))
                return preferred_port
            except OSError:
                print(f"Preferred port {preferred_port} is occupied. Finding another...")

    while True:
        port = random.randint(10000, 60000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue

@app.get("/agents/{agent_id}")
def get_agent_info(agent_id: str):
    agent_folder = os.path.join(".ab", "agents", agent_id)
    with open(os.path.join(agent_folder, "state"), "r") as f:
        state = f.read().strip()
    if os.path.exists(os.path.join(agent_folder, "stdout.log")):
        with open(os.path.join(agent_folder, "stdout.log"), "r") as f:
            stdout_log = f.read()
    else:
        stdout_log = "File not found."
    if os.path.exists(os.path.join(agent_folder, "stderr.log")):
        with open(os.path.join(agent_folder, "stderr.log"), "r") as f:
            stderr_log = f.read()
    else:
        stderr_log = "File not found."
    if os.path.exists(os.path.join(agent_folder, "agent_card")):
        with open(os.path.join(agent_folder, "agent_card"), "r") as f:
            agent_card = f.read()
    else:
        agent_card = "File not found."
    return {
        "state": state,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "agent_card": agent_card,
    }


@app.post("/agents/{agent_id}/reset")
def reset_agent(agent_id: str):
    agent_folder = os.path.join(".ab", "agents", agent_id)
    with open(os.path.join(agent_folder, "state"), "w") as f:
        f.write("reset_requested")
    return {"message": f"Agent {agent_id} reset requested."}


async def get_agent_card(agent_port: int):
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=f"http://localhost:{agent_port}")
    try:
        return await resolver.get_agent_card()
    except Exception:
        return None

def maintain_agent_process(agent_id: str):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    agent_folder = os.path.join(root_dir, ".ab", "agents", agent_id)
    agent_p = None
    agent_port = None
    
    while True:
        try:
            with open(os.path.join(agent_folder, "state"), "r") as f:
                state = f.read().strip()
        except FileNotFoundError:
            time.sleep(0.5)
            try:
                with open(os.path.join(agent_folder, "state"), "r") as f:
                    state = f.read().strip()
            except FileNotFoundError:
                 state = "pending"

        if state == "pending":
            os.makedirs(agent_folder, exist_ok=True)
            agent_port = find_unoccupied_port() # Random port for internal agent
            with open(os.path.join(agent_folder, "port"), "w") as f:
                f.write(str(agent_port))
            
            # Prepare Environment
            env = os.environ.copy()
            env["AGENT_PORT"] = str(agent_port)
            
            # Calculate Proxy URL for the child agent to know its public face
            protocol = "https" if settings.https_enabled else "http"
            public_url_base = os.environ.get("AGENT_URL")
            if public_url_base:
                 public_url_base = public_url_base.rstrip("/")
                 env["AGENT_URL"] = f"{public_url_base}/to_agent/{agent_id}"
            else:
                 env["AGENT_URL"] = f"http://127.0.0.1:{settings.port}/to_agent/{agent_id}"

            # Start Process
            # We assume 'run.sh' exists in CWD (SolverAgent/)
            with open(os.path.join(agent_folder, "stdout.log"), "w") as fout, \
                 open(os.path.join(agent_folder, "stderr.log"), "w") as ferr:
                agent_p = subprocess.Popen(
                    ["./run.sh"],
                    cwd=root_dir, # Run in SolverAgent directory
                    shell=True,
                    stdout=fout,
                    stderr=ferr,
                    env=env,
                )
            with open(os.path.join(agent_folder, "state"), "w") as f:
                f.write("starting")
                
        elif state == "starting":
            if agent_port:
                card = asyncio.run(get_agent_card(agent_port))
                if card:
                    with open(os.path.join(agent_folder, "agent_card"), "w") as f:
                        f.write(card.model_dump_json(indent=2))
                    with open(os.path.join(agent_folder, "state"), "w") as f:
                        f.write("running")
                        
        elif state == "running":
            if agent_p:
                poll = agent_p.poll()
                if poll is not None:
                    with open(os.path.join(agent_folder, "state"), "w") as f:
                        f.write(f"finished({poll})")
        
        elif state == "reset_requested":
            if agent_p:
                print("Resetting agent:", agent_id)
                agent_p.terminate()
                agent_p.wait()
            
            print("Agent process terminated, archiving and restarting...")
            archive_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_folder = os.path.join(".ab", "agents", f"archived_{archive_time}")
            os.makedirs(archive_folder, exist_ok=True)
            
            # Move current folder to archive, but since we are IN the current folder's parent,
            # we need to be careful with paths if we use shutil.move or rename.
            # GreenAgent uses os.rename(agent_folder, ...) which moves the folder.
            os.rename(agent_folder, os.path.join(archive_folder, agent_id))
            
            # Recreate agent folder and set to pending
            os.makedirs(agent_folder, exist_ok=True)
            os.rename(archive_folder, os.path.join(agent_folder, f"archived_{archive_time}"))
            
            with open(os.path.join(agent_folder, "state"), "w") as f:
                f.write("pending")
            
            # Skip sleep to handle pending state immediately
            continue

        elif state.startswith("finished"):
            pass

        time.sleep(1.0)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ab_dir = os.path.join(root_dir, ".ab")
    agents_dir = os.path.join(ab_dir, "agents")
    
    if os.path.exists(agents_dir):
        shutil.rmtree(agents_dir)
    os.makedirs(agents_dir, exist_ok=True)
    
    agent_id = uuid.uuid4().hex
    agent_folder = os.path.join(agents_dir, agent_id)
    os.makedirs(agent_folder, exist_ok=True)
    with open(os.path.join(agent_folder, "state"), "w") as f:
        f.write("pending")
        
    p = Process(target=maintain_agent_process, args=(agent_id,))
    p.start()
    
    # Dynamic Port Allocation for Controller
    final_port = find_unoccupied_port(settings.port)
    if final_port != settings.port:
        print(f"!!! WARNING: Port {settings.port} was busy. Solver Controller starting on {final_port} instead. !!!")
        settings.port = final_port

    uvicorn.run(app, host=settings.host, port=settings.port)

if __name__ == "__main__":
    main()
