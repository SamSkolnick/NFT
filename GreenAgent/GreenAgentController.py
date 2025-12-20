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
from fastapi.responses import RedirectResponse, Response, HTMLResponse, StreamingResponse
import uvicorn
from a2a.client import A2ACardResolver
from a2a.types import AgentCard

from a2a.types import AgentCard

from pydantic_settings import BaseSettings

class ControllerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    https_enabled: bool = False
    cloudrun_host: str | None = None
    agent_maintainer_sleep_n_seconds: float = 1.0


r"""
For the first version, by default, only create one agent
- GET /status  
  - → \# running agents, starting command  
- GET /agents  
  - → dict of \[ids → (url, internal port)\]  
- GET /agents/\<agent\_id\>  
  - → (status, logs)  
  - Status – Is\_ready is done by checking agent card every CHECK\_AGENT\_CARD\_EVERY\_N\_SECONDS, with ASSUME\_AGENT\_READY\_AFTER\_N\_SECONDS delay to announce it is ready  
- POST /agents/\<agent\_id\>/reset  
  - Will reset agent to be not\_ready  
- \[special\] \* /to\_agent/\<agent\_id\>/…   
  - Traffic reroute to agent instances
"""


settings = ControllerSettings()
app = FastAPI()


@app.get("/info", response_class=HTMLResponse)
async def get_info_page():
    """Serve the controller info frontend page"""
    try:
        html_content = files("agentbeats.frontend").joinpath("ctrl_info.html").read_text()
    except Exception:
        # Fallback if package resource not found (running locally)
        html_content = "<html><body><h1>AgentBeats Controller</h1><p>Frontend not available in patched mode.</p></body></html>"
    return html_content


@app.get("/status")
def get_status():
    with open("run.sh", "r") as f:
        starting_command = f.read().strip()
    agents_folder = os.path.join(".ab", "agents")
    maintained_agents = len(os.listdir(agents_folder))
    running_agents = 0
    for agent_id in os.listdir(agents_folder):
        agent_folder = os.path.join(agents_folder, agent_id)
        with open(os.path.join(agent_folder, "state"), "r") as f:
            state = f.read().strip()
        if state == "running":
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
    for agent_id in os.listdir(agents_folder):
        agent_folder = os.path.join(agents_folder, agent_id)
        with open(os.path.join(agent_folder, "port"), "r") as f:
            agent_port = int(f.read().strip())
        with open(os.path.join(agent_folder, "state"), "r") as f:
            state = f.read().strip()
        protocol = "https" if settings.https_enabled else "http"
        
        # Check for explicitly provided public URL (e.g. from Tunnel)
        public_url_base = os.environ.get("AGENT_URL")
        
        if public_url_base:
             # If AGENT_URL is set (e.g. https://foo.trycloudflare.com), use it.
             # Ensure no trailing slash
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


@app.api_route(
    "/to_agent/{agent_id}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_to_agent_root(agent_id: str, request: Request):
    return await proxy_to_agent(agent_id, "", request)


@app.api_route(
    "/to_agent/{agent_id}/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_to_agent(agent_id: str, full_path: str, request: Request):
    agent_folder = os.path.join(".ab", "agents", agent_id)
    with open(os.path.join(agent_folder, "port"), "r") as f:
        agent_port = int(f.read().strip())
    
    agent_url = f"http://localhost:{agent_port}/{full_path}"
    
    # Use a per-request client or shared client. For streaming, we need to be careful with closure.
    client = httpx.AsyncClient(follow_redirects=True, timeout=600)
    
    # Build the request to the internal agent
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

        # Forward headers, but exclude ones that might interfere with stream
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

# --- PATCH START ---
@app.get("/.well-known/agent-card.json")
async def get_root_agent_card(request: Request):
    """Proxy agent card request from root to the single running agent."""
    agents_dir = os.path.join(".ab", "agents")
    if not os.path.exists(agents_dir):
         return Response(status_code=404)
    # Find the active agent folder (ignore archived ones)
    agent_ids = [d for d in os.listdir(agents_dir) if not d.startswith("archived_") and os.path.isdir(os.path.join(agents_dir, d))]
    if not agent_ids:
         return Response(status_code=404)
    # Use the first found agent
    agent_id = agent_ids[0]
    return await proxy_to_agent(agent_id, ".well-known/agent-card.json", request)
# --- PATCH END ---

@app.get("/")
async def root():
    return RedirectResponse(url="/info")


def find_unoccupied_port():
    # find random unoccupied port above 10000
    while True:
        port = random.randint(10000, 60000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue


async def get_agent_card(agent_port: int):
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(
        httpx_client=httpx_client, base_url=f"http://localhost:{agent_port}"
    )
    try:
        card: AgentCard | None = await resolver.get_agent_card()
        return card
    except Exception as _:
        return None


def maintain_agent_process(agent_id: str):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    agent_folder = os.path.join(root_dir, ".ab", "agents", agent_id)
    agent_p = None
    agent_port = None
    skip_sleep = False
    while True:
        # check the agent status
        try:
            with open(os.path.join(agent_folder, "state"), "r") as f:
                state = f.read().strip()
        except FileNotFoundError:
            # Retry a few times if the state file is missing (e.g. during startup race)
            time.sleep(0.5)
            try:
                with open(os.path.join(agent_folder, "state"), "r") as f:
                    state = f.read().strip()
            except FileNotFoundError:
                print(f"Warning: State file not found for agent {agent_id}, assuming pending...")
                state = "pending"

        if state == "pending":
            os.makedirs(agent_folder, exist_ok=True)
            agent_port = find_unoccupied_port()
            port_file = os.path.join(agent_folder, "port")
            with open(port_file, "w") as f:
                f.write(str(agent_port))
            env = os.environ.copy()
            env["AGENT_PORT"] = str(agent_port)
            _protocol = "https" if settings.https_enabled else "http"
            public_url_base = os.environ.get("AGENT_URL")
            if public_url_base:
                 public_url_base = public_url_base.rstrip("/")
                 env["AGENT_URL"] = f"{public_url_base}/to_agent/{agent_id}"
            elif settings.cloudrun_host is not None:
                host = settings.cloudrun_host
                env["AGENT_URL"] = f"{_protocol}://{host}/to_agent/{agent_id}"
            else:
                host = settings.host
                port = settings.port
                env["AGENT_URL"] = f"{_protocol}://{host}:{port}/to_agent/{agent_id}"
            with (
                open(os.path.join(agent_folder, "stdout.log"), "w") as fout,
                open(os.path.join(agent_folder, "stderr.log"), "w") as ferr,
            ):
                agent_p = subprocess.Popen(
                    ["./run.sh"],
                    cwd=os.getcwd(),
                    shell=True,
                    stdout=fout,
                    stderr=ferr,
                    env=env,
                )
            with open(os.path.join(agent_folder, "state"), "w") as f:
                f.write("starting")
        elif state == "starting":
            assert agent_port is not None
            card = asyncio.run(get_agent_card(agent_port))
            if card is not None:
                with open(os.path.join(agent_folder, "agent_card"), "w") as f:
                    f.write(card.model_dump_json(indent=2))
                with open(os.path.join(agent_folder, "state"), "w") as f:
                    f.write("running")
        elif state == "running":
            assert agent_p is not None
            poll_status = agent_p.poll()
            if poll_status is None:
                pass  # still running, all good
            else:
                with open(os.path.join(agent_folder, "state"), "w") as f:
                    f.write(f"finished({poll_status})")
        elif state == "reset_requested":
            assert agent_p is not None
            print("Resetting agent:", agent_id)
            agent_p.terminate()
            agent_p.wait()
            print("Agent process terminated, restarting...")
            archive_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_folder = os.path.join(".ab", "agents", f"archived_{archive_time}")
            os.makedirs(archive_folder, exist_ok=True)
            os.rename(agent_folder, os.path.join(archive_folder, agent_id))
            os.makedirs(agent_folder, exist_ok=True)
            os.rename(
                archive_folder, os.path.join(agent_folder, f"archived_{archive_time}")
            )
            with open(os.path.join(agent_folder, "state"), "w") as f:
                f.write("pending")
            skip_sleep = True
        elif state.startswith("finished"):
            pass  # do nothing, agent has finished
        else:
            raise ValueError(f"Unknown state: {state}")
        if not skip_sleep:
            time.sleep(settings.agent_maintainer_sleep_n_seconds)
        skip_sleep = False


def main():
    # step 1: scan the folder for `run.sh`
    if not os.path.exists("run.sh"):
        raise FileNotFoundError("run.sh not found")
    # step 2: maintain the agent stub files in the folder `.ab/agents`
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ab_dir = os.path.join(root_dir, ".ab")
    agents_dir = os.path.join(ab_dir, "agents")
    
    os.makedirs(ab_dir, exist_ok=True)
    if os.path.exists(agents_dir):
        print("Found existing .ab/agents file from previous runs, cleaning up...")
        # remove the content in the folder
        shutil.rmtree(agents_dir)
    os.makedirs(agents_dir, exist_ok=True)
    # for now, only create one agent
    agent_id = uuid.uuid4().hex
    agent_folder = os.path.join(agents_dir, agent_id)
    os.makedirs(agent_folder, exist_ok=True)
    with open(os.path.join(agent_folder, "state"), "w") as f:
        f.write("pending")
    # step 3: start the agent process
    p = Process(target=maintain_agent_process, args=(agent_id,))
    p.start()
    # step 4: start the controller server
    uvicorn.run(app, host=settings.host, port=settings.port)

if __name__ == "__main__":
    print("DEBUG: Registered Routes:")
    for route in app.routes:
        print(f" - {route.path}")
    main()
