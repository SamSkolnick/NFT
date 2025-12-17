import asyncio
import json
import logging
import os
import sys
import uuid
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings

# Ensure we can import from tau-bench-main
TAU_BENCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tau-bench-main"))
if TAU_BENCH_PATH not in sys.path:
    sys.path.append(TAU_BENCH_PATH)

from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.types import Action, EnvResetResponse, EnvResponse, EnvInfo, Task, RewardResult, RunConfig
from tau_bench.envs.base import Env

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState, TaskStatus, TaskStatusUpdateEvent, Message, Role, TextPart

logger = logging.getLogger(__name__)

class RemoteEnv:
    def __init__(self, session_id: str, base_url: str):
        self.session_id = session_id
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0) # Sync client for Tau Bench compatibility

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        url = f"{self.base_url}/env/{self.session_id}/reset"
        payload = {"task_index": task_index}
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return EnvResetResponse(**resp.json())

    def step(self, action: Action) -> EnvResponse:
        url = f"{self.base_url}/env/{self.session_id}/step"
        # serialized action
        payload = action.model_dump()
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return EnvResponse(**resp.json())
        
    def get_tools_info(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/env/{self.session_id}/tools"
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_wiki(self) -> str:
        url = f"{self.base_url}/env/{self.session_id}/wiki"
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.json()["wiki"]

class TauBenchSolverExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Expects task input: {"green_url": "..."} or uses GREEN_URL env var.
        """
        # HACK: Assume inputs are passed via a Task input mechanism or we default to Green URL from env
        green_controller_url = os.environ.get("GREEN_URL", "http://localhost:8000")
        
        await self._send_status(context, event_queue, TaskState.working, f"Connecting to Green Controller at {green_controller_url}...")
        
        try:
             # 0. Service Discovery: Find the actual agent URL
             async with httpx.AsyncClient() as client:
                 resp = await client.get(f"{green_controller_url}/agents")
                 resp.raise_for_status()
                 agents_map = resp.json()
                 if not agents_map:
                     raise Exception("No active Green Agents found managed by the controller.")
                 
                 # Pick the first one
                 agent_id = list(agents_map.keys())[0]
                 agent_info = agents_map[agent_id]
                 # construct full agent url (which is proxied via controller)
                 # The controller returns "url" field which usually includes /to_agent/{id}
                 green_agent_url = agent_info["url"]
                 logger.info(f"Discovered Agent {agent_id} at {green_agent_url}")

             # 1. Initialize Remote Session
             async with httpx.AsyncClient() as client:
                 init_resp = await client.post(f"{green_agent_url}/env/init", json={"env_name": "retail"})
                 init_resp.raise_for_status()
                 session_id = init_resp.json()["session_id"]
             
             # 2. Setup RemoteEnv
             remote_env = RemoteEnv(session_id, green_agent_url)
             tools_info = remote_env.get_tools_info()
             wiki = remote_env.get_wiki()
             
             # 3. Initialize Agent
             # Needs OPENAI_API_KEY
             agent = ToolCallingAgent(
                 tools_info=tools_info,
                 wiki=wiki,
                 model="gpt-4o",
                 provider="openai",
                 temperature=0.0
             )
             
             # 4. Run Solve Loop (Sync call in async function? Use to_thread)
             await self._send_status(context, event_queue, TaskState.working, f"Solving task (Session {session_id})...")
             
             # Tau Bench agents use 'env.reset()' inside solve.
             # We need to make sure 'RemoteEnv' quacks like 'Env'. It does (reset, step).
             
             res = await asyncio.to_thread(agent.solve, env=remote_env, task_index=0)
             
             await self._send_status(context, event_queue, TaskState.completed, f"Solved! Reward: {res.reward}", final=True)
             
        except Exception as e:
            logger.exception("Tau Bench Execution Failed")
            await self._send_status(context, event_queue, TaskState.failed, f"Error: {e}", final=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
        
    async def _send_status(self, context, event_queue, state, message, final=False):
        status_msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text=message)],
        )
        status = TaskStatusUpdateEvent(
            context_id=context.context_id or "default",
            task_id=context.task_id or "default",
            status=TaskStatus(state=state, message=status_msg),
            final=final
        )
        await event_queue.enqueue_event(status)

def create_app():
    skill = AgentSkill(
        id="tau_bench_solver",
        name="Tau Bench Solver",
        description="Solves Tau Bench tasks using a remote environment.",
        tags=["tau-bench", "solver"],
        inputs={"description": "Task request"},
        outputs={"description": "Assessment Result"},
    )
    capabilities = AgentCapabilities(streaming=True)
    card = AgentCard(
        id=f"tau-solver-{uuid.uuid4().hex[:8]}",
        name="Tau Bench Solver",
        description="Tau Bench Agent",
        version="0.0.1",
        capabilities=capabilities,
        skills=[skill],
        url=os.environ.get("AGENT_URL", "http://0.0.0.0:8005"),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )
    
    executor = TauBenchSolverExecutor()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build()

app = create_app()

from starlette.responses import JSONResponse
async def health_check(request):
    return JSONResponse({"status": "running", "agent": "TauBenchSolver"})
app.add_route("/status", health_check, methods=["GET"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
