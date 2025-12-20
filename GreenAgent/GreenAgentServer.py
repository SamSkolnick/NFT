import asyncio
import copy
import json
import logging
import os
from typing import Any, Dict, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import (
    new_agent_text_message,
    new_data_artifact,
    new_text_artifact,
)

from GreenAgent import GreenAgent

import uvicorn
from starlette.requests import Request
from starlette.responses import Response

from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


class GreenAgentExecutor(AgentExecutor):
    """A2A executor wrapper around the GreenAgent evaluator."""

    MAX_LOG_CHARACTERS = 10_000

    def __init__(self, task_config: Dict[str, Any]):
        self._task_config = copy.deepcopy(task_config)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        try:
            submission = self._extract_submission(context)
        except ValueError as exc:
            await self._send_status(
                context,
                event_queue,
                TaskState.failed,
                str(exc),
                final=True,
            )
            return

        await self._send_status(
            context,
            event_queue,
            TaskState.working,
            "Connecting to remote Solver and running assessment.",
        )

        agent = GreenAgent(copy.deepcopy(self._task_config))
        try:
            result = await asyncio.to_thread(agent.evaluate, submission)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("GreenAgent evaluation failed.")
            await self._send_status(
                context,
                event_queue,
                TaskState.failed,
                f"Evaluation error: {exc}",
                final=True,
            )
            return

        await self._publish_results(context, event_queue, result)
        await self._send_status(
            context,
            event_queue,
            TaskState.completed,
            "Evaluation complete.",
            final=True,
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:  # pragma: no cover - cancel not currently supported
        await self._send_status(
            context,
            event_queue,
            TaskState.rejected,
            "Cancellation is not supported for this agent.",
            final=True,
        )

    def _extract_submission(self, context: RequestContext) -> Dict[str, Any]:
        import re
        metadata = context.metadata or {}
        payload: Any = metadata.get("submission")
        if isinstance(payload, dict):
            submission = payload
        else:
            text = payload if isinstance(payload, str) else context.get_user_input()
            if not text.strip():
                raise ValueError(
                    "Submission payload required. Provide JSON with docker_image and research_artifacts."
                )
            
            # Check if AgentBeats format (XML-like with agent URL)
            if "<white_agent_url>" in text:
                match = re.search(r'<white_agent_url>\s*(https?://[^\s<]+)', text)
                if match:
                    agent_url = match.group(1).strip()
                    logger.info(f"Extracted agent URL from AgentBeats format: {agent_url}")
                    return {
                        "agent_url": agent_url,
                    }
                else:
                    raise ValueError("Could not extract agent URL from AgentBeats format")
            
            # Try JSON parsing
            try:
                submission = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid submission JSON: {exc}") from exc
            if not isinstance(submission, dict):
                raise ValueError("Submission payload must be a JSON object.")

        # Validate based on submission type
        if "agent_url" in submission:
            # Remote agent submission - only needs agent_url
            return submission
        
        # Default submission type
        return submission


    async def _publish_results(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        results: Dict[str, Any],
    ) -> None:
        execution = results.get("execution", {})
        summary = {
            "performance": results.get("performance"),
            "execution": {
                "success": execution.get("success"),
                "predictions": execution.get("predictions"),
                "time_seconds": execution.get("time_seconds"),
            },
        }

        await self._enqueue_artifact(
            event_queue,
            TaskArtifactUpdateEvent(
                context_id=self._context_id(context),
                task_id=self._task_id(context),
                artifact=new_data_artifact(
                    name="evaluation_summary",
                    data=summary,
                    description="Key results from the Green Agent evaluation.",
                ),
                last_chunk=True,
            ),
        )

        logs = execution.get("logs")
        if isinstance(logs, str) and logs:
            display_logs = logs
            if len(logs) > self.MAX_LOG_CHARACTERS:
                display_logs = (
                    f"{logs[: self.MAX_LOG_CHARACTERS]}\n... [truncated {len(logs) - self.MAX_LOG_CHARACTERS} characters]"
                )
            await self._enqueue_artifact(
                event_queue,
                TaskArtifactUpdateEvent(
                    context_id=self._context_id(context),
                    task_id=self._task_id(context),
                    artifact=new_text_artifact(
                        name="container_logs",
                        text=display_logs,
                        description="Stdout/stderr captured from the evaluation run.",
                    ),
                    last_chunk=True,
                ),
            )

    async def _send_status(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        state: TaskState,
        message: Optional[str] = None,
        final: bool = False,
    ) -> None:
        status = TaskStatusUpdateEvent(
            context_id=self._context_id(context),
            task_id=self._task_id(context),
            status=TaskStatus(
                state=state,
                message=new_agent_text_message(message) if message else None,
            ),
            final=final,
        )
        await event_queue.enqueue_event(status)

    async def _enqueue_artifact(
        self,
        event_queue: EventQueue,
        event: TaskArtifactUpdateEvent,
    ) -> None:
        await event_queue.enqueue_event(event)

    @staticmethod
    def _context_id(context: RequestContext) -> str:
        if context.context_id is None:
            raise RuntimeError("Request context missing context_id.")
        return context.context_id

    @staticmethod
    def _task_id(context: RequestContext) -> str:
        if context.task_id is None:
            raise RuntimeError("Request context missing task_id.")
        return context.task_id


def create_green_agent_app(
    task_config: Dict[str, Any],
    *,
    public_url: Optional[str] = None,
    agent_name: str = "Green Agent Evaluator",
    agent_description: Optional[str] = None,
    extended_agent_card: Optional[AgentCard] = None,
) -> A2AStarletteApplication:
    """Build an A2A application exposing the Green Agent as a service."""
    # Priority: Explicit Arg > GREEN_AGENT_PUBLIC_URL > AGENT_URL (Controller) > Default
    url = public_url or os.environ.get("GREEN_AGENT_PUBLIC_URL") or os.environ.get("AGENT_URL") or "http://localhost:8000"
    skill_example = json.dumps(
        {
            "agent_url": "http://solver-agent.com",
        },
        indent=2,
    )

    skill = AgentSkill(
        id="evaluate_solver",
        name="Evaluate Solver Agent",
        description="Connects to a remote Solver Agent and sends the training/validation datasets as A2A artifacts. Scores the solver based on its returned predictions.",
        tags=["evaluation", "benchmark", "nlp"],
        input_schema={
            "type": "object",
            "properties": {
                "agent_url": {"type": "string"},
            }
        }
    )

    capabilities = AgentCapabilities(streaming=True)
    card = AgentCard(
        id=f"green-evaluator-{hash(url) % 10000:04d}",
        name=agent_name,
        description=agent_description
        or "Evaluates Solver Agents by connecting to them and scoring their model predictions.",
        url=url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )

    executor = GreenAgentExecutor(task_config)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
        extended_agent_card=extended_agent_card,
    )

# Create the app instance for Uvicorn/AgentBeats
# This allows 'uvicorn GreenAgentServer:app' to work
try:
    # Load config from default location
    config_path = os.environ.get("TASK_CONFIG", "task_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Fallback default config if file missing
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = {
            "data_path": "/data",
            "test_labels": "/data/test_labels.csv",
            "constraints": {}
        }
    
    app = create_green_agent_app(
        task_config=config,
        public_url=os.environ.get("AGENT_URL") or os.environ.get("PUBLIC_URL"),
    ).build()
except Exception as e:
    logger.error(f"Failed to initialize Green Agent app: {e}")
    raise


# --- TAU BENCH INTEGRATION ---

# --- TAU BENCH INTEGRATION MOVED TO TCP SERVER ---
from TauBenchMCPServer import create_mcp_app
mcp_server = create_mcp_app()
app.mount("/mcp", mcp_server.sse_app)

# Host the data directory as static files for remote readiness
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")


if __name__ == "__main__":
    print(f"App Type: {type(app)}")
    for route in app.routes:
        print(f"Route: {route.path} {getattr(route, 'methods', '')}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
