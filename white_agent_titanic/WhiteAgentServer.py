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

from WhiteAgent import WhiteAgent

logger = logging.getLogger(__name__)


class WhiteAgentExecutor(AgentExecutor):
    """A2A executor wrapper around the White Agent generic evaluator."""

    MAX_LOG_CHARACTERS = 10_000

    def __init__(self, task_config: Dict[str, Any]):
        self._task_config = task_config.copy()

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
            f"Running submitted image for challenge: {self._task_config.get('challenge_name', 'Unknown')}",
        )

        agent = WhiteAgent(self._task_config)
        try:
            # Run evaluation in a separate thread to avoid blocking the async event loop
            import asyncio
            result = await asyncio.to_thread(agent.evaluate, submission)
        except Exception as exc:
            logger.exception("WhiteAgent evaluation failed.")
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
    ) -> None:
        await self._send_status(
            context,
            event_queue,
            TaskState.rejected,
            "Cancellation is not supported.",
            final=True,
        )

    def _extract_submission(self, context: RequestContext) -> Dict[str, Any]:
        metadata = context.metadata or {}
        payload: Any = metadata.get("submission")
        
        if isinstance(payload, dict):
            submission = payload
        else:
            # Try to parse from user input if not in metadata
            text = payload if isinstance(payload, str) else context.get_user_input()
            if not text.strip():
                raise ValueError(
                    "Submission payload required. Provide JSON with 'docker_image'."
                )
            try:
                submission = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid submission JSON: {exc}") from exc
            
        if not isinstance(submission, dict):
             raise ValueError("Submission payload must be a JSON object.")

        if "docker_image" not in submission:
            raise ValueError("Submission missing required field: 'docker_image'")

        return submission

    async def _publish_results(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        results: Dict[str, Any],
    ) -> None:
        summary = {
            "metrics": results.get("metrics"),
            "constraints": results.get("constraints"),
            "execution": {
                "success": results.get("execution", {}).get("success"),
                "time_seconds": results.get("execution", {}).get("time_seconds"),
                "error": results.get("execution", {}).get("error"),
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
                    description="Evaluation metrics and summary.",
                ),
                last_chunk=True,
            ),
        )

        logs = results.get("execution", {}).get("logs", "")
        if logs:
            display_logs = logs
            if len(logs) > self.MAX_LOG_CHARACTERS:
                display_logs = (
                    f"{logs[: self.MAX_LOG_CHARACTERS]}\n... [truncated]"
                )
            await self._enqueue_artifact(
                event_queue,
                TaskArtifactUpdateEvent(
                    context_id=self._context_id(context),
                    task_id=self._task_id(context),
                    artifact=new_text_artifact(
                        name="container_logs",
                        text=display_logs,
                        description="Logs from the submission container.",
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


def create_white_agent_app(
    task_config: Dict[str, Any],
    *,
    public_url: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
    extended_agent_card: Optional[AgentCard] = None,
) -> A2AStarletteApplication:
    """Build an A2A application exposing the Generic White Agent."""
    
    challenge_name = task_config.get("challenge_name", "Kaggle Challenge")
    url = public_url or os.environ.get("WHITE_AGENT_PUBLIC_URL", "http://localhost:8000")
    
    skill_example = json.dumps(
        {
            "docker_image": "my_submission:latest",
        },
        indent=2,
    )

    skill = AgentSkill(
        id="evaluate_submission",
        name=f"Evaluate {challenge_name} Submission",
        description=f"Run a docker image against the {challenge_name} evaluation set.",
        tags=["evaluation", "benchmark", "docker", challenge_name.lower().replace(" ", "-")],
        examples=[skill_example],
    )

    capabilities = AgentCapabilities(streaming=True)
    card = AgentCard(
        id=f"white-agent-{hash(url) % 10000:04d}",
        name=agent_name or f"White Agent - {challenge_name}",
        description=agent_description
        or f"Generic evaluator for {challenge_name}. Expects a docker container generating predictions.csv.",
        url=url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )

    executor = WhiteAgentExecutor(task_config)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
        extended_agent_card=extended_agent_card,
    )


# App entrypoint
try:
    config_path = os.environ.get("TASK_CONFIG", "task_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using empty default.")
        config = {}
    
    app = create_white_agent_app(
        task_config=config,
        public_url=os.environ.get("PUBLIC_URL"),
    ).build()
except Exception as e:
    logger.error(f"Failed to initialize White Agent app: {e}")
    raise
