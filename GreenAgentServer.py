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
            "Running submitted docker image against evaluation data.",
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
            try:
                submission = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid submission JSON: {exc}") from exc
            if not isinstance(submission, dict):
                raise ValueError("Submission payload must be a JSON object.")

        required = {"docker_image", "research_artifacts"}
        missing = sorted(required - submission.keys())
        if missing:
            raise ValueError(f"Submission missing required fields: {', '.join(missing)}")

        submission.setdefault("storage_method", "local")
        return submission

    async def _publish_results(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        results: Dict[str, Any],
    ) -> None:
        execution = results.get("execution", {})
        summary = {
            "research": results.get("research"),
            "constraints": results.get("constraints"),
            "performance": results.get("performance"),
            "execution": {
                "success": execution.get("success"),
                "predictions": execution.get("predictions"),
                "output_dir": execution.get("output_dir"),
                "time_seconds": execution.get("time_seconds"),
                "memory_used_mb": execution.get("memory_used_mb"),
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
    url = public_url or os.environ.get("GREEN_AGENT_PUBLIC_URL", "http://localhost:8000")
    skill_example = json.dumps(
        {
            "docker_image": "white_agent:latest",
            "research_artifacts": "/path/to/research",
            "storage_method": "local",
        },
        indent=2,
    )

    skill = AgentSkill(
        id="evaluate_submission",
        name="Evaluate ML submission",
        description="Run a submitted docker image on hidden evaluation data and return performance metrics.",
        tags=["evaluation", "ml-benchmark", "docker"],
        examples=[skill_example],
    )

    capabilities = AgentCapabilities(streaming=True)
    card = AgentCard(
        name=agent_name,
        description=agent_description
        or "Evaluates ML agent submissions by running their docker images against hidden test data.",
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
