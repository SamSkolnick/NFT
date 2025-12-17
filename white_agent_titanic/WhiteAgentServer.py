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

    def __init__(self, task_configs: list[Dict[str, Any]]):
        # Map skill_id -> config
        self._configs = {}
        self._default_config = None
        
        for cfg in task_configs:
            challenge = cfg.get("challenge_name", "Challenge")
            # Create a simplified ID
            slug = challenge.lower().replace(" ", "_").replace("-", "_")
            skill_id = f"evaluate_{slug}"
            self._configs[skill_id] = cfg
            self._configs[slug] = cfg # Allow lookup by slug too
            if self._default_config is None:
                self._default_config = cfg

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
            # Determine config
        target_skill = None
        # Try to find 'challenge' or 'skill' in submission
        req_challenge = submission.get("challenge") or submission.get("skill")
        
        config = None
        if req_challenge:
            config = self._configs.get(req_challenge) or self._configs.get(f"evaluate_{req_challenge}")
        
        # Fallback to default if only one exists or no specific request
        if not config:
            if len(self._configs) == 0:
                 await self._send_status(context, event_queue, TaskState.failed, "No challenges configured.", final=True)
                 return
            if len(self._configs) <= 2: # Dictionary doubles keys (id + slug), so <= 2 entries means 1 config
                 config = self._default_config
        
        if not config:
             await self._send_status(context, event_queue, TaskState.failed, f"Challenge '{req_challenge}' not found.", final=True)
             return

        await self._send_status(
            context,
            event_queue,
            TaskState.working,
            f"Running submitted image for challenge: {config.get('challenge_name', 'Unknown')}",
        )

        agent = WhiteAgent(config)
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

        # Publish plots if available
        output_dir = results.get("execution", {}).get("output_dir")
        if output_dir and os.path.exists(output_dir):
            import glob
            import base64
            for img_path in glob.glob(os.path.join(output_dir, "*.png")):
                fname = os.path.basename(img_path)
                try:
                    with open(img_path, "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    await self._enqueue_artifact(
                        event_queue,
                        TaskArtifactUpdateEvent(
                            context_id=self._context_id(context),
                            task_id=self._task_id(context),
                            artifact=new_data_artifact(
                                name=fname,
                                data={"base64": b64_data, "mime_type": "image/png"},
                                description=f"Visualization: {fname}",
                            ),
                            last_chunk=True,
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to publish image artifact {fname}: {e}")

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
    task_configs: list[Dict[str, Any]],
    *,
    public_url: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
    extended_agent_card: Optional[AgentCard] = None,
) -> A2AStarletteApplication:
    """Build an A2A application exposing the Generic White Agent."""
    
    url = public_url or os.environ.get("WHITE_AGENT_PUBLIC_URL", "http://localhost:8000")
    skills = []
    
    for cfg in task_configs:
        challenge_name = cfg.get("challenge_name", "Kaggle Challenge")
        slug = challenge_name.lower().replace(" ", "_").replace("-", "_")
        skill_id = f"evaluate_{slug}"
        
        skill_example = json.dumps(
            {
                "docker_image": "my_submission:latest",
                "challenge": slug 
            },
            indent=2,
        )

        skill = AgentSkill(
            id=skill_id,
            name=f"Evaluate {challenge_name} Submission",
            description=f"Run a docker image against the {challenge_name} evaluation set.",
            tags=["evaluation", "benchmark", "docker", slug],
            examples=[skill_example],
        )
        skills.append(skill)

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
        skills=skills,
    )

    executor = WhiteAgentExecutor(task_configs)
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
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    loaded_configs = []
    
    # scan for json configs
    if os.path.exists(configs_dir):
        for fname in os.listdir(configs_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(configs_dir, fname), "r") as f:
                        cfg = json.load(f)
                        loaded_configs.append(cfg)
                except Exception as e:
                    logger.error(f"Failed to load config {fname}: {e}")
    
    # Fallback/Backward compatibility or if directory empty
    if not loaded_configs:
        config_path = os.environ.get("TASK_CONFIG", "task_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                loaded_configs.append(json.load(f))
        else:
            logger.warning("No configs found. Agent will have no skills.")

    app = create_white_agent_app(
        task_configs=loaded_configs,
        public_url=os.environ.get("PUBLIC_URL"),
    ).build()
except Exception as e:
    logger.error(f"Failed to initialize White Agent app: {e}")
    raise
