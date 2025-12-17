import asyncio
import json
import logging
import os
import uuid
import pandas as pd
import joblib
from pathlib import Path
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
)

# Import model logic dependencies
# We assume this script is run from MLEngineer root or we adjust paths
# The model is in white_agent_titanic/model/model.pkl

logger = logging.getLogger(__name__)

class SolverExecutor(AgentExecutor):
    """Executes the Solver Model."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.model_path = base_dir / "model" / "model.pkl"
        self.data_dir = base_dir / "data"
        self.model = None
        
        # Pre-load model if possible
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning(f"Model not found at {self.model_path}")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        
        await self._send_status(
            context, event_queue, TaskState.working, "Solver started. Processing data..."
        )

        if not self.model:
             await self._send_status(
                context, event_queue, TaskState.failed, "Model not loaded.", final=True
            )
             return

        try:
            # 1. Load Data (Test set)
            # Logic adapted from evaluate.load_features
            test_path = self.data_dir / "test" / "test.csv"
            if not test_path.exists():
                test_path = self.data_dir / "test.csv"
            
            if not test_path.exists():
                 raise FileNotFoundError(f"Test data not found in {self.data_dir}")

            # Define columns as per evaluate.py
            COLUMN_NAMES = [
                "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
            ]
            
            # Read and Preprocess
            df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES, na_values=[""])
            if "Survived" in df.columns:
                df = df.drop(columns=["Survived"]) # Should not happen in test but good to be safe
                
            passenger_ids = df.get("PassengerId")
            X_test = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
            
            # 2. Predict
            # Run in thread to not block async loop
            preds = await asyncio.to_thread(self.model.predict, X_test)
            
            # 3. Format Results
            result_df = pd.DataFrame({"prediction": preds})
            if passenger_ids is not None:
                result_df.insert(0, "PassengerId", passenger_ids)
            
            # Convert to list of dicts or CSV string for transmission
            # For A2A, sending as a Data artifact (JSON) is clean for small data, 
            # or CSV string for compat with GreenAgent's pandas expectations.
            # GreenAgent expects a CSV file usually? Or just data.
            # Let's send it as a CSV string inside a "predictions" artifact.
            
            csv_content = result_df.to_csv(index=False)
            
            await self._enqueue_artifact(
                event_queue,
                TaskArtifactUpdateEvent(
                    context_id=self._context_id(context),
                    task_id=self._task_id(context),
                    artifact=new_data_artifact(
                        name="predictions_csv",
                        data={"content": csv_content}, 
                        description="Titanic predictions in CSV format.",
                    ),
                    last_chunk=True,
                ),
            )
            
            await self._send_status(
                context, event_queue, TaskState.completed, "Predictions generated successfully.", final=True
            )

        except Exception as e:
            logger.exception("Solver execution failed")
            await self._send_status(
                context, event_queue, TaskState.failed, f"Error: {str(e)}", final=True
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

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

    async def _enqueue_artifact(self, event_queue: EventQueue, event: TaskArtifactUpdateEvent) -> None:
        await event_queue.enqueue_event(event)

    @staticmethod
    def _context_id(context: RequestContext) -> str:
        return context.context_id or "default_ctx"

    @staticmethod
    def _task_id(context: RequestContext) -> str:
        return context.task_id or "default_task"


def create_solver_app(
    base_dir: Path,
    public_url: Optional[str] = None,
) -> A2AStarletteApplication:
    
    url = public_url or os.environ.get("AGENT_URL") or "http://localhost:8005"
    
    skill = AgentSkill(
        id="solve_titanic",
        name="Solve Titanic Survival",
        description="Predicts survival on the Titanic dataset.",
        tags=["solver", "ml", "titanic"],
    )

    capabilities = AgentCapabilities(streaming=True)
    card = AgentCard(
        id=f"titanic-solver-{hash(url) % 10000:04d}",
        name="Titanic Solver Agent",
        description="A White Agent that solves the Titanic challenge.",
        url=url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skill],
    )

    executor = SolverExecutor(base_dir)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )

# For Uvicorn
# Usage: uvicorn TitanicSolverServer:app --port 8005
try:
    # Assume we are in white_agent_titanic or root?
    # Let's assume this file is placed in white_agent_titanic/
    # so base_dir is current dir.
    # But wait, I'm writing it to... where?
    # I should write it to white_agent_titanic/TitanicSolverServer.py
    
    base_path = Path(__file__).parent
    app = create_solver_app(
        base_dir=base_path,
        public_url=os.environ.get("PUBLIC_URL"),
    ).build()
except Exception as e:
    logger.error(f"Failed to init app: {e}")
    # Don't raise here to allow import without crashing if paths wrong
