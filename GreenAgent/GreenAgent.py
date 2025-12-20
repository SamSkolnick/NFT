import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import asyncio
import httpx
from a2a.client import ClientFactory, ClientConfig
from a2a.types import (
    AgentCard, 
    Message, 
    Role, 
    TextPart, 
    TaskArtifactUpdateEvent, 
    DataPart,
    TaskStatusUpdateEvent,
    TaskState
)
from Memory import ChromaMemory, store_memory

logger = logging.getLogger(__name__)

class GreenAgent:
    def __init__(self, task_config: dict):
        if "train_data_path" in task_config and "test_data_path" in task_config:
            self.train_data_path = Path(task_config["train_data_path"])
            self.test_data_path = Path(task_config["test_data_path"])
        else:
            base_data_path = Path(task_config.get("data_path", "data"))
            self.train_data_path = base_data_path / "train.csv"
            self.test_data_path = base_data_path / "test.csv"

        self.test_labels = task_config.get("test_labels")
        self.target_column = task_config.get("target_column")
        self.constraints = task_config.get("constraints", {})

        self.collection_name = "evaluation_results"
        try:
            self.eval_memory = ChromaMemory(collection_name=self.collection_name)
        except Exception as exc:
            logger.warning(f"Failed to initialize ChromaMemory: {exc}. Run recording might fail.")
            self.eval_memory = None

    
    def evaluate(self, submission: dict) -> dict:
        """
        The main entry point the server calls.
        It runs the remote agent and then checks how well it did.
        """
        # 1. Handle direct predictions if provided
        performance = {}
        execution_result = {"success": True}
        
        predictions = submission.get("predictions")
        predictions_path = submission.get("predictions_path")
        
        if predictions or predictions_path:
            logger.info("Direct predictions provided, skipping remote execution.")
            try:
                performance = self.evaluate_performance(predictions or predictions_path, self.test_labels)
            except Exception as e:
                logger.error(f"Performance evaluation failed: {e}")
                execution_result["success"] = False
                execution_result["error"] = f"Prediction evaluation failed: {e}"
        else:
            # Fall back to remote execution
            agent_url = submission.get("agent_url")
            if not agent_url:
                 raise ValueError("Submission must contain 'agent_url' or direct 'predictions'")
            
            execution_result = self.run_remote_agent_sync(agent_url, self.train_data_path, self.test_data_path)
            
            if execution_result.get("success"):
                predictions_path = execution_result.get("predictions")
                if predictions_path and os.path.exists(predictions_path):
                    try:
                        performance = self.evaluate_performance(predictions_path, self.test_labels)
                    except Exception as e:
                        logger.error(f"Performance evaluation failed: {e}")
                        execution_result["error"] = f"Prediction evaluation failed: {e}"

        # 2. Run research evaluation
        research_report_path = execution_result.get("research_report")
        if not research_report_path:
             # Fallback to local path if provided in submission (deprecated but kept for compatibility)
             research_report_path = submission.get("research_artifacts")
        
        research_score = 0
        if research_report_path:
            try:
                research_score = self._run_research_eval(research_report_path)
                performance["research_quality_score"] = research_score
            except Exception as e:
                logger.error(f"Research evaluation failed: {e}")

        result = {
            "execution": execution_result,
            "performance": performance
        }
        
        if self.eval_memory:
            try:
                self._record_run(result)
            except Exception as e:
                logger.warning(f"Failed to record run: {e}")

        return result

    def _run_research_eval(self, artifacts_path: str) -> int:
        """
        Looks at the research files (like research_report.md) and uses an LLM to score them.
        """
        from LLMModule import call_openrouter_tongyi
        
        report_content = ""
        path = Path(artifacts_path)
        if path.is_file():
            report_content = path.read_text()
        elif path.is_dir():
            # Try to find common report files
            for name in ["research_report.md", "research.md", "report.md", "README.md"]:
                candidate = path / name
                if candidate.exists():
                    report_content = candidate.read_text()
                    break
        
        if not report_content:
            logger.warning(f"No research report found at {artifacts_path}")
            return 0

        prompt = f"""
        Analyze the following machine learning research report for:
        1. Usefulness: Does it provide actionable insights and a clear path forward?
        2. Accuracy: Are the technical claims and results plausible and well-supported?
        
        Report Content:
        {report_content[:5000]}
        
        Provide an integer score between 0 and 100 representing the overall quality. 
        Return ONLY the integer.
        """
        
        try:
            response = call_openrouter_tongyi(prompt)
            # Extract digits and convert to int
            score_str = "".join(filter(str.isdigit, response))
            return int(score_str) if score_str else 0
        except Exception as e:
            logger.error(f"LLM research eval failed: {e}")
            return 0

    def run_remote_agent_sync(self, agent_url: str, train_data_path: Path, test_data_path: Path) -> dict:
        """Wrapper to run async remote agent logic in sync context."""
        return asyncio.run(self.run_remote_agent(agent_url, train_data_path, test_data_path))

    async def run_remote_agent(self, agent_url: str, train_data_path: Path, test_data_path: Path) -> dict:
        """
        Connects to a Solver agent via A2A, gives it a task, and waits for it to finish.
        """
        # A2A Client Setup
        # We need to fetch the agent card first to configure the client
        card_url = f"{agent_url.rstrip('/')}/.well-known/agent-card.json"
        
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(card_url)
            if resp.status_code != 200:
                print(f"DEBUG: Failed to fetch card: {resp.status_code}")
                return {"success": False, "error": f"Failed to fetch card from {card_url}"}
            card_data = resp.json()
            card = AgentCard(**card_data)
        
        print(f"DEBUG: Card fetched. URL: {card.url}")
        # Factory and Client
        config = ClientConfig()
        factory = ClientFactory(config=config)
        client = factory.create(card)
        
        # Read datasets
        train_content = train_data_path.read_text() if train_data_path.exists() else ""
        test_content = test_data_path.read_text() if test_data_path.exists() else ""
        labels_content = ""
        if self.test_labels and os.path.exists(self.test_labels):
            labels_content = Path(self.test_labels).read_text()

        # Construct Payload (Remote Ready)
        my_url = os.environ.get("AGENT_URL", "http://localhost:8000").rstrip("/")
        
        task_payload = {
            "task_description": "Train a model on the provided dataset.",
        }
        
        if self.target_column:
             task_payload["target_column"] = self.target_column
        
        # Inject MCP Server URL
        task_payload["mcp_server_url"] = f"{my_url}/mcp/sse"
        
        print(f"DEBUG: Initiating A2A task with artifacts for training and test data.")

        message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[
                TextPart(text="Solve this task for me."),
                DataPart(data={"filename": train_data_path.name, "content": train_content}),
                DataPart(data={"filename": test_data_path.name, "content": test_content}),
                DataPart(data={"filename": "test_labels.csv", "content": labels_content}),
            ],
            context_id=f"eval_{uuid.uuid4().hex}",
            metadata=task_payload,
        )
        
        print(f"DEBUG: Sending message to {card.url}")
        execution_result = {"success": False}
        events_received = 0
        try:
            async for item in client.send_message(request=message):
                events_received += 1
                
                # Handle unpacking if generic client yields (task, event)
                event = item
                if isinstance(item, tuple):
                    _, event = item

                if isinstance(event, TaskStatusUpdateEvent):
                    print(f"DEBUG: Received Status: {event.status.state}")
                    if event.status.state == TaskState.completed:
                        execution_result["success"] = True
                    elif event.status.state == TaskState.failed:
                        execution_result["success"] = False
                        if event.status.message and hasattr(event.status.message, 'text'):
                             execution_result["error"] = event.status.message.text
                             print(f"DEBUG: Failure Message: {execution_result['error']}")
                elif isinstance(event, TaskArtifactUpdateEvent):
                    print(f"DEBUG: Received Artifact: {event.artifact.name}")
                    if event.artifact.name in ["predictions_csv", "research_report"]:
                        # Save to temp file
                        output_dir = Path(f"/tmp/outputs_{uuid.uuid4().hex}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = "predictions.csv" if event.artifact.name == "predictions_csv" else "research_report.md"
                        artifact_path = output_dir / filename
                        
                        content = ""
                        for part in event.artifact.parts:
                            # A2A parts can be wrapped in pydantic-like structures or have .root
                            p = part.root if hasattr(part, "root") else part
                            if hasattr(p, "data") and isinstance(p.data, dict) and "content" in p.data:
                                content += p.data["content"]
                            elif hasattr(p, "text"):
                                content += p.text
                        
                        if content:
                            artifact_path.write_text(content)
                            if event.artifact.name == "predictions_csv":
                                execution_result["predictions"] = str(artifact_path)
                            else:
                                execution_result["research_report"] = str(artifact_path)
                            print(f"DEBUG: Saved {event.artifact.name} to {artifact_path}")
                
        except Exception as e:
             print(f"DEBUG: Exception in send_message: {e}")
             import traceback
             traceback.print_exc()
             
        print(f"DEBUG: Finished loop. Events: {events_received}. Result: {execution_result}")
        return execution_result

    def evaluate_performance(self, predictions_source: Union[str, Sequence], test_labels: Union[str, Sequence, pd.Series]) -> dict:
        """
        Compares the agent's predictions against the ground truth labels.
        """
        if isinstance(predictions_source, (str, os.PathLike)) and os.path.exists(predictions_source):
             preds_df = pd.read_csv(predictions_source)
             if "prediction" in preds_df.columns:
                 y_pred = preds_df["prediction"]
             else:
                 y_pred = preds_df.iloc[:, -1]
        elif isinstance(predictions_source, (list, Sequence)):
             y_pred = predictions_source
        else:
             # Try reading as simple text file with newlines
             try:
                 content = Path(predictions_source).read_text()
                 y_pred = [line.strip() for line in content.splitlines() if line.strip()]
             except:
                 y_pred = predictions_source

        y_true = [str(x) for x in self._load_labels(test_labels)]
        y_pred = [str(x) for x in y_pred]
        
        from sklearn.metrics import accuracy_score, f1_score
        
        # Ensure lengths match
        if len(y_true) != len(y_pred):
            logger.warning(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
            # If y_pred is shorter, pad it; if longer, truncate it. Simple fallback.
            if len(y_pred) > len(y_true):
                y_pred = y_pred[:len(y_true)]
            else:
                y_true = list(y_true)[:len(y_pred)]

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        }

    @staticmethod
    def _load_labels(labels_source: Union[str, Sequence, pd.Series]) -> Iterable:
        if isinstance(labels_source, pd.Series):
            return labels_source
        if isinstance(labels_source, (str, os.PathLike)):
            if not os.path.exists(labels_source):
                return []
            
            # Read first chunk to decide if it's CSV or simple list
            try:
                content = Path(labels_source).read_text(errors="ignore").strip()
            except:
                return []
            
            if not content:
                return []

            # If it looks like a multi-column CSV
            if "," in content.splitlines()[0]:
                try:
                    labels_df = pd.read_csv(labels_source)
                    for candidate in ("label", "target", "y", "labels"):
                        if candidate in labels_df.columns:
                            return labels_df[candidate]
                    return labels_df.iloc[:, -1]
                except:
                    pass
            
            # Fallback to newline-separated text file
            return [line.strip() for line in content.splitlines() if line.strip()]
        return labels_source

    def _record_run(self, results: dict) -> None:
        """
        Saves a snapshot of this evaluation into Chroma so we can remember it later.
        """
        doc_id = f"run_{uuid.uuid4().hex}"
        summary = {
            "performance": results.get("performance", {}),
            "execution": results.get("execution"),
        }
        payload = json.dumps(summary, default=str)

        self.eval_memory.upsert(
            doc_id=doc_id,
            document=payload,
            metadata={"collection": self.collection_name},
        )
        try:
            store_memory(
                doc_id=doc_id,
                document=payload,
                metadata={"collection": self.collection_name},
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to upsert into shared memory collection: %s", exc)
