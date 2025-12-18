import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import asyncio
from a2a.client import ClientFactory, ClientConfig
from a2a.types import AgentCard, Message, Role, TextPart, TaskArtifactUpdateEvent
from Memory import ChromaMemory, store_memory

logger = logging.getLogger(__name__)

class GreenAgent:
    def __init__(self, task_config: dict):
        # Support explicit paths or fallback to data_path directory logic
        if "train_data_path" in task_config and "test_data_path" in task_config:
            self.train_data_path = Path(task_config["train_data_path"])
            self.test_data_path = Path(task_config["test_data_path"])
        else:
            base_data_path = Path(task_config["data_path"])
            self.train_data_path = base_data_path / "train.csv"
            self.test_data_path = base_data_path / "test.csv"

        self.test_labels = task_config["test_labels"]
        self.target_column = task_config.get("target_column")

        self.collection_name = "evaluation_results"
        try:
            self.eval_memory = ChromaMemory(collection_name=self.collection_name)
        except RuntimeError as exc:
            raise RuntimeError(
                "ChromaDB is required for the Green Agent. Install dependencies with "
                "`pip install -r requirements.txt` before running evaluations."
            ) from exc
        self.eval_collection = self.eval_memory.collection

    
    def evaluate(self, submission: dict) -> dict:
        """
        Entry point called by the server.
        Orchestrates running the remote agent and then evaluating its predictions.
        """
        agent_url = submission.get("agent_url")
        if not agent_url:
             raise ValueError("Submission must contain 'agent_url'")
        
        # 1. Run the remote agent
        execution_result = self.run_remote_agent_sync(agent_url, self.train_data_path, self.test_data_path)
        
        # 2. Evaluate performance if successful
        performance = {}
        if execution_result.get("success"):
            predictions_path = execution_result.get("predictions")
            if predictions_path and os.path.exists(predictions_path):
                try:
                    performance = self.evaluate_performance(predictions_path, self.test_labels)
                except Exception as e:
                    logger.error(f"Performance evaluation failed: {e}")
                    execution_result["error"] = f"Prediction evaluation failed: {e}"
                    # We don't mark success=False necessarily if execution worked, but usually yes.
                    # But let's keep execution success as True (agent ran) but perf calc failed.
        
        # 3. Return structured result for Server
        return {
            "execution": execution_result,
            "performance": performance
        }

    def run_remote_agent_sync(self, agent_url: str, train_data_path: Path, test_data_path: Path) -> dict:
        """Wrapper to run async remote agent logic in sync context."""
        return asyncio.run(self.run_remote_agent(agent_url, train_data_path, test_data_path))

    async def run_remote_agent(self, agent_url: str, train_data_path: Path, test_data_path: Path) -> dict:
        """
        Connects to a remote Solver Agent and requests a solution.
        """
        logger.info(f"Connecting to remote agent at {agent_url}")
        output_dir = Path(f"/tmp/outputs_{uuid.uuid4().hex}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_path = None
        logs = []
        
        start_time = time.time()
        
        try:
            # 1. Setup Client
            config = ClientConfig() 
            factory = ClientFactory(config=config)
            
            card_url = f"{agent_url.rstrip('/')}/.well-known/agent-card.json"
            
            import httpx
            async with httpx.AsyncClient() as http_client:
                 resp = await http_client.get(card_url)
                 if resp.status_code != 200:
                     return {"success": False, "error": f"Failed to fetch card from {card_url}"}
                 card_data = resp.json()
                 
                 # Normalize 0.0.0.0
                 if "0.0.0.0" in card_data.get("url", ""):
                     card_data["url"] = card_data["url"].replace("0.0.0.0", "127.0.0.1")
                 
                 agent_card = AgentCard(**card_data)
                 
            client = factory.create(agent_card)
            
            # Construct the task payload
            task_payload = {
                "train_data_path": str(train_data_path.resolve()),
                "test_data_path": str(test_data_path.resolve()),
                "train_data_path": str(train_data_path.resolve()),
                "test_data_path": str(test_data_path.resolve()),
                "task_description": "Train a model on the provided dataset.",
            }
            if self.target_column:
                task_payload["target_column"] = self.target_column
            
            message = Message(
                message_id=str(uuid.uuid4()),
                role=Role.user,
                parts=[TextPart(text=json.dumps(task_payload))],
                metadata=task_payload,
                context_id=f"eval_{uuid.uuid4().hex}",
            )
            
            async for item in client.send_message(request=message):
                if isinstance(item, tuple):
                    task, event = item
                    if event:
                         if isinstance(event, TaskArtifactUpdateEvent):
                             if event.artifact.name == "predictions_csv":
                                 content = None
                                 if event.artifact.parts:
                                      p = event.artifact.parts[0]
                                      root = p.root if hasattr(p, 'root') else p
                                      if hasattr(root, 'data') and root.data:
                                           content = root.data.get("content")
                                 
                                 if content:
                                      p_path = output_dir / "predictions.csv"
                                      p_path.write_text(content)
                                      predictions_path = str(p_path)
                                      logger.info("Received predictions artifact.")

            elapsed_time = time.time() - start_time
            
            if predictions_path:
                 return {
                    "success": True,
                    "predictions": predictions_path,
                    "output_dir": str(output_dir),
                    "time_seconds": elapsed_time,
                    "memory_used_mb": 0.0, # Remote
                    "logs": "\n".join(logs)
                 }
            else:
                 return {
                    "success": False,
                    "error": "No predictions received from remote agent",
                    "time_seconds": elapsed_time
                 }

        except Exception as e:
            logger.exception("Remote agent execution failed")
            return {
                "success": False, 
                "error": str(e),
                "time_seconds": time.time() - start_time
            }

    def evaluate_performance(self, predictions_path: str, test_labels: Union[str, Sequence, pd.Series]) -> dict:
        """
        Compare predicted labels against hidden ground truth.
        """
        preds = pd.read_csv(predictions_path)
        if "prediction" not in preds.columns:
            raise ValueError("Predictions file must include a 'prediction' column")

        y_true = self._load_labels(test_labels)
        y_pred = preds["prediction"]

        from sklearn.metrics import accuracy_score, f1_score

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        }

    @staticmethod
    def _load_labels(labels_source: Union[str, Sequence, pd.Series]) -> Iterable:
        if isinstance(labels_source, pd.Series):
            return labels_source
        if isinstance(labels_source, (str, os.PathLike)):
            labels_df = pd.read_csv(labels_source)
            for candidate in ("label", "target", "y", "labels"):
                if candidate in labels_df.columns:
                    return labels_df[candidate]
            return labels_df.iloc[:, -1]
        return labels_source

    def _record_run(self, results: dict) -> None:
        """
        Persist a lightweight snapshot of the evaluation to the shared Chroma collection.
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
