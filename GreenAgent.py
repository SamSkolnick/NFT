import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import docker
import pandas as pd
from docker.errors import APIError, ContainerError, DockerException, ImageNotFound, NotFound

from ResearchEval import evaluate_research
from Memory import ChromaMemory, store_memory

logger = logging.getLogger(__name__)


class GreenAgent:
    def __init__(self, task_config: dict):
        self.task_data_path = Path(task_config["data_path"])
        self.test_labels = task_config["test_labels"]

        raw_constraints = task_config.get("constraints") or {}
        default_constraints = {
            "max_time_seconds": 3600,
            "max_memory_mb": 8192,
            "max_cpus": 2.0,
        }
        self.constraints = {
            key: raw_constraints.get(key, default)
            for key, default in default_constraints.items()
        }
        for key, value in raw_constraints.items():
            if key not in self.constraints:
                self.constraints[key] = value

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
        Run an submitted docker image, evaluate its research, enforce constraints, and score performance.
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            execution_future = executor.submit(
                self.run_white_agent,
                docker_image=submission["docker_image"],
                task_path=self.task_data_path,
                command=submission.get("eval_command"),
                auth_config=self._extract_auth_config(submission),
                pull_image=submission.get("pull_image", True),
            )
            research_future = executor.submit(
                evaluate_research,
                submission["research_artifacts"],
                storage=submission.get("storage_method", "local"),
            )

            execution = execution_future.result()
            try:
                research_result = research_future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Research evaluation failed.")
                research_result = {"error": str(exc)}

        results: dict[str, Union[dict, float, int]] = {}
        results["execution"] = execution
        results["research"] = research_result

        results["constraints"] = self.check_constraints(execution, self.constraints)

        if results["constraints"]["passed"] and execution.get("predictions"):
            results["performance"] = self.evaluate_performance(execution["predictions"], self.test_labels)
        else:
            results["performance"] = {"accuracy": 0.0, "f1_score": 0.0}

        # Store a summary in shared memory for later retrieval.
        self._record_run(results)
        return results

    def run_white_agent(
        self,
        docker_image: str,
        task_path: Path,
        command: Optional[Union[str, Sequence[str]]] = None,
        auth_config: Optional[dict] = None,
        pull_image: bool = True,
    ) -> dict:
        """
        Runs the white agent's container against the hidden evaluation data.
        Returns execution metadata and the produced predictions file location.
        """
        client = docker.from_env()

        if isinstance(command, str):
            command = command.strip() or None

        run_command: Optional[Union[str, Sequence[str]]] = command if command else ["python", "evaluate.py"]

        volumes = self._build_volume_map(task_path)
        output_dir = Path(f"/tmp/outputs_{uuid.uuid4().hex}")
        output_dir.mkdir(parents=True, exist_ok=True)
        volumes[str(output_dir)] = {"bind": "/output", "mode": "rw"}

        mem_limit_mb = self.constraints.get("max_memory_mb", 8192)
        cpu_limit = float(self.constraints.get("max_cpus", 2.0))
        timeout_seconds = self.constraints.get("max_time_seconds", 3600)

        container = None
        image_pulled = False
        pull_error: Optional[str] = None

        if pull_image:
            need_pull = True
            try:
                client.images.get(docker_image)
                need_pull = False
                logger.info("Found docker image '%s' locally; skipping pull.", docker_image)
            except ImageNotFound:
                need_pull = True
            except DockerException as exc:
                logger.debug("Error checking local image cache for '%s': %s", docker_image, exc, exc_info=True)

            if need_pull:
                try:
                    client.images.pull(docker_image, auth_config=auth_config)
                    image_pulled = True
                    logger.info("Pulled docker image '%s' for evaluation.", docker_image)
                except ImageNotFound as exc:
                    pull_error = f"Image not found during pull: {exc}"
                    logger.warning("Unable to pull image '%s': %s", docker_image, pull_error)
                except (APIError, DockerException) as exc:
                    pull_error = str(exc)
                    logger.warning("Failed to pull image '%s': %s", docker_image, pull_error)

        try:
            container = client.containers.run(
                docker_image,
                command=run_command,
                network_mode="none",
                volumes=volumes,
                mem_limit=f"{mem_limit_mb}m",
                cpus=cpu_limit,
                detach=True,
                remove=False,
                environment={
                    "EVAL_DATA_DIR": "/data",
                    "EVAL_OUTPUT_DIR": "/output",
                    "EVAL_PREDICTIONS_FILE": "/output/predictions.csv",
                },
            )

            start_time = time.time()
            result = container.wait(timeout=timeout_seconds)
            elapsed_time = time.time() - start_time

            stats = container.stats(stream=False)
            logs = container.logs().decode("utf-8", errors="replace")
            predictions_path = self._locate_predictions_file(output_dir)

            return {
                "success": result.get("StatusCode", 1) == 0,
                "predictions": predictions_path,
                "output_dir": str(output_dir),
                "time_seconds": elapsed_time,
                "memory_used_mb": self._extract_memory_usage(stats),
                "logs": logs,
                "image_pulled": image_pulled,
                "pull_error": pull_error,
            }
        except (ContainerError, APIError, ImageNotFound, NotFound) as exc:
            error_logs = container.logs().decode("utf-8", errors="replace") if container else ""
            return {
                "success": False,
                "error": str(exc),
                "predictions": None,
                "output_dir": str(output_dir),
                "time_seconds": 0.0,
                "memory_used_mb": 0.0,
                "logs": error_logs,
                "image_pulled": image_pulled,
                "pull_error": pull_error or str(exc),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "predictions": None,
                "output_dir": str(output_dir),
                "time_seconds": 0.0,
                "memory_used_mb": 0.0,
                "logs": "",
                "image_pulled": image_pulled,
                "pull_error": pull_error or str(exc),
            }
        finally:
            if container:
                try:
                    container.remove(force=True)
                except NotFound:
                    pass

    def check_constraints(self, execution: dict, constraints: dict) -> dict:
        """
        Validate runtime, memory, and output artifacts.
        """
        if not execution.get("success"):
            return {"passed": False, "violations": ["Execution failed"]}

        violations: list[str] = []

        time_limit = constraints.get("max_time_seconds")
        if time_limit is not None and execution.get("time_seconds", 0) > time_limit:
            violations.append(f"Time limit exceeded: {execution['time_seconds']}s > {time_limit}s")

        memory_limit = constraints.get("max_memory_mb")
        if memory_limit is not None and execution.get("memory_used_mb", 0) > memory_limit:
            violations.append(f"Memory limit exceeded: {execution['memory_used_mb']}MB > {memory_limit}MB")

        predictions_path = execution.get("predictions")
        if not predictions_path:
            violations.append("Predictions file not found")
        else:
            try:
                preds = pd.read_csv(predictions_path)
                if not self._valid_format(preds):
                    violations.append("Invalid prediction format")
            except Exception as exc:  # pylint: disable=broad-except
                violations.append(f"Could not read predictions: {exc}")

        return {"passed": len(violations) == 0, "violations": violations}

    def evaluate_performance(self, predictions_path: str, test_labels: Union[str, Sequence, pd.Series]) -> dict:
        """
        Compare predicted labels against hidden ground truth.
        """
        preds = pd.read_csv(predictions_path)
        if "prediction" not in preds.columns:
            raise ValueError("Predictions file must include a 'prediction' column")

        y_true = self._load_labels(test_labels)
        y_pred = preds["prediction"]

        from sklearn.metrics import accuracy_score, f1_score  # Deferred import to avoid global dependency if unused

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        }

    def _build_volume_map(self, task_path: Path) -> dict:
        volumes: dict[str, dict] = {}
        for split in ("train", "val", "test"):
            split_path = task_path / split
            if split_path.exists():
                volumes[str(split_path.resolve())] = {"bind": f"/data/{split}", "mode": "ro"}
        return volumes

    def _locate_predictions_file(self, output_dir: Path) -> Optional[str]:
        candidates = ["predictions.csv", "preds.csv", "output.csv"]
        for candidate in candidates:
            path = output_dir / candidate
            if path.exists():
                return str(path.resolve())
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            return str(csv_files[0].resolve())
        return None

    def _extract_auth_config(self, submission: dict) -> Optional[dict]:
        """
        Normalize optional docker credential payloads from the submission.
        Supports docker_credentials, docker_auth, and registry_auth keys.
        """
        candidates = [
            submission.get("docker_credentials"),
            submission.get("docker_auth"),
            submission.get("registry_auth"),
        ]
        for candidate in candidates:
            auth = self._normalize_auth(candidate)
            if auth:
                return auth
        return None

    @staticmethod
    def _normalize_auth(candidate: Optional[dict]) -> Optional[dict]:
        if not isinstance(candidate, dict):
            return None
        allowed_keys = {"username", "password", "email", "registry", "identitytoken"}
        normalized = {key: value for key, value in candidate.items() if key in allowed_keys and value}
        return normalized or None

    @staticmethod
    def _extract_memory_usage(stats: dict) -> float:
        try:
            memory_bytes = stats["memory_stats"]["max_usage"]
            return memory_bytes / (1024 * 1024)
        except Exception:  # pylint: disable=broad-except
            return 0.0

    @staticmethod
    def _valid_format(preds: pd.DataFrame) -> bool:
        required_columns = {"prediction"}
        return required_columns.issubset(preds.columns)

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
            "constraints": results.get("constraints", {}),
            "research_score": results.get("research"),
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
