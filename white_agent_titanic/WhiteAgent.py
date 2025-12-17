import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import docker
import pandas as pd
from docker.errors import APIError, ContainerError, DockerException, ImageNotFound, NotFound

# Metrics imports - explicitly importing commonly used ones
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class WhiteAgent:
    """
    Generic White Agent that evaluates submissions based on a configuration.
    """
    def __init__(self, task_config: dict):
        self.config = task_config
        self.data_path = Path(task_config.get("data_path", "/data"))
        self.ground_truth_file = task_config.get("ground_truth_file", "test_labels.csv")
        self.id_column = task_config.get("id_column", "id")
        self.target_column = task_config.get("target_column", "target")
        self.prediction_column = task_config.get("prediction_column", "prediction")
        self.metrics = task_config.get("metrics", ["accuracy"])
        
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

    def evaluate(self, submission: dict) -> dict:
        """
        Main evaluation flow:
        1. Run user container.
        2. Validate constraints (time, memory).
        3. Load predictions and ground truth.
        4. Calculate metrics.
        """
        execution = self.run_container(
            docker_image=submission["docker_image"],
            pull_image=submission.get("pull_image", True),
            auth_config=self._extract_auth_config(submission)
        )

        constraints_result = self.check_constraints(execution, self.constraints)
        
        metrics = {}
        if constraints_result["passed"] and execution.get("predictions"):
            try:
                metrics = self.calculate_metrics(execution["predictions"])
                self.generate_plots(execution["predictions"], execution["output_dir"])
            except Exception as e:
                logger.error(f"Metric calculation or plotting failed: {e}")
                constraints_result["passed"] = False
                constraints_result["violations"].append(f"Metric calculation error: {str(e)}")
        
        return {
            "execution": execution,
            "constraints": constraints_result,
            "metrics": metrics
        }

    def run_container(
        self,
        docker_image: str,
        pull_image: bool = True,
        auth_config: Optional[dict] = None
    ) -> dict:
        client = docker.from_env()
        
        # Determine absolute path for data volume
        # If running in container, we might need mapped paths, but assuming local path for now
        # or that the user has mounted the data correctly if running inside docker.
        # For simplicity, we assume 'data_path' is accessible to the docker daemon.
        
        local_data_path = self.data_path.resolve()
        
        output_dir = Path(f"/tmp/outputs_{uuid.uuid4().hex}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        volumes = {
            str(local_data_path): {"bind": "/data", "mode": "ro"},
            str(output_dir): {"bind": "/output", "mode": "rw"}
        }

        # Env vars for the container to know where to read/write
        environment = {
            "EVAL_DATA_DIR": "/data",
            "EVAL_OUTPUT_DIR": "/output",
            "EVAL_PREDICTIONS_FILE": "/output/predictions.csv",
        }

        mem_limit_mb = self.constraints.get("max_memory_mb", 8192)
        timeout_seconds = self.constraints.get("max_time_seconds", 3600)

        container = None
        logs = ""
        
        try:
            if pull_image:
                try:
                    client.images.pull(docker_image, auth_config=auth_config)
                except Exception as e:
                    logger.warning(f"Pull failed: {e}")
                    # Try running anyway if local

            container = client.containers.run(
                docker_image,
                network_mode="none",
                volumes=volumes,
                environment=environment,
                mem_limit=f"{mem_limit_mb}m",
                detach=True,
            )

            start_time = time.time()
            result = container.wait(timeout=timeout_seconds)
            elapsed_time = time.time() - start_time
            
            logs = container.logs().decode("utf-8", errors="replace")
            
            predictions_path = self._locate_predictions_file(output_dir)

            return {
                "success": result.get("StatusCode", 1) == 0,
                "predictions": predictions_path,
                "output_dir": str(output_dir),
                "time_seconds": elapsed_time,
                "logs": logs,
                "memory_used_mb": 0.0 # TODO: Implement stats collection if needed
            }

        except Exception as exc:
             if container:
                 try:
                     logs = container.logs().decode("utf-8", errors="replace")
                     container.remove(force=True)
                 except: pass
             return {
                "success": False,
                "error": str(exc),
                "logs": logs,
                "time_seconds": 0.0
            }
        finally:
             if container:
                 try: 
                     container.remove(force=True)
                 except: pass

    def calculate_metrics(self, predictions_path: str) -> dict:
        """
        Generically calculate metrics based on config.
        """
        # Load Preds
        pdf = pd.read_csv(predictions_path)
        
        # Load GT
        gt_path = self.data_path / self.ground_truth_file
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        gdf = pd.read_csv(gt_path)

        # Merge
        if self.id_column not in pdf.columns:
            raise ValueError(f"Predictions missing ID column: {self.id_column}")
        if self.id_column not in gdf.columns:
             # Try first column if ID not found? No, strict is better.
             raise ValueError(f"Ground Truth missing ID column: {self.id_column}")

        # Ensure IDs are same type (often string vs int issues)
        pdf[self.id_column] = pdf[self.id_column].astype(str)
        gdf[self.id_column] = gdf[self.id_column].astype(str)

        merged = pd.merge(gdf, pdf, on=self.id_column, suffixes=('_true', '_pred'))
        
        # Determine actual column names after merge
        # If columns were named differently in files, we use config mapping
        # Config: target_column (GT), prediction_column (Preds)
        
        target_col = self.target_column
        pred_col = self.prediction_column
        
        # If merge created suffixes
        if f"{target_col}_true" in merged.columns:
            y_true = merged[f"{target_col}_true"]
        elif target_col in merged.columns:
             y_true = merged[target_col]
        else:
            raise KeyError(f"Target column '{target_col}' not found after merge.")

        if f"{pred_col}_pred" in merged.columns:
             y_pred = merged[f"{pred_col}_pred"]
        elif pred_col in merged.columns:
             y_pred = merged[pred_col]
        else:
             # Fallback: check if user used target_column name for prediction
             if f"{target_col}_pred" in merged.columns:
                y_pred = merged[f"{target_col}_pred"]
             else:
                raise KeyError(f"Prediction column '{pred_col}' not found. Columns: {merged.columns}")

        scores = {}
        for metric in self.metrics:
            scores[metric] = self._compute_one_metric(metric, y_true, y_pred)
            
        return scores

    def _compute_one_metric(self, metric_name: str, y_true: Any, y_pred: Any) -> float:
        """Dispatcher for sklearn metrics."""
        # Clean inputs if classification
        # For simplicity, passing directly to sklearn
        try:
            if metric_name == "accuracy":
                return float(accuracy_score(y_true, y_pred))
            elif metric_name == "f1_score" or metric_name == "f1":
                # Auto detect average
                average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
                return float(f1_score(y_true, y_pred, average=average))
            elif metric_name == "rmse":
                 return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric_name == "mae":
                 return float(mean_absolute_error(y_true, y_pred))
            elif metric_name == "r2":
                 return float(r2_score(y_true, y_pred))
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                return 0.0
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")
            return -1.0

    def check_constraints(self, execution: dict, constraints: dict) -> dict:
        passed = True
        violations = []
        
        if not execution.get("success"):
            passed = False
            violations.append(execution.get("error", "Unknown execution failure"))

        time_limit = constraints.get("max_time_seconds")
        if time_limit and execution.get("time_seconds", 0) > time_limit:
            passed = False
            violations.append("Time limit exceeded")

        if not execution.get("predictions"):
             passed = False
             violations.append("No predictions file generated")

        return {"passed": passed, "violations": violations}

    def _locate_predictions_file(self, output_dir: Path) -> Optional[str]:
        # Priority: literal config path -> predictions.csv -> any csv
        candidates = ["predictions.csv", self.config.get("prediction_file_name", "")]
        for c in candidates:
            if not c: continue
            p = output_dir / c
            if p.exists(): return str(p.resolve())
            
        csvs = list(output_dir.glob("*.csv"))
        if csvs: return str(csvs[0].resolve())
        return None

        return None

    def generate_plots(self, predictions_path: str, output_dir: str):
        """
        Generate relevant plots based on problem type.
        """
        try:
            pdf = pd.read_csv(predictions_path)
            gt_path = self.data_path / self.ground_truth_file
            gdf = pd.read_csv(gt_path)

            pdf[self.id_column] = pdf[self.id_column].astype(str)
            gdf[self.id_column] = gdf[self.id_column].astype(str)

            merged = pd.merge(gdf, pdf, on=self.id_column, suffixes=('_true', '_pred'))
            
            target_col = self.target_column
            pred_col = self.prediction_column

            y_true = merged[f"{target_col}_true"] if f"{target_col}_true" in merged.columns else merged[target_col]
            
            y_pred = None
            if f"{pred_col}_pred" in merged.columns:
                 y_pred = merged[f"{pred_col}_pred"]
            elif pred_col in merged.columns:
                 y_pred = merged[pred_col]
            elif f"{target_col}_pred" in merged.columns:
                y_pred = merged[f"{target_col}_pred"]
                
            if y_true is None or y_pred is None:
                return

            problem_type = self.config.get("problem_type", "classification")
            out = Path(output_dir)

            if problem_type == "classification":
                # Confusion Matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {self.config.get("challenge_name")}')
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                plt.savefig(out / "confusion_matrix.png")
                plt.close()

            elif problem_type == "regression":
                # Residual Plot
                residuals = y_true - y_pred
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_pred, y=residuals)
                plt.axhline(0, color='r', linestyle='--')
                plt.title('Residuals vs Predicted')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                plt.savefig(out / "residual_plot.png")
                plt.close()
                
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
