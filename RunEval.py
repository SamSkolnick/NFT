import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

if sys.version_info < (3, 10):
    python310 = shutil.which("python3.10")
    if python310:
        os.execv(python310, [python310, *sys.argv])
    raise SystemExit(
        "Green Agent requires Python 3.10 or newer. Install dependencies with "
        "python3.10 -m pip install -r requirements.txt and re-run the command."
    )

import uvicorn

from GreenAgentServer import create_green_agent_app


def build_task_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        if not args.data_path or not args.labels:
            raise ValueError("Either --config or both --data-path and --labels must be provided.")

        config = {
            "data_path": str(Path(args.data_path).resolve()),
            "test_labels": str(Path(args.labels).resolve()),
            "constraints": {
                "max_time_seconds": args.max_time_seconds,
                "max_memory_mb": args.max_memory_mb,
                "max_cpus": args.max_cpus,
            },
        }

    constraints = config.setdefault("constraints", {})
    constraints.setdefault("max_time_seconds", args.max_time_seconds)
    constraints.setdefault("max_memory_mb", args.max_memory_mb)
    constraints.setdefault("max_cpus", args.max_cpus)

    if "data_path" not in config:
        if args.data_path:
            config["data_path"] = str(Path(args.data_path).resolve())
        else:
            raise ValueError("Task configuration must include 'data_path'.")

    if "test_labels" not in config:
        if args.labels:
            config["test_labels"] = str(Path(args.labels).resolve())
        else:
            raise ValueError("Task configuration must include 'test_labels'.")

    config["data_path"] = str(Path(config["data_path"]).resolve())
    if isinstance(config["test_labels"], (str, Path)):
        config["test_labels"] = str(Path(config["test_labels"]).resolve())

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Green Agent evaluation service with an A2A server interface."
    )
    parser.add_argument("--config", help="Path to a JSON file defining the task configuration.")
    parser.add_argument("--data-path", help="Directory containing train/val/test splits for evaluation.")
    parser.add_argument("--labels", help="CSV file containing ground-truth labels for the hidden test set.")
    parser.add_argument("--max-time-seconds", type=int, default=3600, help="Max allowed runtime for evaluation.")
    parser.add_argument("--max-memory-mb", type=int, default=8192, help="Max memory in MB for docker container.")
    parser.add_argument("--max-cpus", type=float, default=2.0, help="CPU limit exposed to docker.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the A2A HTTP server.")
    parser.add_argument("--port", type=int, default=9999, help="Port for the A2A HTTP server.")
    parser.add_argument("--public-url", help="Public URL advertised in the agent card.")
    parser.add_argument("--agent-name", default="Green Agent Evaluator", help="Display name for the agent card.")
    parser.add_argument(
        "--agent-description",
        help="Optional override for the agent card description.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the server process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    task_config = build_task_config(args)

    app = create_green_agent_app(
        task_config,
        public_url=args.public_url,
        agent_name=args.agent_name,
        agent_description=args.agent_description,
    )

    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
