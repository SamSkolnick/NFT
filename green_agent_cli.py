from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import uvicorn

from GreenAgent import GreenAgent
from GreenAgentServer import create_green_agent_app


def ensure_python310() -> None:
    """Re-exec under python3.10 if required."""
    if sys.version_info >= (3, 10):
        return

    python310 = shutil.which("python3.10")
    if python310:
        os.execv(python310, [python310, *sys.argv])

    raise SystemExit(
        "The Green Agent requires Python 3.10 or newer. Install dependencies with "
        "`python3.10 -m pip install -r requirements.txt` and rerun the command."
    )


def add_task_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to a JSON file defining the task configuration.")
    parser.add_argument("--data-path", help="Directory containing train/val/test splits for evaluation.")
    parser.add_argument("--labels", help="CSV file containing ground-truth labels for the hidden test set.")
    parser.add_argument("--max-time-seconds", type=int, default=3600, help="Max allowed runtime for evaluation.")
    parser.add_argument("--max-memory-mb", type=int, default=8192, help="Max memory in MB for docker container.")
    parser.add_argument("--max-cpus", type=float, default=2.0, help="CPU limit exposed to docker.")


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
    test_labels = config["test_labels"]
    if isinstance(test_labels, (str, Path)):
        config["test_labels"] = str(Path(test_labels).resolve())

    return config


def load_submission(submission_arg: Optional[str]) -> Optional[Dict[str, Any]]:
    if not submission_arg:
        return None

    submission_path = Path(submission_arg)
    if submission_path.exists():
        content = submission_path.read_text(encoding="utf-8")
    else:
        content = submission_arg

    try:
        loaded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid submission JSON: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError("Submission payload must be a JSON object.")
    return loaded


def build_submission_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    submission = load_submission(args.submission)
    if submission:
        return submission

    if not args.docker_image or not args.research_artifacts:
        raise ValueError("Either --submission or (--docker-image and --research-artifacts) must be provided.")

    submission = {
        "docker_image": args.docker_image,
        "research_artifacts": args.research_artifacts,
        "storage_method": args.storage_method,
        "pull_image": not args.no_pull_image,
    }
    if args.eval_command:
        submission["eval_command"] = args.eval_command

    credentials = normalize_credentials(
        username=args.docker_username,
        password=args.docker_password,
        email=args.docker_email,
        registry=args.docker_registry,
        identitytoken=args.docker_identity_token,
    )
    if credentials:
        submission["docker_credentials"] = credentials

    return submission


def normalize_credentials(**kwargs: Optional[str]) -> Dict[str, str]:
    allowed = {key: value for key, value in kwargs.items() if value}
    return allowed


def run_server(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    task_config = build_task_config(args)

    app = create_green_agent_app(
        task_config,
        public_url=args.public_url,
        agent_name=args.agent_name,
        agent_description=args.agent_description,
    )
    uvicorn.run(app.build(), host=args.host, port=args.port)


def run_evaluation(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level))
    task_config = build_task_config(args)
    submission = build_submission_from_args(args)

    agent = GreenAgent(task_config)
    results = agent.evaluate(submission)
    output = json.dumps(results, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")

    print(output)
    return 0 if results.get("execution", {}).get("success") else 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the Green Agent evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the Green Agent as an A2A HTTP server.")
    add_task_options(serve_parser)
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface for the A2A HTTP server.")
    serve_parser.add_argument("--port", type=int, default=9999, help="Port for the A2A HTTP server.")
    serve_parser.add_argument("--public-url", help="Public URL advertised in the agent card.")
    serve_parser.add_argument("--agent-name", default="Green Agent Evaluator", help="Display name for the agent.")
    serve_parser.add_argument("--agent-description", help="Optional override for the agent description.")
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the server process.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Run a single submission evaluation from the CLI.")
    add_task_options(eval_parser)
    eval_parser.add_argument(
        "--submission",
        help="Path to a JSON file or inline JSON payload describing the submission.",
    )
    eval_parser.add_argument("--docker-image", help="Name of the docker image to evaluate.")
    eval_parser.add_argument("--research-artifacts", help="Path or URI to research artifacts.")
    eval_parser.add_argument("--storage-method", default="local", help="Storage backend for research artifacts.")
    eval_parser.add_argument("--eval-command", help="Override the container command (default: python evaluate.py).")
    eval_parser.add_argument(
        "--no-pull-image",
        action="store_true",
        help="Skip docker pull even if credentials are provided.",
    )
    eval_parser.add_argument("--docker-username", help="Docker registry username.")
    eval_parser.add_argument("--docker-password", help="Docker registry password.")
    eval_parser.add_argument("--docker-email", help="Docker registry email.")
    eval_parser.add_argument("--docker-registry", help="Docker registry URL.")
    eval_parser.add_argument("--docker-identity-token", help="Docker registry identity token.")
    eval_parser.add_argument("--output", help="Optional path to write the evaluation results as JSON.")
    eval_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the evaluation run.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_python310()

    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        run_server(args)
        return 0

    if args.command == "evaluate":
        return run_evaluation(args)

    parser.error(f"Unknown command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
