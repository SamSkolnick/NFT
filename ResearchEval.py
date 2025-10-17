import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from LLMModule import call_openrouter_tongyi


def _normalize_research_payload(payload: Union[str, bytes, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(payload, Mapping):
        data = payload
    else:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        try:
            parsed = json.loads(text)
            data = parsed if isinstance(parsed, Mapping) else {"plan": text}
        except json.JSONDecodeError:
            data = {"plan": text}

    return {
        "task": str(data.get("task", "")),
        "plan": str(data.get("plan", "")),
        "resources": str(data.get("resources", "")),
    }


def _load_local_research(source: Union[str, os.PathLike, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(source, Mapping):
        return _normalize_research_payload(source)

    path = Path(source)
    if path.is_file():
        return _normalize_research_payload(path.read_text(encoding="utf-8"))

    if path.is_dir():
        candidates = [
            path / "research.json",
            path / "research.txt",
            path / "research.md",
        ]
        candidates.extend(sorted(path.glob("*.json")))
        candidates.extend(sorted(path.glob("*.txt")))
        for candidate in candidates:
            if candidate.exists():
                return _normalize_research_payload(candidate.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Could not locate research artifacts at {path}")


def _load_aws_research(source: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(source, Mapping):
        bucket = source.get("bucket")
        key = source.get("key")
    else:
        prefix = "s3://"
        if not str(source).startswith(prefix):
            raise ValueError("AWS research source must be a mapping or s3:// URI")
        bucket_key = str(source)[len(prefix):]
        bucket, _, key = bucket_key.partition("/")

    if not bucket or not key:
        raise ValueError("AWS research source requires both bucket and key")

    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for AWS storage but is not installed") from exc

    client = boto3.client("s3")
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return _normalize_research_payload(body)


def _load_dropbox_research(source: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(source, Mapping):
        path = source.get("path") or source.get("file")
        token = source.get("token") or os.environ.get("DROPBOX_ACCESS_TOKEN")
    else:
        token = os.environ.get("DROPBOX_ACCESS_TOKEN")
        path = str(source)

    if not token:
        raise ValueError("Dropbox access token not provided and DROPBOX_ACCESS_TOKEN is unset")

    if str(path).startswith("dropbox://"):
        path = str(path)[9:]

    if not path:
        raise ValueError("Dropbox research source must include a file path")

    try:
        import dropbox  # type: ignore
    except ImportError as exc:
        raise RuntimeError("dropbox package is required for Dropbox storage but is not installed") from exc

    client = dropbox.Dropbox(token)
    _, response = client.files_download(path)
    return _normalize_research_payload(response.content)


def evaluate_research(input_research: Union[str, os.PathLike, Mapping[str, Any]], storage: str = "local") -> int:
    """
    Intakes research artifacts that may be stored locally, on S3, or in Dropbox.

    Evaluates a proposed ML research plan using an LLM via OpenRouter.
    Returns an integer score (0–100).
    """
    storage_normalized = (storage or "local").lower()
    if storage_normalized == "local":
        research_payload = _load_local_research(input_research)
    elif storage_normalized == "aws":
        research_payload = _load_aws_research(input_research)
    elif storage_normalized == "dropbox":
        research_payload = _load_dropbox_research(input_research)
    else:
        raise ValueError(f"Unsupported storage backend: {storage}")

    task = research_payload.get("task", "")
    plan = research_payload.get("plan", "")
    resources = research_payload.get("resources", "")
    example_research1 = (
        "Task: Image classification for diabetic retinopathy detection.\n"
        "Plan: Uses transfer learning with EfficientNet, applies class-balanced loss, "
        "evaluates on APTOS dataset with 5-fold CV, ensures no data leakage.\n"
        "Resources: Kaggle GPUs, public medical datasets."
    )
    example_research2 = (
        "Task: Predict protein–ligand binding affinity.\n"
        "Plan: Fine-tunes ESM-2 embeddings with GNNs over AlphaFold structures, "
        "benchmarks against PDBBind and CASF-2016, reports MAE/ΔG correlation.\n"
        "Resources: AWS A100s, AlphaFold DB."
    )

    # Prompt template
    input_prompt = f"""
    You are a senior machine learning researcher evaluating the quality of a proposed research plan.

    Scoring criteria (each 0–25 points):
    1. Clarity — Is the task and goal well-defined and measurable?
    2. Rigor — Does the plan include proper baselines, evaluation, and avoidance of data leakage?
    3. Feasibility — Are the methods and resources realistic?
    4. Novelty — Does it extend or combine ideas in a meaningful way?
    5. Accuracy - Is the information correct?
    6. Relevance - Does the research relate to solving the given project 

    Two examples of well-structured research:
    ---
    {example_research1}
    ---
    {example_research2}

    Now evaluate the following proposal and give only an integer score between 0 and 100.

    Task:
    {task}

    Plan:
    {plan}

    Resources:
    {resources}
    """

    # Model call
    response = call_openrouter_tongyi(input_prompt)
    try:
        score = int(''.join([c for c in response if c.isdigit()]))
        score = max(0, min(score, 100))
    except ValueError:
        score = 0

    return score
