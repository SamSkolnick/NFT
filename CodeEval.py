# CodeEval.py
from __future__ import annotations
import json
from typing import List
from LLMModule import call_openrouter_tongyi  # uses your OpenRouter/OpenAI-compatible interface


def evaluate(kaggle_prompt: str, source_code: str, predictions: List[float]) -> float:
    """
    Score a submission using ONLY the LLM.
    Inputs:
      - kaggle_prompt: competition/task plan text
      - source_code:   pipeline/source code as a string
      - predictions:   list of numeric predictions (labels or probabilities)
    Returns:
      - score: float in [0, 1] produced by the LLM
    Raises:
      - ValueError if the LLM response cannot be parsed or doesn't provide a score.
    """
    preds_preview, pred_count = _preview_predictions(predictions)

    prompt = f"""
You are a strict Kaggle pre-submission reviewer. Evaluate the quality of this submission
PURELY as a single numeric score in [0,1] (0 = unusable, 1 = excellent).
Consider: clarity/rigor of the plan, code soundness for competitive ML (sanity, hygiene,
reproducibility risk, leakage risk), and whether the predictions look plausible for the task.

Return STRICT JSON with this schema and NOTHING ELSE:
{{
  "score": <number in [0,1]>
}}

--- KAGGLE_PROMPT ---
{kaggle_prompt}

--- SOURCE_CODE (Python) ---
{source_code}

--- PREDICTIONS ---
# count: {pred_count}
# sample (up to 200):
{preds_preview}
""".strip()

    raw = call_openrouter_tongyi(prompt)  # must return text; we expect JSON with {"score": ...}
    score = _extract_score(raw)
    return score


def _preview_predictions(preds: List[float]) -> tuple[str, int]:
    if preds is None:
        return "[]", 0
    # keep prompt small but informative
    preview = preds[:200]
    return json.dumps(preview), len(preds)


def _extract_score(raw_text: str) -> float:
    # try direct JSON
    try:
        obj = json.loads(raw_text)
        score = float(obj["score"])
        if 0.0 <= score <= 1.0:
            return score
    except Exception:
        pass

    # try to find a JSON object substring
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw_text[start:end+1])
            score = float(obj["score"])
            if 0.0 <= score <= 1.0:
                return score
        except Exception:
            pass

    raise ValueError(f"LLM did not return a valid JSON score in [0,1]. Raw response: {raw_text!r}")
