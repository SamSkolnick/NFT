# CodeEval.py
from __future__ import annotations
import json
from typing import List
from LLMModule import call_openrouter_tongyi


def evaluate(kaggle_prompt: str, source_code: str, predictions: List[float]) -> int:
    """
    LLM-only scoring focused on TWO things:
      1) Code quality / expected behavior for a Kaggle pipeline.
      2) Whether the CSV-style predictions align with the competition's expectations.

    Returns an integer score in [1, 10].
    """
    preds_preview, pred_count = _preview_predictions(predictions)

    prompt = f"""
You are a strict Kaggle pre-submission reviewer. Score ONLY:
(1) code quality & expected behavior, and
(2) whether the predictions/CSV align with the competition requirements.

Ignore scientific novelty, model choice, and performance beyond schema/expectations.

Return ONE integer score in [1,10] (1=unusable, 10=excellent) using this EXACT rubric:

Components (each 0..10; use defaults if info is missing):
- code_quality (default 5): clean structure, sensible imports, readable functions, clear data I/O, fixed seeds, no obviously dangerous calls, minimal hygiene (paths/configs).
- expected_behavior (default 5): pipeline likely runs end-to-end without errors for typical Kaggle limits; uses correct metric/validation shape; produces a submission in the required format.
- csv_alignment (default 5): predictions match the competition's submission expectations (e.g., correct length, type/range, and format). For the HIV task described, expect 692 rows of binary labels 0/1 in the second column; if only a list of predictions is shown here, judge whether values look like valid 0/1 labels and the count matches 692.

Exact formula (no deviation):
S_raw = 0.45*code_quality + 0.35*expected_behavior + 0.20*csv_alignment
S_10  = round( max(1, min(10, S_raw)) )

Return STRICT JSON and NOTHING ELSE:
{{ "score": <integer in [1,10]> }}

--- VERSION ---
code_csv_focus_v1

--- KAGGLE_PROMPT (for context) ---
{_normalize(kaggle_prompt)}

--- SOURCE_CODE (Python) ---
{_normalize(source_code)}

--- PREDICTIONS PREVIEW ---
# count: {pred_count}
# head (up to 20 shown):
{preds_preview}
""".strip()

    raw = call_openrouter_tongyi(prompt)
    score = _extract_score(raw)
    score = int(max(1, min(10, round(score))))
    return score


def _preview_predictions(preds: List[float]) -> tuple[str, int]:
    if not preds:
        return "[]", 0
    preview = preds[:20]
    try:
        return json.dumps([float(x) for x in preview]), len(preds)
    except Exception:
        return json.dumps([str(x) for x in preview]), len(preds)


def _normalize(text: str) -> str:
    if not text:
        return ""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _extract_score(raw_text: str) -> int:
    try:
        obj = json.loads(raw_text)
        return _to_int_1_10(obj["score"])
    except Exception:
        pass
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw_text[start:end+1])
            return _to_int_1_10(obj["score"])
        except Exception:
            pass
    raise ValueError(f"LLM did not return a valid JSON score in [1,10]. Raw response: {raw_text!r}")


def _to_int_1_10(val) -> int:
    try:
        num = float(val)
    except Exception:
        raise ValueError("score is not numeric")
    if num < 1 or num > 10:
        num = max(1.0, min(10.0, num))
    return int(round(num))
