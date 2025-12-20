# Helper to make switching between LLM providers easier.
# This works with both the new openai>=1.0 stuff and the older 0.x versions.
import os
from typing import Any, Dict

try:  # Try using the new SDK if it's there
    from openai import OpenAI as _OpenAIClient  # type: ignore
except ImportError:
    _OpenAIClient = None


def _resolve_api_key() -> str:
    # Look for the OpenRouter key in the environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")
    return api_key


def _call_via_new_sdk(prompt: str, api_key: str, base_url: str) -> str:
    # Use the new OpenAI client style
    client = _OpenAIClient(base_url=base_url, api_key=api_key)  # type: ignore
    model_name = os.environ.get("OPENROUTER_MODEL", "alibaba/tongyi-deepresearch-30b-a3b")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    message = response.choices[0].message.content
    if not message:
        raise RuntimeError("OpenRouter returned an empty response.")
    return message


def _call_via_legacy_sdk(prompt: str, api_key: str, base_url: str) -> str:
    # Fallback for older openai installations
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package isn't installed.") from exc

    openai.api_key = api_key  # type: ignore[attr-defined]
    openai.api_base = base_url  # type: ignore[attr-defined]

    model_name = os.environ.get("OPENROUTER_MODEL", "alibaba/tongyi-deepresearch-30b-a3b")
    response: Dict[str, Any] = openai.ChatCompletion.create(  # type: ignore[attr-defined]
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Empty response from OpenRouter (legacy SDK).")
    message = choices[0].get("message", {}).get("content")
    if not message:
        raise RuntimeError("No message content in legacy response.")
    return str(message)


def call_openrouter_tongyi(prompt: str) -> str:
    """
    Kicks off a chat request to OpenRouter's Tongyi model.
    It automatically handles whichever OpenAI SDK version is installed.
    """
    api_key = _resolve_api_key()
    base_url = "https://openrouter.ai/api/v1"

    if _OpenAIClient is not None:
        return _call_via_new_sdk(prompt, api_key, base_url)

    return _call_via_legacy_sdk(prompt, api_key, base_url)

