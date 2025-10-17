# LLMModule.py
# Interface for calling LLMs for agent to make it easier to switch them out
# pip install openai
from openai import OpenAI
import os

def call_openrouter_tongyi(prompt: str) -> str:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    resp = client.chat.completions.create(
        model="alibaba/tongyi-deepresearch-30b-a3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=0,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content
