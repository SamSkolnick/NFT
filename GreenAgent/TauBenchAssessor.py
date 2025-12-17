import sys
import os
import uuid
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Ensure we can import from tau-bench-main
# Assuming absolute path or relative to MLEngineer
TAU_BENCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tau-bench-main"))
if TAU_BENCH_PATH not in sys.path:
    sys.path.append(TAU_BENCH_PATH)

from tau_bench.envs import get_env
from tau_bench.types import EnvResetResponse, EnvResponse, Action

logger = logging.getLogger(__name__)

class EnvSession:
    def __init__(self, env):
        self.env = env
        self.id = uuid.uuid4().hex
        self.created_at = os.times().elapsed

class TauBenchAssessor:
    def __init__(self):
        self.sessions: Dict[str, EnvSession] = {}
        
    def create_session(self, env_name: str = "retail", task_split: str = "test", user_model: str = "gpt-4o") -> str:
        """Initializes a new Tau Bench Environment and returns a session ID."""
        # Use defaults or config
        env = get_env(
            env_name=env_name,
            user_strategy="llm",
            user_model=user_model,
            task_split=task_split,
            user_provider="openai", # Assuming OpenAI for valid user sim
        )
        session = EnvSession(env)
        self.sessions[session.id] = session
        logger.info(f"Created Tau Bench session {session.id} for env {env_name}")
        return session.id

    def reset(self, session_id: str, task_index: int) -> EnvResetResponse:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        logger.info(f"Resetting session {session_id} to task {task_index}")
        return session.env.reset(task_index=task_index)

    def step(self, session_id: str, action: Action) -> EnvResponse:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        logger.info(f"Stepping session {session_id} with action {action.name}")
        return session.env.step(action)

    def get_tools_info(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session.env.tools_info
        
    def get_wiki(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session.env.wiki
