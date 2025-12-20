
import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from TauBenchAssessor import TauBenchAssessor
from tau_bench.types import Action

# Initialize logger
logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("TauBench Environment")

# Global assessor instance (ensuring state persistence across calls)
# In a real multi-user scenario, we might need a session manager, 
# but GreenAgentServer is currently single-tenant per run or manages sessions via ID.
assessor = TauBenchAssessor()

@mcp.tool()
def init_env(env_name: str = "retail", task_split: str = "test") -> str:
    """Initialize a new TauBench environment session. Returns the session_id."""
    try:
        session_id = assessor.create_session(env_name=env_name, task_split=task_split)
        return json.dumps({"session_id": session_id})
    except Exception as e:
        return f"Error initializing environment: {str(e)}"

@mcp.tool()
def reset_env(session_id: str, task_index: Optional[int] = None) -> str:
    """Reset the environment for the given session to a specific task."""
    try:
        obs = assessor.reset(session_id=session_id, task_index=task_index)
        return obs.model_dump_json()
    except ValueError as e:
        return f"Error resetting environment: {str(e)}"

@mcp.tool()
def step_env(session_id: str, action_json: str) -> str:
    """Execute a step in the environment. action_json must be a valid JSON string for Action."""
    try:
        data = json.loads(action_json)
        action = Action(**data)
        obs = assessor.step(session_id=session_id, action=action)
        return obs.model_dump_json()
    except Exception as e:
        return f"Error executing step: {str(e)}"

@mcp.tool()
def get_tools(session_id: str) -> str:
    """Get the list of tools available in the current environment."""
    try:
        tools = assessor.get_tools_info(session_id)
        return json.dumps(tools)
    except Exception as e:
        return f"Error fetching tools: {str(e)}"

@mcp.resource("wiki://{session_id}/current")
def get_wiki(session_id: str) -> str:
    """Get the wiki/instruction content for the current task."""
    try:
        return assessor.get_wiki(session_id)
    except Exception as e:
        return f"Error fetching wiki: {str(e)}"

def create_mcp_app():
    """
    Returns the Starlette app (SSE handler) for embedding in another service.
    """
    # FastMCP provides an internal mechanism to get the ASGI app, usually via .mount_sse() 
    # but that binds to a parent app. 
    # For integration, we can expose the underlying standardized objects if needed, 
    # but FastMCP is easiest used via its run method or direct mounting.
    
    # Since we want to mount this *inside* GreenAgentServer, 
    # we return the mcp object so we can mount it there.
    return mcp
