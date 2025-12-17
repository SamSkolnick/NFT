import logging
import os
import signal
import sys
import uvicorn
from warnings import filterwarnings

# AgentBeats imports
# We need to define settings locally as we are patching the environment
from pydantic import Field
from pydantic_settings import BaseSettings

class ControllerSettings(BaseSettings):
    """Configuration for the Agent Controller."""
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8005, env="PORT") # Default to 8005 for Solver
    agent_module: str = Field("SolverServer", env="AGENT_MODULE")
    agent_app: str = Field("app", env="AGENT_APP")
    log_level: str = Field("info", env="LOG_LEVEL")
    agent_url: str = Field(None, env="AGENT_URL")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TitanicSolverController")

def start_server():
    """Start the uvicorn server for the agent."""
    settings = ControllerSettings()
    
    # Check if cloudflared is available (optional integration)
    # For local demo, we just focus on local port
    
    logger.info(f"Starting Solver Agent Controller on {settings.host}:{settings.port}")
    
    # If AGENT_URL is not set, set it to localhost
    if not os.environ.get("AGENT_URL"):
        os.environ["AGENT_URL"] = f"http://127.0.0.1:{settings.port}"

    try:
        uvicorn.run(
            f"{settings.agent_module}:{settings.agent_app}",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
            reload=True,  # Enable auto-reload for development
        )
    except KeyboardInterrupt:
        logger.info("Shutting down controller...")
    except Exception as e:
        logger.error(f"Controller failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Handle SIGTERM/SIGINT
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    start_server()
