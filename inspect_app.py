
import os
import sys
from a2a.server.apps import A2AStarletteApplication
from GreenAgentServer import create_green_agent_app

print("Inspecting A2AStarletteApplication...")
try:
    # Create a dummy app
    app_instance = create_green_agent_app({"data_path": "."})
    print(f"Type: {type(app_instance)}")
    print(f"Is callable? {callable(app_instance)}")
    print(f"Dir: {dir(app_instance)}")
    
    if hasattr(app_instance, 'app'):
        print(f"Has .app attribute: {type(app_instance.app)}")
        print(f"Is .app callable? {callable(app_instance.app)}")
        
    if hasattr(app_instance, 'starlette_app'):
        print(f"Has .starlette_app attribute: {type(app_instance.starlette_app)}")
        
except Exception as e:
    print(f"Error: {e}")
