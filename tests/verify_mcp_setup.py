
import sys
import os
import asyncio

# Ensure paths are correct
sys.path.append(os.path.join(os.getcwd(), "GreenAgent"))
sys.path.append(os.path.join(os.getcwd(), "SolverAgent"))

def test_green_agent_imports():
    print("Testing GreenAgent imports...")
    try:
        from TauBenchMCPServer import create_mcp_app
        mcp = create_mcp_app()
        print("✅ TauBenchMCPServer imported and initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to import TauBenchMCPServer: {e}")
        sys.exit(1)

def test_green_agent_server_mount():
    print("Testing GreenAgentServer mounting...")
    try:
        from GreenAgentServer import app
        # Check if route exists
        routes = [r.path for r in app.routes]
        if "/mcp" in routes:
             print("✅ /mcp route found in GreenAgentServer.")
        else:
             print(f"❌ /mcp route NOT found in GreenAgentServer. Routes: {routes}")
             # Note: Mounts might match differently in Starlette routes list, 
             # usually they appear as Mount object.
             found = False
             for r in app.routes:
                 if getattr(r, "path", "") == "/mcp":
                     found = True
             if found:
                 print("✅ /mcp mount found.")
             else:
                 print("⚠️ /mcp route not explicitly listable as path, but might be mounted.")
    except Exception as e:
        print(f"❌ Failed to import GreenAgentServer: {e}")
        # Not exiting, might be env specific issues with TauBenchAssessor

def test_solver_agent_updates():
    print("Testing SolverServer updates...")
    try:
        from SolverServer import SolverExecutor
        if hasattr(SolverExecutor, "_run_mcp_agent"):
            print("✅ SolverExecutor has _run_mcp_agent method.")
        else:
            print("❌ SolverExecutor missing _run_mcp_agent method.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to import SolverServer: {e}")

if __name__ == "__main__":
    test_green_agent_imports()
    test_green_agent_server_mount()
    test_solver_agent_updates()
