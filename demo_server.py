
import json
import os
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from pathlib import Path
import sys

# Add GreenAgent to path
sys.path.append(os.path.join(os.getcwd(), "GreenAgent"))
from GreenAgent import GreenAgent

# Configuration
SPAMHAM_LABELS = "/Users/samuelskolnick/MLEngineer/Datasets/spamham/spamham_demo_labels.csv"
SPAMHAM_REPORT_GOOD = "/Users/samuelskolnick/MLEngineer/Datasets/spamham/research_report.md"
SPAMHAM_REPORT_BAD = "/Users/samuelskolnick/MLEngineer/Datasets/spamham/bad_research_report.md"

agent = GreenAgent({"test_labels": SPAMHAM_LABELS})

async def homepage(request):
    return FileResponse("demo_index.html")

async def evaluate_agent(request):
    """
    Endpoint to trigger evaluation.
    Expects JSON: { "agent_type": "baseline" | "optimized" | "garbage" }
    """
    try:
        data = await request.json()
        agent_type = data.get("agent_type", "baseline")
        
        preds_file = f"demo_{agent_type}_preds.csv"
        report_file = SPAMHAM_REPORT_BAD if agent_type == "baseline" else SPAMHAM_REPORT_GOOD
        
        submission = {
            "predictions_path": os.path.abspath(preds_file),
            "research_artifacts": report_file
        }
        
        result = agent.evaluate(submission)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_report(request):
    agent_type = request.query_params.get("agent_type", "baseline")
    report_file = SPAMHAM_REPORT_BAD if agent_type == "baseline" else SPAMHAM_REPORT_GOOD
    return FileResponse(report_file)

routes = [
    Route("/", homepage),
    Route("/evaluate", evaluate_agent, methods=["POST"]),
    Route("/report", get_report),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    print("GreenAgent Demo Server starting at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
