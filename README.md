# AgentBeats Integration Demo

This project demonstrates the integration between a **Green Agent** (Evaluator) and a **Solver Agent** (White Agent) using the AgentBeats protocol.

## Directory Structure

*   `GreenAgent/`: Contains the Green Agent (Evaluator) code, controller, and server.
*   `SolverAgent/`: Contains the White Agent code, model, and server.
*   `configs/`: Configuration files for evaluation tasks.
*   `manage_agents.py`: Unified CLI to manage agents and run demonstrations.
*   `requirements.txt`: Project dependencies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Ensure you have `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` set if you want to use the research evaluation features of the Green Agent.

## Usage

Use the `manage_agents.py` script to control the ecosystem.

### 1. Run Agent-to-Agent (A2A) Demo
This demonstrates the Green Agent connecting to a remote Solver Agent (running locally on port 8005), requesting a solution for the Titanic challenge, and evaluating the result.

```bash
python manage_agents.py demo-a2a
```
*   **Green Agent**: Starts on Port 8000.
*   **Solver Agent**: Starts on Port 8005.
*   **Action**: Green Agent sends a task to Solver Agent, receives predictions, and prints the evaluation score.

### 2. Manual Agent Management (Decoupled Workflow)
This workflow allows you to start agents independently and hook them up explicitly.

**Step 1: Start Green Agent**
```bash
python manage_agents.py start-green --tunnel
# Starts on localhost, but exposes a PUBLIC URL via Cloudflare.
# Look for: [Tunnel] Verified URL: https://....trycloudflare.com
```

**Step 2: Start Solver Agent**
In a new terminal:
```bash
python manage_agents.py start-solver --tunnel
# Starts on localhost, but exposes a PUBLIC URL via Cloudflare.
# Look for: [Tunnel] Verified URL: https://....trycloudflare.com
```

**Step 3: Trigger Evaluation**
Use the public URLs printed above:
```bash
python manage_agents.py submit-task --green-url https://<green-url>.trycloudflare.com --solver-url https://<solver-url>.trycloudflare.com
```

### 3. Utility Commands
```bash
# Stop all agents
python manage_agents.py stop-all
```

## Adding New Skills
To add a new challenge:
1.  Add a generic config in `configs/<challenge_name>.json`.
2.  Update `SolverAgent` to load the appropriate model/logic based on the challenge name (currently hardcoded for Titanic but architected for generic support).
