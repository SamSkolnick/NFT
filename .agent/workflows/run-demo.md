---
description: How to run the GreenAgent Evaluation Demo for presentation
---

To run the GreenAgent Evaluation Demo:

1. **Start the Demo Server**:
   Ensure you are in the project root directory and run:
   ```bash
   python3 demo_server.py
   ```
   *Note: If a server is already running, you may need to stop it first or use `pkill -f demo_server.py`.*

2. **Open the Dashboard**:
   Open your browser and navigate to:
   [http://localhost:8080](http://localhost:8080)

3. **Presentation Flow**:
   - **Select "Baseline Agent (Bad)"**: Review the naive research report, then click **START EVALUATION** to show the failure (low accuracy, rejected verdict).
   - **Select "SOTA Agent (Good)"**: Review the professional research report, then click **START EVALUATION** to show the success (>70% accuracy, passed verdict).
