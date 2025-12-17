#!/bin/bash
uvicorn WhiteAgentServer:app --host 0.0.0.0 --port $AGENT_PORT
