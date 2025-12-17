#!/bin/bash
uvicorn GreenAgentServer:app --host ${HOST:-0.0.0.0} --port ${AGENT_PORT:-8000}
