#!/bin/bash

# Green Agent with Cloudflare Tunnel
# Automatically creates a public URL for your A2A server

set -e

echo "======================================"
echo "  Green Agent - Cloudflare Setup"
echo "======================================"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "⚠️  Cloudflared not found. Installing..."
    
    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install cloudflared
        else
            echo "Error: Homebrew not found. Please install from https://brew.sh/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        sudo dpkg -i cloudflared-linux-amd64.deb
        rm cloudflared-linux-amd64.deb
    else
        echo "Please install cloudflared manually from:"
        echo "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        exit 1
    fi
    
    echo "✓ Cloudflared installed"
fi

# Load environment if exists
if [ -f ".env" ]; then
    source .env
    echo "✓ Loaded API keys from .env"
fi

# Get port
PORT=${1:-9999}
echo "Using port: $PORT"
echo ""

# Create log directory
mkdir -p logs

# Start Green Agent A2A server in background
echo "Starting Green Agent A2A server..."
python -m green_agent_cli serve \
    --config task_config.json \
    --host 127.0.0.1 \
    --port $PORT \
    > logs/green_agent.log 2>&1 &

SERVER_PID=$!
echo "✓ Green Agent started (PID: $SERVER_PID)"

# Wait for server to be ready
echo "Waiting for server to start..."
sleep 3

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ Server failed to start. Check logs/green_agent.log"
    exit 1
fi

# Start Cloudflare tunnel
echo ""
echo "Starting Cloudflare tunnel..."
echo "======================================"

# Run cloudflared and capture the URL
cloudflared tunnel --url http://127.0.0.1:$PORT > logs/cloudflare.log 2>&1 &
CF_PID=$!

# Wait for URL to be available
echo "Waiting for tunnel URL..."
for i in {1..30}; do
    # Look for the URL in the log file
    if grep -q "trycloudflare.com" logs/cloudflare.log; then
        grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' logs/cloudflare.log | head -1 > /tmp/cf_url.txt
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

if [ ! -s /tmp/cf_url.txt ]; then
    echo "❌ Failed to get Cloudflare URL"
    kill $SERVER_PID $CF_PID 2>/dev/null
    exit 1
fi

PUBLIC_URL=$(cat /tmp/cf_url.txt)
echo ""
echo "======================================"
echo "  ✓ Green Agent is PUBLIC!"
echo "======================================"
echo ""
echo "🌐 Public URL: $PUBLIC_URL"
echo "📋 Agent Card: $PUBLIC_URL/.well-known/agent-card.json"
echo ""
echo "Example A2A request:"
echo "  curl $PUBLIC_URL/.well-known/agent-card.json"
echo ""
echo "Logs:"
echo "  - Green Agent: logs/green_agent.log"
echo "  - Cloudflare: logs/cloudflare.log"
echo ""
echo "Press Ctrl+C to stop both services"
echo "======================================"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SERVER_PID $CF_PID 2>/dev/null
    rm -f /tmp/cf_url.txt
    echo "✓ Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
wait $CF_PID
