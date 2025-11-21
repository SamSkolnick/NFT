#!/bin/bash

# Green Agent - AgentBeats Controller Tunnel
# This script:
# 1. Starts a Cloudflare tunnel for the AgentBeats Controller (port 8010)
# 2. Captures the public URL
# 3. Starts the AgentBeats Controller with that URL

set -e

echo "🚀 Starting AgentBeats Controller with Cloudflare Tunnel"
echo "======================================================"

# 0. Cleanup previous runs
pkill -f "agentbeats run_ctrl" || true

# 1. Start Cloudflare Tunnel for Controller (Port 8010)
echo "1️⃣  Starting Cloudflare tunnel (Port 8010)..."
cloudflared tunnel --url http://127.0.0.1:8010 > /tmp/cf_tunnel.log 2>&1 &
CF_PID=$!

# Wait for the URL
echo "   Waiting for tunnel URL..."
PUBLIC_URL=""
for i in {1..60}; do
    if grep -q "https://.*\.trycloudflare\.com" /tmp/cf_tunnel.log; then
        PUBLIC_URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1)
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

if [ -z "$PUBLIC_URL" ]; then
    echo "❌ Failed to get Cloudflare URL. Check /tmp/cf_tunnel.log"
    kill $CF_PID 2>/dev/null
    exit 1
fi

echo "✅ Controller URL: $PUBLIC_URL"
echo ""

# 2. Start AgentBeats Controller
echo "2️⃣  Starting AgentBeats Controller..."
# Load API keys
if [ -f ".env" ]; then
    source .env
fi

# Export PUBLIC_URL so the Agent (started by Controller) inherits it
export PUBLIC_URL="$PUBLIC_URL"

# Configure AgentBeats Controller to use the public URL
# It uses CLOUDRUN_HOST and HTTPS_ENABLED to generate the correct proxy URLs
DOMAIN=$(echo "$PUBLIC_URL" | sed 's/https:\/\///')
export CLOUDRUN_HOST="$DOMAIN"
export HTTPS_ENABLED=true

echo "   (Press Ctrl+C to stop)"
echo ""
echo "======================================================"
echo "🎉 READY FOR AGENTBEATS!"
echo "======================================================"
echo ""
echo "👉 Register this URL on AgentBeats:"
echo "   $PUBLIC_URL"
echo ""
echo "   (The agent card is at: $PUBLIC_URL/.well-known/agent-card.json)"
echo ""

# Run Controller (this will start the agent via run.sh)
# Run Patched Controller (this will start the agent via run.sh)
python GreenAgentController.py

# Cleanup (runs when agentbeats exits)
kill $CF_PID 2>/dev/null
