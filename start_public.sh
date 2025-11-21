#!/bin/bash

# Cloudflare Tunnel with Correct URL in Agent Card

echo "🚀 Starting Green Agent with Cloudflare Tunnel (Correct URL)"
echo ""

# Load API keys
if [ -f ".env" ]; then
    source .env
fi

# Start Cloudflare tunnel FIRST to get URL
echo "1️⃣  Starting Cloudflare tunnel..."
cloudflared tunnel --url http://127.0.0.1:9999 > /tmp/cf_tunnel.log 2>&1 &
CF_PID=$!

# Wait for URL
echo "Waiting for tunnel URL..."
sleep 5

# Extract URL
PUBLIC_URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' /tmp/cf_tunnel.log | head -1)

if [ -z "$PUBLIC_URL" ]; then
    echo "❌ Failed to get Cloudflare URL. Check /tmp/cf_tunnel.log"
    kill $CF_PID 2>/dev/null
    exit 1
fi

echo "✅ Tunnel URL: $PUBLIC_URL"
echo ""

# Start Green Agent with the public URL
echo "2️⃣  Starting Green Agent server with public URL..."
python -m green_agent_cli serve \
    --config task_config.json \
    --host 127.0.0.1 \
    --port 9999 \
    --public-url "$PUBLIC_URL" &

SERVER_PID=$!
sleep 3

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ Server failed to start"
    kill $CF_PID 2>/dev/null
    exit 1
fi

echo "✅ Green Agent running (PID: $SERVER_PID)"
echo ""

echo "======================================"
echo "  ✅ All Services Running"
echo "======================================"
echo ""
echo "🌐 Public URL: $PUBLIC_URL"
echo "📋 Agent Card: $PUBLIC_URL/.well-known/agent-card.json"
echo ""
echo "Test it:"
echo "  curl $PUBLIC_URL/.well-known/agent-card.json"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Cleanup on exit
trap "echo ''; echo 'Stopping...'; kill $SERVER_PID $CF_PID 2>/dev/null; rm -f /tmp/cf_tunnel.log; exit" SIGINT SIGTERM

# Wait
wait
