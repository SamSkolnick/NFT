#!/bin/bash

# Simple Cloudflare Tunnel for Green Agent
# Just run: ./cloudflare_tunnel.sh

echo "🚀 Starting Green Agent with Cloudflare Tunnel"
echo ""

# Load API keys
if [ -f ".env" ]; then
    source .env
fi

# Start Green Agent in background
echo "1️⃣  Starting Green Agent server..."
python -m green_agent_cli serve \
    --config task_config.json \
    --host 127.0.0.1 \
    --port 9999 &

SERVER_PID=$!
sleep 3

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ Server failed to start"
    exit 1
fi

echo "✅ Green Agent running (PID: $SERVER_PID)"
echo ""

# Start Cloudflare tunnel
echo "2️⃣  Starting Cloudflare tunnel..."
echo ""
cloudflared tunnel --url http://127.0.0.1:9999 &
CF_PID=$!

echo ""
echo "======================================"
echo "  ✅ Services Running"
echo "======================================"
echo ""
echo "📝 Look for your public URL above (https://...trycloudflare.com)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Cleanup on exit
trap "echo ''; echo 'Stopping...'; kill $SERVER_PID $CF_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait
wait
