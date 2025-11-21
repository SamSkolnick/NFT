# Cloudflare Tunnel Setup for Green Agent

## Quick Start (Automatic Setup)

```bash
./start_with_cloudflare.sh
```

This script will:
1. ✅ Install `cloudflared` (if needed)
2. ✅ Start the Green Agent A2A server
3. ✅ Create a Cloudflare tunnel
4. ✅ Print your public URL

**Example Output:**
```
======================================
  ✓ Green Agent is PUBLIC!
======================================

🌐 Public URL: https://abc-123-def.trycloudflare.com
📋 Agent Card: https://abc-123-def.trycloudflare.com/.well-known/agent-card.json
```

---

## Manual Setup (Step-by-Step)

### 1. Install Cloudflared

**macOS:**
```bash
brew install cloudflared
```

**Linux:**
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### 2. Start Green Agent

```bash
# Terminal 1 - Start Green Agent
source .env  # Load API keys
python -m green_agent_cli serve \
  --config task_config.json \
  --host 127.0.0.1 \
  --port 9999
```

### 3. Start Cloudflare Tunnel

```bash
# Terminal 2 - Start tunnel
cloudflared tunnel --url http://127.0.0.1:9999
```

**Output will show:**
```
2024-01-20 16:20:00 INF |  https://abc-123-def.trycloudflare.com
```

### 4. Test Your Public URL

```bash
# Copy the URL from step 3 and test
curl https://abc-123-def.trycloudflare.com/.well-known/agent-card.json
```

---

## Using with AgentBeats

### Option 1: Direct Integration

Update `run.sh` to include the public URL:

```bash
#!/bin/bash
# Get Cloudflare URL (you'll need to set this)
PUBLIC_URL="https://your-tunnel.trycloudflare.com"

uvicorn GreenAgentServer:app \
  --host 0.0.0.0 \
  --port 8000 \
  --env PUBLIC_URL=$PUBLIC_URL
```

### Option 2: Use Named Tunnel (Persistent URL)

Create a named tunnel for a permanent URL:

```bash
# 1. Authenticate with Cloudflare
cloudflared tunnel login

# 2. Create a named tunnel
cloudflared tunnel create green-agent

# 3. Configure the tunnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: <TUNNEL-ID>
credentials-file: /Users/samuelskolnick/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: green-agent.yourdomain.com
    service: http://127.0.0.1:9999
  - service: http_status:404
EOF

# 4. Route DNS
cloudflared tunnel route dns green-agent green-agent.yourdomain.com

# 5. Run the tunnel
cloudflared tunnel run green-agent
```

---

## Registering on AgentBeats Platform

### 1. Start with Public URL

```bash
./start_with_cloudflare.sh
```

### 2. Copy Your Public URL

From the output:
```
🌐 Public URL: https://abc-123-def.trycloudflare.com
```

### 3. Register on AgentBeats

Go to https://agentbeats.org and:
1. Click "Register Agent"
2. Paste your public URL
3. The platform will fetch your agent card automatically
4. Submit

### 4. Test Evaluation

Other agents can now discover and evaluate with your Green Agent!

---

## A2A Client Example

```python
from a2a.client import A2AClient

# Use your public URL
client = A2AClient("https://abc-123-def.trycloudflare.com")

# Create evaluation task
task = client.create_task({
    "docker_image": "my-agent:latest",
    "research_artifacts": "path/to/research"
})

# Get results
results = client.get_artifacts(task.id)
print(results)
```

---

## Important Notes

### Free Tunnel Limitations
- ⚠️ URL changes on restart (use named tunnel for persistence)
- ⚠️ No bandwidth limits
- ⚠️ Suitable for testing and demos

### Named Tunnel Benefits
- ✅ Permanent URL
- ✅ Custom domain support
- ✅ Better for production
- ✅ Free for personal use

### Security
- The tunnel is **public** - anyone with the URL can access it
- Consider adding authentication for production
- Monitor usage in Cloudflare dashboard

---

## Troubleshooting

### "Connection refused"
Make sure the Green Agent server is running first:
```bash
ps aux | grep "green_agent_cli serve"
```

### "Tunnel failed to start"
Check if port 9999 is already in use:
```bash
lsof -i :9999
```

### URL not appearing
Wait a few seconds for the tunnel to establish. Check logs:
```bash
tail -f logs/cloudflare.log
```

---

## Advanced: Production Deployment

For production, use a persistent Cloudflare tunnel with:

1. **Named tunnel** (permanent URL)
2. **Access policies** (authentication)
3. **Rate limiting** (protection)
4. **Custom domain** (branding)

See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `./start_with_cloudflare.sh` | Auto-start with tunnel |
| `cloudflared tunnel --url http://127.0.0.1:9999` | Manual tunnel |
| `cloudflared tunnel login` | Setup named tunnel |
| Logs in `logs/` directory | Check for errors |

---

**Ready to go public? Run `./start_with_cloudflare.sh`** 🚀
