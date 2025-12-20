# Green Agent - Quick Start Guide

## Super Easy Setup (Recommended)

### 1. Run the Startup Script

```bash
./start_green_agent.sh
```

The script will:
- ✅ Prompt for your API keys
- ✅ Optionally save them to `.env` file for future use
- ✅ Provide a menu to run evaluations
- ✅ Handle all environment configuration

### 2. Enter Your API Keys

When prompted:
- **Anthropic API Key**: Required for research evaluation (get from https://console.anthropic.com/)
- **OpenRouter API Key**: Optional (alternative LLM provider)

### 3. Choose an Action

The menu offers:
1. **Run evaluation** - Quick Titanic example
2. **Run custom evaluation** - Evaluate your own agent
3. **Interactive test runner** - Manual testing interface
4. **AgentBeats controller** - Production deployment
5. **A2A server** - Start as A2A service
6. **Reconfigure keys** - Update API keys
7. **Exit**

---

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Create `.env` file

```bash
cat > .env << 'EOF'
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export ANTHROPIC_MODEL="claude-sonnet-4-5"
export OPENROUTER_API_KEY="sk-or-v1-..."  # Optional
export TOKENIZERS_PARALLELISM=false
EOF
```

### 2. Load Environment

```bash
source .env
```

### 3. Run Evaluation

```bash
python -m green_agent_cli evaluate \
  --config task_config.json \
  --agent-url http://localhost:8005
```

---

## First Time Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from GreenAgent import GreenAgent; print('✓ Green Agent ready')"
```

---

## Common Commands

### Using the Startup Script
```bash
./start_green_agent.sh
```

### Quick Evaluation
```bash
source .env  # If you have .env file
python -m green_agent_cli evaluate \
  --config task_config.json \
  --agent-url http://localhost:8005
```

### Interactive Testing
```bash
source .env
python3 interactive_runner.py
```

### AgentBeats Mode
```bash
source .env
agentbeats run_ctrl
```

---

## Troubleshooting

### "401 Unauthorized" Error
Your ANTHROPIC_API_KEY is invalid or expired. Update it:
```bash
./start_green_agent.sh  # Choose option 6 to reconfigure
```

### "FileNotFoundError: task_config.json"
Create the config file:
```bash
cat > task_config.json << EOF
{
  "data_path": "/path/to/your/data",
  "test_labels": "/path/to/test_labels.csv",
  "constraints": {
    "max_time_seconds": 1200,
    "max_memory_mb": 4096,
    "max_cpus": 2.0
  }
}
EOF
```

### Research Evaluation Skipped
This is normal if ANTHROPIC_API_KEY is not set. The evaluation will still complete with execution and performance metrics.

---

## Next Steps

- 📖 Read the full guide: `GREEN_AGENT_GUIDE.md`
- 🧪 Try interactive testing: `python3 interactive_runner.py`
- 🚀 Deploy with AgentBeats: `agentbeats run_ctrl`

---

## Getting API Keys

### Anthropic (Required for research evaluation)
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy and paste into the startup script

### OpenRouter (Optional alternative)
1. Go to https://openrouter.ai/
2. Sign up and get credits
3. Get your API key
4. Use in startup script

---

**Need help?** See `GREEN_AGENT_GUIDE.md` for comprehensive documentation.
