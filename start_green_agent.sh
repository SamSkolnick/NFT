#!/bin/bash

# Green Agent Startup Script
# This script helps configure API keys and run evaluations

set -e

echo "======================================"
echo "  Green Agent Evaluator - Setup"
echo "======================================"
echo ""

# Check if .env file exists
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    echo "Found existing .env file. Loading..."
    source "$ENV_FILE"
    echo "✓ Loaded environment variables from .env"
    echo ""
fi

# Function to prompt for API key
prompt_for_key() {
    local var_name=$1
    local description=$2
    local current_value=${!var_name}
    
    if [ -n "$current_value" ]; then
        echo "$description is already set: ${current_value:0:20}..."
        read -p "Update it? (y/N): " update
        if [[ ! $update =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    read -p "Enter $description (or press Enter to skip): " new_value
    if [ -n "$new_value" ]; then
        export $var_name="$new_value"
        echo "✓ Set $var_name"
    fi
}

# Prompt for API keys
echo "API Key Configuration"
echo "---------------------"
prompt_for_key "ANTHROPIC_API_KEY" "Anthropic API Key (for research evaluation)"
prompt_for_key "OPENROUTER_API_KEY" "OpenRouter API Key (optional)"

echo ""
echo "Optional Model Configuration"
echo "----------------------------"
prompt_for_key "ANTHROPIC_MODEL" "Anthropic Model (default: claude-sonnet-4-5)"
prompt_for_key "OPENROUTER_MODEL" "OpenRouter Model (default: anthropic/claude-3.5-sonnet)"

# Set default models if not provided
export ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-"claude-sonnet-4-5"}
export OPENROUTER_MODEL=${OPENROUTER_MODEL:-"anthropic/claude-3.5-sonnet"}

# Disable tokenizers warning
export TOKENIZERS_PARALLELISM=false

# Ask if user wants to save to .env
echo ""
read -p "Save these settings to .env file? (Y/n): " save_env
if [[ ! $save_env =~ ^[Nn]$ ]]; then
    echo "# Green Agent Environment Variables" > "$ENV_FILE"
    echo "# Generated on $(date)" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"
    
    [ -n "$ANTHROPIC_API_KEY" ] && echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" >> "$ENV_FILE"
    [ -n "$OPENROUTER_API_KEY" ] && echo "export OPENROUTER_API_KEY=\"$OPENROUTER_API_KEY\"" >> "$ENV_FILE"
    [ -n "$ANTHROPIC_MODEL" ] && echo "export ANTHROPIC_MODEL=\"$ANTHROPIC_MODEL\"" >> "$ENV_FILE"
    [ -n "$OPENROUTER_MODEL" ] && echo "export OPENROUTER_MODEL=\"$OPENROUTER_MODEL\"" >> "$ENV_FILE"
    echo "export TOKENIZERS_PARALLELISM=false" >> "$ENV_FILE"
    
    echo "✓ Saved to $ENV_FILE"
    echo "  (You can source this file in the future: source $ENV_FILE)"
fi

# Show configuration summary
echo ""
echo "======================================"
echo "  Current Configuration"
echo "======================================"
echo "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:+Set (${ANTHROPIC_API_KEY:0:20}...)}"
echo "ANTHROPIC_MODEL: $ANTHROPIC_MODEL"
echo "OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+Set (${OPENROUTER_API_KEY:0:20}...)}"
echo "OPENROUTER_MODEL: $OPENROUTER_MODEL"
echo ""

# Main menu
while true; do
    echo "======================================"
    echo "  What would you like to do?"
    echo "======================================"
    echo "1) Run evaluation (Titanic example)"
    echo "2) Run custom evaluation"
    echo "3) Start interactive test runner"
    echo "4) Start AgentBeats controller"
    echo "5) Start A2A server"
    echo "6) Reconfigure API keys"
    echo "7) Exit"
    echo ""
    read -p "Choose option (1-7): " choice
    
    case $choice in
        1)
            echo ""
            echo "Running Titanic evaluation..."
            python -m green_agent_cli evaluate \
                --config task_config.json \
                --docker-image titanic-white-agent:latest \
                --research-artifacts white_agent_titanic/research \
                --no-pull-image
            echo ""
            ;;
        2)
            echo ""
            read -p "Config file path (default: task_config.json): " config_path
            config_path=${config_path:-task_config.json}
            
            read -p "Docker image name: " docker_image
            read -p "Research artifacts path: " research_path
            
            read -p "Pull image from registry? (y/N): " pull_choice
            pull_flag=""
            [[ ! $pull_choice =~ ^[Yy]$ ]] && pull_flag="--no-pull-image"
            
            echo ""
            echo "Running evaluation..."
            python -m green_agent_cli evaluate \
                --config "$config_path" \
                --docker-image "$docker_image" \
                --research-artifacts "$research_path" \
                $pull_flag
            echo ""
            ;;
        3)
            echo ""
            echo "Starting interactive test runner..."
            python3 interactive_runner.py
            echo ""
            ;;
        4)
            echo ""
            echo "Starting AgentBeats controller..."
            echo "(Press Ctrl+C to stop)"
            agentbeats run_ctrl
            echo ""
            ;;
        5)
            echo ""
            read -p "Host (default: 0.0.0.0): " host
            host=${host:-0.0.0.0}
            
            read -p "Port (default: 9999): " port
            port=${port:-9999}
            
            read -p "Public URL (optional): " public_url
            
            echo ""
            echo "Starting A2A server on $host:$port..."
            echo "(Press Ctrl+C to stop)"
            
            if [ -n "$public_url" ]; then
                python -m green_agent_cli serve \
                    --config task_config.json \
                    --host "$host" \
                    --port "$port" \
                    --public-url "$public_url"
            else
                python -m green_agent_cli serve \
                    --config task_config.json \
                    --host "$host" \
                    --port "$port"
            fi
            echo ""
            ;;
        6)
            echo ""
            # Re-run key prompts
            prompt_for_key "ANTHROPIC_API_KEY" "Anthropic API Key (for research evaluation)"
            prompt_for_key "OPENROUTER_API_KEY" "OpenRouter API Key (optional)"
            echo ""
            ;;
        7)
            echo ""
            echo "Exiting Green Agent. Goodbye!"
            exit 0
            ;;
        *)
            echo ""
            echo "Invalid option. Please choose 1-7."
            echo ""
            ;;
    esac
done
