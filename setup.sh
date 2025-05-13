#!/bin/bash

# Setup script for Telecom Support Chatbot
# This script helps set up the environment and dependencies

echo "Setting up Telecom Support Chatbot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Python $REQUIRED_VERSION or higher is required. You have Python $PYTHON_VERSION."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install venv package and try again."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

# Check if .env file exists, create if not
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# Hugging Face API Key - You can get a free API key from https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=

# Hugging Face Model - This model is available for free inference
# You can use any of these free models:
# - google/flan-t5-large (general purpose)
# - facebook/bart-large-cnn (summarization)
# - gpt2 (text generation)
# - EleutherAI/gpt-neo-1.3B (larger model)
HF_MODEL_NAME=google/flan-t5-large

# Local Model Options (Ollama)
# Set USE_LOCAL_MODEL=true to use Ollama instead of Hugging Face
USE_LOCAL_MODEL=false
LOCAL_MODEL_NAME=llama2
OLLAMA_BASE_URL=http://localhost:11434

# Configuration settings
DEBUG=False
LOG_LEVEL=INFO
EOL
    echo "Created .env file. Please edit it to add your API keys."
fi

# Check for Ollama installation if local model is enabled
if grep -q "USE_LOCAL_MODEL=true" .env; then
    if ! command -v ollama &> /dev/null; then
        echo "Warning: Ollama is not installed but USE_LOCAL_MODEL is set to true."
        echo "Please install Ollama from https://ollama.ai/download"
    else
        echo "Ollama is installed. Checking for required model..."
        MODEL_NAME=$(grep "LOCAL_MODEL_NAME" .env | cut -d= -f2)
        if ! ollama list | grep -q "$MODEL_NAME"; then
            echo "Model $MODEL_NAME not found. Pulling model..."
            ollama pull $MODEL_NAME
        fi
    fi
fi

echo "Setup complete! You can now run the chatbot with:"
echo "  cd src && streamlit run app.py"
echo ""
echo "Or use the command line interface:"
echo "  cd src && python main.py"
