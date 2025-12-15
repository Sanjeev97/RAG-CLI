#!/bin/bash

################################################################################
# RAG System - Simple Installation Script
# Author: Sanjeev
# Date: December 2024
################################################################################

set -e  # Exit on any error

echo "=========================================="
echo "RAG System - Automated Setup"
echo "=========================================="
echo ""

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Found Python: $PYTHON_CMD"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/Scripts/activate
echo "✓ Virtual environment activated"

# # Upgrade pip
# echo ""
# echo "Upgrading pip..."
# pip install --upgrade pip --quiet
# echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found"
    exit 1
fi
echo "✓ Dependencies installed"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; import faiss; import sentence_transformers; from ctransformers import AutoModelForCausalLM" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All libraries verified"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p models
# mkdir -p data
echo "✓ Directories created"

# Download model
echo ""
echo "Downloading TinyLlama model (~600MB)..."
MODEL_PATH="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model already exists"
else
    MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL"
    else
        echo "WARNING: wget/curl not found"
        echo "Please manually download from:"
        echo "$MODEL_URL"
        echo "Save to: $MODEL_PATH"
    fi
    
    if [ -f "$MODEL_PATH" ]; then
        echo "✓ Model downloaded"
    fi
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To run the system:"
echo "  source venv/bin/activate"
echo "  python src/cli.py --model $MODEL_PATH --data ./data"
echo "or"
echo "  python src/cli_enhanced.py --model $MODEL_PATH --data ./data to run with query reformulation"
echo ""
echo "Sample documents are in data/ directory"
echo ""
