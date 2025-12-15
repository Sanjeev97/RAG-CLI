#!/bin/bash

################################################################################
# RAG System - Simple Installation Script for Linux/Unix
# Author: Sanjeev (Modified for Linux by AI)
# Date: December 2024
################################################################################

set -e  # Exit immediately if a command exits with a non-zero status.

echo "=========================================="
echo "🐧 RAG System - Automated Setup for Linux"
echo "=========================================="
echo ""

# --- 1. FIND PYTHON ---
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Found Python: $PYTHON_CMD"

# --- 2. CREATE VIRTUAL ENVIRONMENT ---
echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
echo "✓ Virtual environment created"

# --- 3. ACTIVATE VIRTUAL ENVIRONMENT (Linux Standard) ---
echo ""
echo "Activating virtual environment..."
# Note: Use venv/bin/activate for Linux/Unix
source venv/bin/activate
echo "✓ Virtual environment activated"

# --- 4. INSTALL DEPENDENCIES ---
echo ""
echo "Installing dependencies (this may take a few minutes)..."
if [ -f "requirements.txt" ]; then
    # Use -q for quiet install, but remove it for better visibility if needed
    pip install -r requirements.txt --quiet 
else
    echo "ERROR: requirements.txt not found"
    exit 1
fi
echo "✓ Dependencies installed"

# --- 5. VERIFY INSTALLATION ---
echo ""
echo "Verifying installation..."
# Silence stderr (2>/dev/null) to clean up terminal output from potential warnings
python -c "import torch; import faiss; import sentence_transformers; from ctransformers import AutoModelForCausalLM" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All core libraries verified (torch, faiss, ctransformers, s-t)"
else
    echo "ERROR: Installation verification failed."
    echo "Ensure necessary system libraries for ctransformers/faiss are installed."
    # We skip exit 1 here to allow the script to continue to the model download
fi

# --- 6. CREATE DIRECTORIES ---
echo ""
echo "Creating required 'models' directory..."
mkdir -p models
echo "✓ Directories created"

# --- 7. DOWNLOAD MODEL ---
echo ""
echo "Downloading TinyLlama GGUF model (~600MB)..."
MODEL_PATH="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model already exists"
else
    MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Use wget or curl, prioritizing wget as it's often more verbose/reliable for large files
    if command -v wget &> /dev/null; then
        echo "Using wget to download..."
        wget -O "$MODEL_PATH" "$MODEL_URL" --show-progress
    elif command -v curl &> /dev/null; then
        echo "Using curl to download..."
        curl -L -o "$MODEL_PATH" "$MODEL_URL"
    else
        echo "WARNING: Neither wget nor curl found."
        echo "Please manually download the model from the URL below and save it to: $MODEL_PATH"
        echo "$MODEL_URL"
    fi
    
    if [ -f "$MODEL_PATH" ]; then
        echo "✓ Model downloaded"
    fi
fi

# --- 8. FINAL INSTRUCTIONS ---
echo ""
echo "=========================================="
echo "🚀 Installation Complete!"
echo "=========================================="
echo ""
echo "To run the system, first activate the environment:"
echo "  source venv/bin/activate"
echo "Then, run the RAG CLI:"
echo "  python src/cli.py --model $MODEL_PATH --data ./data"
echo "or"
echo "  python src/cli_enhanced.py --model $MODEL_PATH --data ./data to run with query reformulation"
echo ""
echo "You can deactivate the environment with 'deactivate'"