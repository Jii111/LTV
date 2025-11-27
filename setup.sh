#!/bin/bash
mkdir .log
mkdir .cache
mkdir result

echo "Simple setup script - no version checks"

# Use existing Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Using $(python3 --version)"
else
    echo "Error: Python 3 not found"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (compatible with driver 535)
echo "Installing PyTorch 2.6.0 with CUDA 12.1..."
pip install --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "PyTorch with CUDA 12.1 installed (compatible with driver 535+)"