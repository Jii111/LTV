#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p .log .cache result

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

# Install PyTorch 2.5.1 with CUDA 12.1 (compatible with Driver 535 / CUDA 12.2)
echo "Installing PyTorch 2.5.1 + CUDA 12.1..."
pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "  PyTorch 2.5.1 + CUDA 12.1"
echo "  transformers 4.44.2 (Llama 3.1 support)"
echo "  Driver 535 / CUDA 12.2 compatible"
echo "=========================================="
echo ""
echo "Activate with: source .venv/bin/activate"