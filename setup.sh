#!/bin/bash
mkdir -p .log .cache result

echo "Simple setup script - run_our_ltv focused"

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

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "Environment ready for run_our_ltv"
