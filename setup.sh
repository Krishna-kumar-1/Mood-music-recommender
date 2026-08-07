#!/bin/bash
# One-click setup: creates a virtual env and installs everything needed.
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Run 'source venv/bin/activate' then 'python inference.py' to test the pretrained model."
