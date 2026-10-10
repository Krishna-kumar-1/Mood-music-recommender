#!/bin/bash
# Smart One-Click Setup & Launcher for Linux/macOS
set -e
cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Creating virtual environment..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV_DIR" --python 3.11
    else
        python3 -m venv "$VENV_DIR"
    fi
fi

source "$VENV_DIR/bin/activate"
python run.py "$@"
