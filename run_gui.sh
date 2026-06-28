#!/usr/bin/env bash
# Launch igram_finder_gui.py inside the project virtual environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Virtual environment not found at $VENV"
    echo "Create it with:  python3 -m venv venv && source venv/bin/activate && pip install opencv-python numpy scipy pillow"
    exit 1
fi

source "$VENV/bin/activate"
exec python "$SCRIPT_DIR/igram_finder_gui.py" "$@"
