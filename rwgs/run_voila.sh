#!/bin/bash
set -e

# Navigate to project directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DIR/.." && pwd)"
cd "$DIR"

# Ensure root virtual environment is used
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
elif [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
fi

PORT="${1:-8867}"
echo "=========================================================="
echo " Starting Comp-Chem-Lab: Reverse Water-Gas Shift Studio"
echo " Method: Quantum Chemistry DFT (wB97X-V/def2-TZVP / ORCA)"
echo " URL: http://localhost:$PORT"
echo "=========================================================="

exec voila rwgs_web.ipynb --port="$PORT" --no-browser --strip_sources=True
