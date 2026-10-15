#!/bin/bash
set -e

# Navigate to project directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Ensure virtual environment is used
if [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
fi

PORT="${1:-8866}"
echo "=================================================="
echo " Starting Quantum Chemical Reaction Studio (Voilà)"
echo " URL: http://localhost:$PORT"
echo "=================================================="

exec voila quantum_chem_web.ipynb --port="$PORT" --no-browser --strip_sources=True
