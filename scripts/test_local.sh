#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHONPATH=src "$PYTHON_BIN" scripts/validate_tracking.py
PYTHONPATH=src "$PYTHON_BIN" -m unittest discover -s tests -v
