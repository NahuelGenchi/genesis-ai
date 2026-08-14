#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python scripts/validate_tracking.py
PYTHONPATH=src python -m unittest discover -s tests -v
