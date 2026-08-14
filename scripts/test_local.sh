#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python3 scripts/validate_tracking.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
