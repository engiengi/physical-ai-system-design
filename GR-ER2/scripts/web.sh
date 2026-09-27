#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec .venv/bin/python src/app.py serve "$@"
