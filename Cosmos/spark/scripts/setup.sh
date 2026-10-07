#!/usr/bin/env bash
set -euo pipefail
COSMOS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COSMOS_ROOT"
if [[ ! -x .tools/bin/uv ]]; then
    python3 -m venv .tools
    .tools/bin/pip install uv==0.12.23
fi
if [[ ! -d RoboLab/.git ]]; then
    git clone --no-checkout https://github.com/NVlabs/RoboLab.git RoboLab
    git -C RoboLab checkout --detach ad45d4f974725d020f82c2b0d77d78533aeba2b3
fi
if [[ "$(git -C RoboLab rev-parse HEAD)" != ad45d4f974725d020f82c2b0d77d78533aeba2b3 ]]; then
    echo "RoboLab revision differs from the tested revision; review before updating." >&2
    exit 1
fi
git -C RoboLab lfs pull
unset VIRTUAL_ENV
.tools/bin/uv sync --locked
