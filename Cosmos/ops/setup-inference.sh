#!/usr/bin/env bash
# Fresh inference environment; never replaces an existing environment.
set -euo pipefail
COSMOS_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COSMOS_INFERENCE_ROOT="${1:-$COSMOS_PROJECT/thor}"
mkdir -p "$COSMOS_INFERENCE_ROOT"
COSMOS_INFERENCE_ROOT="$(cd "$COSMOS_INFERENCE_ROOT" && pwd -P)"
if [[ -e "$COSMOS_INFERENCE_ROOT/.venv" ]]; then
    echo 'An inference .venv already exists. Use it or select a new workspace; no files changed.' >&2
    exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first; see 실행방법.md.' >&2; exit 1; }
COSMOS_FRAMEWORK="$COSMOS_INFERENCE_ROOT/external/cosmos-framework"
COSMOS_REVISION=18d90eacbda778e666e6978dc1ab6ba8635400e3
mkdir -p "$COSMOS_INFERENCE_ROOT/external"
if [[ ! -e "$COSMOS_FRAMEWORK" ]]; then
    git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS_FRAMEWORK"
    git -C "$COSMOS_FRAMEWORK" checkout --detach "$COSMOS_REVISION"
fi
[[ "$(git -C "$COSMOS_FRAMEWORK" rev-parse HEAD)" == "$COSMOS_REVISION" ]] || {
    echo 'Framework revision differs from this tutorial. Existing source was not changed.' >&2; exit 1;
}
cd "$COSMOS_FRAMEWORK"
UV_PROJECT_ENVIRONMENT="$COSMOS_INFERENCE_ROOT/.venv" uv sync --frozen --python 3.13 \
    --all-extras --group cu130-train --group policy-server
LD_LIBRARY_PATH= "$COSMOS_INFERENCE_ROOT/.venv/bin/python" -c \
    'import torch; print("torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available()'
