#!/usr/bin/env bash
# Independent, pinned environment for the official-path comparison.
set -euo pipefail
COSMOS_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COSMOS_REFERENCE_ROOT="${1:-$COSMOS_PROJECT/thor}"
COSMOS_REFERENCE="$COSMOS_REFERENCE_ROOT/external/cosmos-framework-reference-20261005"
COSMOS_REFERENCE_REVISION=cf5d68c00d97ccd2480a2320ed652b92dec63102
command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 1; }
mkdir -p "$COSMOS_REFERENCE_ROOT/external"
if [[ ! -e "$COSMOS_REFERENCE" ]]; then
    git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS_REFERENCE"
    git -C "$COSMOS_REFERENCE" checkout --detach "$COSMOS_REFERENCE_REVISION"
fi
[[ "$(git -C "$COSMOS_REFERENCE" rev-parse HEAD)" == "$COSMOS_REFERENCE_REVISION" ]] || {
    echo 'Reference framework revision differs. Existing checkout was not changed.' >&2; exit 1;
}
cd "$COSMOS_REFERENCE"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
uv sync --frozen --python 3.13 --all-extras --group cu130-train --group policy-server
