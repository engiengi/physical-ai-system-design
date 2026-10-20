# Source this file from Bash: source ops/tutorial-env.sh local|cloud|cloud-gui
COSMOS_TUTORIAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT="$COSMOS_TUTORIAL_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
case "${1:-}" in
    local)
        unset COSMOS_ARTIFACT_TRANSPORT COSMOS_THOR_HOST COSMOS_THOR_WORKSPACE COSMOS_THOR_URI
        SIM_MODULE=cosmos_spark.session
        SIM_ARGS=(--enable_cameras)
        ;;
    cloud|cloud-gui)
        # Use each wheel's CUDA libraries instead of the DLAMI's multiple toolkits.
        export LD_LIBRARY_PATH=
        export COSMOS_THOR_WORKSPACE="$PROJECT_ROOT/thor"
        export COSMOS_THOR_URI=ws://127.0.0.1:8000
        export COSMOS_ARTIFACT_TRANSPORT=local
        SIM_MODULE=cosmos_spark.runner
        SIM_ARGS=(--headless --enable_cameras)
        if [[ "$1" == cloud-gui ]]; then
            export DISPLAY=:99
            SIM_MODULE=cosmos_spark.session
            SIM_ARGS=(--enable_cameras)
        fi
        ;;
    *) echo 'Usage: source ops/tutorial-env.sh local|cloud|cloud-gui' >&2; return 1 ;;
esac
cd "$PROJECT_ROOT"
