#!/bin/bash

# ==============================================================================
# Generic vLLM Server Launcher with Hardcoded Environment Settings
# ==============================================================================
#
# This script launches the vLLM OpenAI-compatible server.
# It forwards all command-line arguments it receives directly to the
# vLLM entrypoint, after adding static environment settings.
#
# Hardcoded Settings:
# - Port: 8000
# - vLLM Request Logging: Disabled
# - Uvicorn Access Logging: Disabled
#
# ==============================================================================

set -e


PORT=8000
STATIC_ARGS="--port $PORT --disable-log-requests --disable-uvicorn-access-log --no-enable-prefix-caching"

USER_ARGS="$@"

CMD="python3 -m vllm.entrypoints.openai.api_server $STATIC_ARGS $@"

echo "=============================================================================="
echo "Launching vLLM Server..."
echo "Executing command: $CMD"
echo "=============================================================================="

export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname -i)

$CMD