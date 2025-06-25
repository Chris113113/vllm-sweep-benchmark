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

# CMD="python3 -m vllm.entrypoints.openai.api_server"

USER_ARGS="$@"

echo "=============================================================================="
echo "Launching vLLM Server..."
echo "Executing command: $CMD $STATIC_ARGS $USER_ARGS"
echo "HF Token: $HF_TOKEN"
echo "HF Home: $HF_HOME"
echo "VLLM Cache Root: $VLLM_CACHE_ROOT"
echo "=============================================================================="

export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1
export VLLM_HOST_IP=$(hostname -i)
export MASTER_ADDR=$(hostname -i)

CMD="python3 -m vllm.entrypoints.openai.api_server $STATIC_ARGS $@"

$CMD $STATIC_ARGS $USER_ARGS