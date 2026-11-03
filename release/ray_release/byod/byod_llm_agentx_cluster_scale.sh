#!/bin/bash
# This script is used to build an extra layer on top of the base llm image
# to run the AgentX cluster-scale benchmark (llm_tests/serve/agentx_cluster_scale).

set -exo pipefail

# Dynamo's Python MockerWorker (https://github.com/jeffreywang88/dynamo/commit/dc9d9a0fe)
# is not in an ai-dynamo-runtime release yet. Set this to the wheel built from it.
DYNAMO_MOCKER_WHEEL_URL=""
if [[ -z "${DYNAMO_MOCKER_WHEEL_URL}" ]]; then
  echo "DYNAMO_MOCKER_WHEEL_URL is not set" >&2
  exit 1
fi
# --no-deps: ai-dynamo-runtime constrains pydantic and uvloop, both already
# pinned by the base llm image's vLLM install.
pip3 install --no-cache-dir --no-deps "${DYNAMO_MOCKER_WHEEL_URL}"

# AgentX gets its own venv so its dependencies leave the image's untouched.
python3 -m venv "${HOME}/agentx-venv"
"${HOME}/agentx-venv/bin/pip" install --no-cache-dir agentx-harness==1.0.12
