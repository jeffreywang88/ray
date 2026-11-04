#!/bin/bash
# This script is used to build an extra layer on top of the base llm image
# to run the AgentX cluster-scale benchmark (llm_tests/serve/agentx_cluster_scale).

set -exo pipefail

# Dynamo with the Python MockerWorker the benchmark's engines use, built from
# ai-dynamo/dynamo main @ 2d126e0 plus jeffreywang88/dynamo@295c175
# (kvbench-mocker-main); no Dynamo release has it yet.
# --no-deps: ai-dynamo-runtime constrains pydantic and uvloop, both already
# pinned by the base llm image's vLLM install.
cat > /tmp/ai-dynamo-runtime.txt <<'EOF'
ai-dynamo-runtime @ https://air-example-data.s3.us-west-2.amazonaws.com/rayllm-ossci/dynamo/295c175/ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_35_x86_64.whl \
    --hash=sha256:7db4e71dcf1e76db9e05f80724c0116f3bcd09913f2bea72976127cb1f76d8ef
EOF
pip3 install --no-cache-dir --no-deps --require-hashes -r /tmp/ai-dynamo-runtime.txt

# AgentX gets its own venv so its dependencies leave the image's untouched.
python3 -m venv "${HOME}/agentx-venv"
"${HOME}/agentx-venv/bin/pip" install --no-cache-dir agentx-harness==1.0.12
