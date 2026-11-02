# Environment for every kvbench command on the head node: source kvbench/env.sh
# KVBENCH_PY: the cluster's Python (the one with Ray installed). KVBENCH_REPO: clone of the fork.
export KVBENCH_PY=${KVBENCH_PY:-python}
export KVBENCH_REPO=${KVBENCH_REPO:-$HOME/ray-kv}
export KVBENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export KVBENCH_NODE_DIR=${KVBENCH_NODE_DIR:-$HOME/kvbench_node}
export KVBENCH_AGENTX=${KVBENCH_AGENTX:-$HOME/agentx-venv/bin/agentx}
export KVBENCH_DATA=${KVBENCH_DATA:-$HOME/kvbench-data/cc-k1}
# Serve settings under test (app.py passes these to the controller, proxy and replicas).
export RAY_SERVE_ENABLE_HA_PROXY=1
export RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1
export RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY=1
export RAY_SERVE_INGRESS_ROUTER_REPLICAS_PER_NODE=${ROUTERS_PER_NODE_DEFAULT:-2}  # point.sh sets it per run
export RAY_SERVE_HAPROXY_MAXCONN=65536
export RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE=524288
export RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED=1
export RAY_SERVE_EVENT_LOOP_MONITORING_INTERVAL_S=1
export KVBENCH_DYNAMO_WHEEL=${KVBENCH_DYNAMO_WHEEL:-$(ls $HOME/kvbench-wheels/ai_dynamo_runtime-*.whl 2>/dev/null | head -1)}
