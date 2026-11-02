#!/bin/bash
# Build the Dynamo wheel the mock engines need (MockerWorker + SelectionService) on the
# head node, so it links against the cluster image's glibc. Output: ~/kvbench-wheels/*.whl
# Needs: Rust (rustup), clang, protoc, cmake. GCC >= 15 segfaults in Dynamo's vendored
# libzmq on bind, so build with clang.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SRC=${DYNAMO_SRC:-$HOME/dynamo-kv}
[ -d "$SRC" ] || git clone -b kvbench-mocker https://github.com/jeffreywang88/dynamo.git "$SRC"
git -C "$SRC" checkout -q dc9d9a0fec
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y; source "$HOME/.cargo/env"; }
$KVBENCH_PY -m pip install -q "maturin==1.15.0"
cd "$SRC/lib/bindings/python"
CC=clang CXX=clang++ $KVBENCH_PY -m maturin build --release -o "$HOME/kvbench-wheels" \
  --features kv-indexer,slot-tracker,select-service
ls -la "$HOME/kvbench-wheels"
