#!/bin/bash
# Build the AgentX input: the public semianalysisai/cc-traces-weka-062126 traces, token-scaled
# to 1 token per hash block (scale_traces.py), one file per trace. Output: $KVBENCH_DATA
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
D=$(dirname "$KVBENCH_DATA"); mkdir -p "$D"
"$(dirname "$KVBENCH_AGENTX")/python" - "$D" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("semianalysisai/cc-traces-weka-062126", "traces.jsonl",
                      repo_type="dataset", local_dir=sys.argv[1]))
PYEOF
python3 "$KVBENCH_DIR/scale_traces.py" "$D/traces.jsonl" "$D/k1.jsonl" --tokens-per-block 1
echo "b93e7b393affd19b5d1818662e3cda53  $D/k1.jsonl" | md5sum -c -
python3 "$KVBENCH_DIR/scale_traces.py" --split "$D/k1.jsonl" "$KVBENCH_DATA"
ls "$KVBENCH_DATA" | wc -l
