"""KVAwareRouter2 (payload token forwarding) side channel.

kv1 forwards the fused-select prompt ids to the engine via a per-request
``get_prompt_tokens`` RPC from the engine replica back to the singleton
``KVRouterActor`` (a second synchronous hit on the routing funnel). kv2 instead
materializes the ids at select time and rides them to the engine *with the
request* (route response -> Lua header -> engine), eliminating that RPC.

``KVAwareRouter.choose_replicas`` and the ingress ``LLMRouter.route`` both run
in the same ingress replica process (the DeploymentHandle's router executes in
the caller's event loop), so this in-process, single-threaded (asyncio) map
carries the ids from the routing decision to the ``/internal/route`` response.
Bounded so an orphaned stash (request errored between the two) cannot leak.
"""
from collections import OrderedDict
from typing import List, Optional

# request_id -> prompt_token_ids. Written by choose_replicas, popped by route().
_FUSED_IDS: "OrderedDict[str, List[int]]" = OrderedDict()
_MAX_ENTRIES = 8192  # normal flow pops immediately; cap only bounds leaks


def stash_ids(request_id: Optional[str], ids: Optional[List[int]]) -> None:
    if not request_id or not ids:
        return
    _FUSED_IDS[request_id] = ids
    _FUSED_IDS.move_to_end(request_id)
    while len(_FUSED_IDS) > _MAX_ENTRIES:
        _FUSED_IDS.popitem(last=False)


def pop_ids(request_id: Optional[str]) -> Optional[List[int]]:
    if not request_id:
        return None
    return _FUSED_IDS.pop(request_id, None)
