"""Small cluster helpers for the multi-node campaign.

  scale N C          ask the autoscaler for N worker nodes with >= C CPUs and wait for them
  release            drop the resource request so the cluster can scale down
  placement N R      check the deployed app: N engines on N distinct worker nodes, and R
                     LLMRouter replicas on every node running a Serve proxy; prints totals
  metrics-urls       print every alive node's Prometheus endpoint, one per line
  ingress-urls       print http://<ip>:8000 for every node that hosts an engine
"""

import sys
import time

import ray


def _head():
    gcs = ray.get_runtime_context().gcs_address.split(":")[0]
    alive = [n for n in ray.nodes() if n["Alive"]]
    for n in alive:
        if n.get("Resources", {}).get("node:__internal_head__"):
            return n
    return next(n for n in alive if n["NodeManagerAddress"] == gcs)


def workers(cpus):
    hid = _head()["NodeID"]
    return [n for n in ray.nodes() if n["Alive"] and n["NodeID"] != hid
            and n["Resources"].get("CPU", 0) >= cpus]


def scale(n, cpus, timeout_s=3600):
    from ray.autoscaler.sdk import request_resources

    head_cpus = _head()["Resources"].get("CPU", 0)
    if head_cpus >= cpus:
        sys.exit(f"head node has {head_cpus} CPUs >= engine CPUs {cpus}: an engine could land "
                 "on the head. Use an engine CPU count above the head's.")
    request_resources(bundles=[{"CPU": cpus}] * n)
    t = time.time()
    while len(workers(cpus)) < n:
        if time.time() - t > timeout_s:
            sys.exit(f"only {len(workers(cpus))}/{n} worker nodes after {timeout_s}s")
        time.sleep(10)
    print(f"{len(workers(cpus))} worker nodes with >= {cpus} CPUs up after {time.time() - t:.0f}s")


def _alive_actors(fragment):
    from ray._private.state import actors as actor_table  # GCS; needs no dashboard

    return [a["Address"]["NodeID"] for a in actor_table().values()
            if a["State"] == "ALIVE" and fragment in a["Name"]]


def _ip(node_id):
    return next(n["NodeManagerAddress"] for n in ray.nodes() if n["NodeID"] == node_id)


def placement(n, r):
    from ray import serve

    hid = _head()["NodeID"]
    engines = _alive_actors("LLMServer:mock")
    routers = _alive_actors("LLMRouter")
    proxies = set(serve.status().proxies)
    problems = []
    if len(engines) != n or len(set(engines)) != n or hid in engines:
        problems.append(f"engines: {len(engines)} on {len(set(engines))} nodes "
                        f"(head has {engines.count(hid)}); want {n} on {n} worker nodes")
    missing = set(engines) - proxies
    if missing:
        problems.append(f"{len(missing)} engine nodes have no Serve proxy")
    bad = {p: routers.count(p) for p in proxies if routers.count(p) != r}
    if bad or len(routers) != r * len(proxies):
        problems.append(f"routers: {len(routers)} on {len(proxies)} proxy nodes, "
                        f"want {r} per proxy node; off: {bad}")
    if problems:
        sys.exit("placement check failed: " + "; ".join(problems))
    print(f"placement ok: {n} engines on {n} worker nodes; {len(routers)} routers "
          f"({r} on each of {len(proxies)} proxy nodes, head {'included' if hid in proxies else 'not included'})")


def main():
    ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "scale":
        scale(int(args[0]), float(args[1]))
    elif cmd == "release":
        from ray.autoscaler.sdk import request_resources

        request_resources(bundles=[])
    elif cmd == "placement":
        placement(int(args[0]), int(args[1]))
    elif cmd == "metrics-urls":
        for node in ray.nodes():
            if node["Alive"]:
                print(f"http://{node['NodeManagerAddress']}:{node['MetricsExportPort']}/metrics")
    elif cmd == "ingress-urls":
        for node_id in sorted(set(_alive_actors("LLMServer:mock"))):
            print(f"http://{_ip(node_id)}:8000")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
