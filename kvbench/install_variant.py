"""Install one benchmark variant on every alive node of the Ray cluster.

The three variants differ only in pure-Python files under python/ray/serve and
python/ray/llm, all branched from the same master commit (BASE). For the given
git ref this writes, on every node (head included):

- each file that any variant changed relative to BASE, with the ref's content
  (or deletes it when the ref does not have it), into the installed `ray` package;
- the harness modules (kvbench/*.py) into KVBENCH_NODE_DIR, which app.py puts on
  the replicas' PYTHONPATH;
- optionally a Dynamo wheel (pip install --no-deps --force-reinstall).

It refuses to run if a node's installed Ray is not BASE, and checks that every
node ends up with identical file hashes.

usage: python install_variant.py --repo ~/ray-kv --ref bench/prom-mesh-zmq [--dynamo-wheel W]
       (--dynamo-wheel defaults to $KVBENCH_DYNAMO_WHEEL)
"""

import argparse
import glob
import hashlib
import os
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = "fb7c757706aafd22025a2103268586cf138a143d"
VARIANT_REFS = ("bench/prom-nomesh", "bench/prom-mesh-serve", "bench/prom-mesh-zmq")
PREFIXES = ("python/ray/serve/", "python/ray/llm/")


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True).stdout


def resolve(repo, name):
    """A variant branch, local or under any remote."""
    for cand in (name, *[f"{r}/{name}" for r in git(repo, "remote").decode().split()]):
        if subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify", cand],
                          capture_output=True).returncode == 0:
            return cand
    raise SystemExit(f"cannot find branch {name} in {repo}; run git fetch")


def overlay(repo, ref):
    """{path under the ray package: bytes, or None to delete}."""
    paths = set()
    for v in VARIANT_REFS:
        out = git(repo, "diff", "--name-only", BASE, resolve(repo, v)).decode().split()
        paths |= {p for p in out if p.startswith(PREFIXES)}
    have = set(git(repo, "ls-tree", "-r", "--name-only", ref, "--", *PREFIXES).decode().split())
    files = {}
    for p in sorted(paths):
        rel = p[len("python/ray/"):]
        files[rel] = git(repo, "show", f"{ref}:{p}") if p in have else None
    return files


@ray.remote(num_cpus=0)
def install(files, harness, node_dir, wheel_name, wheel):
    import importlib.metadata
    import ray as installed

    if installed.__commit__ != BASE:
        return {"error": f"installed ray is {installed.__commit__}, need {BASE}"}
    pkg = os.path.dirname(installed.__file__)
    digest = hashlib.sha256()
    for rel, data in sorted(files.items()):
        path = os.path.join(pkg, rel)
        if data is None:
            if os.path.exists(path):
                os.remove(path)
            digest.update(f"{rel}:deleted".encode())
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.kvbench-{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        with open(path, "rb") as f:
            digest.update(rel.encode() + hashlib.sha256(f.read()).digest())
    os.makedirs(node_dir, exist_ok=True)
    for name, data in harness.items():
        with open(os.path.join(node_dir, name), "wb") as f:
            f.write(data)
    if wheel is not None:
        # Install once per node; nodes the autoscaler adds later get it on their first run.
        marker = os.path.join(node_dir, "dynamo-wheel.sha256")
        sha = hashlib.sha256(wheel).hexdigest()
        if not os.path.exists(marker) or open(marker).read() != sha:
            import sys

            path = os.path.join(node_dir, wheel_name)
            with open(path, "wb") as f:
                f.write(wheel)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                            "--force-reinstall", path], check=True)
            with open(marker, "w") as f:
                f.write(sha)
    try:
        dynamo = importlib.metadata.version("ai-dynamo-runtime")
    except Exception:
        dynamo = None
    return {"digest": digest.hexdigest(), "dynamo": dynamo}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="clone of the fork with the bench branches fetched")
    p.add_argument("--ref", required=True, help="variant branch, e.g. bench/prom-mesh-zmq")
    p.add_argument("--dynamo-wheel", default=os.environ.get("KVBENCH_DYNAMO_WHEEL") or None)
    a = p.parse_args()
    repo = os.path.expanduser(a.repo)
    files = overlay(repo, resolve(repo, a.ref))
    here = os.path.dirname(os.path.abspath(__file__))
    harness = {os.path.basename(f): open(f, "rb").read() for f in glob.glob(f"{here}/*.py")}
    node_dir = os.environ["KVBENCH_NODE_DIR"]
    wheel = open(a.dynamo_wheel, "rb").read() if a.dynamo_wheel else None
    wheel_name = os.path.basename(a.dynamo_wheel) if a.dynamo_wheel else None

    ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
    files_ref, harness_ref = ray.put(files), ray.put(harness)
    wheel_ref = ray.put(wheel)
    nodes = [n for n in ray.nodes() if n["Alive"]]
    results = ray.get([
        install.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))
        .remote(files_ref, harness_ref, node_dir, wheel_name, wheel_ref)
        for n in nodes
    ])
    errors = [(n["NodeManagerAddress"], r["error"]) for n, r in zip(nodes, results) if "error" in r]
    if errors:
        raise SystemExit(f"install failed: {errors}")
    digests = {r["digest"] for r in results}
    dynamos = {r["dynamo"] for r in results}
    if len(digests) != 1 or len(dynamos) != 1 or None in dynamos:
        raise SystemExit(f"nodes differ: digests={digests} dynamo={dynamos}")
    changed = sum(v is not None for v in files.values())
    print(f"installed {a.ref} on {len(nodes)} nodes: {changed} files written, "
          f"{len(files) - changed} removed, digest {digests.pop()[:12]}, dynamo {dynamos.pop()}")


if __name__ == "__main__":
    main()
