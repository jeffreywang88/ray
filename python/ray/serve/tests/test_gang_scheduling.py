import os
import sys

import pytest

import ray
from ray import serve
from ray._common.test_utils import wait_for_condition
from ray.serve._private.test_utils import check_apps_running
from ray.serve.config import GangPlacementStrategy, GangSchedulingConfig
from ray.tests.conftest import *  # noqa


class TestGangScheduling:
    """Tests for gang scheduling with placement groups."""

    def test_sufficient_resources(self, ray_cluster):
        """Verifies that gang scheduling succeeds when cluster has sufficient resources."""
        cluster = ray_cluster
        cluster.add_node(num_cpus=1)
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(
            num_replicas=8,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        )
        class GangDeployment:
            def __call__(self):
                return ray.get_runtime_context().get_node_id()

        handle = serve.run(GangDeployment.bind(), name="gang_app_success")
        wait_for_condition(
            check_apps_running,
            apps=["gang_app_success"],
        )

        # Verify all replicas are running and responding
        refs = [handle.remote() for _ in range(8)]
        results = [ref.result() for ref in refs]
        assert len(results) == 8

        serve.delete("gang_app_success")
        serve.shutdown()

    def test_sufficient_resources_with_options(self, ray_cluster):
        """Verifies gang scheduling via .options() succeeds and responds to requests."""
        cluster = ray_cluster
        cluster.add_node(num_cpus=1)
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0})
        class GangDeployment:
            def __call__(self):
                return ray.get_runtime_context().get_node_id()

        app = GangDeployment.options(
            num_replicas=8,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        ).bind()

        handle = serve.run(app, name="gang_app_options")
        wait_for_condition(
            check_apps_running,
            apps=["gang_app_options"],
        )

        # Verify all replicas are running and responding
        refs = [handle.remote() for _ in range(8)]
        results = [ref.result() for ref in refs]
        assert len(results) == 8

        serve.delete("gang_app_options")
        serve.shutdown()

    def test_incomplete_deployment(self, ray_cluster):
        """
        Verifies that schedulable gangs serve traffic while unschedulable gangs wait for resources.
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=1)
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment
        class IncompleteGangDeployment:
            def __call__(self):
                return ray.get_runtime_context().get_node_id()

        app = IncompleteGangDeployment.options(
            num_replicas=12,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        ).bind()

        handle = serve._run(app, name="gang_partial_app", _blocking=False)

        # The deployment should NOT fail. 2 of 3 gangs should be scheduled,
        # and those 8 replicas should serve traffic. The deployment stays
        # DEPLOYING because it hasn't reached 12 replicas.
        def check_replicas_running(expected_count: int):
            try:
                app_status = serve.status().applications["gang_partial_app"]
                # Should be DEPLOYING
                if app_status.status == "DEPLOY_FAILED":
                    raise AssertionError(
                        "Deployment should not fail with partial gang scheduling"
                    )
                # Check that some replicas are running
                dep_status = list(app_status.deployments.values())[0]
                running = dep_status.replica_states.get("RUNNING", 0)
                assert running == expected_count
                return True
            except KeyError:
                return False

        wait_for_condition(check_replicas_running, expected_count=8, timeout=60)

        # Verify the running replicas can serve traffic.
        results = set()
        for _ in range(40):
            results.add(handle.remote().result())
        assert len(results) > 0

        # Verify deployment is still DEPLOYING
        app_status = serve.status().applications["gang_partial_app"]
        assert app_status.status == "DEPLOYING"

        # Now add a 3rd node so the remaining gang can be scheduled.
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()

        # The deployment should become RUNNING with all 12 replicas.
        wait_for_condition(
            check_apps_running,
            apps=["gang_partial_app"],
            timeout=60,
        )

        # Verify all 12 replicas serve traffic.
        results = set()
        for _ in range(100):
            results.add(handle.remote().result())
        assert len(results) == 3

        serve.delete("gang_partial_app")
        serve.shutdown()

    def test_no_partial_gang(self, ray_cluster):
        """Verifies atomic gang scheduling: no partial gangs are created."""
        cluster = ray_cluster
        # 2 CPUs total: enough for 2 full gangs (1.6 CPUs) but not 3 (2.4 CPUs).
        # The leftover 0.4 CPUs must NOT produce a partial gang.
        cluster.add_node(num_cpus=1)
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment
        class AtomicGangDeployment:
            def __call__(self):
                return ray.get_runtime_context().get_node_id()

        app = AtomicGangDeployment.options(
            num_replicas=12,
            ray_actor_options={"num_cpus": 0.2},
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        ).bind()

        handle = serve._run(app, name="atomic_gang_app", _blocking=False)

        # Wait until exactly 8 replicas (2 gangs) are running.
        def check_replicas_running(expected_count: int):
            try:
                app_status = serve.status().applications["atomic_gang_app"]
                if app_status.status == "DEPLOY_FAILED":
                    raise AssertionError(
                        "Deployment should not fail — partial gangs should "
                        "serve traffic while waiting for resources."
                    )
                dep_status = list(app_status.deployments.values())[0]
                running = dep_status.replica_states.get("RUNNING", 0)
                assert running == expected_count
                return True
            except KeyError:
                return False

        wait_for_condition(check_replicas_running, expected_count=8, timeout=60)

        # Deployment should still be DEPLOYING (not RUNNING, not DEPLOY_FAILED).
        app_status = serve.status().applications["atomic_gang_app"]
        assert app_status.status == "DEPLOYING"

        # Verify the 8 running replicas can serve traffic.
        results = set()
        for _ in range(80):
            results.add(handle.remote().result())
        assert len(results) > 0

        # Add 1 more CPU so the 3rd gang (0.8 CPUs) can be scheduled.
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()

        # The deployment should become RUNNING with all 12 replicas.
        wait_for_condition(check_apps_running, apps=["atomic_gang_app"], timeout=60)

        # All 12 replicas should now serve traffic.
        app_status = serve.status().applications["atomic_gang_app"]
        dep_status = list(app_status.deployments.values())[0]
        running = dep_status.replica_states.get("RUNNING", 0)
        assert running == 12

        serve.delete("atomic_gang_app")
        serve.shutdown()

    def test_pack_strategy(self, ray_cluster):
        """Verifies that PACK strategy places gang replicas on the same node."""
        cluster = ray_cluster
        cluster.add_node(num_cpus=1)
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment
        def PackDeployment():
            return os.environ.get(
                "RAY_NODE_ID", ray.get_runtime_context().get_node_id()
            )

        # 1 gang with PACK strategy - all replicas should be on same node
        app = PackDeployment.options(
            num_replicas=4,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=4,
                gang_placement_strategy=GangPlacementStrategy.PACK,
            ),
        ).bind()

        handle = serve.run(app, name="gang_pack_app")
        wait_for_condition(check_apps_running, apps=["gang_pack_app"])

        # Query multiple times to hit all replicas and collect node IDs
        node_ids = set()
        for _ in range(40):
            result = handle.remote().result()
            node_ids.add(result)

        # With PACK strategy, all 4 replicas should be on the same node
        assert len(node_ids) == 1

        serve.delete("gang_pack_app")
        serve.shutdown()

    def test_gang_scheduling_spread_strategy(self, ray_cluster):
        """Verifies that SPREAD strategy places gang replicas on different nodes."""
        cluster = ray_cluster
        cluster.add_node(num_cpus=1)
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment
        def SpreadDeployment():
            return os.environ.get(
                "RAY_NODE_ID", ray.get_runtime_context().get_node_id()
            )

        # 1 gang with SPREAD strategy - replicas should be on different nodes
        app = SpreadDeployment.options(
            num_replicas=2,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=2,
                gang_placement_strategy=GangPlacementStrategy.SPREAD,
            ),
        ).bind()

        handle = serve.run(app, name="gang_spread_app")
        wait_for_condition(check_apps_running, apps=["gang_spread_app"])

        # Query multiple times to hit all replicas and collect node IDs
        node_ids = set()
        for _ in range(40):
            result = handle.remote().result()
            node_ids.add(result)

        # With SPREAD strategy, 2 replicas should be on 2 different nodes
        assert len(node_ids) == 2

        serve.delete("gang_spread_app")
        serve.shutdown()

    def test_gang_context(self, ray_cluster):
        """Verifies GangContext is correctly populated in ReplicaContext."""
        cluster = ray_cluster
        cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment
        class GangContextDeployment:
            def __call__(self):
                ctx = ray.serve.context._get_internal_replica_context()
                gc = ctx.gang_context
                if gc is None:
                    return None
                return {
                    "gang_id": gc.gang_id,
                    "rank": gc.rank,
                    "world_size": gc.world_size,
                    "member_replica_ids": gc.member_replica_ids,
                    "replica_id": ctx.replica_id.unique_id,
                }

        app = GangContextDeployment.options(
            num_replicas=4,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        ).bind()

        handle = serve.run(app, name="gang_context_app")
        wait_for_condition(check_apps_running, apps=["gang_context_app"])

        # Collect gang contexts from all replicas
        # Query enough times to hit all 4 replicas
        contexts_by_replica = {}
        for _ in range(100):
            result = handle.remote().result()
            assert result is not None
            replica_id = result["replica_id"]
            if replica_id not in contexts_by_replica:
                contexts_by_replica[replica_id] = result
            if len(contexts_by_replica) == 4:
                break
        assert len(contexts_by_replica) == 4

        # Group replicas by gang_id
        gangs = {}
        for replica_id, ctx in contexts_by_replica.items():
            gang_id = ctx["gang_id"]
            gangs.setdefault(gang_id, []).append(ctx)

        assert len(gangs) == 2

        for gang_id, members in gangs.items():
            assert len(members) == 2
            assert all(member["world_size"] == 2 for member in members)
            assert members[0]["member_replica_ids"] == members[1]["member_replica_ids"]

            expected_ids = sorted([m["replica_id"] for m in members])
            actual_ids = sorted(members[0]["member_replica_ids"])
            assert actual_ids == expected_ids

            ranks = sorted([m["rank"] for m in members])
            assert ranks == [0, 1]

        # Across gangs: gang_ids should be different
        gang_ids = list(gangs.keys())
        assert gang_ids[0] != gang_ids[1]

        # Across gangs: member_replica_ids should be different
        gang_members_list = list(gangs.values())
        assert sorted(gang_members_list[0][0]["member_replica_ids"]) != sorted(
            gang_members_list[1][0]["member_replica_ids"]
        )

        serve.delete("gang_context_app")
        serve.shutdown()


class TestGangScaling:
    """Integration tests for gang-aware scaling and rolling updates."""

    @staticmethod
    def _send_requests_background(handle, stop_event, errors, successes):
        """Continuously send requests until *stop_event* is set.

        Any request that raises an exception is recorded in *errors*.
        """
        import time

        while not stop_event.is_set():
            try:
                handle.remote().result(timeout_s=10)
                successes.append(1)
            except Exception as e:
                errors.append(str(e))
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Upscale: 4 → 8, gang_size=2 → adds 2 complete gangs, no downtime
    # ------------------------------------------------------------------
    def test_upscale_gang_boundary(self, ray_cluster):
        """Upscaling adds only complete gangs.  Existing replicas keep serving."""
        import threading

        cluster = ray_cluster
        cluster.add_node(num_cpus=2)
        cluster.add_node(num_cpus=2)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        # Pin version so that changing only num_replicas is a pure scale-up
        # (no rolling restart).
        @serve.deployment(
            name="D",
            version="v1",
            num_replicas=4,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class D:
            def __call__(self):
                ctx = ray.serve.context._get_internal_replica_context()
                gc = ctx.gang_context
                return {"pid": os.getpid(), "gang_id": gc.gang_id if gc else None}

        handle = serve.run(D.bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])

        # Collect the initial gang_ids (should be 2 gangs of size 2).
        initial_gang_ids = set()
        for _ in range(40):
            resp = handle.remote().result()
            if resp["gang_id"] is not None:
                initial_gang_ids.add(resp["gang_id"])
        assert (
            len(initial_gang_ids) == 2
        ), f"Expected 2 initial gangs, got {len(initial_gang_ids)}"

        # Start background request sender *before* the scale-up.
        errors, successes = [], []
        stop_event = threading.Event()
        t = threading.Thread(
            target=self._send_requests_background,
            args=(handle, stop_event, errors, successes),
            daemon=True,
        )
        t.start()

        # Upscale: 4 → 8 (same code + version, only num_replicas changes).
        handle = serve.run(D.options(num_replicas=8).bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])

        # Verify 8 replicas running.
        dep = list(serve.status().applications["app"].deployments.values())[0]
        assert dep.replica_states.get("RUNNING", 0) == 8

        stop_event.set()
        t.join(timeout=5)

        assert len(errors) == 0, f"Request errors during upscale: {errors}"
        assert len(successes) > 0, "Background sender never succeeded"

        # Verify the final 8 replicas belong to exactly 4 gangs, and the
        # original 2 gangs are preserved (subset of the final set).
        final_gang_ids = set()
        seen_pids = set()
        for _ in range(80):
            resp = handle.remote().result()
            if resp["gang_id"] is not None:
                final_gang_ids.add(resp["gang_id"])
            seen_pids.add(resp["pid"])
            if len(seen_pids) >= 8:
                break
        assert (
            len(final_gang_ids) == 4
        ), f"Expected 4 final gangs, got {len(final_gang_ids)}: {final_gang_ids}"
        assert initial_gang_ids.issubset(final_gang_ids), (
            f"Initial gang_ids {initial_gang_ids} are not a subset of "
            f"final gang_ids {final_gang_ids}"
        )

        serve.delete("app")
        serve.shutdown()

    # ------------------------------------------------------------------
    # Downscale: 8 → 4, gang_size=2 → removes 2 complete gangs, no downtime
    # ------------------------------------------------------------------
    def test_downscale_gang_boundary(self, ray_cluster):
        """Downscaling removes only complete gangs.  Surviving replicas keep serving."""
        import threading

        cluster = ray_cluster
        cluster.add_node(num_cpus=2)
        cluster.add_node(num_cpus=2)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        # Pin version="v1" so that changing only num_replicas does NOT
        # trigger a rolling restart (otherwise each serve.run generates a
        # random code_version and all replicas are replaced).
        @serve.deployment(
            name="D",
            version="v1",
            num_replicas=8,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class D:
            def __call__(self):
                ctx = ray.serve.context._get_internal_replica_context()
                gc = ctx.gang_context
                return {"pid": os.getpid(), "gang_id": gc.gang_id if gc else None}

        handle = serve.run(D.bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])

        dep = list(serve.status().applications["app"].deployments.values())[0]
        assert dep.replica_states.get("RUNNING", 0) == 8

        # Collect the initial gang_ids (should be 4 gangs of size 2).
        initial_gang_ids = set()
        for _ in range(80):
            resp = handle.remote().result()
            if resp["gang_id"] is not None:
                initial_gang_ids.add(resp["gang_id"])
        assert (
            len(initial_gang_ids) == 4
        ), f"Expected 4 initial gangs, got {len(initial_gang_ids)}"

        # Start background request sender *before* the scale-down.
        errors, successes = [], []
        stop_event = threading.Event()
        t = threading.Thread(
            target=self._send_requests_background,
            args=(handle, stop_event, errors, successes),
            daemon=True,
        )
        t.start()

        # Downscale: 8 → 4 (same code + version, only num_replicas changes).
        handle = serve.run(D.options(num_replicas=4).bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])

        dep = list(serve.status().applications["app"].deployments.values())[0]
        assert dep.replica_states.get("RUNNING", 0) == 4

        stop_event.set()
        t.join(timeout=5)

        assert len(errors) == 0, f"Request errors during downscale: {errors}"
        assert len(successes) > 0, "Background sender never succeeded"

        # Verify the surviving 4 replicas belong to exactly 2 complete gangs
        # and those gang_ids are a subset of the original 4.
        surviving_gang_ids = set()
        seen_pids = set()
        for _ in range(80):
            resp = handle.remote().result()
            if resp["gang_id"] is not None:
                surviving_gang_ids.add(resp["gang_id"])
            seen_pids.add(resp["pid"])
            if len(seen_pids) >= 4:
                break
        assert len(surviving_gang_ids) == 2, (
            f"Expected 2 surviving gangs, got {len(surviving_gang_ids)}: "
            f"{surviving_gang_ids}"
        )
        assert surviving_gang_ids.issubset(initial_gang_ids), (
            f"Surviving gang_ids {surviving_gang_ids} are not a subset of "
            f"initial gang_ids {initial_gang_ids}"
        )

        serve.delete("app")
        serve.shutdown()

    # ------------------------------------------------------------------
    # Rolling update: same num_replicas, new code, gang_size replicas at a time
    # ------------------------------------------------------------------
    def test_rolling_update_gang_size_at_a_time(self, ray_cluster):
        """Rolling update replaces one gang at a time.  No downtime."""
        import threading

        cluster = ray_cluster
        cluster.add_node(num_cpus=2)
        cluster.add_node(num_cpus=2)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(
            name="D",
            num_replicas=4,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class V1:
            def __call__(self):
                return "v1"

        handle = serve.run(V1.bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])
        assert handle.remote().result() == "v1"

        # Start background request sender *before* the update.
        errors, successes = [], []
        stop_event = threading.Event()
        t = threading.Thread(
            target=self._send_requests_background,
            args=(handle, stop_event, errors, successes),
            daemon=True,
        )
        t.start()

        # New code version → triggers requires_actor_restart → rolling update.
        @serve.deployment(
            name="D",
            num_replicas=4,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class V2:
            def __call__(self):
                return "v2"

        handle = serve.run(V2.bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])

        # All replicas should now return "v2".
        for _ in range(20):
            assert handle.remote().result() == "v2"

        stop_event.set()
        t.join(timeout=5)

        assert len(errors) == 0, f"Request errors during rolling update: {errors}"
        assert len(successes) > 0, "Background sender never succeeded"

        serve.delete("app")
        serve.shutdown()

    # ------------------------------------------------------------------
    # Rolling update *with* scale change: 4→8 replicas + new code
    # ------------------------------------------------------------------
    def test_rolling_update_with_scale_change(self, ray_cluster):
        """Rolling update + upscale works correctly.  No downtime."""
        import threading

        cluster = ray_cluster
        cluster.add_node(num_cpus=2)
        cluster.add_node(num_cpus=2)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(
            name="D",
            num_replicas=4,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class V1:
            def __call__(self):
                return "v1"

        handle = serve.run(V1.bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])
        assert handle.remote().result() == "v1"

        # Start background request sender *before* the update.
        errors, successes = [], []
        stop_event = threading.Event()
        t = threading.Thread(
            target=self._send_requests_background,
            args=(handle, stop_event, errors, successes),
            daemon=True,
        )
        t.start()

        # New code version + more replicas.
        @serve.deployment(
            name="D",
            num_replicas=8,
            ray_actor_options={"num_cpus": 0.25},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class V2:
            def __call__(self):
                return "v2"

        handle = serve.run(V2.bind(), name="app")
        wait_for_condition(check_apps_running, apps=["app"])

        dep = list(serve.status().applications["app"].deployments.values())[0]
        assert dep.replica_states.get("RUNNING", 0) == 8

        for _ in range(20):
            assert handle.remote().result() == "v2"

        stop_event.set()
        t.join(timeout=5)

        assert len(errors) == 0, f"Errors during rolling update + scale: {errors}"
        assert len(successes) > 0, "Background sender never succeeded"

        serve.delete("app")
        serve.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
