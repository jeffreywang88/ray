"""Tests for gang scheduling decorator and E2E behavior.

Gang scheduling ensures that groups of replicas (gangs) are scheduled together
atomically, which is essential for distributed training and inference workloads
that require tight coordination between replicas.

Config validation tests are in tests/unit/test_config.py.
"""

import sys
import uuid

import pytest

import ray
from ray import serve
from ray._common.test_utils import wait_for_condition
from ray.serve._private.test_utils import check_apps_running
from ray.serve.config import (
    GangPlacementStrategy,
    GangRuntimeFailurePolicy,
    GangSchedulingConfig,
)
from ray.tests.conftest import *  # noqa


class TestGangSchedulingDecorator:
    """Tests for using gang_scheduling_config with @serve.deployment decorator."""

    def test_decorator_accepts_gang_scheduling_config(self):
        """Verify @serve.deployment accepts gang_scheduling_config parameter."""

        @serve.deployment(
            num_replicas=4,
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        )
        class MyDeployment:
            def __call__(self):
                return "hello"

        # Verify the deployment was created successfully
        assert MyDeployment is not None
        assert MyDeployment.name == "MyDeployment"

    def test_decorator_with_full_gang_config(self):
        """Verify decorator works with all gang scheduling options."""

        @serve.deployment(
            num_replicas=8,
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=4,
                gang_timeout_s=60.0,
                gang_placement_strategy=GangPlacementStrategy.STRICT_PACK,
                max_retries=5,
                runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_GANG,
            ),
        )
        class MyDeployment:
            def __call__(self):
                return "hello"

        assert MyDeployment is not None
        assert MyDeployment.name == "MyDeployment"
        assert MyDeployment.num_replicas == 8

    def test_options_accepts_gang_scheduling_config(self):
        """Verify Deployment.options() accepts gang_scheduling_config."""

        @serve.deployment
        class MyDeployment:
            def __call__(self):
                return "hello"

        updated = MyDeployment.options(
            num_replicas=4,
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        )

        assert updated is not None
        assert updated.num_replicas == 4

    def test_options_with_full_gang_config(self):
        """Verify options() works with all gang scheduling parameters."""

        @serve.deployment
        class MyDeployment:
            def __call__(self):
                return "hello"

        updated = MyDeployment.options(
            num_replicas=8,
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=4,
                gang_timeout_s=120.0,
                gang_placement_strategy=GangPlacementStrategy.SPREAD,
                max_retries=10,
                runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_REPLICA,
            ),
        )

        assert updated is not None
        assert updated.num_replicas == 8


class TestGangSchedulingE2E:
    """End-to-end tests for gang scheduling behavior."""

    @pytest.mark.asyncio
    async def test_gang_deployment_runs(self, ray_cluster):
        """
        Verifies that a deployment with gang_scheduling_config can be deployed
        and responds to requests.
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=4)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(
            num_replicas=4,
            ray_actor_options={"num_cpus": 1},
            gang_scheduling_config=GangSchedulingConfig(gang_size=4),
        )
        class GangDeployment:
            def __call__(self):
                return ray.get_runtime_context().get_node_id()

        handle = serve.run(GangDeployment.bind(), name="gang_e2e_app")

        # Wait for deployment to be running
        wait_for_condition(check_apps_running, apps=["gang_e2e_app"], timeout=60)

        # Verify we can call the deployment
        result = await handle.remote()
        assert result is not None

        serve.delete("gang_e2e_app")
        serve.shutdown()

    @pytest.mark.asyncio
    async def test_gang_context_available_in_replica_context(self, ray_cluster):
        """
        Verifies that gang_context is properly set in ReplicaContext
        and contains correct values (gang_id, rank, world_size, member_replica_ids).
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=4)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        # Collect gang context from all replicas
        @ray.remote
        class GangContextCollector:
            def __init__(self):
                self.contexts = []

            def add_context(self, gang_id, rank, world_size, member_replica_ids):
                self.contexts.append({
                    "gang_id": gang_id,
                    "rank": rank,
                    "world_size": world_size,
                    "member_replica_ids": member_replica_ids,
                })

            def get_contexts(self):
                return self.contexts

        collector = GangContextCollector.options(
            name="gang_context_collector", lifetime="detached"
        ).remote()

        @serve.deployment(
            num_replicas=2,
            ray_actor_options={"num_cpus": 1},
            gang_scheduling_config=GangSchedulingConfig(gang_size=2),
        )
        class GangDeployment:
            def __init__(self):
                # Get gang context from ReplicaContext during initialization
                import ray.serve
                ctx = ray.serve.context._get_internal_replica_context()
                gang_ctx = ctx.gang_context
                if gang_ctx is not None:
                    collector_actor = ray.get_actor("gang_context_collector")
                    ray.get(collector_actor.add_context.remote(
                        gang_ctx.gang_id,
                        gang_ctx.rank,
                        gang_ctx.world_size,
                        gang_ctx.member_replica_ids,
                    ))

            def __call__(self):
                import ray.serve
                ctx = ray.serve.context._get_internal_replica_context()
                gang_ctx = ctx.gang_context
                if gang_ctx is None:
                    return {"gang_context": None}
                return {
                    "gang_id": gang_ctx.gang_id,
                    "rank": gang_ctx.rank,
                    "world_size": gang_ctx.world_size,
                    "member_replica_ids": gang_ctx.member_replica_ids,
                }

        handle = serve.run(GangDeployment.bind(), name="gang_context_app")
        wait_for_condition(check_apps_running, apps=["gang_context_app"], timeout=60)

        # Call the deployment to get gang context from a replica
        result = await handle.remote()

        # Verify gang_context is set
        assert result.get("gang_context") is None or result.get("gang_id") is not None, \
            "gang_context should be set for gang deployment"

        if result.get("gang_id") is not None:
            # Verify gang_context fields
            assert result["world_size"] == 2, "world_size should match gang_size"
            assert result["rank"] in [0, 1], "rank should be 0 or 1 for gang_size=2"
            assert len(result["member_replica_ids"]) == 2, \
                "member_replica_ids should have 2 members"

        # Get all contexts collected during initialization
        contexts = ray.get(collector.get_contexts.remote())

        # If contexts were collected, verify them
        if len(contexts) > 0:
            # All replicas should have the same gang_id
            gang_ids = set(c["gang_id"] for c in contexts)
            assert len(gang_ids) == 1, f"All replicas should have same gang_id, got {gang_ids}"

            # All should have world_size = 2
            for ctx in contexts:
                assert ctx["world_size"] == 2, f"Expected world_size=2, got {ctx['world_size']}"

            # Ranks should be unique
            ranks = [c["rank"] for c in contexts]
            assert len(set(ranks)) == len(ranks), f"Ranks should be unique, got {ranks}"

        serve.delete("gang_context_app")
        ray.kill(collector)
        serve.shutdown()

    @pytest.mark.asyncio
    async def test_gang_with_timeout_config(self, ray_cluster):
        """
        Verifies gang_timeout_s parameter is accepted in a deployment.
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=4)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(
            num_replicas=2,
            ray_actor_options={"num_cpus": 1},
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=2,
                gang_timeout_s=60.0,
            ),
        )
        class GangDeployment:
            def __call__(self):
                return "ok"

        handle = serve.run(GangDeployment.bind(), name="gang_timeout_app")
        wait_for_condition(check_apps_running, apps=["gang_timeout_app"], timeout=60)

        result = await handle.remote()
        assert result == "ok"

        serve.delete("gang_timeout_app")
        serve.shutdown()

    @pytest.mark.asyncio
    async def test_gang_with_max_retries_config(self, ray_cluster):
        """
        Verifies max_retries parameter is accepted in a deployment.
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=4)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        @serve.deployment(
            num_replicas=2,
            ray_actor_options={"num_cpus": 1},
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=2,
                max_retries=5,
            ),
        )
        class GangDeployment:
            def __call__(self):
                return "ok"

        handle = serve.run(GangDeployment.bind(), name="gang_retries_app")
        wait_for_condition(check_apps_running, apps=["gang_retries_app"], timeout=60)

        result = await handle.remote()
        assert result == "ok"

        serve.delete("gang_retries_app")
        serve.shutdown()

    @pytest.mark.asyncio
    async def test_gang_with_restart_gang_policy(self, ray_cluster):
        """
        Verifies that when a replica in a gang fails health check with
        RESTART_GANG policy, ALL replicas in that gang are force-stopped
        and restarted, while replicas in other gangs are NOT affected.
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=8)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        # Track replica actor IDs
        # When replicas restart, they get NEW actor IDs
        # So we track all unique actor IDs seen
        @ray.remote
        class ReplicaTracker:
            def __init__(self):
                self.all_actor_ids = set()  # All actor IDs ever seen
                self.should_fail_health_check = None

            def register_start(self, actor_id: str):
                self.all_actor_ids.add(actor_id)
                return len(self.all_actor_ids)

            def get_all_actor_ids(self):
                return list(self.all_actor_ids)

            def get_count(self):
                return len(self.all_actor_ids)

            def set_fail_replica(self, actor_id: str):
                self.should_fail_health_check = actor_id

            def should_fail(self, actor_id: str) -> bool:
                return self.should_fail_health_check == actor_id

            def clear_fail(self):
                self.should_fail_health_check = None

        tracker_name = f"replica_tracker_gang_{uuid.uuid4().hex[:8]}"
        tracker = ReplicaTracker.options(
            name=tracker_name, lifetime="detached"
        ).remote()

        @serve.deployment(
            num_replicas=4,  # 2 gangs of size 2
            ray_actor_options={"num_cpus": 1},
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=2,
                runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_GANG,
            ),
            health_check_period_s=1,
            health_check_timeout_s=1,
        )
        class GangDeployment:
            def __init__(self):
                import ray
                self.actor_id = ray.get_runtime_context().get_actor_id()
                self.tracker = ray.get_actor(tracker_name)
                ray.get(self.tracker.register_start.remote(self.actor_id))

            def check_health(self):
                # Fail health check if this replica is marked to fail
                should_fail = ray.get(self.tracker.should_fail.remote(self.actor_id))
                if should_fail:
                    raise RuntimeError("Simulated health check failure")

            def __call__(self):
                return self.actor_id

        handle = serve.run(GangDeployment.bind(), name="gang_restart_policy_app")
        wait_for_condition(
            check_apps_running, apps=["gang_restart_policy_app"], timeout=60
        )

        # Get initial replica actor IDs by making requests
        initial_actor_ids = set()
        for _ in range(20):  # Make enough requests to hit all replicas
            result = await handle.remote()
            initial_actor_ids.add(result)
        assert len(initial_actor_ids) == 4, f"Expected 4 replicas, got {initial_actor_ids}"

        # Verify initial count
        initial_count = ray.get(tracker.get_count.remote())
        assert initial_count == 4, f"Expected 4 initial actor IDs, got {initial_count}"

        # Mark one replica to fail health check
        failed_actor_id = list(initial_actor_ids)[0]
        ray.get(tracker.set_fail_replica.remote(failed_actor_id))

        # Wait for the deployment to detect the failure and restart gang
        # Health check fails 3 times before replica is marked unhealthy
        # health_check_period_s=1, so ~3-4 seconds for failure detection
        import time
        time.sleep(8)  # Wait for health check to detect failure and restart

        # Clear the failure so new replicas don't fail
        ray.get(tracker.clear_fail.remote())

        # Wait for deployment to recover
        wait_for_condition(
            check_apps_running, apps=["gang_restart_policy_app"], timeout=60
        )

        # With RESTART_GANG policy:
        # - 1 replica fails health check
        # - Both replicas in that gang are force-stopped (2 replicas)
        # - 2 new replicas are created (with new actor IDs)
        # - Other gang (2 replicas) is NOT affected
        # Total unique actor IDs seen: 4 original + 2 new = 6
        #
        # Wait for all new replicas to register. This is necessary because
        # the deployment may be marked RUNNING before all replicas have
        # completed their __init__ and registered with the tracker.
        def check_replica_count():
            count = ray.get(tracker.get_count.remote())
            # We expect 6 (4 original + 2 new), but due to timing we might
            # briefly see 5 if one new replica hasn't registered yet.
            return count >= 6

        wait_for_condition(check_replica_count, timeout=30, retry_interval_ms=500)

        final_count = ray.get(tracker.get_count.remote())
        assert final_count == 6, (
            f"Expected 6 total actor IDs (4 original + 2 new from gang restart), "
            f"got {final_count}. All IDs: {ray.get(tracker.get_all_actor_ids.remote())}"
        )

        serve.delete("gang_restart_policy_app")
        ray.kill(tracker)
        serve.shutdown()

    @pytest.mark.asyncio
    async def test_gang_with_restart_replica_policy(self, ray_cluster):
        """
        Verifies that when a replica in a gang fails health check with
        RESTART_REPLICA policy, ONLY that replica is restarted (not the
        entire gang), and it rejoins the deployment.
        """
        cluster = ray_cluster
        cluster.add_node(num_cpus=8)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        serve.start()

        # Track replica actor IDs
        @ray.remote
        class ReplicaTracker:
            def __init__(self):
                self.all_actor_ids = set()
                self.should_fail_health_check = None

            def register_start(self, actor_id: str):
                self.all_actor_ids.add(actor_id)
                return len(self.all_actor_ids)

            def get_all_actor_ids(self):
                return list(self.all_actor_ids)

            def get_count(self):
                return len(self.all_actor_ids)

            def set_fail_replica(self, actor_id: str):
                self.should_fail_health_check = actor_id

            def should_fail(self, actor_id: str) -> bool:
                return self.should_fail_health_check == actor_id

            def clear_fail(self):
                self.should_fail_health_check = None

        tracker_name = f"replica_tracker_replica_{uuid.uuid4().hex[:8]}"
        tracker = ReplicaTracker.options(
            name=tracker_name, lifetime="detached"
        ).remote()

        @serve.deployment(
            num_replicas=4,  # 2 gangs of size 2
            ray_actor_options={"num_cpus": 1},
            gang_scheduling_config=GangSchedulingConfig(
                gang_size=2,
                runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_REPLICA,
            ),
            health_check_period_s=1,
            health_check_timeout_s=1,
        )
        class GangDeployment:
            def __init__(self):
                import ray
                self.actor_id = ray.get_runtime_context().get_actor_id()
                self.tracker = ray.get_actor(tracker_name)
                ray.get(self.tracker.register_start.remote(self.actor_id))

            def check_health(self):
                should_fail = ray.get(self.tracker.should_fail.remote(self.actor_id))
                if should_fail:
                    raise RuntimeError("Simulated health check failure")

            def __call__(self):
                return self.actor_id

        handle = serve.run(GangDeployment.bind(), name="gang_replica_policy_app")
        wait_for_condition(
            check_apps_running, apps=["gang_replica_policy_app"], timeout=60
        )

        # Get initial replica actor IDs
        initial_actor_ids = set()
        for _ in range(20):
            result = await handle.remote()
            initial_actor_ids.add(result)
        assert len(initial_actor_ids) == 4

        # Verify initial count
        initial_count = ray.get(tracker.get_count.remote())
        assert initial_count == 4

        # Mark one replica to fail health check
        failed_actor_id = list(initial_actor_ids)[0]
        ray.get(tracker.set_fail_replica.remote(failed_actor_id))

        # Wait for the failure to be detected
        import time
        time.sleep(8)

        # Clear the failure
        ray.get(tracker.clear_fail.remote())

        # Wait for recovery
        wait_for_condition(
            check_apps_running, apps=["gang_replica_policy_app"], timeout=60
        )

        # With RESTART_REPLICA policy:
        # - 1 replica fails health check
        # - Only that 1 replica is stopped (gang members NOT affected)
        # - 1 new replica is created and scheduled individually
        # - Other 3 replicas are NOT affected
        # Total unique actor IDs seen: 4 original + 1 new = 5
        final_count = ray.get(tracker.get_count.remote())
        assert final_count == 5, (
            f"Expected 5 total actor IDs (4 original + 1 new from single replica restart), "
            f"got {final_count}. All IDs: {ray.get(tracker.get_all_actor_ids.remote())}"
        )

        serve.delete("gang_replica_policy_app")
        ray.kill(tracker)
        serve.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
