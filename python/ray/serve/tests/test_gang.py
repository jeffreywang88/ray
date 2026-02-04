"""Tests for gang scheduling decorator and E2E behavior.

Gang scheduling ensures that groups of replicas (gangs) are scheduled together
atomically, which is essential for distributed training and inference workloads
that require tight coordination between replicas.

Config validation tests are in tests/unit/test_config.py.
"""

import sys

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
        Verifies runtime_failure_policy=RESTART_GANG is accepted and deployment runs.
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
                runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_GANG,
            ),
        )
        class GangDeployment:
            def __call__(self):
                return "ok"

        handle = serve.run(GangDeployment.bind(), name="gang_restart_policy_app")
        wait_for_condition(
            check_apps_running, apps=["gang_restart_policy_app"], timeout=60
        )

        result = await handle.remote()
        assert result == "ok"

        serve.delete("gang_restart_policy_app")
        serve.shutdown()

    @pytest.mark.asyncio
    async def test_gang_with_restart_replica_policy(self, ray_cluster):
        """
        Verifies runtime_failure_policy=RESTART_REPLICA is accepted and deployment runs.
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
                runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_REPLICA,
            ),
        )
        class GangDeployment:
            def __call__(self):
                return "ok"

        handle = serve.run(GangDeployment.bind(), name="gang_replica_policy_app")
        wait_for_condition(
            check_apps_running, apps=["gang_replica_policy_app"], timeout=60
        )

        result = await handle.remote()
        assert result == "ok"

        serve.delete("gang_replica_policy_app")
        serve.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
