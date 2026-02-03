import ray
from ray import serve
from ray._common.test_utils import wait_for_condition
from ray.serve.gang import GangRuntimeFailurePolicy, GangSchedulingConfig


@ray.remote
class TargetReplica:
    def __init__(self):
        self._target_replica_tag = None

    def set(self, replica_tag: str):
        self._target_replica_tag = replica_tag

    def get(self):
        return self._target_replica_tag


def test_gang_restart_on_health_check_failure(serve_instance):
    gang_size = 2
    num_replicas = 4
    target_actor = TargetReplica.remote()

    @serve.deployment(
        num_replicas=num_replicas,
        health_check_period_s=0.5,
        health_check_timeout_s=1,
        gang_scheduling_config=GangSchedulingConfig(
            gang_size=gang_size,
            runtime_failure_policy=GangRuntimeFailurePolicy.RESTART_GANG,
        ),
    )
    class GangDeployment:
        def __init__(self, target_actor):
            self._target_actor = target_actor

        def check_health(self):
            target = ray.get(self._target_actor.get.remote())
            replica_tag = serve.get_replica_context().replica_tag
            if target == replica_tag:
                raise RuntimeError("intentional health check failure")

        def __call__(self):
            context = serve.get_replica_context()
            rank = context.rank.rank
            return {
                "replica_tag": context.replica_tag,
                "rank": rank,
                "gang_id": rank // gang_size,
            }

    handle = serve.run(GangDeployment.bind(target_actor))

    rank_to_tag = {}

    def collect_initial_replicas():
        results = ray.get([handle.remote() for _ in range(40)])
        for result in results:
            rank_to_tag[result["rank"]] = result["replica_tag"]
        return len(rank_to_tag) == num_replicas

    wait_for_condition(collect_initial_replicas, timeout=30)
    initial_rank_to_tag = dict(rank_to_tag)

    target_rank = 0
    ray.get(target_actor.set.remote(initial_rank_to_tag[target_rank]))

    def gang_restarted():
        updated = {}
        results = ray.get([handle.remote() for _ in range(40)])
        for result in results:
            updated[result["rank"]] = result["replica_tag"]
        if len(updated) != num_replicas:
            return False
        return (
            updated[0] != initial_rank_to_tag[0]
            and updated[1] != initial_rank_to_tag[1]
            and updated[2] == initial_rank_to_tag[2]
            and updated[3] == initial_rank_to_tag[3]
        )

    wait_for_condition(gang_restarted, timeout=60)