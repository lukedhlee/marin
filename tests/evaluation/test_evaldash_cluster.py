# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

from iris.resources.identity import (
    AttemptIdentity,
    JobIdentity,
    NodeIdentity,
    ResourceKey,
    ResourceKind,
    TaskIdentity,
)
from iris.resources.source import Page
from iris.rpc import job_pb2
from rigging.timing import Timestamp

discovery = ModuleType("infra.evaldash.src.discovery")
discovery.resolve_internal_ip = lambda *_args, **_kwargs: "127.0.0.1"
sys.modules[discovery.__name__] = discovery

from infra.evaldash.src import cluster  # noqa: E402


def _attempt(task_key: ResourceKey, number: int, uid: str, *, node=None, failed: bool = False):
    return SimpleNamespace(
        identity=AttemptIdentity(task_key, number, uid),
        state=job_pb2.TASK_STATE_WORKER_FAILED if failed else job_pb2.TASK_STATE_RUNNING,
        node=node,
        exit_code=1 if failed else None,
        error_message="worker lost" if failed else "",
        started_at=Timestamp.from_ms(40 if failed else 70),
        finished_at=Timestamp.from_ms(50) if failed else None,
    )


def _task(job: JobIdentity, index: int, *, node: NodeIdentity | None, attempts: tuple):
    key = ResourceKey("iris", ResourceKind.TASK, f"/owner/eval/{index}")
    current_attempt = attempts[-1]
    return SimpleNamespace(
        summary=SimpleNamespace(
            identity=TaskIdentity(key, f"task-uid-{index}"),
            job=job,
            state=job_pb2.TASK_STATE_RUNNING,
            current_attempt=current_attempt.identity,
            current_node=node,
            started_at=Timestamp.from_ms(70),
            finished_at=None,
            error_message="",
        ),
        attempts=attempts,
    )


def test_job_status_reads_all_tasks_through_resource_api(monkeypatch) -> None:
    job_key = ResourceKey("iris", ResourceKind.JOB, "/owner/eval")
    job_identity = JobIdentity(job_key, "job-uid")
    job = SimpleNamespace(
        summary=SimpleNamespace(
            identity=job_identity,
            state=job_pb2.JOB_STATE_RUNNING,
            started_at=Timestamp.from_ms(20),
            finished_at=None,
            error_message="",
            pending_reason="warming workers",
            exit_code=17,
        ),
        spec=SimpleNamespace(name="evaluation"),
    )
    node = NodeIdentity(ResourceKey("iris", ResourceKind.NODE, "worker-a"), "gpu", "worker-uid")
    first_key = ResourceKey("iris", ResourceKind.TASK, "/owner/eval/0")
    first_task = _task(
        job_identity,
        0,
        node=node,
        attempts=(
            _attempt(first_key, 0, "attempt-0", node=node, failed=True),
            _attempt(first_key, 1, "attempt-1", node=node),
        ),
    )
    second_key = ResourceKey("iris", ResourceKind.TASK, "/owner/eval/1")
    second_task = _task(
        job_identity,
        1,
        node=None,
        attempts=(_attempt(second_key, 0, "attempt-second"),),
    )

    class FakeResourceClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def list_jobs(self, _query):
            return Page((job.summary,), None, ())

        def describe_job(self, _key):
            return job

        def list_tasks(self, query):
            if query.page_token is None:
                return Page((first_task.summary,), "next", ())
            return Page((second_task.summary,), None, ())

        def describe_tasks(self, keys):
            details = {
                first_task.summary.identity.key: first_task,
                second_task.summary.identity.key: second_task,
            }
            return tuple(details[key] for key in keys)

        def close(self) -> None:
            pass

    monkeypatch.setattr(cluster, "ResourceRpcClient", FakeResourceClient)
    gateway = cluster.ClusterGateway()
    monkeypatch.setattr(gateway, "_resolve", lambda *_args: "http://controller")

    result = gateway.job_status("/owner/eval")

    assert result["job"] == {
        "state": "JOB_STATE_RUNNING",
        "error": "",
        "exit_code": 17,
        "started_at": {"epoch_ms": 20},
        "name": "evaluation",
        "status_message": "warming workers",
    }
    assert [task["task_id"] for task in result["tasks"]] == ["/owner/eval/0", "/owner/eval/1"]
    first, second = result["tasks"]
    assert (first["worker_id"], first["current_attempt_id"]) == ("worker-a", 1)
    assert [(attempt["attempt_uid"], attempt["is_worker_failure"]) for attempt in first["attempts"]] == [
        ("attempt-0", True),
        ("attempt-1", False),
    ]
    assert (second["worker_id"], second["current_attempt_id"]) == ("", 0)
