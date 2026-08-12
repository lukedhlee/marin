# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import MagicMock

from iris.client import IrisClient
from iris.resources.execution import Entrypoint, EnvironmentSpec, GpuDevice, ResourceSpec
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.job import JobQuery, JobSummary
from iris.resources.names import JobName
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskSummary
from iris.testing.resources import make_job_summary
from rigging.timing import Timestamp


def _job(job_id: str, uid: str) -> JobSummary:
    return make_job_summary(job_id, job_uid=uid)


def test_current_job_uses_an_exact_resource_query() -> None:
    exact = _job("/alice/train", "exact-uid")
    sibling = _job("/alice/train-longer", "sibling-uid")
    cluster = MagicMock()
    cluster.list_jobs.side_effect = lambda query: Page(
        (exact,) if query == JobQuery(resource_id="/alice/train", page_size=1) else (sibling,),
        None,
        (),
    )
    client = IrisClient(cluster)

    job = client.current_job(JobName.from_wire("/alice/train"))

    assert job.identity == exact.identity


def test_job_wait_returns_the_terminal_summary(monkeypatch) -> None:
    running = _job("/alice/train", "exact-uid")
    succeeded = replace(running, state=JobState.SUCCEEDED)
    cluster = MagicMock()
    cluster.list_jobs.return_value = Page((running,), None, ())
    cluster.job_state.side_effect = (JobState.RUNNING, JobState.SUCCEEDED)
    cluster.describe_job.return_value = MagicMock(summary=succeeded)
    monkeypatch.setattr("iris.client.client.time.sleep", lambda _seconds: None)
    job = IrisClient(cluster).current_job(JobName.from_wire("/alice/train"))

    status = job.wait(timeout=1, poll_interval=0)

    assert status == succeeded


def test_current_task_resolves_a_task_handle_from_its_wire_id() -> None:
    job = _job("/alice/train", "job-uid")
    task_id = JobName.from_wire("/alice/train/7")
    task_identity = TaskIdentity(ResourceKey("test", ResourceKind.TASK, task_id.to_wire()), "task-uid")
    summary = TaskSummary(
        identity=task_identity,
        job=job.identity,
        task_index=7,
        state=TaskState.RUNNING,
        execution_cluster_id="test",
        backend_id="default",
        current_attempt=AttemptIdentity(task_identity.key, 0, "attempt-uid"),
        current_node=None,
        failure_count=0,
        preemption_count=0,
        submitted_at=Timestamp.from_ms(1),
        started_at=Timestamp.from_ms(2),
        finished_at=None,
        status_message="",
        error_message="",
    )
    cluster = MagicMock()
    cluster.list_jobs.return_value = Page((job,), None, ())
    cluster.describe_job.return_value = MagicMock(summary=job)
    cluster.list_tasks.return_value = Page((summary,), None, ())
    client = IrisClient(cluster)

    task = client.current_task(task_id)

    assert task.identity == task_identity


def test_high_level_submit_applies_accelerator_cpu_floor_without_changing_direct_specs() -> None:
    submitted = []

    class Cluster:
        def submit_job(self, spec, *, bundle=None):
            submitted.append(spec)
            return JobIdentity(ResourceKey("test", ResourceKind.JOB, spec.name), "job-uid")

    cluster = Cluster()
    client = IrisClient(cluster)
    requested = ResourceSpec(cpu=0.5, device=GpuDevice("H100"))

    client.submit(
        Entrypoint.from_command("python", "train.py"),
        "train",
        requested,
        environment=EnvironmentSpec(setup_scripts=()),
        user="alice",
    )

    assert submitted[-1].resources.cpu == 4
    assert requested.cpu == 0.5

    client.submit(
        Entrypoint.from_command("python", "train.py"),
        "large-train",
        ResourceSpec(cpu=6, device=GpuDevice("H100")),
        environment=EnvironmentSpec(setup_scripts=()),
        user="alice",
    )
    assert submitted[-1].resources.cpu == 6

    direct = replace(submitted[0], resources=requested)
    client.submit_job(direct)
    assert submitted[-1].resources.cpu == 0.5
