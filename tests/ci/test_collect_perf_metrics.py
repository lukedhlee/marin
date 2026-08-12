# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pyarrow as pa
from finelog.client import LogClient
from iris.client import IrisClient
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from rigging.timing import Timestamp

from scripts.ci import collect_perf_metrics


def _task(job, index: int):
    key = ResourceKey("test", ResourceKind.TASK, f"/owner/perf/{index}")
    identity = TaskIdentity(key, f"task-{index}")
    attempt = SimpleNamespace(
        identity=AttemptIdentity(key, 0, f"attempt-{index}"),
        exit_code=index,
        error_message="",
    )
    summary = SimpleNamespace(
        identity=identity,
        task_index=index,
        state=TaskState.SUCCEEDED,
        current_node=None,
        failure_count=0,
        preemption_count=0,
        started_at=Timestamp.from_ms(20),
        finished_at=Timestamp.from_ms(30),
        status_message="",
        error_message="",
    )
    return summary, SimpleNamespace(attempts=(attempt,))


def test_fetch_job_summary_includes_tasks_from_every_page() -> None:
    job_key = ResourceKey("test", ResourceKind.JOB, "/owner/perf")
    job = SimpleNamespace(
        identity=JobIdentity(job_key, "job-uid"),
        state=JobState.SUCCEEDED,
        num_tasks=3,
        submitted_at=Timestamp.from_ms(1),
        started_at=Timestamp.from_ms(2),
        finished_at=Timestamp.from_ms(40),
        error_message="",
    )
    rows = [_task(job, index) for index in range(3)]

    class CurrentJob:
        def status(self):
            return job

    class FakeClient:
        def current_job(self, _job_id):
            return CurrentJob()

        def list_tasks(self, query):
            if query.page_token is None:
                return Page(tuple(summary for summary, _ in rows[:2]), "next", ())
            return Page((rows[2][0],), None, ())

        def describe_tasks(self, keys):
            details = {summary.identity.key: detail for summary, detail in rows}
            return tuple(details[key] for key in keys)

    client = FakeClient()

    result = collect_perf_metrics.fetch_job_summary(cast(IrisClient, client), job_key.resource_id)

    assert result is not None
    assert [(task["task_id"], task["exit_code"]) for task in result["tasks"]] == [
        ("/owner/perf/0", 0),
        ("/owner/perf/1", 1),
        ("/owner/perf/2", 2),
    ]


def test_peak_worker_memory_comes_from_task_measurements() -> None:
    class FakeLogClient:
        def query(self, *_args, **_kwargs):
            return pa.table({"peak_worker_memory_mb": [73_421]})

    peak = collect_perf_metrics.fetch_peak_worker_memory_mb(cast(LogClient, FakeLogClient()), "/owner/perf")
    report = collect_perf_metrics.build_report(
        job_id="/owner/perf",
        summary=None,
        job_tree=None,
        leaf_summaries=[],
        peak_worker_memory_mb=peak,
        status=None,
        workflow_env={},
    )

    assert report.peak_worker_memory_mb == 73_421
    assert "finelog peak worker memory unavailable" not in report.warnings


def test_missing_task_measurements_are_not_reported_as_zero_memory() -> None:
    class FakeLogClient:
        def query(self, *_args, **_kwargs):
            return pa.table({"peak_worker_memory_mb": [None]})

    peak = collect_perf_metrics.fetch_peak_worker_memory_mb(cast(LogClient, FakeLogClient()), "/owner/perf")
    report = collect_perf_metrics.build_report(
        job_id="/owner/perf",
        summary=None,
        job_tree=None,
        leaf_summaries=[],
        peak_worker_memory_mb=peak,
        status=None,
        workflow_env={},
    )

    assert report.peak_worker_memory_mb == 0
    assert "finelog peak worker memory unavailable" in report.warnings
