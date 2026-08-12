# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import cast

import pyarrow as pa
from finelog.client import LogClient
from iris.client import IrisClient
from iris.resources.identity import ResourceKey, ResourceKind
from iris.resources.job import JobSummary
from iris.resources.source import Page
from iris.resources.state import JobState, TaskState
from iris.resources.task import TaskDetail, TaskSummary
from iris.testing.resources import make_attempt_summary, make_job_summary, make_task_detail, make_task_summary
from rigging.timing import Timestamp

from scripts.ci import collect_perf_metrics


def _task(job: JobSummary, index: int) -> tuple[TaskSummary, TaskDetail]:
    summary = make_task_summary(
        job.identity,
        index,
        task_uid=f"task-{index}",
        state=TaskState.SUCCEEDED,
        started_at=Timestamp.from_ms(20),
        finished_at=Timestamp.from_ms(30),
    )
    attempt = make_attempt_summary(
        summary.identity,
        0,
        attempt_uid=f"attempt-{index}",
        state=TaskState.SUCCEEDED,
        started_at=Timestamp.from_ms(20),
        finished_at=Timestamp.from_ms(30),
        exit_code=index,
    )
    summary = replace(summary, current_attempt=attempt.identity)
    return summary, make_task_detail(summary, (attempt,))


def test_fetch_job_summary_includes_tasks_from_every_page() -> None:
    job_key = ResourceKey("test", ResourceKind.JOB, "/owner/perf")
    job = make_job_summary(
        job_key.resource_id,
        cluster_id=job_key.cluster_id,
        state=JobState.SUCCEEDED,
        num_tasks=3,
        submitted_at=Timestamp.from_ms(1),
        started_at=Timestamp.from_ms(2),
        finished_at=Timestamp.from_ms(40),
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
        def query(self, sql: str, *, max_rows: int):
            assert sql == (
                'SELECT MAX(memory_peak_mb) AS peak_worker_memory_mb FROM "iris.task" '
                "WHERE task_id LIKE '/owner/o''clock\\_100\\%/%' ESCAPE '\\'"
            )
            assert max_rows == 1
            return pa.table({"peak_worker_memory_mb": [73_421]})

    peak = collect_perf_metrics.fetch_peak_worker_memory_mb(cast(LogClient, FakeLogClient()), "/owner/o'clock_100%")
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
        def query(self, _sql: str, *, max_rows: int):
            assert max_rows == 1
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
