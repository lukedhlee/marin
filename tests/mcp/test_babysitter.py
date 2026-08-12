# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import marin.mcp.babysitter as babysitter
import pytest
from iris.resources.endpoint import ProfileResult
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.source import Page
from iris.rpc import job_pb2
from marin.mcp.babysitter import (
    IrisBabysitter,
    IrisConnectionConfig,
    _token_provider,
    classify_diagnosis,
    parse_zephyr_progress,
    parse_zephyr_thread_state,
    task_status_to_json,
)
from rigging.credentials import MARIN_CLUSTER_TOKEN_ENV
from rigging.timing import Timestamp

_NOW = Timestamp(1_000)


def _job_identity() -> JobIdentity:
    return JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")


def _task_identity() -> TaskIdentity:
    return TaskIdentity(ResourceKey("prod", ResourceKind.TASK, "/alice/train/0"), "task-uid")


def _attempt(
    attempt_number: int,
    attempt_uid: str,
    *,
    state: int = job_pb2.TASK_STATE_RUNNING,
    finished_at: Timestamp | None = None,
    exit_code: int | None = None,
    error_message: str = "",
):
    return SimpleNamespace(
        identity=AttemptIdentity(_task_identity().key, attempt_number, attempt_uid),
        state=state,
        node=None,
        started_at=_NOW,
        finished_at=finished_at,
        exit_code=exit_code,
        error_message=error_message,
        terminal_reason="application" if error_message else "",
    )


def _running_task_detail():
    first = _attempt(1, "attempt-one")
    second = _attempt(2, "attempt-two")
    return SimpleNamespace(
        summary=SimpleNamespace(
            identity=_task_identity(),
            state=job_pb2.TASK_STATE_RUNNING,
            current_attempt=second.identity,
            current_node=None,
            failure_count=1,
            preemption_count=0,
            started_at=_NOW,
            finished_at=None,
            status_message="running",
            error_message="",
        ),
        attempts=(first, second),
        source_statuses=(),
        root_cause_highlights=(),
    )


def _job_detail():
    return SimpleNamespace(
        summary=SimpleNamespace(
            identity=_job_identity(),
            owner_id="alice",
            parent=None,
            state=job_pb2.JOB_STATE_RUNNING,
            execution_cluster_id="prod",
            backend_id="east",
            num_tasks=1,
            submitted_at=_NOW,
            started_at=_NOW,
            finished_at=None,
            error_message="",
            pending_reason="",
        ),
        spec=SimpleNamespace(
            name="train",
            resources=SimpleNamespace(cpu_millicores=1_000, memory=1024, disk=2048, device=None),
            ports=(),
        ),
    )


class _Closeable:
    def close(self) -> None:
        pass


def _service(monkeypatch, resources, controller=None) -> IrisBabysitter:
    monkeypatch.setattr(babysitter, "ResourceRpcClient", lambda *_args, **_kwargs: resources)
    monkeypatch.setattr(babysitter, "ControllerServiceClientSync", lambda *_args, **_kwargs: controller or _Closeable())
    monkeypatch.setattr(babysitter, "LogServiceClientSync", lambda *_args, **_kwargs: _Closeable())
    return IrisBabysitter(IrisConnectionConfig("http://controller.test", cluster="prod"))


def test_task_status_json_preserves_exact_identity_and_attempt_history():
    task_identity = _task_identity()
    attempt_identity = AttemptIdentity(task_identity.key, 1, "attempt-uid")
    attempt = _attempt(
        1,
        "attempt-uid",
        state=job_pb2.TASK_STATE_FAILED,
        finished_at=Timestamp(2_500),
        exit_code=137,
        error_message="OOMKilled",
    )
    task = SimpleNamespace(
        summary=SimpleNamespace(
            identity=task_identity,
            state=job_pb2.TASK_STATE_FAILED,
            current_attempt=attempt_identity,
            current_node=None,
            failure_count=1,
            preemption_count=0,
            started_at=_NOW,
            finished_at=Timestamp(2_500),
            status_message="",
            error_message="OOMKilled",
        ),
        attempts=(attempt,),
        source_statuses=(),
        root_cause_highlights=("container exited 137",),
    )

    payload = task_status_to_json(task)

    assert payload["task_id"] == "/alice/train/0"
    assert payload["task_uid"] == "task-uid"
    assert payload["state"] == "failed"
    assert payload["started_at_ms"] == 1_000
    assert payload["finished_at_ms"] == 2_500
    assert payload["duration_ms"] == 1_500
    assert payload["attempts"][0]["attempt_uid"] == "attempt-uid"
    assert payload["attempts"][0]["exit_code"] == 137
    assert payload["root_cause_highlights"] == ["container exited 137"]


def test_job_summary_payload_preserves_summary_task_fields():
    running_task = SimpleNamespace(
        identity=_task_identity(),
        task_index=0,
        state=job_pb2.TASK_STATE_RUNNING,
        current_node=None,
        failure_count=0,
        preemption_count=0,
        started_at=_NOW,
        finished_at=None,
        status_message="running",
        error_message="",
    )

    payload = babysitter._job_summary_payload(_job_detail(), [running_task])

    assert payload["tasks"][0]["index"] == "0"
    assert payload["job_uid"] == "job-uid"
    assert payload["tasks"][0]["task_uid"] == "task-uid"
    assert "resource_requests" in payload
    assert "resource_usage" not in payload


def test_job_summary_returns_the_selected_job(monkeypatch):
    class Resources:
        def describe_job(self, key: ResourceKey):
            if key != ResourceKey("prod", ResourceKind.JOB, "/alice/train"):
                raise AssertionError(f"unexpected Job key: {key}")
            return _job_detail()

        def list_tasks(self, _query):
            return Page((), None, ())

        def close(self) -> None:
            pass

    service = _service(monkeypatch, Resources())

    payload = service.job_summary("/alice/train")

    assert payload["data"]["job_id"] == "/alice/train"


@pytest.mark.parametrize(
    ("target", "expected_text"),
    [
        ("/alice/train/0", "attempt-two"),
        ("/alice/train/0:1", "attempt-one"),
    ],
)
def test_task_profile_targets_the_exact_resource_attempt(monkeypatch, target, expected_text):
    class Resources:
        def describe_task(self, _key: ResourceKey):
            return _running_task_detail()

        def profile_attempt(self, identity, *, profile, duration) -> ProfileResult:
            return ProfileResult(identity.attempt_uid.encode(), "")

        def close(self) -> None:
            pass

    class LegacyController:
        def profile_task(self, _request):
            return job_pb2.ProfileTaskResponse(profile_data=b"legacy-task-profile")

        def close(self) -> None:
            pass

    service = _service(monkeypatch, Resources(), LegacyController())

    payload = service.profile_task(target=target)

    assert payload["data"] == {"text": expected_text, "encoding": "utf-8"}


def test_system_profile_stays_on_the_process_control_boundary(monkeypatch):
    class Resources:
        def describe_task(self, _key: ResourceKey):
            raise AssertionError("system profiling must not resolve a Task")

        def close(self) -> None:
            pass

    class LegacyController:
        def profile_task(self, _request):
            return job_pb2.ProfileTaskResponse(profile_data=b"controller-profile")

        def close(self) -> None:
            pass

    service = _service(monkeypatch, Resources(), LegacyController())

    payload = service.profile_task(target="/system/process")

    assert payload["data"] == {"text": "controller-profile", "encoding": "utf-8"}


def test_token_provider_uses_env_override(monkeypatch):
    # Pure-IAP: the controller mints no user token, so the only Authorization bearer
    # is the explicit $MARIN_CLUSTER_TOKEN escape hatch (e.g. a worker JWT for CI).
    monkeypatch.setenv(MARIN_CLUSTER_TOKEN_ENV, "env-token")
    provider = _token_provider()
    assert provider is not None
    assert provider.get_token() == "env-token"


def test_token_provider_none_without_env(monkeypatch):
    monkeypatch.delenv(MARIN_CLUSTER_TOKEN_ENV, raising=False)
    assert _token_provider() is None


def test_parse_zephyr_progress_keeps_latest_stage_snapshot():
    lines = [
        "noise: pull_task worker-7",
        "[stage0-Map -> Scatter] 12/20 complete, 3 in-flight, 5 queued, 8/9 workers alive, 1 dead",
        "[stage1-Reduce] 4/10 complete, 1 in-flight, 5 queued, 8/8 workers alive, 0 dead",
        "[stage0-Map -> Scatter] 15/20 complete, 2 in-flight, 3 queued, 8/9 workers alive, 1 dead",
    ]

    progress = parse_zephyr_progress(lines)

    assert len(progress) == 2
    assert progress[0] == {
        "stage": "stage0-Map -> Scatter",
        "completed": 15,
        "total": 20,
        "in_flight": 2,
        "queued": 3,
        "workers_alive": 8,
        "workers_total": 9,
        "workers_dead": 1,
    }
    assert progress[1]["stage"] == "stage1-Reduce"


def test_parse_zephyr_thread_state_classifies_active_and_zombie_dumps():
    active = parse_zephyr_thread_state(
        """
        Thread actor-method_0:
          File "zephyr/execution.py", line 873, in _wait_for_stage
        Thread zephyr-coordinator-loop:
          File "zephyr/execution.py", line 444, in _coordinator_loop
        """
    )
    zombie = parse_zephyr_thread_state(
        """
        Thread worker-pool-0:
          File "concurrent/futures/thread.py", line 58, in _worker
        """
    )

    assert active["state"] == "active"
    assert "waiting for stage completion" in active["evidence"]
    assert zombie["state"] == "zombie_suspected"
    assert "worker pool frames without coordinator loop" in zombie["evidence"]


def test_classify_diagnosis_reports_common_babysitting_signals():
    job = {
        "state": "failed",
        "error": "Terminated by user",
        "failure_count": 3,
        "preemption_count": 1,
        "pending_reason": "Quota exceeded for v5litepod",
        "tasks": [
            {
                "task_id": "/alice/train/0",
                "state": "failed",
                "exit_code": 137,
                "error": "container OOMKilled",
                "pending_reason": "",
                "attempts": [{"attempt_id": 0}, {"attempt_id": 1}, {"attempt_id": 2}],
            }
        ],
    }
    logs = [
        {"task_id": "/alice/train/0", "data": "RESOURCE_EXHAUSTED: TPU quota exceeded"},
        {"task_id": "/alice/train/0", "data": "XLA detected bad TPU node"},
    ]
    workers = [
        {
            "worker_id": "worker-a",
            "healthy": False,
            "status_message": "Heartbeat timeout",
        }
    ]

    thread_dump = 'File "concurrent/futures/thread.py", line 58, in _worker'

    signals = classify_diagnosis(job=job, logs=logs, workers=workers, thread_dump=thread_dump)
    names = {signal["signal"] for signal in signals}

    assert "oom_or_exit_137" in names
    assert "quota_or_backoff" in names
    assert "tpu_xla_bad_node" in names
    assert "dead_worker" in names
    assert "zombie_coordinator" in names
    assert "repeated_retries" in names
    assert "misleading_terminated_by_user" in names
