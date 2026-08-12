# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

from click.testing import CliRunner
from iris.cli.job import job
from iris.cli.process_status import process_group
from iris.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.resources.endpoint import ProfileResult
from iris.resources.identity import AttemptIdentity, JobIdentity, ResourceKey, ResourceKind, TaskIdentity
from rigging.timing import Timestamp

_NOW = Timestamp(1_000)


def _job_identity() -> JobIdentity:
    return JobIdentity(ResourceKey("prod", ResourceKind.JOB, "/alice/train"), "job-uid")


def _task_identity(index: int) -> TaskIdentity:
    return TaskIdentity(ResourceKey("prod", ResourceKind.TASK, f"/alice/train/{index}"), f"task-uid-{index}")


def _receipt() -> ActionReceipt:
    return ActionReceipt(
        action_id="action-7",
        kind=ActionKind.CANCEL_JOB,
        target=_job_identity().key,
        expected_target_uid="job-uid",
        expected_attempt_uid=None,
        state=ActionState.ACCEPTED,
        result_code=ActionResult.NONE,
        result_message="",
        created_at=_NOW,
        updated_at=_NOW,
        completed_at=None,
    )


def test_job_cancel_uses_the_described_exact_identity(monkeypatch) -> None:
    accepted_identity: list[JobIdentity] = []

    class Client:
        def describe_job(self, _key):
            return SimpleNamespace(summary=SimpleNamespace(identity=_job_identity()))

        def cancel_job(self, identity, *, idempotency_key):
            accepted_identity.append(identity)
            assert idempotency_key == "request-7"
            return _receipt()

    monkeypatch.setattr("iris.cli.job.resource_client_for_ctx", lambda _ctx: nullcontext(Client()))

    result = CliRunner().invoke(
        job,
        ["cancel", "/alice/train", "--idempotency-key", "request-7"],
        obj={"cluster_name": "prod", "controller_url": "unused"},
    )

    assert result.exit_code == 0, result.output
    assert accepted_identity == [_job_identity()]


def test_process_profile_for_a_task_uses_the_exact_attempt_resource(monkeypatch) -> None:
    expected = AttemptIdentity(_task_identity(7).key, 2, "attempt-uid-2")
    profiled: list[AttemptIdentity] = []

    class Client:
        def describe_attempt(self, _locator):
            return SimpleNamespace(summary=SimpleNamespace(identity=expected))

        def profile_attempt(self, identity, *, profile, duration):
            profiled.append(identity)
            return ProfileResult(f"profile for {identity.attempt_uid}".encode(), "")

    monkeypatch.setattr("iris.cli.process_status.resource_client_for_ctx", lambda _ctx: nullcontext(Client()))
    monkeypatch.setattr(
        "iris.cli.process_status.rpc_client_for_ctx",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("task profile used legacy RPC")),
    )

    result = CliRunner().invoke(
        process_group,
        ["profile", "--target", "/alice/train/7:2", "threads"],
        obj={"cluster_name": "prod", "controller_url": "unused"},
    )

    assert result.exit_code == 0, result.output
    assert profiled == [expected]
