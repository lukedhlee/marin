# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.rpc import job_pb2, worker_pb2
from iris.rpc.worker_codec import (
    reconcile_request_from_proto,
    worker_metadata_from_proto,
)


def test_worker_metadata_with_present_empty_device_decodes_device_less() -> None:
    wire = job_pb2.WorkerMetadata(hostname="cpu-worker")
    wire.device.SetInParent()

    assert worker_metadata_from_proto(wire).device is None


def test_worker_reconcile_run_without_repeated_spec_remains_run_intent() -> None:
    wire = worker_pb2.Worker.ReconcileRequest(
        worker_id="worker-1",
        desired=[worker_pb2.Worker.DesiredAttempt(attempt_uid="abc", run=worker_pb2.Worker.AttemptSpec())],
    )

    request = reconcile_request_from_proto(wire)

    assert request.desired[0].is_run
    assert request.desired[0].launch is None
