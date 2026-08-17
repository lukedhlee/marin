# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Which backend an fsspec filesystem speaks.

Backend dispatch tests the protocol. ``filesystem_for`` returns
:class:`~rigging.filesystem.cross_region.CrossRegionGuardedFS` for a GCS URL, and that
guard proxies the real filesystem without subclassing it, so an ``isinstance`` check
misses and silently drops the caller onto a generic path.
"""

from typing import Any


def _protocols(fs: Any) -> tuple[str, ...]:
    """The protocols *fs* declares, as a tuple whether it declares one or several."""
    protocol = getattr(fs, "protocol", ())
    return (protocol,) if isinstance(protocol, str) else tuple(protocol)


def is_s3_filesystem(fs: Any) -> bool:
    """Whether *fs* speaks S3, through any wrapper that forwards its protocol."""
    return "s3" in _protocols(fs)


def is_gcs_filesystem(fs: Any) -> bool:
    """Whether *fs* speaks GCS, through any wrapper that forwards its protocol."""
    declared = _protocols(fs)
    return "gcs" in declared or "gs" in declared
