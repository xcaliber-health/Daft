"""Public surface for Iceberg maintenance APIs.

Re-exports the result types, exception types, and helper utilities that
callers of ``rewrite_data_files``, ``rewrite_manifests``,
``expire_snapshots``, and ``remove_orphan_files`` need to handle return
values and recover from errors.
"""

from collections.abc import Mapping
from typing import Union

from daft.io.iceberg._common import CommitRetryExhausted
from daft.io.iceberg._compact import (
    RewriteConflict,
    RewriteFailedException,
    RewriteResult,
)
from daft.io.iceberg._expire import (
    ExpireResult,
    ExpireSnapshotsFailedException,
)
from daft.io.iceberg._remove_orphan import (
    PrefixMismatchError,
    RemoveOrphanResult,
)
from daft.io.iceberg._rewrite_manifests import (
    RewriteManifestsFailedException,
    RewriteManifestsResult,
)

IcebergMaintenanceOptions = Mapping[str, Union[str, int, float, bool]]
"""Tuning knobs for the maintenance APIs.

A mapping of option name to a scalar value. Recognized keys and their
defaults are documented on each maintenance method; unknown keys are
rejected at validation time.
"""

__all__ = [
    "CommitRetryExhausted",
    "ExpireResult",
    "ExpireSnapshotsFailedException",
    "IcebergMaintenanceOptions",
    "PrefixMismatchError",
    "RemoveOrphanResult",
    "RewriteConflict",
    "RewriteFailedException",
    "RewriteManifestsFailedException",
    "RewriteManifestsResult",
    "RewriteResult",
]
