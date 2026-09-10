"""Public surface for Iceberg maintenance APIs.

Re-exports the result types, exception types, and helper utilities that
callers of ``rewrite_data_files``, ``rewrite_position_delete_files``,
``rewrite_manifests``, ``expire_snapshots``, and ``remove_orphan_files`` need to handle return
values and recover from errors.
"""

from typing import TypeAlias
from typing import Union

from daft.io.iceberg._common import MaintenanceOptions, CommitRetryExhausted
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
from daft.io.iceberg._rewrite_position_deletes import (
    RewritePositionDeletesFailedException,
    RewritePositionDeletesResult,
)

# Tuning knobs for the maintenance APIs, by option name. Recognized keys and
# their defaults are documented on each maintenance method; unknown keys are
# rejected at validation time.
IcebergMaintenanceOptions: TypeAlias = MaintenanceOptions

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
    "RewritePositionDeletesFailedException",
    "RewritePositionDeletesResult",
    "RewriteResult",
]
