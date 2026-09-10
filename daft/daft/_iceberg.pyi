from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypedDict

if TYPE_CHECKING:
    import pyarrow as pa

#: A planning option value as accepted from the caller.
OptionValue: TypeAlias = str | int | float | bool

class CandidateRecord(TypedDict):
    """A data file offered to the planner, with the row-level deletes that apply to it."""

    path: str
    size_bytes: int
    partition_key: str
    partition_spec_id: int
    positional_delete_paths: list[str]
    equality_delete_paths: list[str]
    record_count: int
    deleted_record_count: int

class FileGroupRecord(TypedDict):
    """A set of candidate files the planner rewrites together into one partition's outputs."""

    partition_key: str
    output_spec_id: int
    total_bytes: int
    expected_output_files: int
    input_split_size: int
    files: list[CandidateRecord]

def plan_file_groups_py(
    candidates: list[CandidateRecord],
    options: dict[str, OptionValue],
    current_spec_id: int,
) -> list[FileGroupRecord]: ...
def validate_options_py(options: dict[str, OptionValue]) -> dict[str, OptionValue]: ...
def build_zorder_key_py(
    arrays: list[pa.Array],
    var_length_contribution: int,
    max_output_size: int,
) -> pa.Array: ...
