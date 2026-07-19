from __future__ import annotations

import pytest

from daft.runners.remote_runner import normalize_address


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("daft://host:9494", "grpc://host:9494"),
        ("grpc://host:1234", "grpc://host:1234"),
        ("http://host:1234", "grpc://host:1234"),
        ("host:1234", "grpc://host:1234"),
        ("host", "grpc://host:9494"),
        ("daft://host", "grpc://host:9494"),
        ("[::1]:1234", "grpc://[::1]:1234"),
        ("daft://[::1]", "grpc://[::1]:9494"),
    ],
)
def test_normalize_address_accepts_supported_forms(address: str, expected: str) -> None:
    assert normalize_address(address) == expected


@pytest.mark.parametrize(
    "address",
    ["", "ftp://host:1", "daft://", "unix://socket"],
)
def test_normalize_address_rejects_invalid_forms(address: str) -> None:
    with pytest.raises(ValueError):
        normalize_address(address)
