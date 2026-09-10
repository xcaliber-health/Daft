"""User-defined functions executed through the serving endpoint.

Covers the UDF variants a client can ship: scalar functions, stateful
classes, asynchronous functions, pooled execution with a concurrency hint,
process isolation, and accelerator resource requests.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import daft
from daft import col

from .conftest import assert_remote_matches_native

if TYPE_CHECKING:
    from daft.runners.native_runner import NativeRunner
    from daft.runners.remote_runner import RemoteRunner


def test_scalar_function_with_captured_state(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    factor = 7

    @daft.func(return_dtype=daft.DataType.int64())
    def scale(v: int) -> int:
        return v * factor

    df = daft.from_pydict({"v": [1, 2, 3]}).select(scale(col("v")).alias("scaled"))
    assert_remote_matches_native(remote_runner, native_runner, df)


def test_class_udf(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    @daft.cls
    class Scaler:
        def __init__(self) -> None:
            self.factor = 10

        def __call__(self, x: int) -> int:
            return x * self.factor

    df = daft.from_pydict({"x": [1, 2, 3]}).select(Scaler()(col("x")).alias("scaled"))
    assert_remote_matches_native(remote_runner, native_runner, df)


def test_async_function_udf(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    @daft.func(return_dtype=daft.DataType.int64())
    async def slow_add(a: int) -> int:
        await asyncio.sleep(0.001)
        return a + 100

    df = daft.from_pydict({"a": [1, 2, 3]}).select(slow_add(col("a")).alias("plus"))
    assert_remote_matches_native(remote_runner, native_runner, df)


def test_pooled_udf_with_concurrency(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    @daft.udf(return_dtype=daft.DataType.int64(), concurrency=2)
    class Doubler:
        def __init__(self) -> None:
            self.factor = 2

        def __call__(self, data: daft.Series) -> list[int]:
            return [v * self.factor for v in data.to_pylist()]

    df = daft.from_pydict({"v": list(range(16))}).select(Doubler(col("v")).alias("doubled"))
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="doubled")


def test_process_isolated_udf(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    @daft.func(return_dtype=daft.DataType.int64(), use_process=True)
    def negate(v: int) -> int:
        return -v

    df = daft.from_pydict({"v": [1, 2, 3]}).select(negate(col("v")).alias("neg"))
    assert_remote_matches_native(remote_runner, native_runner, df, sort_key="neg")


def test_gpu_resource_request_behaves_like_native(remote_runner: RemoteRunner, native_runner: NativeRunner) -> None:
    """A GPU request on a machine without one must fail identically, not hang.

    Whichever way the engine treats an unsatisfiable accelerator request,
    the remote path must mirror the local path: same success or same error
    category, surfaced promptly to the client.
    """

    @daft.udf(return_dtype=daft.DataType.int64(), num_gpus=1)
    def on_gpu(data: daft.Series) -> list[int]:
        return [v + 1 for v in data.to_pylist()]

    df = daft.from_pydict({"v": [1, 2, 3]}).select(on_gpu(col("v")).alias("out"))

    native_error: type[Exception] | None = None
    try:
        list(native_runner.run_iter_tables(df._builder))
    except Exception as e:
        native_error = type(e)

    if native_error is None:
        assert_remote_matches_native(remote_runner, native_runner, df)
    else:
        remote_error: type[Exception] | None = None
        try:
            list(remote_runner.run_iter_tables(df._builder))
        except Exception as e:
            remote_error = type(e)
        assert remote_error is not None, "native rejected the GPU request but remote accepted it"
