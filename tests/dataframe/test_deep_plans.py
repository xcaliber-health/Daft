"""A plan of any depth plans, renders and runs without overflowing a stack.

A frame built from many chained operations nests one plan level per operation,
and an expression chained many times nests as deep. Optimizing, rendering,
serializing, translating and executing them recurse once per level, so each
grows its stack as it needs: a deep plan answers on a thread with a small
stack, and a server answering one keeps serving.

Each case runs in a fresh interpreter: an overflowed stack ends the whole
process, which must fail the case rather than the test run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

# Each depth is past where the shape overflowed a 2 MiB stack before.
SHAPES = {
    "chained offsets": ("offset", 1000),
    "chained filters": ("filter", 1500),
    "chained column rewrites": ("with_column", 400),
    "one chained expression": ("expr", 1000),
    "one chained predicate": ("predicate", 2000),
    "chained concatenations": ("concat", 1000),
}

PROGRAM = textwrap.dedent(
    """
    import io, json, sys, threading
    import daft
    from daft import col

    shape, depth, where = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    def build():
        base = daft.from_pydict({"x": list(range(2000))})
        if shape == "offset":
            frame = base.sort("x")
            for _ in range(depth):
                frame = frame.offset(1).select("x")
            return frame.limit(3)
        if shape == "filter":
            frame = base
            for i in range(depth):
                frame = frame.where(col("x") >= i % 7)
            return frame
        if shape == "with_column":
            frame = base
            for _ in range(depth):
                frame = frame.with_column("x", col("x") + 1)
            return frame.limit(3)
        if shape == "expr":
            e = col("x")
            for _ in range(depth):
                e = e + 1
            return base.select(e.alias("x")).limit(3)
        if shape == "predicate":
            e = col("x") >= 0
            for i in range(depth):
                e = e & (col("x") >= -i)
            return base.where(e).limit(3)
        frame = base.limit(1)
        for _ in range(depth):
            frame = frame.concat(base.limit(1))
        return frame

    answer = {}

    def in_process():
        frame = build()
        frame.explain(show_all=True, file=io.StringIO())
        answer["x"] = frame.to_pydict()["x"]

    if where == "in process":
        # Far smaller than the main thread's stack, as a worker thread's may be.
        threading.stack_size(512 * 1024)
        thread = threading.Thread(target=in_process)
        thread.start()
        thread.join()
    else:
        from daft.runners.remote_runner import RemoteRunner
        from daft.serve import ServeSettings, start_server

        server = start_server(ServeSettings(host="127.0.0.1", port=0))
        runner = RemoteRunner(server.address())
        answer["x"] = [x for part in runner.run_iter_tables(build()._builder) for x in part.to_pydict()["x"]]
        answer["next"] = [x for part in runner.run_iter_tables(daft.from_pydict({"x": [7]})._builder) for x in part.to_pydict()["x"]]
        server.shutdown(drain_timeout_secs=1)
    print(json.dumps(answer))
    """
)


def _expected(shape: str, depth: int) -> list[int]:
    if shape == "offset":
        return [depth, depth + 1, depth + 2]
    if shape == "filter":
        return list(range(6, 2000))
    if shape in ("with_column", "expr"):
        return [depth, depth + 1, depth + 2]
    if shape == "predicate":
        return [0, 1, 2]
    return [0] * (depth + 1)


@pytest.mark.parametrize("where", ["in process", "served"])
@pytest.mark.parametrize("case", sorted(SHAPES))
def test_a_deep_plan_answers(case: str, where: str) -> None:
    shape, depth = SHAPES[case]

    ran = subprocess.run(
        [sys.executable, "-c", PROGRAM, shape, str(depth), where],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert ran.returncode == 0, f"exited {ran.returncode}: {ran.stderr[-2000:]}"
    answer = json.loads(ran.stdout.strip().splitlines()[-1])
    assert answer["x"] == _expected(shape, depth)
    if where == "served":
        assert answer["next"] == [7]
