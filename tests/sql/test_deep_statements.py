"""A SQL statement of any length answers or is refused, and never overflows a stack.

A chain of infix operators (`a AND b AND …`, `x + 1 + 1 …`), a `CASE` with many
branches and a chain of set operations each nest one level per term. A statement
within the nesting limits answers; one past them is refused with an error that
names the limit, however far past it is.

Each case runs in a fresh interpreter: an overflowed stack ends the whole
process, which must fail the case rather than the test run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

# Expressions may nest 1,000 levels and queries 2,000. Each answered length is
# past where its shape overflowed a stack before.
ANSWERED = {
    "chained AND": ("and", 900),
    "chained OR": ("or", 900),
    "chained +": ("plus", 900),
    "CASE branches": ("case", 600),
    "chained UNION ALL": ("union", 1900),
}

# One term past each shape's limit.
REFUSED = {
    "chained AND": ("and", 1_001),
    "chained OR": ("or", 1_001),
    "chained +": ("plus", 1_001),
    "CASE branches": ("case", 1_001),
    "chained UNION ALL": ("union", 2_001),
}

PROGRAM = textwrap.dedent(
    """
    import json, sys, threading
    import daft

    shape, n, where = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    def statement():
        if shape == "and":
            return "SELECT 1 AS x WHERE " + " AND ".join(f"{i} >= 0" for i in range(n))
        if shape == "or":
            return "SELECT 1 AS x WHERE " + " OR ".join(f"{i} < 0" for i in range(n)) + " OR TRUE"
        if shape == "plus":
            return "SELECT 0" + " + 1" * n + " AS x"
        if shape == "case":
            return "SELECT CASE " + " ".join(f"WHEN {i} < 0 THEN {i}" for i in range(n)) + " ELSE 7 END AS x"
        return " UNION ALL ".join("SELECT 1 AS x" for _ in range(n))

    answer = {}

    def in_process():
        try:
            answer["x"] = daft.sql(statement()).to_pydict()["x"]
        except Exception as failure:
            answer["refused"] = f"{type(failure).__name__}: {failure}"

    if where == "in process":
        # Far smaller than the main thread's stack, as a worker thread's may be.
        threading.stack_size(512 * 1024)
        thread = threading.Thread(target=in_process)
        thread.start()
        thread.join()
    else:
        from daft.daft import DaftServeClient
        from daft.recordbatch import MicroPartition
        from daft.serve import ServeSettings, start_server

        def served(sql, query_id):
            parts = DaftServeClient(server.address()).run_sql(sql, query_id)
            return [x for part in parts for x in MicroPartition._from_pymicropartition(part).to_pydict()["x"]]

        server = start_server(ServeSettings(host="127.0.0.1", port=0))
        try:
            answer["x"] = served(statement(), "deep")
        except Exception as failure:
            answer["refused"] = f"{type(failure).__name__}: {failure}"
        answer["next"] = served("SELECT 7 AS x", "next")
        server.shutdown(drain_timeout_secs=1)
    print(json.dumps(answer))
    """
)


def _run(shape: str, n: int, where: str) -> dict[str, list[int] | str]:
    ran = subprocess.run(
        [sys.executable, "-c", PROGRAM, shape, str(n), where],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert ran.returncode == 0, f"exited {ran.returncode}: {ran.stderr[-2000:]}"
    return json.loads(ran.stdout.strip().splitlines()[-1])


def _expected(shape: str, n: int) -> list[int]:
    return {"plus": [n], "case": [7], "union": [1] * n}.get(shape, [1])


@pytest.mark.parametrize("where", ["in process", "served"])
@pytest.mark.parametrize("case", sorted(ANSWERED))
def test_a_statement_within_the_limits_answers(case: str, where: str) -> None:
    shape, n = ANSWERED[case]

    answer = _run(shape, n, where)

    assert answer.get("x") == _expected(shape, n), answer.get("refused")
    if where == "served":
        assert answer["next"] == [7]


@pytest.mark.parametrize("past", ["just past", "far past"])
@pytest.mark.parametrize("where", ["in process", "served"])
@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_statement_past_the_limits_is_refused(case: str, where: str, past: str) -> None:
    shape, n = REFUSED[case]

    answer = _run(shape, n if past == "just past" else 200_000, where)

    assert "x" not in answer
    assert "may nest at most" in answer["refused"]
    if where == "served":
        assert answer["next"] == [7]
