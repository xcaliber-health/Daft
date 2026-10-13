"""A fresh server answers the burst of requests that first reaches it.

A server whose process has not yet used the engine sets the engine's context up
on first use. Requests arriving together reach it from several threads at once,
and every one is answered.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

REQUESTS = 8

# Each server runs in its own interpreter, so the engine's context is not yet set up.
SERVE = textwrap.dedent(
    """
    import json, threading
    import daft
    from daft.runners.remote_runner import RemoteRunner
    from daft.serve import ServeSettings, start_server

    class Host:
        kinds = ("q",)

        def plan(self, request):
            return daft.from_pydict({"n": [1]})

    server = start_server(
        ServeSettings(host="127.0.0.1", port=0, max_concurrent_queries=REQUESTS, disable_plan_payload=True),
        request_host=Host(),
    )
    ended = []

    def send(n):
        try:
            client = RemoteRunner(server.address()).client
            list(client.run_request("q", b"", f"q-{n}"))
            ended.append("answered")
        except Exception as failed:
            ended.append(str(getattr(failed, "error_class", type(failed).__name__)))

    senders = [threading.Thread(target=send, args=(n,)) for n in range(REQUESTS)]
    for sender in senders:
        sender.start()
    for sender in senders:
        sender.join()
    server.shutdown(drain_timeout_secs=5)
    print(json.dumps(ended))
    """
).replace("REQUESTS", str(REQUESTS))


@pytest.mark.parametrize("server", range(6))
def test_a_fresh_server_answers_every_request_of_its_first_burst(server: int) -> None:
    ran = subprocess.run([sys.executable, "-c", SERVE], capture_output=True, text=True, timeout=300, check=False)

    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout.strip().splitlines()[-1]) == ["answered"] * REQUESTS, ran.stderr
    assert "panicked" not in ran.stderr
