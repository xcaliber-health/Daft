"""A proxy before the local object store that notes each request the engine sends it."""

from __future__ import annotations

import http.client
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import daft

STORE = ("127.0.0.1", 9000)


@dataclass(frozen=True)
class Request:
    """One request the engine sent to the store, as the store answered it."""

    method: str
    path: str
    content_range: str

    @property
    def parquet_footer(self) -> bool:
        """Whether it read a Parquet file's end, where its footer is."""
        found = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", self.content_range)
        return (
            self.method == "GET"
            and self.path.endswith(".parquet")
            and found is not None
            and int(found[2]) == int(found[3]) - 1
        )

    @property
    def parquet_size(self) -> bool:
        """Whether it asked a Parquet file's size."""
        return self.method == "HEAD" and self.path.endswith(".parquet")

    @property
    def manifest_list(self) -> bool:
        """Whether it read a table snapshot's manifest list."""
        return self.method == "GET" and self.path.rsplit("/", 1)[-1].startswith("snap-")

    @property
    def manifest(self) -> bool:
        """Whether it read one of a table's manifests."""
        name = self.path.rsplit("/", 1)[-1]
        return self.method == "GET" and name.endswith(".avro") and not name.startswith("snap-")

    @property
    def metadata_size(self) -> bool:
        """Whether it asked a table metadata file's size."""
        return self.method == "HEAD" and self.path.endswith(".avro")


class _Forward(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    noted: ClassVar[list[Request]] = []

    def _forward(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        upstream = http.client.HTTPConnection(*STORE)
        upstream.request(self.command, self.path, body, dict(self.headers))
        answer = upstream.getresponse()
        content = answer.read()
        self.noted.append(Request(self.command, self.path.split("?")[0], answer.getheader("Content-Range") or ""))
        self.send_response(answer.status)
        for name, value in answer.getheaders():
            if name.lower() not in {"transfer-encoding", "connection"}:
                self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    do_GET = _forward
    do_HEAD = _forward
    do_PUT = _forward
    do_POST = _forward
    do_DELETE = _forward

    def log_message(self, format: str, *args: object) -> None:
        """Log nothing; the requests are noted instead."""


def io_config(port: int = STORE[1], **overrides: object) -> daft.io.IOConfig:
    """The engine's settings for the local store, reached on ``port``."""
    return daft.io.IOConfig(
        s3=daft.io.S3Config(
            endpoint_url=f"http://127.0.0.1:{port}",
            key_id="minioadmin",
            access_key="minioadmin",
            use_ssl=False,
            **overrides,
        )
    )


@contextmanager
def counting_proxy() -> Iterator[tuple[int, list[Request]]]:
    """Yield the port of a proxy before the local store, and the requests it notes."""
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), _Forward)
    proxy.daemon_threads = True
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    _Forward.noted = []
    try:
        yield proxy.server_address[1], _Forward.noted
    finally:
        proxy.shutdown()
