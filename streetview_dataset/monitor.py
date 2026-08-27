from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
import time
from typing import Literal

from .monitor_page import HTML_PAGE


log = logging.getLogger(__package__ or __name__)
ProgressState = Literal["running", "complete", "stopped"]
ProgressUnit = Literal["panorama", "image"]


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Opt-in settings for the local harvest monitor."""

    port: int = 0
    on_start: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 0 <= self.port <= 65_535:
            raise ValueError("port must be an integer from 0 through 65535")


@dataclass(frozen=True, slots=True)
class _ProgressSnapshot:
    """The only harvest data exposed to the monitor process boundary."""

    state: ProgressState
    current: int
    target: int
    added: int
    queries: int
    unit: ProgressUnit
    last_update: float
    rate_per_second: float

    def as_json(self) -> str:
        """Serialize the explicit scalar status whitelist."""
        return json.dumps(
            {
                "state": self.state,
                "current": self.current,
                "target": self.target,
                "added": self.added,
                "queries": self.queries,
                "unit": self.unit,
                "last_update": self.last_update,
                "rate_per_second": self.rate_per_second,
            },
            separators=(",", ":"),
        )


class _SnapshotHolder:
    """A lock-protected mutable holder required for handler-thread reads."""

    def __init__(self, initial: _ProgressSnapshot) -> None:
        self._snapshot = initial
        self._lock = threading.Lock()

    def publish(self, snapshot: _ProgressSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def read(self) -> _ProgressSnapshot:
        with self._lock:
            return self._snapshot


class _MonitorServer(ThreadingHTTPServer):
    """Loopback-only HTTP server for one monitor instance."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, holder: _SnapshotHolder, port: int) -> None:
        super().__init__(("127.0.0.1", port), _MonitorHandler)
        self.holder = holder
        self.socket.settimeout(1.0)


class _MonitorHandler(BaseHTTPRequestHandler):
    """Serve only the monitor's three read-only routes."""

    protocol_version = "HTTP/1.0"
    server_version = "streetview-dataset"
    sys_version = ""
    server: _MonitorServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(2.0)

    def do_GET(self) -> None:
        route = self.path
        if route == "/":
            self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode("utf-8"))
        elif route == "/api/v1/progress":
            body = self.server.holder.read().as_json().encode("ascii")
            self._send(200, "application/json; charset=utf-8", body)
        elif route == "/healthz":
            self._send(200, "text/plain; charset=utf-8", b"ok\n")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def do_HEAD(self) -> None:
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:
        self._method_not_allowed()

    def do_POST(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def do_CONNECT(self) -> None:
        self._method_not_allowed()

    def do_TRACE(self) -> None:
        self._method_not_allowed()

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        if code == 501:
            self._method_not_allowed()
            return
        super().send_error(code, message, explain)

    def _method_not_allowed(self) -> None:
        self._send(405, "text/plain; charset=utf-8", b"method not allowed\n")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: str) -> None:
        return


class Monitor:
    """Own one monitor server and its scalar progress snapshot."""

    def __init__(self, config: MonitorConfig, initial: _ProgressSnapshot | None = None) -> None:
        if initial is None:
            initial = _ProgressSnapshot("running", 0, 0, 0, 0, "panorama", time.time(), 0.0)
        self._holder = _SnapshotHolder(initial)
        self._config = config
        self._server: _MonitorServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None
        self._lifecycle_lock = threading.Lock()

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("monitor has not started")
        return self._url

    def start(self, initial: _ProgressSnapshot | None = None) -> None:
        """Bind the loopback server and notify the caller of its URL."""
        if initial is not None:
            self._holder.publish(initial)
        with self._lifecycle_lock:
            if self._server is not None:
                raise RuntimeError("monitor is already running")
            server = _MonitorServer(self._holder, self._config.port)
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="streetview-monitor",
                daemon=True,
            )
            self._server = server
            self._thread = thread
            self._url = f"http://127.0.0.1:{server.server_port}/"
            thread.start()
        log.info("Harvest monitor listening at %s", self.url)
        if self._config.on_start is not None:
            self._config.on_start(self.url)

    def publish(self, snapshot: _ProgressSnapshot) -> None:
        """Replace the scalar snapshot read by future HTTP requests."""
        self._holder.publish(snapshot)

    def stop(self) -> None:
        """Stop the server and close its listening socket."""
        with self._lifecycle_lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._url = None
        if server is None:
            return
        if thread is not None:
            server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join()
