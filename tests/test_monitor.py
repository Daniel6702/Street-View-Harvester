import json
import logging
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

import streetview_dataset.dataset as dataset_module
from streetview_dataset import HarvestResult, MonitorConfig, Panorama, StreetViewDataset
from streetview_dataset.monitor import Monitor, _ProgressSnapshot


def _snapshot(
    *,
    state: str = "running",
    current: int = 2,
    target: int = 5,
    added: int = 0,
    queries: int = 3,
    unit: str = "panorama",
    rate_per_second: float = 0.0,
) -> _ProgressSnapshot:
    return _ProgressSnapshot(
        state=state,
        current=current,
        target=target,
        added=added,
        queries=queries,
        unit=unit,
        last_update=1_700_000_000.0,
        rate_per_second=rate_per_second,
    )


def _request(url: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    request = Request(url, method=method)
    try:
        with urlopen(request, timeout=2.0) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


def test_monitor_rejects_invalid_ports() -> None:
    for port in (-1, 65_536, True):
        with pytest.raises(ValueError):
            MonitorConfig(port=port)


def test_monitor_routes_headers_and_scalar_payload() -> None:
    monitor = Monitor(MonitorConfig(port=0))
    monitor.start(_snapshot())
    try:
        assert monitor.url.startswith("http://127.0.0.1:")
        assert monitor.url.endswith("/")

        status, headers, body = _request(monitor.url)
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert int(headers["Content-Length"]) == len(body)
        assert b"Harvest monitor" in body
        assert b"Pace" in body

        status, headers, body = _request(f"{monitor.url}api/v1/progress")
        assert status == 200
        payload = json.loads(body)
        assert set(payload) == {
            "state",
            "current",
            "target",
            "added",
            "queries",
            "unit",
            "last_update",
            "rate_per_second",
        }
        assert payload["state"] == "running"
        assert payload["current"] == 2
        assert payload["target"] == 5
        assert payload["added"] == 0
        assert payload["queries"] == 3
        assert payload["unit"] == "panorama"
        assert payload["rate_per_second"] == 0.0
        assert all(isinstance(value, (str, int, float)) for value in payload.values())
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"

        status, headers, body = _request(f"{monitor.url}healthz")
        assert status == 200
        assert body == b"ok\n"
        assert int(headers["Content-Length"]) == len(body)

        status, _, _ = _request(f"{monitor.url}missing")
        assert status == 404
        status, _, _ = _request(f"{monitor.url}api/v1/progress?extra=1")
        assert status == 404
        status, _, _ = _request(monitor.url, method="POST")
        assert status == 405
    finally:
        monitor.stop()


def test_monitor_updates_only_whitelisted_scalar_status() -> None:
    monitor = Monitor(MonitorConfig(), _snapshot())
    monitor.start()
    try:
        monitor.publish(
            _ProgressSnapshot(
                state="complete",
                current=5,
                target=5,
                added=3,
                queries=8,
                unit="image",
                last_update=1_700_000_001.0,
                rate_per_second=2.5,
            )
        )
        _, _, body = _request(f"{monitor.url}api/v1/progress")
        payload = json.loads(body)
        assert payload["state"] == "complete"
        assert payload["current"] == 5
        assert payload["unit"] == "image"
        assert payload["rate_per_second"] == 2.5
        assert "source" not in payload
        assert "source_value" not in payload
        assert "secret" not in body.decode()
    finally:
        monitor.stop()


def test_monitor_callback_and_package_logger_receive_loopback_url(caplog) -> None:
    urls: list[str] = []
    monitor = Monitor(MonitorConfig(on_start=urls.append), _snapshot())
    with caplog.at_level(logging.INFO, logger="streetview_dataset"):
        monitor.start()
    try:
        assert urls == [monitor.url]
        assert monitor.url in caplog.text
    finally:
        monitor.stop()


def test_monitor_stop_closes_port() -> None:
    monitor = Monitor(MonitorConfig(), _snapshot())
    monitor.start()
    url = monitor.url
    monitor.stop()

    with pytest.raises((URLError, ConnectionError, TimeoutError)):
        _request(url)


class _FakeMonitor:
    instances: list["_FakeMonitor"] = []

    def __init__(self, _config: MonitorConfig, initial: _ProgressSnapshot | None = None) -> None:
        self.publish_thread_ids: list[int] = []
        self.published_snapshots: list[_ProgressSnapshot] = []
        self.start_thread_ids: list[int] = []
        self.stop_thread_ids: list[int] = []
        self.initial = initial
        self.instances.append(self)

    def start(self, initial: _ProgressSnapshot | None = None) -> None:
        self.initial = initial
        self.start_thread_ids.append(threading.get_ident())

    def publish(self, snapshot: _ProgressSnapshot) -> None:
        self.publish_thread_ids.append(threading.get_ident())
        self.published_snapshots.append(snapshot)

    def stop(self) -> None:
        self.stop_thread_ids.append(threading.get_ident())


def test_dataset_publishes_monitor_progress_from_coordinator(monkeypatch, tmp_path) -> None:
    worker_ids: set[int] = set()

    def find_nearest(_self, lat: float, lon: float) -> Panorama:
        worker_ids.add(threading.get_ident())
        return Panorama(f"{lat:.12f}:{lon:.12f}", lat, lon)

    _FakeMonitor.instances.clear()
    monkeypatch.setattr(dataset_module, "Monitor", _FakeMonitor)
    monkeypatch.setattr("streetview_dataset.client.StreetViewClient.find_nearest", find_nearest)

    coordinator_id = threading.get_ident()
    dataset = StreetViewDataset(tmp_path, workers=2, seed=1)
    dataset.radius(0.0, 0.0, 10.0, 2, monitor=MonitorConfig(), max_queries=2)

    monitor = _FakeMonitor.instances[0]
    assert monitor.start_thread_ids == [coordinator_id]
    assert monitor.initial is not None
    assert monitor.initial.rate_per_second == 0.0
    assert any(snapshot.rate_per_second > 0.0 for snapshot in monitor.published_snapshots)
    assert all(snapshot.rate_per_second >= 0.0 for snapshot in monitor.published_snapshots)
    assert set(monitor.publish_thread_ids) == {coordinator_id}
    assert monitor.stop_thread_ids == [coordinator_id]
    assert coordinator_id not in worker_ids


def test_dataset_stops_monitor_after_harvest_cleanup(monkeypatch, tmp_path) -> None:
    urls: list[str] = []

    def find_nearest(_self, lat: float, lon: float) -> Panorama:
        return Panorama(f"{lat:.12f}:{lon:.12f}", lat, lon)

    monkeypatch.setattr("streetview_dataset.client.StreetViewClient.find_nearest", find_nearest)
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)
    result = dataset.radius(
        0.0,
        0.0,
        10.0,
        1,
        monitor=MonitorConfig(on_start=urls.append),
        max_queries=1,
    )

    assert result.complete
    assert len(urls) == 1
    with pytest.raises((URLError, ConnectionError, TimeoutError)):
        _request(urls[0])


def test_dataset_rate_counts_only_new_additions(monkeypatch, tmp_path) -> None:
    clock = iter((100.0, 102.0, 104.0))

    def find_nearest(_self, lat: float, lon: float) -> Panorama:
        return Panorama(f"{lat:.12f}:{lon:.12f}", lat, lon)

    _FakeMonitor.instances.clear()
    monkeypatch.setattr(dataset_module, "Monitor", _FakeMonitor)
    monkeypatch.setattr(dataset_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr("streetview_dataset.client.StreetViewClient.find_nearest", find_nearest)

    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)
    dataset._append_rows([{"panoid": "resumed"}])
    dataset.radius(0.0, 0.0, 10.0, 2, monitor=MonitorConfig(), max_queries=1)

    monitor = _FakeMonitor.instances[0]
    final_snapshot = monitor.published_snapshots[-1]
    assert monitor.initial is not None
    assert monitor.initial.current == 1
    assert monitor.initial.rate_per_second == 0.0
    assert final_snapshot.current == 2
    assert final_snapshot.added == 1
    assert final_snapshot.rate_per_second == 0.25


def test_live_http_monitor_exposes_progress_before_harvest_finishes(monkeypatch, tmp_path) -> None:
    url_holder: list[str] = []
    second_render_started = threading.Event()
    release_second_render = threading.Event()
    progress_published = threading.Event()
    finished = threading.Event()
    render_calls = [0]

    def find_nearest(_self, lat: float, lon: float) -> Panorama:
        return Panorama(f"{lat:.12f}:{lon:.12f}", lat, lon)

    def render(_self, *_args, **_kwargs) -> bytes:
        render_calls[0] += 1
        if render_calls[0] == 1:
            return b"image"
        second_render_started.set()
        release_second_render.wait(timeout=2.0)
        return b"image"

    real_publish = Monitor.publish

    def observe_publish(self, snapshot: _ProgressSnapshot) -> None:
        real_publish(self, snapshot)
        if snapshot.added >= 1:
            progress_published.set()

    monkeypatch.setattr("streetview_dataset.client.StreetViewClient.find_nearest", find_nearest)
    monkeypatch.setattr(StreetViewDataset, "_render_image_bytes", render)
    monkeypatch.setattr(Monitor, "publish", observe_publish)
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)
    result: list[HarvestResult] = []

    def harvest() -> None:
        try:
            result.append(
                dataset.radius(
                    0.0,
                    0.0,
                    10.0,
                    2,
                    download="flat",
                    yaw=0.0,
                    monitor=MonitorConfig(on_start=url_holder.append),
                    max_queries=2,
                )
            )
        finally:
            finished.set()

    thread = threading.Thread(target=harvest)
    thread.start()
    assert second_render_started.wait(timeout=2.0)
    assert progress_published.wait(timeout=2.0)
    assert not finished.is_set()
    assert len(url_holder) == 1

    status, _, body = _request(f"{url_holder[0]}api/v1/progress")
    payload = json.loads(body)
    assert status == 200
    assert payload["state"] == "running"
    assert payload["current"] == 1
    assert payload["added"] == 1
    assert payload["rate_per_second"] >= 0.0

    release_second_render.set()
    assert finished.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert len(result) == 1


def test_real_monitor_closes_url_after_image_worker_failure(monkeypatch, tmp_path) -> None:
    url_holder: list[str] = []

    def find_nearest(_self, lat: float, lon: float) -> Panorama:
        return Panorama(f"{lat:.12f}:{lon:.12f}", lat, lon)

    def fail_render(_self, *_args, **_kwargs) -> bytes:
        raise OSError("image endpoint rejected request")

    monkeypatch.setattr("streetview_dataset.client.StreetViewClient.find_nearest", find_nearest)
    monkeypatch.setattr(StreetViewDataset, "_render_image_bytes", fail_render)
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)

    with pytest.raises(OSError, match="image endpoint rejected request"):
        dataset.radius(
            0.0,
            0.0,
            10.0,
            1,
            download="flat",
            yaw=0.0,
            monitor=MonitorConfig(on_start=url_holder.append),
            max_queries=1,
        )

    assert len(url_holder) == 1
    with pytest.raises((URLError, ConnectionError, TimeoutError)):
        _request(url_holder[0])
