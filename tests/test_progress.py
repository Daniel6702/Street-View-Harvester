import inspect
import threading

import pytest

import streetview_dataset.dataset as dataset_module
from streetview_dataset import MonitorConfig, Panorama, StreetViewDataset


class FakeProgress:
    instances: list["FakeProgress"] = []

    def __init__(
        self,
        *,
        total: int,
        initial: int,
        unit: str,
        disable: bool | None,
    ) -> None:
        self.total = total
        self.initial = initial
        self.unit = unit
        self.disable = disable
        self.update_calls: list[int] = []
        self.update_thread_ids: list[int] = []
        self.close_calls = 0
        self.instances.append(self)

    def update(self, count: int = 1) -> None:
        self.update_calls.append(count)
        self.update_thread_ids.append(threading.get_ident())

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_tqdm(monkeypatch):
    FakeProgress.instances.clear()
    monkeypatch.setattr(dataset_module, "tqdm", FakeProgress, raising=False)
    return FakeProgress


@pytest.fixture
def unique_lookup(monkeypatch):
    worker_ids: set[int] = set()

    def find_nearest(_self, lat: float, lon: float) -> Panorama:
        worker_ids.add(threading.get_ident())
        return Panorama(f"{lat:.12f}:{lon:.12f}", lat, lon)

    monkeypatch.setattr("streetview_dataset.client.StreetViewClient.find_nearest", find_nearest)
    return worker_ids


@pytest.mark.parametrize("method_name", ["country", "radius", "bbox", "geojson"])
def test_aggregate_methods_accept_progress(method_name):
    method = getattr(StreetViewDataset, method_name)

    assert inspect.signature(method).parameters["progress"].default is None
    assert inspect.signature(method).parameters["monitor"].default is None


def test_progress_uses_target_existing_count_and_closes(fake_tqdm, unique_lookup, tmp_path):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)
    dataset._append_rows([{"panoid": "already-seen"}])

    result = dataset.radius(0.0, 0.0, 10.0, 3, progress=True, max_queries=2)

    progress = fake_tqdm.instances[0]
    assert result.added_count == 2
    assert (progress.total, progress.initial, progress.unit) == (3, 1, "panorama")
    assert progress.disable is False
    assert progress.update_calls == [2]
    assert progress.close_calls == 1


def test_resume_complete_creates_no_progress_bar(fake_tqdm, unique_lookup, tmp_path):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)
    dataset._append_rows([{"panoid": "already-seen"}])

    result = dataset.radius(0.0, 0.0, 10.0, 1, progress=True)

    assert result.complete
    assert fake_tqdm.instances == []


def test_resume_complete_creates_no_monitor(fake_tqdm, unique_lookup, monkeypatch, tmp_path):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)
    dataset._append_rows([{"panoid": "already-seen"}])

    def fail_if_started(*_args, **_kwargs):
        pytest.fail("a no-op resume must not start a monitor")

    monkeypatch.setattr(dataset_module, "Monitor", fail_if_started)
    result = dataset.radius(0.0, 0.0, 10.0, 1, monitor=MonitorConfig())

    assert result.complete
    assert fake_tqdm.instances == []


def test_progress_false_disables_bar(fake_tqdm, unique_lookup, tmp_path):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)

    dataset.radius(0.0, 0.0, 10.0, 1, progress=False, max_queries=1)

    assert fake_tqdm.instances[0].disable is True


def test_metadata_append_failure_does_not_advance_progress(
    fake_tqdm,
    unique_lookup,
    monkeypatch,
    tmp_path,
):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1, checkpoint=1)

    def fail_append(_rows):
        raise OSError("metadata append failed")

    monkeypatch.setattr(dataset, "_append_rows", fail_append)

    with pytest.raises(OSError, match="metadata append failed"):
        dataset.radius(0.0, 0.0, 10.0, 1, progress=True, max_queries=1)

    progress = fake_tqdm.instances[0]
    assert progress.update_calls == []
    assert progress.close_calls == 1


def test_download_worker_failure_preserves_error_and_closes_progress(
    fake_tqdm,
    unique_lookup,
    monkeypatch,
    tmp_path,
):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)

    def fail_render(*_args, **_kwargs):
        raise OSError("image endpoint rejected request")

    monkeypatch.setattr(dataset, "_render_image_bytes", fail_render)

    with pytest.raises(OSError, match="image endpoint rejected request"):
        dataset.radius(
            0.0,
            0.0,
            10.0,
            1,
            download="flat",
            yaw=0.0,
            progress=True,
            max_queries=1,
        )

    progress = fake_tqdm.instances[0]
    assert progress.update_calls == []
    assert progress.close_calls == 1


def test_progress_updates_only_in_coordinator_thread(fake_tqdm, unique_lookup, tmp_path):
    dataset = StreetViewDataset(tmp_path, workers=3, seed=1)
    coordinator_id = threading.get_ident()

    dataset.radius(0.0, 0.0, 10.0, 3, progress=True, max_queries=3)

    progress = fake_tqdm.instances[0]
    assert progress.update_calls == [3]
    assert set(progress.update_thread_ids) == {coordinator_id}
    assert coordinator_id not in unique_lookup


@pytest.mark.parametrize(
    ("download", "storage", "unit"),
    [
        ("flat", "files", "image"),
        ("half", "files", "image"),
        ("panorama", "files", "image"),
        ("flat", "zip", "image"),
    ],
)
def test_progress_updates_after_image_storage(
    fake_tqdm,
    unique_lookup,
    monkeypatch,
    tmp_path,
    download,
    storage,
    unit,
):
    dataset = StreetViewDataset(
        tmp_path,
        workers=2,
        seed=1,
        storage=storage,
        shard_size=100,
    )
    monkeypatch.setattr(dataset, "_render_image_bytes", lambda *_args, **_kwargs: b"image")

    result = dataset.radius(
        0.0,
        0.0,
        10.0,
        3,
        download=download,
        yaw=0.0,
        progress=True,
        max_queries=3,
    )

    progress = fake_tqdm.instances[0]
    assert result.added_count == 3
    assert progress.unit == unit
    assert progress.update_calls == [1, 1, 1]
    assert progress.close_calls == 1


def test_progress_none_does_not_render_in_non_tty(unique_lookup, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    dataset = StreetViewDataset(tmp_path, workers=1, seed=1)

    dataset.radius(0.0, 0.0, 10.0, 1, progress=None, max_queries=1)

    assert capsys.readouterr().err == ""
