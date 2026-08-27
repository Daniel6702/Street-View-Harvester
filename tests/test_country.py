import json
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest

from streetview_dataset import Panorama, StreetViewDataset
from streetview_dataset.geo import CountryIndex


def test_bundled_denmark_sampling():
    index = CountryIndex()
    denmark = index.get("Denmark")
    pts = index.sample(np.random.default_rng(1), denmark, 30)
    assert len(pts) == 30
    assert all(denmark.contains(lat, lon) for lat, lon in pts)


def test_country_returns_after_download_worker_finishes(tmp_path, monkeypatch):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=42)
    download_started = Event()
    allow_download_to_finish = Event()
    result_box = []

    monkeypatch.setattr(
        dataset.client,
        "find_nearest",
        lambda _lat, _lon: Panorama("pano", 55.6761, 12.5683),
    )

    def render_image(*_args, **_kwargs):
        download_started.set()
        assert allow_download_to_finish.wait(timeout=1)
        return b"image"

    monkeypatch.setattr(dataset, "_render_image_bytes", render_image)

    caller = Thread(
        target=lambda: result_box.append(
            dataset.country("Denmark", 1, download="flat", yaw=0)
        )
    )
    caller.start()
    assert download_started.wait(timeout=1)
    assert caller.is_alive()

    allow_download_to_finish.set()
    caller.join(timeout=1)

    assert not caller.is_alive()
    assert result_box[0].complete
    assert result_box[0].added_count == 1
    assert len(list((tmp_path / "images").rglob("*.jpg"))) == 1


def test_country_raises_download_worker_error(tmp_path, monkeypatch):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=42)
    monkeypatch.setattr(
        dataset.client,
        "find_nearest",
        lambda _lat, _lon: Panorama("pano", 55.6761, 12.5683),
    )

    def render_image(*_args, **_kwargs):
        raise OSError("image endpoint rejected request")

    monkeypatch.setattr(dataset, "_render_image_bytes", render_image)

    with pytest.raises(OSError, match="image endpoint rejected request"):
        dataset.country("Denmark", 1, download="flat", yaw=0)


def test_country_redownloads_missing_flat_image(tmp_path, monkeypatch):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=42)
    dataset._append_rows(
        [
            {
                "panoid": "missing",
                "image_type": "flat",
                "image_path": "images/missing.jpg",
            }
        ]
    )
    monkeypatch.setattr(
        dataset.client,
        "find_nearest",
        lambda _lat, _lon: Panorama("replacement", 55.6761, 12.5683),
    )
    monkeypatch.setattr(dataset, "_render_image_bytes", lambda *_args, **_kwargs: b"image")

    result = dataset.country("Denmark", 1, download="flat", yaw=0)

    assert result.existing_count == 0
    assert result.added_count == 1
    assert list((tmp_path / "images").glob("*/replacement.jpg"))


def test_country_writes_unsharded_flat_images_when_disabled(tmp_path, monkeypatch):
    dataset = StreetViewDataset(tmp_path, workers=1, seed=42, file_sharding=False)
    monkeypatch.setattr(
        dataset.client,
        "find_nearest",
        lambda _lat, _lon: Panorama("pano", 55.6761, 12.5683),
    )
    monkeypatch.setattr(dataset, "_render_image_bytes", lambda *_args, **_kwargs: b"image")

    result = dataset.country("Denmark", 1, download="flat", yaw=0)

    assert result.added_count == 1
    assert (tmp_path / "images" / "pano.jpg").read_bytes() == b"image"


def test_geojson_accepts_resolved_panorama_on_boundary(tmp_path: Path, monkeypatch):
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        ),
        encoding="utf-8",
    )
    dataset = StreetViewDataset(workers=1, seed=42)
    monkeypatch.setattr(
        dataset.client,
        "find_nearest",
        lambda _lat, _lon: Panorama("boundary", 0.0, 0.0),
    )

    result = dataset.geojson(path, 1, max_queries=1)

    assert result.complete
    assert result.added_count == 1
    assert result.rows is not None
    assert result.rows[0]["source"] == "geojson"


def test_geojson_rejects_resolved_panorama_outside(tmp_path: Path, monkeypatch):
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        ),
        encoding="utf-8",
    )
    dataset = StreetViewDataset(workers=1, seed=42)
    monkeypatch.setattr(
        dataset.client,
        "find_nearest",
        lambda _lat, _lon: Panorama("outside", 2.0, 2.0),
    )

    result = dataset.geojson(path, 1, max_queries=1)

    assert not result.complete
    assert result.added_count == 0
    assert result.queries == 1
