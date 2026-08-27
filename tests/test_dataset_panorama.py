from io import BytesIO

import numpy as np
from PIL import Image

from streetview_dataset import StreetViewDataset


def _jpeg_image() -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((24, 24, 3), 200, dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, float | int | str]] = []

    def image_bytes(
        self,
        panoid: str,
        *,
        width: int,
        height: int,
        fov: float,
        yaw: float,
        pitch: float,
    ) -> bytes:
        self.calls.append(
            {
                "panoid": panoid,
                "width": width,
                "height": height,
                "fov": fov,
                "yaw": yaw,
                "pitch": pitch,
            },
        )
        return _jpeg_image()


def test_download_panorama_forwards_explicit_grid_and_dimensions(tmp_path, monkeypatch):
    dataset = StreetViewDataset(tmp_path, workers=1)
    client = RecordingClient()
    monkeypatch.setattr(dataset.client, "image_bytes", client.image_bytes)

    output = dataset.download_panorama(
        "pano",
        tmp_path / "panorama.jpg",
        span=90.0,
        fov=30.0,
        vertical_span=90.0,
        pitch_overlap=0.30,
        view_width=24,
        view_height=24,
        parallel_views=False,
    )

    with Image.open(output) as image:
        assert image.size == (72, 72)
    assert len({call["pitch"] for call in client.calls}) > 1
    assert {call["width"] for call in client.calls} == {24}
    assert {call["height"] for call in client.calls} == {24}


def test_download_half_panorama_forwards_explicit_grid(tmp_path, monkeypatch):
    dataset = StreetViewDataset(tmp_path, workers=1)
    client = RecordingClient()
    monkeypatch.setattr(dataset.client, "image_bytes", client.image_bytes)

    output = dataset.download_half_panorama(
        "pano",
        tmp_path / "half.jpg",
        fov=30.0,
        vertical_span=90.0,
        view_width=24,
        view_height=24,
        parallel_views=False,
    )

    with Image.open(output) as image:
        assert image.size == (144, 72)
    assert len({call["pitch"] for call in client.calls}) > 1
