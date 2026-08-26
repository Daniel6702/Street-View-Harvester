import numpy as np

from streetview_dataset.stitch import planned_yaws, stitch_views


def test_planned_full_yaws_cover_multiple_views():
    yaws = planned_yaws(span=360, center_yaw=0, fov=90, overlap=0.2)
    assert len(yaws) >= 4
    assert all(0 <= yaw < 360 for yaw in yaws)


def test_planned_half_yaws():
    yaws = planned_yaws(span=180, center_yaw=90, fov=90, overlap=0.2)
    assert len(yaws) >= 2


def test_stitch_synthetic_images():
    # Solid-color views are enough to verify geometry/shape and that the blend
    # produces non-empty output without relying on network access.
    images = [
        np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8),
        np.full((64, 64, 3), (0, 255, 0), dtype=np.uint8),
        np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8),
        np.full((64, 64, 3), (255, 255, 0), dtype=np.uint8),
    ]
    result = stitch_views(images, [0, 90, 180, 270], span=360, fov=100, output_width=256, output_height=64)
    arr = np.asarray(result)
    assert arr.shape == (64, 256, 3)
    assert arr.max() > 0
