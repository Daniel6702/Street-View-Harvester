from itertools import pairwise

import numpy as np
import pytest

from streetview_dataset.stitch import planned_pitches, planned_yaws, stitch_views


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


def test_legacy_scalar_pitch_keeps_sizing_and_auto_crop_behavior():
    images = [np.full((32, 32, 3), 200, dtype=np.uint8)] * 4

    uncropped = stitch_views(
        images,
        [0.0, 90.0, 180.0, 270.0],
        span=360.0,
        fov=100.0,
        output_width=128,
        output_height=32,
        auto_crop=False,
    )
    cropped = stitch_views(
        images,
        [0.0, 90.0, 180.0, 270.0],
        span=360.0,
        fov=100.0,
        output_width=128,
        output_height=32,
        auto_crop=True,
    )

    assert uncropped.size == (128, 32)
    assert cropped.size[0] == uncropped.size[0]
    assert cropped.size[1] < uncropped.size[1]


def test_planned_pitches_cover_requested_span_with_conservative_overlap():
    pitches = planned_pitches(span=90.0, center_pitch=0.0, vfov=30.0, max_yaw_offset=12.0)

    assert pitches[0] - 15.0 <= -45.0
    assert pitches[-1] + 15.0 >= 45.0
    assert all(right > left for left, right in pairwise(pitches))


@pytest.mark.parametrize(
    ("span", "center_pitch", "vfov", "overlap"),
    [
        (0.0, 0.0, 30.0, 0.30),
        (90.0, 50.0, 30.0, 0.30),
        (90.0, 0.0, float("nan"), 0.30),
        (90.0, 0.0, 30.0, 1.0),
    ],
)
def test_planned_pitches_reject_invalid_ranges(
    span: float,
    center_pitch: float,
    vfov: float,
    overlap: float,
):
    with pytest.raises(ValueError):
        planned_pitches(
            span=span,
            center_pitch=center_pitch,
            vfov=vfov,
            overlap=overlap,
        )


def test_pair_yaws_require_explicit_vertical_span():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="vertical_span"):
        stitch_views([image], [(0.0, 0.0)], span=30.0, fov=30.0)


def test_explicit_pair_views_preserve_requested_output_height():
    images = [np.full((32, 32, 3), value, dtype=np.uint8) for value in (192, 128, 64)]
    pairs = [(0.0, 30.0), (0.0, 0.0), (0.0, -30.0)]

    result = stitch_views(
        images,
        pairs,
        span=30.0,
        vertical_span=90.0,
        fov=60.0,
        output_width=32,
        output_height=96,
    )

    assert result.size == (32, 96)
    array = np.asarray(result)
    assert array[4].mean() > array[48].mean() > array[91].mean()


def test_explicit_low_fov_grid_has_no_uncovered_pixels():
    yaws = planned_yaws(span=360.0, center_yaw=0.0, fov=30.0, overlap=0.20)
    pitches = planned_pitches(span=90.0, center_pitch=0.0, vfov=30.0, max_yaw_offset=12.0)
    pairs = [(yaw, pitch) for pitch in pitches for yaw in yaws]
    images = [
        np.full((32, 32, 3), 150 - int(pitch), dtype=np.uint8)
        for _, pitch in pairs
    ]

    result = stitch_views(
        images,
        pairs,
        span=360.0,
        vertical_span=90.0,
        fov=30.0,
        output_width=384,
        output_height=96,
    )

    array = np.asarray(result)
    assert np.all(array > 0)
    assert array[4].mean() < array[48].mean() < array[91].mean()


def test_explicit_fov45_half_grid_covers_yaw_edge_pixels():
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    yaws = planned_yaws(span=180.0, center_yaw=0.0, fov=45.0, overlap=0.20)
    pitches = planned_pitches(span=90.0, center_pitch=0.0, vfov=45.0, max_yaw_offset=22.5)
    pairs = [(yaw, pitch) for pitch in pitches for yaw in yaws]

    result = stitch_views(
        [image] * len(pairs),
        pairs,
        span=180.0,
        vertical_span=90.0,
        fov=45.0,
        output_width=1024,
        output_height=512,
    )

    assert np.all(np.asarray(result) > 0)


def test_explicit_fov45_half_grid_covers_yaw_edges_without_pitch_overlap():
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    yaws = planned_yaws(span=180.0, center_yaw=0.0, fov=45.0, overlap=0.20)
    pitches = planned_pitches(
        span=90.0,
        center_pitch=0.0,
        vfov=45.0,
        overlap=0.0,
        max_yaw_offset=22.5,
    )
    pairs = [(yaw, pitch) for pitch in pitches for yaw in yaws]

    result = stitch_views(
        [image] * len(pairs),
        pairs,
        span=180.0,
        vertical_span=90.0,
        fov=45.0,
        output_width=1024,
        output_height=512,
    )

    assert np.all(np.asarray(result) > 0)


def test_explicit_vertical_span_equal_to_vfov_covers_yaw_edges():
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    yaws = planned_yaws(span=180.0, center_yaw=0.0, fov=45.0, overlap=0.20)
    pitches = planned_pitches(
        span=45.0,
        center_pitch=0.0,
        vfov=45.0,
        overlap=0.0,
        max_yaw_offset=22.5,
    )
    pairs = [(yaw, pitch) for pitch in pitches for yaw in yaws]

    result = stitch_views(
        [image] * len(pairs),
        pairs,
        span=180.0,
        vertical_span=45.0,
        fov=45.0,
        output_width=1024,
        output_height=256,
    )

    assert np.all(np.asarray(result) > 0)


def test_explicit_off_center_positive_pitch_span_covers_yaw_edges():
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    yaws = planned_yaws(span=180.0, center_yaw=0.0, fov=45.0, overlap=0.20)
    pitches = planned_pitches(
        span=45.0,
        center_pitch=45.0,
        vfov=45.0,
        overlap=0.0,
        max_yaw_offset=22.5,
    )
    pairs = [(yaw, pitch) for pitch in pitches for yaw in yaws]

    result = stitch_views(
        [image] * len(pairs),
        pairs,
        span=180.0,
        center_yaw=0.0,
        pitch=45.0,
        vertical_span=45.0,
        fov=45.0,
        output_width=1024,
        output_height=256,
    )

    assert np.all(np.asarray(result) > 0)


def test_explicit_off_center_negative_pitch_span_covers_yaw_edges():
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    yaws = planned_yaws(span=180.0, center_yaw=0.0, fov=45.0, overlap=0.20)
    pitches = planned_pitches(
        span=45.0,
        center_pitch=-45.0,
        vfov=45.0,
        overlap=0.0,
        max_yaw_offset=22.5,
    )
    pairs = [(yaw, pitch) for pitch in pitches for yaw in yaws]

    result = stitch_views(
        [image] * len(pairs),
        pairs,
        span=180.0,
        center_yaw=0.0,
        pitch=-45.0,
        vertical_span=45.0,
        fov=45.0,
        output_width=1024,
        output_height=256,
    )

    assert np.all(np.asarray(result) > 0)


def test_explicit_stitch_fails_when_target_is_uncovered():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="cover"):
        stitch_views(
            [image],
            [(0.0, 0.0)],
            span=180.0,
            vertical_span=120.0,
            fov=90.0,
            output_width=64,
            output_height=64,
        )
