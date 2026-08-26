import numpy as np

from streetview_dataset.geo import haversine_m, sample_bbox, sample_radius


def test_haversine_zero():
    assert haversine_m(56, 10, 56, 10) == 0


def test_radius_samples_are_inside_radius():
    rng = np.random.default_rng(123)
    center = (56.1629, 10.2039)
    pts = sample_radius(rng, *center, 25, 500)
    assert len(pts) == 500
    assert max(haversine_m(*center, lat, lon) for lat, lon in pts) <= 25_000.001


def test_bbox_samples_are_inside():
    rng = np.random.default_rng(123)
    pts = sample_bbox(rng, 9, 55, 11, 57, 500)
    assert len(pts) == 500
    assert all(55 <= lat <= 57 and 9 <= lon <= 11 for lat, lon in pts)
