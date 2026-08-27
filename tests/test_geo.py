import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import shape

from streetview_dataset.geo import (
    GeoJSONGeometry,
    haversine_m,
    load_geojson,
    sample_bbox,
    sample_radius,
)


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


@pytest.mark.parametrize("wrapper", ["geometry", "feature", "collection"])
def test_load_geojson_accepts_polygon_wrappers(tmp_path: Path, wrapper: str):
    polygon = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
    }
    if wrapper == "geometry":
        document = polygon
    elif wrapper == "feature":
        document = {"type": "Feature", "properties": {}, "geometry": polygon}
    else:
        document = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": polygon},
            ],
        }
    path = tmp_path / f"{wrapper}.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    geometry = load_geojson(path)

    assert isinstance(geometry, GeoJSONGeometry)
    assert geometry.geometry_4326.equals(shape(polygon))
    assert all(
        geometry.covers(lat, lon)
        for lat, lon in geometry.sample(np.random.default_rng(4), 20)
    )


def test_load_geojson_accepts_bare_multipolygon(tmp_path: Path):
    document = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[3, 0], [4, 0], [4, 1], [3, 1], [3, 0]]],
        ],
    }
    path = tmp_path / "multipolygon.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    geometry = load_geojson(path)

    assert geometry.covers(0.5, 0.5)
    assert geometry.covers(0.5, 3.5)
    assert not geometry.covers(0.5, 2.0)


def test_load_geojson_uses_longitude_latitude_coordinate_order(tmp_path: Path):
    document = {
        "type": "Polygon",
        "coordinates": [[[10, 50], [14, 50], [14, 52], [10, 52], [10, 50]]],
    }
    path = tmp_path / "coordinate-order.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    geometry = load_geojson(path)

    assert geometry.covers(51, 12)
    assert not geometry.covers(12, 51)


def test_geojson_sampling_preserves_holes_and_multipart(tmp_path: Path):
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                        [[1, 1], [1, 3], [3, 3], [3, 1], [1, 1]],
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 0], [11, 0], [11, 1], [10, 1], [10, 0]]],
                },
            },
        ],
    }
    path = tmp_path / "multipart.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    geometry = load_geojson(path)
    points = geometry.sample(np.random.default_rng(8), 1000)

    assert len(points) == 1000
    assert all(geometry.covers(lat, lon) for lat, lon in points)
    assert all(not (1 < lon < 3 and 1 < lat < 3) for lat, lon in points)
    assert any(lon > 9 for _, lon in points)


def test_geojson_sampling_is_area_weighted(tmp_path: Path):
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2, 0], [5, 0], [5, 3], [2, 3], [2, 0]]],
                },
            },
        ],
    }
    path = tmp_path / "weighted.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    points = load_geojson(path).sample(np.random.default_rng(12), 2000)

    large_polygon_points = sum(lon > 2 for _, lon in points)
    assert 0.75 < large_polygon_points / len(points) < 0.95


def test_load_geojson_normalizes_declared_crs(tmp_path: Path):
    document = {
        "type": "Feature",
        "properties": {},
        "crs": {"type": "name", "properties": {"code": 3857}},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [111319, 0], [111319, 111325], [0, 111325], [0, 0]]],
        },
    }
    path = tmp_path / "projected.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    geometry = load_geojson(path)

    assert geometry.geometry_4326.bounds == pytest.approx((0, 0, 1, 1), abs=1e-5)


def test_load_geojson_rejects_unsupported_declared_crs(tmp_path: Path):
    document = {
        "type": "Polygon",
        "crs": {"type": "name", "properties": {"href": "https://example.test/crs"}},
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    path = tmp_path / "unsupported-crs.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="name or numeric code"):
        load_geojson(path)


def test_load_geojson_rejects_antimeridian_crossing(tmp_path: Path):
    document = {
        "type": "Polygon",
        "coordinates": [[[179, 10], [-179, 10], [-179, 11], [179, 11], [179, 10]]],
    }
    path = tmp_path / "dateline.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="antimeridian"):
        load_geojson(path)


@pytest.mark.parametrize(
    "document",
    [
        {"type": "Feature", "properties": {}, "geometry": None},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"type": "Polygon", "coordinates": []},
        {"type": "FeatureCollection", "features": []},
    ],
)
def test_load_geojson_rejects_empty_or_non_polygon(tmp_path: Path, document):
    path = tmp_path / "invalid.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        load_geojson(path)


def test_load_geojson_rejects_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_geojson(tmp_path / "missing.geojson")
