from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import contains, points
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


@dataclass(frozen=True, slots=True)
class EqualAreaSampler:
    geometry: BaseGeometry
    to_wgs84: Transformer

    def sample(self, rng: np.random.Generator, n: int) -> list[tuple[float, float]]:
        if n <= 0:
            return []

        minx, miny, maxx, maxy = self.geometry.bounds
        accepted_x: list[np.ndarray] = []
        accepted_y: list[np.ndarray] = []
        remaining = n
        rounds = 0

        while remaining > 0:
            rounds += 1
            if rounds > 10_000:
                raise RuntimeError("Could not sample enough points from GeoJSON geometry")

            batch = max(1024, remaining * 4)
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            mask = contains(self.geometry, points(xs, ys))
            if not np.any(mask):
                continue

            xs_good = xs[mask][:remaining]
            ys_good = ys[mask][:remaining]
            accepted_x.append(xs_good)
            accepted_y.append(ys_good)
            remaining -= len(xs_good)

        x = np.concatenate(accepted_x)
        y = np.concatenate(accepted_y)
        lon, lat = self.to_wgs84.transform(x, y)
        return list(zip(np.asarray(lat).tolist(), np.asarray(lon).tolist()))


@dataclass(frozen=True, slots=True)
class GeoJSONGeometry:
    """A polygonal GeoJSON geometry in WGS84 and equal-area coordinates."""

    geometry_4326: BaseGeometry
    geometry_equal_area: BaseGeometry
    _to_wgs84: Transformer

    def covers(self, lat: float, lon: float) -> bool:
        """Return whether a WGS84 latitude/longitude lies in the geometry."""
        return bool(self.geometry_4326.covers(Point(lon, lat)))

    def sample(self, rng: np.random.Generator, n: int) -> list[tuple[float, float]]:
        """Sample WGS84 latitude/longitude pairs uniformly by projected area."""
        return EqualAreaSampler(self.geometry_equal_area, self._to_wgs84).sample(rng, n)


def _polygonal_geometries(value: JSONValue) -> list[Polygon | MultiPolygon]:
    if not isinstance(value, dict):
        return []

    geometry_type = value.get("type")
    if not isinstance(geometry_type, str):
        return []

    if geometry_type in {"Polygon", "MultiPolygon"}:
        coordinates = value.get("coordinates")
        if not isinstance(coordinates, list) or not coordinates:
            return []
        parsed = shape(value)
        if isinstance(parsed, (Polygon, MultiPolygon)) and not parsed.is_empty:
            return [parsed]
        return []

    if geometry_type == "Feature":
        geometry = value.get("geometry")
        return _polygonal_geometries(geometry) if isinstance(geometry, dict) else []

    if geometry_type == "FeatureCollection":
        features = value.get("features")
        if not isinstance(features, list):
            return []
        polygons: list[Polygon | MultiPolygon] = []
        for feature in features:
            polygons.extend(_polygonal_geometries(feature))
        return polygons

    return []


def _source_crs(value: JSONValue) -> str | int:
    if not isinstance(value, dict):
        return 4326
    crs = value.get("crs")
    if crs is None:
        return 4326
    if not isinstance(crs, dict):
        raise ValueError("GeoJSON CRS must be an object")
    properties = crs.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("GeoJSON CRS must contain properties")
    name = properties.get("name")
    if isinstance(name, str):
        return name
    code = properties.get("code")
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    raise ValueError("GeoJSON CRS must contain a name or numeric code")


def _crosses_antimeridian(geometry: Polygon | MultiPolygon) -> bool:
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else (geometry,)
    for polygon in polygons:
        rings = (polygon.exterior, *polygon.interiors)
        for ring in rings:
            longitudes = np.asarray(ring.coords)[:, 0]
            if np.any(np.abs(np.diff(longitudes)) > 180.0):
                return True
    return False


def load_geojson(path: str | Path) -> GeoJSONGeometry:
    """Load polygonal GeoJSON and prepare WGS84 and EPSG:6933 geometries."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {source}")

    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("GeoJSON file is empty")

    document = json.loads(text)
    polygons = _polygonal_geometries(document)
    if not polygons:
        raise ValueError("GeoJSON must contain a non-empty Polygon or MultiPolygon")

    source_crs = _source_crs(document)
    geometry_4326 = gpd.GeoSeries(polygons, crs=source_crs).to_crs(4326).union_all()
    if geometry_4326.is_empty or not isinstance(geometry_4326, (Polygon, MultiPolygon)):
        raise ValueError("GeoJSON must contain a non-empty Polygon or MultiPolygon")
    if _crosses_antimeridian(geometry_4326):
        raise ValueError("GeoJSON polygons crossing the antimeridian are not supported")

    geometry_equal_area = gpd.GeoSeries([geometry_4326], crs=4326).to_crs(6933).iloc[0]
    to_wgs84 = Transformer.from_crs(6933, 4326, always_xy=True)
    return GeoJSONGeometry(geometry_4326, geometry_equal_area, to_wgs84)
