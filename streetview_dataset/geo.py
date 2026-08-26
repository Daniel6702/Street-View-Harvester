from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import math

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import contains, points
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def sample_radius(
    rng: np.random.Generator,
    lat: float,
    lon: float,
    radius_km: float,
    n: int,
) -> list[tuple[float, float]]:
    """Uniformly sample by surface area inside a spherical radius."""
    if radius_km <= 0:
        raise ValueError("radius_km must be > 0")
    if n <= 0:
        return []

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    max_delta = radius_km * 1000.0 / EARTH_RADIUS_M

    # Uniform area on a spherical cap: cos(delta) is uniform.
    u = rng.random(n)
    cos_delta = 1.0 - u * (1.0 - math.cos(max_delta))
    delta = np.arccos(np.clip(cos_delta, -1.0, 1.0))
    bearing = rng.uniform(0.0, 2.0 * math.pi, n)

    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_delta, cos_delta_arr = np.sin(delta), np.cos(delta)

    lat2 = np.arcsin(
        sin_lat1 * cos_delta_arr + cos_lat1 * sin_delta * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * sin_delta * cos_lat1,
        cos_delta_arr - sin_lat1 * np.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi

    return list(zip(np.degrees(lat2).tolist(), np.degrees(lon2).tolist()))


def sample_bbox(
    rng: np.random.Generator,
    west: float,
    south: float,
    east: float,
    north: float,
    n: int,
) -> list[tuple[float, float]]:
    """Uniformly sample by surface area inside a latitude/longitude box."""
    if not -90 <= south < north <= 90:
        raise ValueError("Require -90 <= south < north <= 90")
    if not -180 <= west < east <= 180:
        raise ValueError("Require -180 <= west < east <= 180")
    if n <= 0:
        return []

    lon = rng.uniform(west, east, n)
    sin_south = math.sin(math.radians(south))
    sin_north = math.sin(math.radians(north))
    sin_lat = rng.uniform(sin_south, sin_north, n)
    lat = np.degrees(np.arcsin(sin_lat))
    return list(zip(lat.tolist(), lon.tolist()))


@dataclass(frozen=True, slots=True)
class CountryGeometry:
    name: str
    geometry_4326: BaseGeometry
    geometry_equal_area: BaseGeometry

    def contains(self, lat: float, lon: float) -> bool:
        return bool(self.geometry_4326.covers(Point(lon, lat)))


class CountryIndex:
    """Country lookup and equal-area sampling.

    If a ``ne_10m_admin_0_countries.geojson`` file exists in the current working
    directory it is preferred automatically. Otherwise the bundled Natural Earth
    low-resolution fallback is used. A custom file can also be passed explicitly.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            local_high_res = Path("ne_10m_admin_0_countries.geojson")
            if local_high_res.exists():
                path = local_high_res
            else:
                path = Path(__file__).with_name("data") / "naturalearth_lowres.geojson"

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Country boundary file not found: {self.path}")

        gdf = gpd.read_file(self.path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

        self._gdf = gdf
        self._name_columns = [
            c for c in ("ADMIN", "NAME_LONG", "NAME", "name", "SOVEREIGNT") if c in gdf.columns
        ]
        self._code_columns = [
            c for c in ("ISO_A2", "ISO_A3", "ADM0_A3", "iso_a3") if c in gdf.columns
        ]
        if not self._name_columns:
            raise ValueError("Country file must contain a country name column")

        self._to_equal_area = Transformer.from_crs(4326, 6933, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(6933, 4326, always_xy=True)
        self._cache: dict[str, CountryGeometry] = {}

    def _match_rows(self, country: str) -> gpd.GeoDataFrame:
        key = country.strip().casefold()
        mask = np.zeros(len(self._gdf), dtype=bool)

        for column in self._name_columns + self._code_columns:
            values = self._gdf[column].astype(str).str.strip().str.casefold()
            mask |= values.eq(key).to_numpy()

        matched = self._gdf.loc[mask]
        if not matched.empty:
            return matched

        names: list[str] = []
        for column in self._name_columns:
            names.extend(self._gdf[column].dropna().astype(str).tolist())
        suggestions = get_close_matches(country, sorted(set(names)), n=5, cutoff=0.55)
        suffix = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"Unknown country: {country!r}.{suffix}")

    def get(self, country: str) -> CountryGeometry:
        cache_key = country.strip().casefold()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        rows = self._match_rows(country)
        geometry = rows.geometry.union_all()
        canonical_name = str(rows.iloc[0][self._name_columns[0]])

        projected = gpd.GeoSeries([geometry], crs=4326).to_crs(6933).iloc[0]
        result = CountryGeometry(canonical_name, geometry, projected)
        self._cache[cache_key] = result
        return result

    def sample(
        self,
        rng: np.random.Generator,
        country: CountryGeometry,
        n: int,
    ) -> list[tuple[float, float]]:
        """Sample uniformly by area from a country polygon using EPSG:6933."""
        if n <= 0:
            return []

        geom = country.geometry_equal_area
        minx, miny, maxx, maxy = geom.bounds
        accepted_x: list[np.ndarray] = []
        accepted_y: list[np.ndarray] = []
        remaining = n

        # Vectorized rejection sampling. Oversampling adapts reasonably well to
        # irregular countries while keeping the implementation simple.
        rounds = 0
        while remaining > 0:
            rounds += 1
            if rounds > 10_000:
                raise RuntimeError("Could not sample enough points from country geometry")

            batch = max(1024, remaining * 4)
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            mask = contains(geom, points(xs, ys))
            if not np.any(mask):
                continue

            xs_good = xs[mask][:remaining]
            ys_good = ys[mask][:remaining]
            accepted_x.append(xs_good)
            accepted_y.append(ys_good)
            remaining -= len(xs_good)

        x = np.concatenate(accepted_x)
        y = np.concatenate(accepted_y)
        lon, lat = self._to_wgs84.transform(x, y)
        return list(zip(np.asarray(lat).tolist(), np.asarray(lon).tolist()))
