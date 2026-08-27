# streetview-dataset

A deliberately small Python package for creating Street View image datasets using the same endpoints as the original script.

It supports:

- nearest panorama to a specific `(lat, lon)`
- random unique panoramas inside a country, radius, bounding box, or GeoJSON polygon
- safe threaded lookup/download with retries and connection reuse
- tqdm progress reporting, CSV checkpointing, and panorama-ID deduplication/resume
- flat perspective images, stitched 180° half panoramas, and stitched 360° panoramas

There is intentionally no database layer, provider abstraction, GUI, or distributed worker system.

## Install

```bash
pip install -e .
```

## Basic use

```python
from streetview_dataset import StreetViewDataset

sv = StreetViewDataset("datasets/denmark", workers=32, seed=42)
sv.country("Denmark", 10_000)
```

The dataset is stored as:

```text
datasets/denmark/
├── metadata.csv
└── images/
```

File storage uses two-character hash subdirectories under `images/` by default. Set `file_sharding=False` to write ordinary files directly to `images/`:

```python
sv = StreetViewDataset("datasets/denmark", storage="files", file_sharding=False)
```

## Harvest progress

Each aggregate harvest (`country`, `radius`, `bbox`, and `geojson`) displays one tqdm progress bar in an interactive terminal. It shows accepted panoramas or downloaded images out of the requested target, plus tqdm's default rate display.

```python
sv.geojson("boundaries/aarhus.geojson", count=5_000, download="flat")
```

Use `progress=True` to force a bar when output is redirected, `progress=False` to disable it, or leave the default `progress=None` for TTY-aware behavior. The bar resumes from existing metadata and is updated only by the coordinator thread, so it is safe with threaded lookup/rendering and file or ZIP storage. For ZIP output, an item advances once it has been written to the active shard; the shard is atomically published at finalization.

If `metadata.csv` already exists, its panorama IDs are loaded automatically. Calling the same command again resumes toward the requested total count.

## Country + images

```python
sv.country("Denmark", 10_000, download="flat", width=1024, height=1024, fov=90)
sv.country("Denmark", 10_000, download="half")
sv.country("Denmark", 10_000, download="panorama")
```

`yaw="random"` is the default for flat and half-panorama dataset samples. Use a fixed heading with `sv.country("Denmark", 1000, download="flat", yaw=90)`.

## Radius

```python
sv.radius(lat=56.1629, lon=10.2039, radius_km=25, count=5000, download="flat")
```

Candidate points are uniform by surface area within the radius, and the resolved panorama itself is checked to ensure it remains inside the requested radius.

## Bounding box

```python
sv.bbox(west=9.0, south=55.0, east=11.0, north=57.0, count=5000)
```

## GeoJSON polygon

```python
sv.geojson("boundaries/aarhus.geojson", count=5000, download="flat")
```

The input may be a bare `Polygon` or `MultiPolygon`, a `Feature`, or a `FeatureCollection`. Polygonal features in a collection are unioned; holes and multipart areas are preserved. GeoJSON coordinates use `(longitude, latitude)` order. The geometry is normalized to EPSG:4326 and sampled uniformly by area in EPSG:6933. A panorama is kept only when its resolved location is covered by the geometry, including its boundary. Antimeridian-crossing polygons are not supported.

Resume and panorama-ID deduplication are root-wide: `metadata.csv` in the dataset root is shared by all selectors. Use a separate root when independent regions must each reach their own target count.

## Specific point

```python
pano = sv.nearest(56.1629, 10.2039)
print(pano)

sv.download_view(pano, "view.jpg", yaw=45, pitch=0, fov=90, width=1024, height=1024)
```

Optionally reject results that resolve too far from the requested point:

```python
pano = sv.nearest(56.1629, 10.2039, max_distance_m=100)
```

## 180° / 360° panorama stitching

```python
sv.download_half_panorama(pano, "half.jpg", center_yaw=90, view_width=1024, view_height=1024)
sv.download_panorama(pano, "full.jpg", span=360, view_width=1024, view_height=1024)
```

The stitcher uses the requested yaw, pitch, and FOV to project perspective images onto a common equirectangular strip. The automatic source-view count uses 20% overlap; override it with `views=8` if needed.

For a low-FOV panorama with full vertical coverage, opt into a pitch grid:

```python
sv.download_panorama(
    pano,
    "detailed.jpg",
    span=360,
    fov=30,
    vertical_span=90.0,
    pitch_overlap=0.30,
    view_width=1024,
    view_height=1024,
)
```

`vertical_span=None` preserves the legacy single-pitch behavior.

## Country boundaries

The package prefers `ne_10m_admin_0_countries.geojson` in the current working directory, then falls back to bundled low-resolution Natural Earth data. Supply another boundary file with:

```python
sv = StreetViewDataset("dataset", country_data="/path/to/ne_10m_admin_0_countries.geojson")
```

Country sampling uses EPSG:6933 rather than Web Mercator to avoid latitude-dependent sampling bias.

## Metadata columns

`metadata.csv` contains:

```text
panoid
pano_lat
pano_lon
pano_date
query_lat
query_lon
snap_distance_m
source
source_value
image_type
yaw
pitch
fov
image_path
```

The query coordinate is retained separately from the resolved panorama coordinate.

## Sparse regions

The default allows roughly 25 lookup attempts per requested new panorama, with a minimum of 1000 total attempts. Increase this for sparse coverage:

```python
sv.country("Some Country", 1000, max_queries=100_000)
```

## Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Compatibility helpers

```python
from streetview_dataset import find_nearest_streetview, download_streetview_image
```

`find_nearest_streetview()` returns the original four-value tuple.
