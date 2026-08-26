# streetview-dataset

A deliberately small Python package for creating Street View image datasets using the same endpoints as the original script.

It supports:

- nearest panorama to a specific `(lat, lon)`
- random unique panoramas inside a country
- random unique panoramas inside a radius
- random unique panoramas inside a bounding box
- safe threaded lookup/download with retries and connection reuse
- CSV checkpointing + automatic panorama-ID deduplication/resume
- flat perspective images
- stitched 180° half panoramas
- stitched 360° horizontal panoramas

There is intentionally no database layer, provider abstraction, GUI, or distributed worker system.

## Install

From the package directory:

```bash
pip install -e .
```

## Basic use

```python
from streetview_dataset import StreetViewDataset

sv = StreetViewDataset(
    "datasets/denmark",
    workers=32,
    seed=42,
)

# Collect metadata only.
sv.country("Denmark", 10_000)
```

The dataset is stored as:

```text
datasets/denmark/
├── metadata.csv
└── images/
```

File storage uses two-character hash subdirectories under `images/` by default.
Set `file_sharding=False` to write all file-backed images directly to `images/`:

```python
sv = StreetViewDataset(
    "datasets/denmark",
    storage="files",
    file_sharding=False,
)
```

If `metadata.csv` already exists, its panorama IDs are loaded automatically. Calling the same command again resumes toward the requested total count.

## Country + images

Flat images with random headings:

```python
sv.country(
    "Denmark",
    10_000,
    download="flat",
    width=1024,
    height=1024,
    fov=90,
)
```

Half panoramas:

```python
sv.country(
    "Denmark",
    10_000,
    download="half",
)
```

Full horizontal panoramas:

```python
sv.country(
    "Denmark",
    10_000,
    download="panorama",
)
```

`yaw="random"` is the default for flat and half-panorama dataset samples. You can instead use a fixed heading:

```python
sv.country("Denmark", 1000, download="flat", yaw=90)
```

## Radius

```python
sv.radius(
    lat=56.1629,
    lon=10.2039,
    radius_km=25,
    count=5000,
    download="flat",
)
```

Candidate points are uniform by surface area within the radius, and the resolved panorama itself is checked to ensure it is still inside the requested radius.

## Bounding box

```python
sv.bbox(
    west=9.0,
    south=55.0,
    east=11.0,
    north=57.0,
    count=5000,
)
```

## Specific point

```python
pano = sv.nearest(56.1629, 10.2039)
print(pano)
```

Optionally reject results that resolve too far from the requested point:

```python
pano = sv.nearest(56.1629, 10.2039, max_distance_m=100)
```

Download one ordinary view:

```python
sv.download_view(
    pano,
    "view.jpg",
    yaw=45,
    pitch=0,
    fov=90,
    width=1024,
    height=1024,
)
```

## 180° / 360° panorama stitching

```python
sv.download_half_panorama(
    pano,
    "half.jpg",
    center_yaw=90,
    view_width=1024,
    view_height=1024,
)
```

```python
sv.download_panorama(
    pano,
    "full.jpg",
    span=360,
    view_width=1024,
    view_height=1024,
)
```

The stitcher does **not** use feature matching. It knows the yaw, pitch, and FOV used for each downloaded perspective image, projects them onto a common equirectangular strip, and feathers overlaps. This is more deterministic than `cv2.Stitcher` for this use case.

The automatic number of source views is based on the requested FOV and a 20% overlap. You can override it:

```python
sv.download_panorama(pano, "full.jpg", views=8)
```

You can also request any horizontal span from 1° to 360°:

```python
sv.download_panorama(
    pano,
    "wide.jpg",
    span=240,
    center_yaw=180,
)
```

## Country boundaries

The package prefers a file named:

```text
ne_10m_admin_0_countries.geojson
```

in your working directory, matching the file used by the original script. If it is not present, a bundled lower-resolution Natural Earth country dataset is used so the package still works standalone.

You can explicitly supply a country file:

```python
sv = StreetViewDataset(
    "dataset",
    country_data="/path/to/ne_10m_admin_0_countries.geojson",
)
```

Country sampling uses an equal-area projection (`EPSG:6933`), rather than Web Mercator, to avoid latitude-dependent sampling bias.

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

The query coordinate is retained separately from the actual panorama coordinate, which is useful for checking how much nearest-panorama lookup changes the intended sampling distribution.

## Sparse regions

By default the harvester allows up to roughly 25 lookup attempts per requested new panorama (with a minimum of 1000 total attempts). For sparse areas you can increase this:

```python
sv.country("Some Country", 1000, max_queries=100_000)
```

## Logging

The package uses Python logging rather than hard-coded prints:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Compatibility with the original helpers

The original-style functions are also exported:

```python
from streetview_dataset import (
    find_nearest_streetview,
    download_streetview_image,
)
```

`find_nearest_streetview()` returns the original four-value tuple.
