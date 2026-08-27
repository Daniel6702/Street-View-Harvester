from __future__ import annotations

import csv
import logging
import math
import os
import re
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .client import StreetViewClient
from .geo import CountryIndex, haversine_m, load_geojson, sample_bbox, sample_radius
from .models import HarvestResult, Panorama
from .stitch import (
    _max_yaw_offset,
    decode_rgb,
    planned_pitches,
    planned_yaws,
    stitch_views,
)
from .storage import FileImageStore, ZipShardStore

log = logging.getLogger(__name__)
ImageMode = Literal["none", "flat", "half", "panorama"]
StorageMode = Literal["files", "zip"]

METADATA_COLUMNS = [
    "panoid",
    "pano_lat",
    "pano_lon",
    "pano_date",
    "query_lat",
    "query_lon",
    "snap_distance_m",
    "source",
    "source_value",
    "image_type",
    "yaw",
    "pitch",
    "fov",
    "image_path",
]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned[:180] or "panorama"


class StreetViewDataset:
    """High-level Street View dataset harvester.

    Large persistent harvests are intentionally bounded-memory: collected rows
    are checkpointed to CSV instead of being retained in a Python list. The
    return value is a compact :class:`HarvestResult`; call ``result.to_dataframe()``
    only when you explicitly want the metadata table loaded into RAM.

    Parameters
    ----------
    root:
        Dataset directory. Existing ``metadata.csv`` is used automatically for
        panorama-ID deduplication/resume.
    workers:
        Network worker threads for lookup and flat-image downloads.
    storage:
        ``"files"`` stores ordinary JPEGs in 256 hash-sharded directories.
        ``"zip"`` stores JPEGs in sequential ZIP shards using ``ZIP_STORED``.
    file_sharding:
        Whether ``storage="files"`` places JPEGs in two-character hash
        subdirectories. Enabled by default.
    shard_size:
        Number of images per ZIP shard when ``storage="zip"``.
    panorama_workers:
        Maximum number of stitched panoramas built concurrently. Stitching is
        much more memory-intensive than flat-image downloading, so this is
        deliberately lower than ``workers`` by default.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        workers: int = 32,
        seed: int | None = None,
        country_data: str | Path | None = None,
        lookup_timeout: float = 10.0,
        image_timeout: float = 20.0,
        retries: int = 3,
        checkpoint: int = 100,
        storage: StorageMode = "files",
        file_sharding: bool = True,
        shard_size: int = 2000,
        panorama_workers: int = 4,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if checkpoint < 1:
            raise ValueError("checkpoint must be >= 1")
        if storage not in {"files", "zip"}:
            raise ValueError("storage must be 'files' or 'zip'")
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        if panorama_workers < 1:
            raise ValueError("panorama_workers must be >= 1")

        self.root = Path(root) if root is not None else None
        self.workers = int(workers)
        self.panorama_workers = int(panorama_workers)
        self.checkpoint = int(checkpoint)
        self.storage = storage
        self.file_sharding = file_sharding
        self.shard_size = int(shard_size)
        self.rng = np.random.default_rng(seed)
        self.client = StreetViewClient(
            lookup_timeout=lookup_timeout,
            image_timeout=image_timeout,
            retries=retries,
        )
        self._country_index = CountryIndex(country_data)

        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)

    @property
    def metadata_path(self) -> Path:
        if self.root is None:
            raise ValueError("This StreetViewDataset has no root directory")
        return self.root / "metadata.csv"

    @property
    def images_dir(self) -> Path:
        if self.root is None:
            raise ValueError("This StreetViewDataset has no root directory")
        return self.root / "images"

    def metadata(self) -> pd.DataFrame:
        """Explicitly load the full metadata CSV into RAM."""
        if self.root is None or not self.metadata_path.exists():
            return pd.DataFrame(columns=METADATA_COLUMNS)
        return pd.read_csv(self.metadata_path)

    def nearest(
        self,
        lat: float,
        lon: float,
        *,
        max_distance_m: float | None = None,
    ) -> Panorama | None:
        """Resolve the nearest panorama to a specific point."""
        pano = self.client.find_nearest(lat, lon)
        if pano is None:
            return None
        if max_distance_m is not None:
            distance = haversine_m(lat, lon, pano.lat, pano.lon)
            if distance > max_distance_m:
                return None
        return pano

    def download_view(
        self,
        pano: Panorama | str,
        out_path: str | Path,
        *,
        width: int = 1024,
        height: int = 1024,
        fov: float = 90.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> Path:
        """Download one ordinary flat perspective view."""
        panoid = pano.panoid if isinstance(pano, Panorama) else pano
        path = self._resolve_output_path(out_path)

        return self.client.download_image(
            panoid,
            path,
            width=width,
            height=height,
            fov=fov,
            yaw=yaw,
            pitch=pitch,
        )
    
    def _resolve_output_path(self, out_path: str | Path) -> Path:
        path = Path(out_path)

        if not path.is_absolute() and self.root is not None:
            path = self.root / path

        return path

    def _panorama_image(
        self,
        panoid: str,
        *,
        span: float,
        center_yaw: float,
        pitch: float,
        fov: float,
        view_width: int,
        view_height: int,
        overlap: float,
        views: int | None,
        output_width: int | None,
        output_height: int | None,
        parallel_views: bool,
        auto_crop: bool,
        vertical_span: float | None = None,
        pitch_overlap: float = 0.30,
    ) -> Image.Image:
        scalar_yaws = planned_yaws(
            span=span,
            center_yaw=center_yaw,
            fov=fov,
            overlap=overlap,
            views=views,
        )
        if vertical_span is None:
            planned_views: Sequence[float | tuple[float, float]] = scalar_yaws
        else:
            hfov = math.radians(fov)
            vfov = math.degrees(2.0 * math.atan(math.tan(hfov / 2.0) * view_height / view_width))
            pitches = planned_pitches(
                span=vertical_span,
                center_pitch=pitch,
                vfov=vfov,
                overlap=pitch_overlap,
                max_yaw_offset=_max_yaw_offset(
                    scalar_yaws,
                    span=span,
                    center_yaw=center_yaw,
                ),
            )
            planned_views = [(yaw, source_pitch) for source_pitch in pitches for yaw in scalar_yaws]

        def fetch(view: float | tuple[float, float]):
            if isinstance(view, tuple):
                yaw, source_pitch = view
            else:
                yaw, source_pitch = view, pitch
            data = self.client.image_bytes(
                panoid,
                width=view_width,
                height=view_height,
                fov=fov,
                yaw=yaw,
                pitch=source_pitch,
            )
            return view, decode_rgb(data)

        by_view: dict[float | tuple[float, float], np.ndarray] = {}
        if parallel_views and len(planned_views) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(planned_views))) as executor:
                futures = [executor.submit(fetch, view) for view in planned_views]
                for future in as_completed(futures):
                    source_view, image = future.result()
                    by_view[source_view] = image
        else:
            for source_view in planned_views:
                fetched_view, image = fetch(source_view)
                by_view[fetched_view] = image

        return stitch_views(
            [by_view[source_view] for source_view in planned_views],
            planned_views,
            span=span,
            center_yaw=center_yaw,
            pitch=pitch,
            fov=fov,
            output_width=output_width,
            output_height=output_height,
            vertical_span=vertical_span,
            auto_crop=auto_crop,
        )

    def download_panorama(
        self,
        pano: Panorama | str,
        out_path: str | Path,
        *,
        span: float = 360.0,
        center_yaw: float = 0.0,
        pitch: float = 0.0,
        fov: float = 90.0,
        view_width: int = 1024,
        view_height: int = 1024,
        overlap: float = 0.20,
        views: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        jpeg_quality: int = 95,
        parallel_views: bool = True,
        auto_crop: bool = True,
        vertical_span: float | None = None,
        pitch_overlap: float = 0.30,
    ) -> Path:
        """Download and stitch a 1-360 degree panorama strip."""
        panoid = pano.panoid if isinstance(pano, Panorama) else pano
        stitched = self._panorama_image(
            panoid,
            span=span,
            center_yaw=center_yaw,
            pitch=pitch,
            fov=fov,
            view_width=view_width,
            view_height=view_height,
            overlap=overlap,
            views=views,
            output_width=output_width,
            output_height=output_height,
            parallel_views=parallel_views,
            auto_crop=auto_crop,
            vertical_span=vertical_span,
            pitch_overlap=pitch_overlap,
        )

        path = self._resolve_output_path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
        image_format = formats.get(suffix)
        if image_format is None:
            raise ValueError("Panorama output must use .jpg, .jpeg, .png, or .webp")

        temp = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
        save_kwargs = {"quality": jpeg_quality, "subsampling": 0} if image_format == "JPEG" else {}
        try:
            stitched.save(temp, format=image_format, **save_kwargs)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def download_half_panorama(self, pano: Panorama | str, out_path: str | Path, **kwargs) -> Path:
        """Convenience wrapper for a 180-degree stitched panorama."""
        kwargs.pop("span", None)
        return self.download_panorama(pano, out_path, span=180.0, **kwargs)

    def read_image_bytes(self, image_path: str) -> bytes:
        """Read an image reference from metadata, including ZIP-shard references.

        For high-throughput training, keep ZIP handles open in the training
        loader instead of calling this helper for every sample.
        """
        if self.root is None:
            raise ValueError("Reading stored images requires a dataset root")
        if "::" not in image_path:
            return (self.root / image_path).read_bytes()
        archive, member = image_path.split("::", 1)
        with zipfile.ZipFile(self.root / archive, "r") as zf:
            return zf.read(member)

    def country(
        self,
        country: str,
        count: int,
        *,
        download: ImageMode = "none",
        yaw: float | Literal["random"] = "random",
        pitch: float = 0.0,
        fov: float = 90.0,
        width: int = 1024,
        height: int = 1024,
        max_queries: int | None = None,
        progress: bool | None = None,
    ) -> HarvestResult:
        """Collect random unique panoramas whose resolved coordinates are in a country."""
        geometry = self._country_index.get(country)

        def sampler(n: int):
            return self._country_index.sample(self.rng, geometry, n)

        def valid(pano: Panorama):
            return geometry.contains(pano.lat, pano.lon)

        return self._collect(
            count,
            sampler=sampler,
            validator=valid,
            source="country",
            source_value=geometry.name,
            download=download,
            yaw=yaw,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            max_queries=max_queries,
            progress=progress,
        )

    def radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        count: int,
        *,
        download: ImageMode = "none",
        yaw: float | Literal["random"] = "random",
        pitch: float = 0.0,
        fov: float = 90.0,
        width: int = 1024,
        height: int = 1024,
        max_queries: int | None = None,
        progress: bool | None = None,
    ) -> HarvestResult:
        """Collect random unique panoramas within a distance of a point."""

        def sampler(n: int):
            return sample_radius(self.rng, lat, lon, radius_km, n)

        def valid(pano: Panorama):
            return haversine_m(lat, lon, pano.lat, pano.lon) <= radius_km * 1000.0

        return self._collect(
            count,
            sampler=sampler,
            validator=valid,
            source="radius",
            source_value=f"{lat},{lon},{radius_km}km",
            download=download,
            yaw=yaw,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            max_queries=max_queries,
            progress=progress,
        )

    def bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        count: int,
        *,
        download: ImageMode = "none",
        yaw: float | Literal["random"] = "random",
        pitch: float = 0.0,
        fov: float = 90.0,
        width: int = 1024,
        height: int = 1024,
        max_queries: int | None = None,
        progress: bool | None = None,
    ) -> HarvestResult:
        """Collect random unique panoramas inside a latitude/longitude box."""

        def sampler(n: int):
            return sample_bbox(self.rng, west, south, east, north, n)

        def valid(pano: Panorama):
            return west <= pano.lon <= east and south <= pano.lat <= north

        return self._collect(
            count,
            sampler=sampler,
            validator=valid,
            source="bbox",
            source_value=f"{west},{south},{east},{north}",
            download=download,
            yaw=yaw,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            max_queries=max_queries,
            progress=progress,
        )

    def geojson(
        self,
        path: str | Path,
        count: int,
        *,
        download: ImageMode = "none",
        yaw: float | Literal["random"] = "random",
        pitch: float = 0.0,
        fov: float = 90.0,
        width: int = 1024,
        height: int = 1024,
        max_queries: int | None = None,
        progress: bool | None = None,
    ) -> HarvestResult:
        """Collect random unique panoramas inside polygonal GeoJSON."""
        geometry = load_geojson(path)

        def sampler(n: int) -> list[tuple[float, float]]:
            return geometry.sample(self.rng, n)

        def valid(pano: Panorama) -> bool:
            return geometry.covers(pano.lat, pano.lon)

        return self._collect(
            count,
            sampler=sampler,
            validator=valid,
            source="geojson",
            source_value=str(path),
            download=download,
            yaw=yaw,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            max_queries=max_queries,
            progress=progress,
        )

    def _load_seen(self, download: ImageMode) -> tuple[set[str], int]:
        """Stream metadata instead of constructing a second large DataFrame."""
        if self.root is None or not self.metadata_path.exists():
            return set(), 0
        seen: set[str] = set()
        count = 0
        with self.metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "panoid" not in reader.fieldnames:
                return set(), 0
            for row in reader:
                panoid = row.get("panoid")
                if not panoid:
                    continue
                if download != "none":
                    image_path = row.get("image_path")
                    if row.get("image_type") != download or not image_path:
                        continue
                    stored_path = self.root / image_path.split("::", 1)[0]
                    if not stored_path.is_file():
                        continue
                count += 1
                seen.add(panoid)
        return seen, count

    def _append_rows(self, rows: list[dict]) -> None:
        if self.root is None or not rows:
            return
        path = self.metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
            if header:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

    def _filename(self, pano: Panorama, mode: ImageMode) -> str:
        safe = _safe_id(pano.panoid)
        if mode == "flat":
            return f"{safe}.jpg"
        if mode == "half":
            return f"{safe}_half.jpg"
        if mode == "panorama":
            return f"{safe}_360.jpg"
        raise ValueError("mode does not produce an image")

    def _render_image_bytes(
        self,
        pano: Panorama,
        mode: ImageMode,
        *,
        yaw: float,
        pitch: float,
        fov: float,
        width: int,
        height: int,
    ) -> bytes:
        if mode == "flat":
            return self.client.image_bytes(
                pano.panoid,
                width=width,
                height=height,
                fov=fov,
                yaw=yaw,
                pitch=pitch,
            )

        if mode == "half":
            span = 180.0
            center_yaw = yaw
        elif mode == "panorama":
            span = 360.0
            center_yaw = 0.0
        else:
            raise ValueError("mode does not produce an image")

        image = self._panorama_image(
            pano.panoid,
            span=span,
            center_yaw=center_yaw,
            pitch=pitch,
            fov=fov,
            view_width=width,
            view_height=height,
            overlap=0.20,
            views=None,
            output_width=None,
            output_height=None,
            # Dataset-level panorama workers already provide concurrency. Avoid
            # nested executors multiplying requests and RAM usage.
            parallel_views=False,
            auto_crop=True,
        )
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95, subsampling=0)
        return buffer.getvalue()

    def _collect(
        self,
        count: int,
        *,
        sampler: Callable[[int], list[tuple[float, float]]],
        validator: Callable[[Panorama], bool],
        source: str,
        source_value: str,
        download: ImageMode,
        yaw: float | Literal["random"],
        pitch: float,
        fov: float,
        width: int,
        height: int,
        max_queries: int | None,
        progress: bool | None,
    ) -> HarvestResult:
        if count < 1:
            raise ValueError("count must be >= 1")
        if download not in {"none", "flat", "half", "panorama"}:
            raise ValueError("download must be one of: none, flat, half, panorama")
        if download != "none" and self.root is None:
            raise ValueError("download != 'none' requires StreetViewDataset(root=...)")
        if yaw != "random":
            yaw = float(yaw)

        seen, existing_count = self._load_seen(download)
        needed = max(0, count - existing_count) if self.root is not None else count
        if needed == 0:
            return HarvestResult(
                target_count=count,
                existing_count=existing_count,
                added_count=0,
                total_count=existing_count,
                queries=0,
                complete=True,
                metadata_path=self.metadata_path if self.root is not None else None,
            )

        if max_queries is None:
            max_queries = max(1000, needed * 25)
        if max_queries < needed:
            raise ValueError("max_queries must be >= the number of requested new samples")

        # Persistent datasets never retain all rows in RAM. For the uncommon
        # root=None case, rows are retained because there is nowhere else to put
        # the result and the caller explicitly chose an ephemeral collection.
        ephemeral_rows: list[dict] | None = [] if self.root is None else None
        pending_save: list[dict] = []
        queries = 0
        added = 0
        batch_size = max(64, self.workers * 8)

        file_store = None
        zip_store = None
        if download != "none" and self.root is not None:
            if self.storage == "files":
                file_store = FileImageStore(self.root, file_sharding=self.file_sharding)
            else:
                zip_store = ZipShardStore(self.root, self.shard_size)

        def commit_rows(rows: list[dict]) -> None:
            nonlocal pending_save
            if not rows:
                return
            if self.root is None:
                assert ephemeral_rows is not None
                ephemeral_rows.extend(rows)
                return
            pending_save.extend(rows)
            if len(pending_save) >= self.checkpoint:
                self._append_rows(pending_save)
                pending_save = []

        def lookup(query_lat: float, query_lon: float):
            try:
                return query_lat, query_lon, self.client.find_nearest(query_lat, query_lon)
            except Exception as exc:
                log.debug("Lookup failed at %.6f, %.6f: %s", query_lat, query_lon, exc)
                return query_lat, query_lon, None

        lookup_executor = ThreadPoolExecutor(max_workers=self.workers)
        image_workers = self.workers if download == "flat" else min(self.workers, self.panorama_workers)
        download_executor = ThreadPoolExecutor(max_workers=max(1, image_workers)) if download != "none" else None
        interrupted = False
        progress_bar = tqdm(
            total=count,
            initial=existing_count,
            unit="panorama" if download == "none" else "image",
            disable=not progress if progress is not None else None,
        )

        try:
            while added < needed and queries < max_queries:
                remaining_queries = max_queries - queries
                n = min(batch_size, remaining_queries)
                coords = sampler(n)
                queries += len(coords)
                futures = [lookup_executor.submit(lookup, lat, lon) for lat, lon in coords]
                batch_new: list[tuple[dict, Panorama, float]] = []
                batch_capacity = needed - added

                for future in as_completed(futures):
                    query_lat, query_lon, pano = future.result()
                    if len(batch_new) >= batch_capacity:
                        continue
                    if pano is None or pano.panoid in seen or not validator(pano):
                        continue

                    seen.add(pano.panoid)
                    chosen_yaw = float(self.rng.uniform(0.0, 360.0)) if yaw == "random" else float(yaw)
                    row = {
                        "panoid": pano.panoid,
                        "pano_lat": pano.lat,
                        "pano_lon": pano.lon,
                        "pano_date": pano.capture_date.isoformat() if pano.capture_date else None,
                        "query_lat": query_lat,
                        "query_lon": query_lon,
                        "snap_distance_m": round(haversine_m(query_lat, query_lon, pano.lat, pano.lon), 3),
                        "source": source,
                        "source_value": source_value,
                        "image_type": download,
                        "yaw": chosen_yaw if download in {"flat", "half"} else None,
                        "pitch": pitch if download != "none" else None,
                        "fov": fov if download != "none" else None,
                        "image_path": None,
                    }
                    batch_new.append((row, pano, chosen_yaw))

                if download == "none":
                    rows = [row for row, _, _ in batch_new]
                    commit_rows(rows)
                    added += len(rows)
                    if rows:
                        progress_bar.update(len(rows))
                else:
                    assert download_executor is not None
                    download_futures = {
                        download_executor.submit(
                            self._render_image_bytes,
                            pano,
                            download,
                            yaw=chosen_yaw,
                            pitch=pitch,
                            fov=fov,
                            width=width,
                            height=height,
                        ): (row, pano)
                        for row, pano, chosen_yaw in batch_new
                    }

                    for future in as_completed(download_futures):
                        row, pano = download_futures[future]
                        data = future.result()
                        filename = self._filename(pano, download)
                        if file_store is not None:
                            stored = file_store.store(pano.panoid, filename, data)
                            row["image_path"] = stored.image_path
                            commit_rows([row])
                        else:
                            assert zip_store is not None
                            finalized_rows = zip_store.add(filename, data, row)
                            # ZIP metadata is committed only after the shard
                            # has been closed and atomically published.
                            commit_rows(finalized_rows)
                        added += 1
                        progress_bar.update()

                log.info(
                    "Collected %d/%d new panoramas (%d queries, %d total unique IDs seen)",
                    added,
                    needed,
                    queries,
                    len(seen),
                )

        except KeyboardInterrupt:
            interrupted = True
            log.warning("Harvest interrupted; finalizing completed work")
        finally:
            try:
                lookup_executor.shutdown(wait=True, cancel_futures=True)
                if download_executor is not None:
                    download_executor.shutdown(wait=True, cancel_futures=True)

                if zip_store is not None:
                    try:
                        # Graceful interruption publishes the final short shard.
                        commit_rows(zip_store.finalize())
                    except Exception:
                        zip_store.abort()
                        raise

                if pending_save:
                    self._append_rows(pending_save)
                    pending_save = []
            finally:
                progress_bar.close()

        total_count = existing_count + added if self.root is not None else added
        complete = added >= needed and not interrupted
        if not complete and queries >= max_queries:
            log.warning(
                "Stopped after %d queries with %d/%d requested new panoramas; "
                "increase max_queries if coverage is sparse.",
                queries,
                added,
                needed,
            )

        return HarvestResult(
            target_count=count,
            existing_count=existing_count,
            added_count=added,
            total_count=total_count,
            queries=queries,
            complete=complete,
            metadata_path=self.metadata_path if self.root is not None else None,
            rows=tuple(ephemeral_rows) if ephemeral_rows is not None else None,
        )
