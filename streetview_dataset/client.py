from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import random
import re
import threading
import time

import requests

from .models import Panorama


_LOOKUP_RE = re.compile(
    r'\[\d+,"([^"]+)"\].+?\[\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)',
    re.S,
)
_DATE_RE = re.compile(r'\[(20\d{2}),(\d{1,2})\]')


class StreetViewClient:
    """HTTP client for the two endpoints used by the original script.

    A separate requests.Session is lazily created per worker thread so TCP
    connections are reused without sharing mutable Session state across threads.
    """

    def __init__(
        self,
        *,
        lookup_timeout: float = 10.0,
        image_timeout: float = 20.0,
        retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.lookup_timeout = float(lookup_timeout)
        self.image_timeout = float(image_timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": "streetview-dataset/0.1"})
            self._local.session = session
        return session

    def _get(self, url: str, *, timeout: float) -> requests.Response:
        last_error: Exception | None = None
        attempts = self.retries + 1

        for attempt in range(attempts):
            try:
                response = self._session().get(url, timeout=timeout)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                delay = self.backoff * (2**attempt)
                delay += random.uniform(0.0, max(0.05, delay * 0.2))
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    def find_nearest(self, lat: float, lon: float) -> Panorama | None:
        """Resolve the nearest panorama to ``lat, lon`` using the original endpoint."""
        url = (
            "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
            f"?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}!2d50!"
            "3m10!2m2!1sen!2sUS!9m1!1e2!11m4!1m3!1e2!2b1!3e2!"
            "4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2"
            "&callback=_xdc_._x"
        )
        response = self._get(url, timeout=self.lookup_timeout)
        match = _LOOKUP_RE.search(response.text)
        if not match:
            return None

        panoid = match.group(1)
        pano_lat = float(match.group(2))
        pano_lon = float(match.group(3))

        capture_date: date | None = None
        date_matches = _DATE_RE.findall(response.text)
        if date_matches:
            valid_dates: list[date] = []
            for year, month in date_matches:
                month_int = int(month)
                if 1 <= month_int <= 12:
                    valid_dates.append(date(int(year), month_int, 1))
            if valid_dates:
                capture_date = max(valid_dates)

        return Panorama(panoid, pano_lat, pano_lon, capture_date)

    def image_bytes(
        self,
        panoid: str,
        *,
        width: int = 1024,
        height: int = 1024,
        fov: float = 90.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> bytes:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if not 1.0 <= fov < 180.0:
            raise ValueError("fov must be in [1, 180)")

        url = (
            "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
            f"?panoid={panoid}&cb_client=maps_sv.tactile.gps"
            f"&w={int(width)}&h={int(height)}&yaw={float(yaw):.6f}"
            f"&pitch={float(pitch):.6f}&thumbfov={float(fov):g}"
        )
        response = self._get(url, timeout=self.image_timeout)
        return response.content

    def download_image(
        self,
        panoid: str,
        out_path: str | Path,
        *,
        width: int = 1024,
        height: int = 1024,
        fov: float = 90.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> Path:
        """Download one flat perspective image, using an atomic final rename."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.image_bytes(
            panoid,
            width=width,
            height=height,
            fov=fov,
            yaw=yaw,
            pitch=pitch,
        )

        temp_path = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            temp_path.write_bytes(data)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return path


def find_nearest_streetview(lat: float, lon: float):
    """Compatibility wrapper matching the tuple returned by the original script."""
    pano = StreetViewClient().find_nearest(lat, lon)
    if pano is None:
        return None, None, None, None
    return pano.panoid, pano.lat, pano.lon, pano.capture_date


def download_streetview_image(
    id: str,
    out_path: Path,
    width: int = 1024,
    height: int = 1024,
    fov: float = 90,
    yaw: float = 0.0,
    pitch: float = 0.0,
):
    """Compatibility wrapper matching the original function name/signature."""
    return StreetViewClient().download_image(
        id,
        out_path,
        width=width,
        height=height,
        fov=fov,
        yaw=yaw,
        pitch=pitch,
    )
