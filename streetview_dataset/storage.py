from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import re
import zipfile


@dataclass(frozen=True, slots=True)
class StoredImage:
    """Reference written into metadata.csv."""

    image_path: str


class FileImageStore:
    """Store images as ordinary files, optionally sharded into hash directories."""

    def __init__(self, root: Path, *, file_sharding: bool = True) -> None:
        self.root = root
        self.images_dir = root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.file_sharding = file_sharding

    def store(self, panoid: str, filename: str, data: bytes) -> StoredImage:
        directory = self.images_dir
        if self.file_sharding:
            bucket = hashlib.sha1(panoid.encode("utf-8")).hexdigest()[:2]
            directory /= bucket
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        temp = path.with_name(path.name + f".tmp-{os.getpid()}")
        try:
            temp.write_bytes(data)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return StoredImage(str(path.relative_to(self.root)))


class ZipShardStore:
    """Single-writer ZIP shard storage using ZIP_STORED.

    A shard is written to ``.partial`` and atomically renamed only after it is
    closed. Metadata rows for that shard are returned only after finalization,
    so metadata never intentionally references an unfinished ZIP.
    """

    _SHARD_RE = re.compile(r"^shard_(\d{6})\.zip$")

    def __init__(self, root: Path, shard_size: int = 2000) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.root = root
        self.images_dir = root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)

        # A hard crash may leave one unfinished archive. It contains no committed
        # metadata and is therefore safe to discard on resume.
        for partial in self.images_dir.glob("shard_*.zip.partial"):
            partial.unlink(missing_ok=True)

        self._index = self._next_index()
        self._zip: zipfile.ZipFile | None = None
        self._temp_path: Path | None = None
        self._final_path: Path | None = None
        self._count = 0
        self._rows: list[dict] = []

    def _next_index(self) -> int:
        maximum = -1
        for path in self.images_dir.glob("shard_*.zip"):
            match = self._SHARD_RE.match(path.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def _open(self) -> None:
        if self._zip is not None:
            return
        name = f"shard_{self._index:06d}.zip"
        self._final_path = self.images_dir / name
        self._temp_path = self.images_dir / f"{name}.partial"
        self._zip = zipfile.ZipFile(
            self._temp_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
        self._count = 0
        self._rows = []

    def add(self, filename: str, data: bytes, row: dict) -> list[dict]:
        """Add one image; return metadata rows if this completed a shard."""
        self._open()
        assert self._zip is not None
        assert self._final_path is not None

        self._zip.writestr(filename, data, compress_type=zipfile.ZIP_STORED)
        row["image_path"] = (
            f"{self._final_path.relative_to(self.root)}::{filename}"
        )
        self._rows.append(row)
        self._count += 1

        if self._count >= self.shard_size:
            return self.finalize()
        return []

    def finalize(self) -> list[dict]:
        """Close and atomically publish the current shard."""
        if self._zip is None:
            return []

        assert self._temp_path is not None
        assert self._final_path is not None
        self._zip.close()
        self._zip = None
        self._temp_path.replace(self._final_path)

        rows = self._rows
        self._rows = []
        self._count = 0
        self._index += 1
        self._temp_path = None
        self._final_path = None
        return rows

    def abort(self) -> None:
        """Discard only the currently unfinished shard."""
        if self._zip is not None:
            self._zip.close()
            self._zip = None
        if self._temp_path is not None:
            self._temp_path.unlink(missing_ok=True)
        self._rows = []
        self._count = 0
        self._temp_path = None
        self._final_path = None
