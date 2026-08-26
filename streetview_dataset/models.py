from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Panorama:
    """Metadata for one resolved Street View panorama."""

    panoid: str
    lat: float
    lon: float
    capture_date: date | None = None


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """Small, bounded-memory summary returned by large harvest operations.

    The harvester deliberately does not retain every collected row in memory when
    a dataset root is configured. Call ``to_dataframe()`` only when you explicitly
    want to load the metadata table into RAM.
    """

    target_count: int
    existing_count: int
    added_count: int
    total_count: int
    queries: int
    complete: bool
    metadata_path: Path | None = None
    rows: tuple[dict[str, Any], ...] | None = None

    def to_dataframe(self):
        """Load metadata into a pandas DataFrame on demand."""
        import pandas as pd

        if self.metadata_path is not None and self.metadata_path.exists():
            return pd.read_csv(self.metadata_path)
        return pd.DataFrame(self.rows or ())
