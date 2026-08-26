from __future__ import annotations

from io import BytesIO
import math
from typing import Sequence

import numpy as np
from PIL import Image


def planned_yaws(
    *,
    span: float,
    center_yaw: float,
    fov: float,
    overlap: float = 0.20,
    views: int | None = None,
) -> list[float]:
    """Choose perspective-view headings that cover a horizontal angular span."""
    if not 0 < span <= 360:
        raise ValueError("span must be in (0, 360]")
    if not 1 <= fov < 180:
        raise ValueError("fov must be in [1, 180)")
    if not 0 <= overlap < 0.9:
        raise ValueError("overlap must be in [0, 0.9)")

    if views is not None:
        if views < 1:
            raise ValueError("views must be >= 1")
        n = int(views)
    elif math.isclose(span, 360.0):
        n = max(3, math.ceil(span / (fov * (1.0 - overlap))))
    elif span <= fov:
        n = 1
    else:
        usable_step = fov * (1.0 - overlap)
        n = math.ceil((span - fov) / usable_step) + 1

    if math.isclose(span, 360.0):
        return [((center_yaw + i * 360.0 / n) % 360.0) for i in range(n)]
    if n == 1:
        return [center_yaw % 360.0]

    left_center = center_yaw - span / 2.0 + fov / 2.0
    right_center = center_yaw + span / 2.0 - fov / 2.0
    if right_center < left_center:
        left_center = right_center = center_yaw
    return [float(v % 360.0) for v in np.linspace(left_center, right_center, n)]


def decode_rgb(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    u = np.clip(u, 0.0, w - 1.0)
    v = np.clip(v, 0.0, h - 1.0)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (u - x0)[..., None]
    wy = (v - y0)[..., None]
    top = image[y0, x0].astype(np.float32) * (1.0 - wx) + image[y0, x1].astype(np.float32) * wx
    bottom = image[y1, x0].astype(np.float32) * (1.0 - wx) + image[y1, x1].astype(np.float32) * wx
    return top * (1.0 - wy) + bottom * wy


def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, indices.size]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(indices[starts[best]]), int(indices[ends[best] - 1]) + 1


def stitch_views(
    images: Sequence[np.ndarray],
    yaws: Sequence[float],
    *,
    span: float,
    center_yaw: float = 0.0,
    pitch: float = 0.0,
    fov: float = 90.0,
    output_width: int | None = None,
    output_height: int | None = None,
    chunk_rows: int = 128,
    auto_crop: bool = True,
) -> Image.Image:
    """Project perspective images onto an equirectangular strip.

    With ``auto_crop=True`` (default), the result is cropped vertically to the
    largest contiguous band that has real source coverage across every output
    column. This removes the curved/scalloped black regions produced near the
    edges of rectilinear source views.
    """
    if len(images) != len(yaws) or not images:
        raise ValueError("images and yaws must be non-empty and have equal length")
    if not 0 < span <= 360:
        raise ValueError("span must be in (0, 360]")
    if not 1 <= fov < 180:
        raise ValueError("fov must be in [1, 180)")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")

    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("images must be RGB arrays with shape HxWx3")
    src_h, src_w, _ = first.shape
    for image in images:
        if np.asarray(image).shape != first.shape:
            raise ValueError("all source images must have the same dimensions")

    hfov = math.radians(fov)
    tan_h = math.tan(hfov / 2.0)
    vfov = 2.0 * math.atan(tan_h * src_h / src_w)
    tan_v = math.tan(vfov / 2.0)

    if output_width is None:
        output_width = max(1, int(round(src_w * span / fov)))
    if output_height is None:
        output_height = src_h
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output dimensions must be > 0")

    out = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    fully_covered_rows = np.zeros(output_height, dtype=bool)

    x_rel = ((np.arange(output_width, dtype=np.float32) + 0.5) / output_width - 0.5) * math.radians(span)
    yaw_world = math.radians(center_yaw) + x_rel
    sin_yaw = np.sin(yaw_world)[None, :]
    cos_yaw = np.cos(yaw_world)[None, :]
    pitch_top = math.radians(pitch) + vfov / 2.0
    pitch_bottom = math.radians(pitch) - vfov / 2.0
    source_arrays = [np.asarray(image, dtype=np.uint8) for image in images]

    for row0 in range(0, output_height, chunk_rows):
        row1 = min(output_height, row0 + chunk_rows)
        rows = np.arange(row0, row1, dtype=np.float32)
        frac = (rows + 0.5) / output_height
        pitch_world = (pitch_top + frac * (pitch_bottom - pitch_top))[:, None]
        cos_pitch = np.cos(pitch_world)
        world_x = sin_yaw * cos_pitch
        world_y = np.sin(pitch_world) * np.ones_like(sin_yaw)
        world_z = cos_yaw * cos_pitch
        accum = np.zeros((row1 - row0, output_width, 3), dtype=np.float32)
        weights = np.zeros((row1 - row0, output_width), dtype=np.float32)

        for image, source_yaw_deg in zip(source_arrays, yaws):
            yaw0 = math.radians(source_yaw_deg)
            pitch0 = math.radians(pitch)
            sy, cy = math.sin(yaw0), math.cos(yaw0)
            sp, cp = math.sin(pitch0), math.cos(pitch0)
            forward = (sy * cp, sp, cy * cp)
            right = (cy, 0.0, -sy)
            up = (-sp * sy, cp, -sp * cy)
            local_x = world_x * right[0] + world_z * right[2]
            local_y = world_x * up[0] + world_y * up[1] + world_z * up[2]
            local_z = world_x * forward[0] + world_y * forward[1] + world_z * forward[2]
            safe_z = np.where(local_z > 1e-6, local_z, 1.0)
            nx = local_x / safe_z
            ny = local_y / safe_z
            valid = (local_z > 1e-6) & (np.abs(nx) <= tan_h) & (np.abs(ny) <= tan_v)
            if not np.any(valid):
                continue
            u = (nx / tan_h + 1.0) * 0.5 * (src_w - 1)
            v = (1.0 - ny / tan_v) * 0.5 * (src_h - 1)
            sampled = _bilinear(image, u, v)
            edge = np.clip(np.abs(nx) / tan_h, 0.0, 1.0)
            weight = np.cos(edge * (math.pi / 2.0)) ** 2
            weight = np.where(valid, np.maximum(weight, 1e-3), 0.0).astype(np.float32)
            accum += sampled * weight[..., None]
            weights += weight

        valid_out = weights > 0.0
        fully_covered_rows[row0:row1] = np.all(valid_out, axis=1)
        chunk = np.zeros_like(accum, dtype=np.uint8)
        if np.any(valid_out):
            values = accum[valid_out] / weights[valid_out, None]
            chunk[valid_out] = np.clip(values, 0, 255).astype(np.uint8)
        out[row0:row1] = chunk

    if auto_crop:
        crop = _largest_true_run(fully_covered_rows)
        if crop is None:
            raise ValueError(
                "Source views do not fully cover any complete panorama row; "
                "increase views/overlap or use a larger source FOV."
            )
        top, bottom = crop
        out = out[top:bottom]

    return Image.fromarray(out, mode="RGB")
