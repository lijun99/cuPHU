"""Build a boolean invalid/water mask from a water-distance raster."""

from __future__ import annotations

import numpy as np

from cuphu._check import check_2d_shape
from cuphu._unwrap import _dilate_mask

__all__ = ["build_water_mask"]


def build_water_mask(
    water_distance,
    *,
    ocean_water_buffer: float | None = None,
    inland_water_buffer: float | None = None,
    water_buffer: int | None = None,
):
    """Return a boolean invalid-pixel mask derived from a water-distance raster.

    Two mutually exclusive modes, selected by which arguments are given:

    Differentiated mode (`ocean_water_buffer` and `inland_water_buffer`,
    both required together): follows the NISAR water-mask convention, where
    `water_distance` encodes 0=land, 1-100=distance from the coastline
    (ocean), 101-200=100+distance from an inland-water boundary. A pixel is
    marked invalid only if its water distance exceeds the corresponding
    buffer -- i.e. the buffers extend the valid/solvable region a short way
    into the water near each boundary. Buffers are in the same real-world
    units as `water_distance` already; no unit conversion is performed here.

    Uniform mode (`water_buffer` only): treats `water_distance` as a plain
    nonzero-is-water indicator (no ocean/inland distinction) and grows the
    valid (non-water) region by `water_buffer` pixels via the same
    dilation used for cuphu's own `mask_buffer`.

    Exactly one of the two modes' arguments must be given.
    """
    check_2d_shape(water_distance, "water_distance")
    water_distance = np.asarray(water_distance)

    differentiated = ocean_water_buffer is not None or inland_water_buffer is not None
    uniform = water_buffer is not None

    if differentiated and uniform:
        raise ValueError(
            "give either ocean_water_buffer/inland_water_buffer or "
            "water_buffer, not both"
        )
    if differentiated and (ocean_water_buffer is None or inland_water_buffer is None):
        raise ValueError(
            "ocean_water_buffer and inland_water_buffer must be given together"
        )
    if not differentiated and not uniform:
        raise ValueError(
            "must give either ocean_water_buffer/inland_water_buffer or "
            "water_buffer"
        )

    if differentiated:
        inland_water_invalid = water_distance > (inland_water_buffer + 100)
        ocean_water_invalid = (water_distance > ocean_water_buffer) & (
            water_distance <= 100
        )
        return inland_water_invalid | ocean_water_invalid

    is_water = water_distance != 0
    valid_grown = _dilate_mask(~is_water, int(water_buffer))
    return ~valid_grown
